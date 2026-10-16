"""Deterministic synthetic fixture and conservative UltraFeedback preprocessing."""
from __future__ import annotations
import collections
import random
import unicodedata
from .provenance import digest
from .tokenization import ByteTokenizer, tokenize_pair


def normalized_prompt(prompt: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", prompt).split()).casefold()


def identify(prompt: str, chosen: str, rejected: str) -> dict:
    return {"id": digest([prompt, chosen, rejected]), "prompt_hash": digest(normalized_prompt(prompt)),
            "prompt": prompt, "chosen": chosen, "rejected": rejected}


def synthetic(cfg) -> tuple[dict, dict]:
    result = {}
    cursor = 0
    for split, n in [("train", cfg.train_pairs), ("validation", cfg.validation_pairs), ("test", cfg.test_pairs)]:
        result[split] = []
        for i in range(cursor, cursor + n):
            # Distinct numeric prompt identities; known preferred/rejected rule.
            row = identify(f"Item {i:03d}: say yes.", "yes", "no" if i % 2 == 0 else "never")
            result[split].append(tokenize_pair(row, ByteTokenizer(), cfg.max_length))
        cursor += n
    return result, manifest(result, cfg, {"generated": cursor, "excluded": 0}, "synthetic-v1")


def extract_single_turn(row: dict) -> dict:
    c, r = row.get("chosen"), row.get("rejected")
    if not isinstance(c, list) or not isinstance(r, list) or len(c) != 2 or len(r) != 2:
        raise ValueError("unsupported conversation structure")
    if c[0] != r[0] or c[0].get("role") != "user" or any(x[-1].get("role") != "assistant" for x in [c, r]):
        raise ValueError("inconsistent context or roles")
    prompt, chosen, rejected = c[0].get("content"), c[1].get("content"), r[1].get("content")
    if any(not isinstance(x, str) or not x.strip() for x in [prompt, chosen, rejected]):
        raise ValueError("empty text")
    if chosen.strip() == rejected.strip():
        raise ValueError("identical completions")
    if row.get("prompt") is not None and normalized_prompt(row["prompt"]) != normalized_prompt(prompt):
        raise ValueError("prompt field mismatch")
    for ck, rk in [("score_chosen", "score_rejected"), ("chosen_score", "rejected_score")]:
        if ck in row and rk in row:
            import math
            if not all(isinstance(row[x], (int, float)) and math.isfinite(row[x]) for x in [ck, rk]):
                raise ValueError("invalid preference scores")
            if row[ck] <= row[rk]:
                raise ValueError("non-strict or reversed scores")
    return identify(prompt, chosen, rejected)


def prepare_pretrained(raw_splits: dict, tokenizer, cfg) -> tuple[dict, dict]:
    stats = collections.Counter()
    accepted = {}
    for split in ["test_prefs", "train_prefs"]:
        seen = set()
        rows = []
        for raw in raw_splits[split]:
            stats[split + ":original"] += 1
            try:
                row = extract_single_turn(raw)
                if row["id"] in seen:
                    raise ValueError("duplicate example")
                seen.add(row["id"])
                rows.append(tokenize_pair(row, tokenizer, cfg.max_length, cfg.profile == "standard"))
            except ValueError as e:
                stats[split + ":excluded:" + str(e)] += 1
        accepted[split] = rows
        stats[split + ":eligible"] = len(rows)
    # Protect all official test prompt groups, including over-length examples.
    test_groups = set()
    for raw in raw_splits["test_prefs"]:
        try:
            test_groups.add(extract_single_turn(raw)["prompt_hash"])
        except ValueError:
            pass
    pool = []
    for row in accepted["train_prefs"]:
        if row["prompt_hash"] in test_groups:
            stats["train_prefs:excluded:test prompt conflict"] += 1
        else:
            pool.append(row)
    # One example per normalized prompt throughout: conservative group split.
    def unique_groups(rows):
        seen = set()
        out = []
        for row in rows:
            if row["prompt_hash"] not in seen:
                seen.add(row["prompt_hash"])
                out.append(row)
            else:
                stats["excluded:duplicate prompt group"] += 1
        return out
    rng = random.Random(cfg.split_seed)
    pool = unique_groups(sorted(pool, key=lambda x: x["id"]))
    test = unique_groups(sorted(accepted["test_prefs"], key=lambda x: x["id"]))
    rng.shuffle(pool)
    rng.shuffle(test)
    nval = min(cfg.validation_pairs, max(1, len(pool) // 5))
    result = {"validation": pool[:nval], "train": pool[nval:nval + cfg.train_pairs],
              "test": test[:cfg.test_pairs]}
    if any(not rows for rows in result.values()):
        raise ValueError("Insufficient eligible examples for all splits")
    return result, manifest(result, cfg, dict(stats), cfg.dataset_revision)


def manifest(splits: dict, cfg, statistics: dict, revision: str) -> dict:
    import numpy as np
    groups = {k: {r["prompt_hash"] for r in v} for k, v in splits.items()}
    keys = list(groups)
    assert all(not groups[a] & groups[b] for i, a in enumerate(keys) for b in keys[i + 1:])
    detail = {}
    for k, rows in splits.items():
        lengths = [len(r[label + "_ids"]) for r in rows for label in ["chosen", "rejected"]]
        detail[k] = {"count_pairs": len(rows), "ids": [r["id"] for r in rows],
                     "split_hash": digest(rows), "prompt_groups": sorted(groups[k]),
                     "lengths_min_median_max": [min(lengths), float(np.median(lengths)), max(lengths)]}
    return {"dataset_revision": revision, "split_seed": cfg.split_seed, "preprocessing_version": "v1",
            "normalization": "Unicode NFKC, whitespace collapse, casefold",
            "split_protocol": "official test protected; one pair per normalized prompt group; fixed shuffle",
            "tokenization": "generation prompt + suffix-encoded completion + one assistant EOS; no truncation",
            "statistics": statistics, "splits": detail, "max_length": cfg.max_length}
