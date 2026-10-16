"""Atomic ID-addressed reference caches with complete token/mask identity."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from .provenance import atomic_json, digest, read_json
from .scoring import model_scores
from .tokenization import collate


def token_identity(pair: dict) -> str:
    return digest({k: pair[k] for k in ["id", "chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask"]})


def cache_context(cfg, reference_hash: str, tokenizer) -> dict:
    return {"reference_state_sha256": reference_hash, "model": cfg.model_id, "revision": cfg.revision,
            "tokenizer_revision": cfg.revision, "template": getattr(tokenizer, "chat_template", "byte-v1"),
            "enable_thinking": False if cfg.profile == "standard" else None,
            "ending_id": tokenizer.eos_token_id, "ending_policy": "exactly one assistant end token scored",
            "preprocessing": "v1:no-truncation:suffix-boundary", "weight_dtype": cfg.weight_dtype,
            "compute_dtype": cfg.compute_dtype, "reduction_dtype": cfg.reduction_dtype,
            "attention_backend": cfg.attention_backend, "device": cfg.device,
            "microbatch_pairs": cfg.microbatch_pairs, "max_length": cfg.max_length}


class ReferenceCache:
    def __init__(self, payload: dict, pairs: list[dict], context: dict):
        if len({p["id"] for p in pairs}) != len(pairs):
            raise ValueError("Duplicate IDs in expected cache inputs")
        if payload.get("version") != 1 or payload.get("context") != context:
            raise ValueError("Stale or incompatible cache context")
        rows = payload.get("rows", [])
        if payload.get("sha256") != digest({"context": context, "rows": rows}):
            raise ValueError("Corrupt cache checksum")
        if len(rows) != len(pairs) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Missing or duplicate cache IDs")
        self.rows = {r["id"]: r for r in rows}
        for pair in pairs:
            row = self.rows.get(pair["id"])
            if row is None or row["token_sha256"] != token_identity(pair):
                raise ValueError("Missing ID or changed token/mask content")
            if any(not math.isfinite(row[k]) for k in ["chosen", "rejected"]):
                raise ValueError("Nonfinite cached reference score")
        self.context = context

    def scores(self, pairs: list[dict], device: str) -> torch.Tensor:
        for p in pairs:
            if p["id"] not in self.rows or token_identity(p) != self.rows[p["id"]]["token_sha256"]:
                raise ValueError("Cache lookup ID/content mismatch")
        dtype = getattr(torch, self.context["reduction_dtype"])
        return torch.tensor([self.rows[p["id"]][label] for label in ["chosen", "rejected"] for p in pairs],
                            dtype=dtype, device=device)

    @classmethod
    def load(cls, path: Path, pairs: list[dict], context: dict):
        if not path.is_file() or path.suffix == ".tmp":
            raise ValueError("Reference cache missing or interrupted write")
        try:
            return cls(read_json(path), pairs, context)
        except (KeyError, TypeError, __import__("json").JSONDecodeError) as e:
            raise ValueError("Malformed or partial reference cache") from e


@torch.no_grad()
def build_cache(path: Path, reference, pairs: list[dict], context: dict, pad_id: int = 0,
                device: str = "cpu", microbatch: int = 1) -> ReferenceCache:
    if len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate IDs while constructing cache")
    rows = []
    for start in range(0, len(pairs), microbatch):
        part = pairs[start:start + microbatch]
        scores = model_scores(reference, collate(part, pad_id, device)).cpu().tolist()
        for i, pair in enumerate(part):
            rows.append({"id": pair["id"], "token_sha256": token_identity(pair),
                         "chosen": scores[i], "rejected": scores[i + len(part)]})
    payload = {"version": 1, "context": context, "rows": rows,
               "sha256": digest({"context": context, "rows": rows})}
    cache = ReferenceCache(payload, pairs, context)
    atomic_json(path, payload)
    return cache
