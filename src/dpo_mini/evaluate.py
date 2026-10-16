from __future__ import annotations
import torch
import numpy as np
from .loss import dpo_loss
from .scoring import model_scores
from .tokenization import collate


@torch.no_grad()
def evaluate(model, reference, cache, pairs: list[dict], cfg, pad_id: int) -> dict:
    rows = []
    for start in range(0, len(pairs), cfg.microbatch_pairs):
        part = pairs[start:start + cfg.microbatch_pairs]
        batch = collate(part, pad_id, cfg.device)
        n = len(part)
        p = model_scores(model, batch)
        r = cache.scores(part, cfg.device) if cache is not None else model_scores(reference, batch)
        losses = dpo_loss(p[:n], p[n:], r[:n], r[n:], cfg.beta, "none")
        margins = cfg.beta * ((p[:n] - r[:n]) - (p[n:] - r[n:]))
        for i, pair in enumerate(part):
            rows.append({"id": pair["id"], "loss": float(losses[i]), "reward_margin": float(margins[i]),
                         "policy_chosen": float(p[i]), "policy_rejected": float(p[i + n]),
                         "reference_chosen": float(r[i]), "reference_rejected": float(r[i + n])})
    margins = np.array([r["reward_margin"] for r in rows])
    wins = int((margins > cfg.tie_tolerance).sum())
    ties = int((np.abs(margins) <= cfg.tie_tolerance).sum())
    return {"pairs": len(rows), "dpo_loss": float(np.mean([r["loss"] for r in rows])),
            "strict_positive_accuracy": wins / len(rows), "tie_rate": ties / len(rows),
            "tie_adjusted_accuracy": (wins + 0.5 * ties) / len(rows),
            "tie_tolerance": cfg.tie_tolerance, "margin_mean": float(margins.mean()),
            "margin_quantiles": np.quantile(margins, [0, .25, .5, .75, 1]).tolist(), "examples": rows}


@torch.no_grad()
def generate(model, tokenizer, pairs: list[dict], cfg, max_new_tokens: int = 24) -> list[dict]:
    rows = []
    for pair in pairs[:8]:
        ids = list(pair["prompt_ids"])
        out = []
        limit = min(max_new_tokens, cfg.max_length - len(ids))
        for _ in range(limit):
            t = torch.tensor([ids], dtype=torch.long, device=cfg.device)
            result = model(input_ids=t, attention_mask=torch.ones_like(t))
            logits = result if isinstance(result, torch.Tensor) else result.logits
            token = int(logits[0, -1].argmax())
            ids.append(token)
            out.append(token)
            if token == tokenizer.eos_token_id:
                break
        rows.append({"id": pair["id"], "prompt": pair["prompt"], "token_ids": out,
                     "text": tokenizer.decode(out), "decoding": "greedy argmax", "max_new_tokens": limit,
                     "stop": "assistant EOS or length cap", "seed": cfg.seed})
    return rows
