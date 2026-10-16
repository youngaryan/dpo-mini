"""Independent optimizer loop; checkpoints occur only at accumulation boundaries."""
from __future__ import annotations
import math
import os
import random
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from .loss import dpo_loss
from .scoring import model_scores
from .tokenization import collate


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Trainer:
    def __init__(self, model, reference, cache, cfg, pairs: list[dict], pad_id: int,
                 total_updates: int | None = None):
        if (reference is None) == (cache is None):
            raise ValueError("Choose exactly one reference instance or verified cache")
        self.model, self.reference, self.cache = model, reference, cache
        self.cfg, self.pairs, self.pad_id = cfg, pairs, pad_id
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        self.total_updates = total_updates or math.ceil(len(pairs) / cfg.effective_batch_pairs)
        self.warmup = max(1, math.ceil(0.03 * self.total_updates))
        self.update_index = self.position = self.reference_forward_calls = self.policy_forward_calls = 0
        self.trace = []
        self.guard = None
        self.deadline = None
        self.progress_callback = None

    def reference_scores(self, pairs: list[dict], batch: dict) -> torch.Tensor:
        if self.cache is not None:
            return self.cache.scores(pairs, self.cfg.device)
        self.reference_forward_calls += 1
        with torch.no_grad():
            return model_scores(self.reference, batch)

    def update(self, pairs: list[dict]) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        response_tokens = 0
        for start in range(0, len(pairs), self.cfg.microbatch_pairs):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise TimeoutError("Budget exhausted between preference microbatches")
            if self.guard is not None:
                self.guard()
            part = pairs[start:start + self.cfg.microbatch_pairs]
            batch = collate(part, self.pad_id, self.cfg.device)
            scores = model_scores(self.model, batch)
            self.policy_forward_calls += 1
            ref = self.reference_scores(part, batch)
            n = len(part)
            loss = dpo_loss(scores[:n], scores[n:], ref[:n], ref[n:], self.cfg.beta)
            (loss * n / len(pairs)).backward()
            loss_sum += loss.detach().item() * n
            response_tokens += int(batch["completion_mask"].sum().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        if self.update_index < self.warmup:
            scale = (self.update_index + 1) / self.warmup
        else:
            scale = (self.total_updates - self.update_index) / max(1, self.total_updates - self.warmup)
        lr = self.cfg.learning_rate * max(0.0, scale)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.update_index += 1
        self.position += len(pairs)
        row = {"update": self.update_index, "pairs": len(pairs), "loss": loss_sum / len(pairs),
               "response_tokens": response_tokens, "learning_rate": lr, "grad_norm": float(grad_norm)}
        self.trace.append(row)
        if self.progress_callback is not None:
            self.progress_callback(self)
        return row

    def run(self, max_updates: int | None = None, deadline: float | None = None) -> list[dict]:
        self.deadline = deadline
        while self.position < len(self.pairs):
            if max_updates is not None and self.update_index >= max_updates:
                break
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Experiment time budget exhausted at update boundary")
            self.update(self.pairs[self.position:self.position + self.cfg.effective_batch_pairs])
        return self.trace

    def save(self, path: Path, identity: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"identity": identity, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": {"total_updates": self.total_updates, "warmup": self.warmup}, "scaler": None,
                 "update_index": self.update_index, "data_position": self.position, "trace": self.trace,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        os.close(fd)
        try:
            with open(name, "wb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def restore(self, path: Path, identity: dict) -> None:
        # Only project-generated, identity-checked checkpoints; never remote pickle files.
        state = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint manifest/configuration mismatch")
        if state["scheduler"] != {"total_updates": self.total_updates, "warmup": self.warmup}:
            raise ValueError("Checkpoint scheduler does not match planned updates")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_index, self.position, self.trace = state["update_index"], state["data_position"], state["trace"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
