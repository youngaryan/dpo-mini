"""Pinned PEFT reference strategy and explicitly scoped tensor identities."""
from contextlib import contextmanager
import hashlib
import math
import time
import torch
from ..provenance import canonical
from ..scoring import model_scores
from ..loss import dpo_loss
from ..tokenization import collate
from ..train import Trainer


def tensor_digest(tensors):
    """All supplied names, including aliases, with shape/dtype and contiguous bytes."""
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        header = canonical([name, list(value.shape), str(value.dtype)])
        h.update(len(header).to_bytes(8, 'big'))
        h.update(header)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(raw).to_bytes(8, 'big'))
        h.update(raw)
    return h.hexdigest()


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if 'lora_' in n}


def base_state(model):
    # remove_duplicate=False explicitly covers tied aliases, plus nonpersistent buffers.
    return {**{n: p for n, p in model.named_parameters(remove_duplicate=False) if 'lora_' not in n},
            **{'BUFFER:' + n: b for n, b in model.named_buffers(remove_duplicate=False)}}


def restore_adapter(model, state):
    target = dict(model.named_parameters())
    if set(state) != {n for n in target if 'lora_' in n}:
        raise ValueError('Adapter fixture names differ')
    with torch.no_grad():
        for n, p in state.items():
            target[n].copy_(p)


@contextmanager
def disabled_reference(model):
    """Restore PEFT enablement, modes, flags and existing gradient objects on all exits.

    Single, unmerged LoRA adapter only. The base is frozen. Reference execution
    precedes policy graph construction in StrategyTrainer.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    parameters = list(model.parameters())
    modules = list(model.modules())
    layers = [m for m in modules if isinstance(m, BaseTunerLayer)]
    if not layers or any(m.merged for m in layers):
        raise ValueError('Shared reference requires unmerged LoRA layers')
    if any(p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n):
        raise ValueError('Shared base must be frozen')
    flags = [p.requires_grad for p in parameters]
    grads = [p.grad for p in parameters]
    modes = [m.training for m in modules]
    enabled = [not m.disable_adapters for m in layers]
    active = [list(m.active_adapters) for m in layers]
    try:
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            yield model
    finally:
        for layer, adapters, on in zip(layers, active, enabled):
            layer.set_adapter(adapters)
            layer.enable_adapters(on)
        for p, requires, grad in zip(parameters, flags, grads):
            p.requires_grad_(requires)
            p.grad = grad
        for m, mode in zip(modules, modes):
            m.training = mode


class SharedReference:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, **kwargs):
        with disabled_reference(self.policy) as base:
            return base(**kwargs)


class StrategyTrainer(Trainer):
    """Same optimizer and accumulation as historical Trainer; reference first."""
    def update(self, pairs):
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        response_tokens = 0
        for start in range(0, len(pairs), self.cfg.microbatch_pairs):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise TimeoutError('Experiment budget exhausted')
            if self.guard is not None:
                self.guard()
            part = pairs[start:start + self.cfg.microbatch_pairs]
            batch = collate(part, self.pad_id, self.cfg.device)
            ref = self.reference_scores(part, batch)
            scores = model_scores(self.model, batch)
            self.policy_forward_calls += 1
            n = len(part)
            loss = dpo_loss(scores[:n], scores[n:], ref[:n], ref[n:], self.cfg.beta)
            (loss * n / len(pairs)).backward()
            loss_sum += loss.detach().item() * n
            response_tokens += int(batch['completion_mask'].sum())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        scale = ((self.update_index + 1) / self.warmup if self.update_index < self.warmup
                 else (self.total_updates - self.update_index) / max(1, self.total_updates - self.warmup))
        lr = self.cfg.learning_rate * max(0., scale)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.update_index += 1
        self.position += len(pairs)
        row = dict(update=self.update_index, pairs=len(pairs), loss=loss_sum / len(pairs),
                   response_tokens=response_tokens, learning_rate=lr, grad_norm=float(norm))
        self.trace.append(row)
        if self.progress_callback:
            self.progress_callback(self)
        return row
