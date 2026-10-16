"""Causal shift and response-only sum; prompt remains visible to attention."""
import torch


def completion_scores(logits: torch.Tensor, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or input_ids.shape != logits.shape[:2] or input_ids.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and input IDs [B,L]")
    if attention_mask.shape != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError("Attention and completion masks must match input IDs")
    if any(not torch.all((m == 0) | (m == 1)) for m in [attention_mask, completion_mask]):
        raise ValueError("Masks must be binary")
    if torch.any(completion_mask.bool() & ~attention_mask.bool()):
        raise ValueError("A scored token cannot be padding")
    if torch.any(completion_mask[:, 0]):
        raise ValueError("Completion must have a preceding conditioning token")
    mask = (attention_mask[:, 1:].bool() & completion_mask[:, 1:].bool())
    if input_ids.shape[1] < 2 or torch.any(mask.sum(-1) == 0):
        raise ValueError("Every sequence needs a nonempty scored completion")
    # Select rows before logsoftmax: invalid masked targets/logits are never gathered,
    # and masked -inf or NaN values cannot produce 0 * infinity artifacts.
    selected = logits[:, :-1][mask]
    targets = input_ids[:, 1:][mask]
    if targets.dtype != torch.long or torch.any((targets < 0) | (targets >= logits.shape[-1])):
        raise ValueError("Scored target ID outside vocabulary or not int64")
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite logits at a scored prediction position")
    token_scores = selected.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    rows = torch.arange(logits.shape[0], device=logits.device)[:, None].expand_as(mask)[mask]
    scores = token_scores.new_zeros(logits.shape[0]).index_add(0, rows, token_scores)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite completion score")
    return scores


def model_scores(model, batch: dict) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out if isinstance(out, torch.Tensor) else out.logits
    return completion_scores(logits, batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
