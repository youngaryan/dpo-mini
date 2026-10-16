"""Original sigmoid DPO: summed response scores, zero label smoothing."""
import math
import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float = 0.1, reduction: str = "mean") -> torch.Tensor:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    values = [policy_chosen, policy_rejected, reference_chosen, reference_rejected]
    if not policy_chosen.numel() or any(x.shape != policy_chosen.shape or x.ndim != 1 for x in values):
        raise ValueError("DPO scores must be nonempty matching vectors, one entry per pair")
    if any(not torch.isfinite(x).all() for x in values):
        raise ValueError("DPO scores must be finite")
    margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    losses = -F.logsigmoid(beta * margin)
    if reduction == "none":
        return losses
    if reduction != "mean":
        raise ValueError("Supported reductions: mean, none")
    return losses.mean()
