"""Composite Loss Calculator and reusable loss terms."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def supervised_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def masked_ce(logits: torch.Tensor, pseudo: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on pseudo-labels, averaged over the *whole* unlabeled batch
    (FixMatch convention) so that the loss scale reflects the mask rate."""
    if logits.numel() == 0:
        return logits.sum()
    return (F.cross_entropy(logits, pseudo, reduction="none") * mask.float()).mean()


def kl_consistency(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                   temperature: float = 1.0) -> torch.Tensor:
    """KL(teacher || student) with temperature scaling (Hinton et al.)."""
    if student_logits.numel() == 0:
        return student_logits.sum()
    t = temperature
    return F.kl_div(F.log_softmax(student_logits / t, dim=1),
                    F.softmax(teacher_logits.detach() / t, dim=1),
                    reduction="batchmean") * (t * t)


class CompositeLoss:
    """Weighted sum of named loss terms.

    >>> loss = CompositeLoss(sup=1.0, unsup=1.0, kl=0.5)
    >>> total, logs = loss(sup=l_s, unsup=l_u, kl=l_kl)
    """

    def __init__(self, **weights: float):
        self.weights = weights

    def __call__(self, **terms: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total, logs = None, {}
        for name, value in terms.items():
            if value is None:
                continue
            w = self.weights.get(name, 1.0)
            if w == 0:
                continue
            total = w * value if total is None else total + w * value
            logs[f"loss_{name}"] = value.detach()      # tensors: converted once per round
        if total is None:
            raise ValueError("CompositeLoss received no active terms")
        logs["loss"] = total.detach()
        return total, logs
