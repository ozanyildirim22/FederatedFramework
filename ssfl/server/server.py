"""Server-Side Aggregation Module: orchestration, broadcast, aggregation, evaluation."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..client import Client, ClientUpdate
from .aggregators import Aggregator


class Server:
    def __init__(self, model: nn.Module, aggregator: Aggregator, clients: Sequence[Client],
                 client_fraction: float, device: torch.device, seed: int = 0):
        self.model = model
        self.aggregator = aggregator
        self.clients = list(clients)
        self.fraction = client_fraction
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---------------------------------------------------- orchestration phase
    def select_clients(self) -> List[Client]:
        m = max(1, int(round(self.fraction * len(self.clients))))
        idx = self.rng.choice(len(self.clients), size=m, replace=False)
        return [self.clients[i] for i in sorted(idx)]

    # ------------------------------------------------------------- downlink
    def broadcast(self) -> Dict[str, torch.Tensor]:
        # In simulation "serialisation" is a defensive copy of W_t.
        return {k: v.clone() for k, v in self.global_state.items()}

    # ----------------------------------------------------------- aggregation
    def aggregate(self, updates: List[ClientUpdate]) -> None:
        self.global_state = self.aggregator.aggregate(self.global_state, updates)

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.load_state_dict(self.global_state)
        self.model.to(self.device).eval()
        prec = getattr(self, "precision", "fp32")      # set by Experiment from the strategy
        dtype = torch.bfloat16 if prec == "bf16" else torch.float16
        loss = torch.zeros((), device=self.device)
        conf = None                                     # confusion matrix [true, predicted]
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=dtype, enabled=prec != "fp32"):
                logits = self.model(x)
            logits = logits.float()
            n_cls = logits.shape[1]
            if conf is None:
                conf = torch.zeros(n_cls * n_cls, dtype=torch.long, device=self.device)
            loss += nn.functional.cross_entropy(logits, y, reduction="sum")
            conf += torch.bincount(y * n_cls + logits.argmax(1), minlength=n_cls * n_cls)
        return classification_metrics(conf.view(n_cls, n_cls).cpu(), float(loss))


def classification_metrics(confusion: torch.Tensor, loss_sum: float = 0.0) -> Dict[str, float]:
    """Top-1 accuracy (correct / total) and **macro-averaged** precision, recall and F1
    over the classes (the ADD's Eq. 6, 8, 9 for the multi-class case), from a confusion
    matrix ``[true, predicted]``. Classes that never occur in the test set are skipped for
    recall / F1; a class never predicted has precision 0."""
    c = confusion.double()
    total = c.sum().clamp_min(1)
    tp = c.diag()
    support, predicted = c.sum(1), c.sum(0)
    present = support > 0
    precision = torch.where(predicted > 0, tp / predicted.clamp_min(1), torch.zeros_like(tp))
    recall = tp / support.clamp_min(1)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp_min(1e-12),
                     torch.zeros_like(tp))
    return {"test_acc": float(tp.sum() / total), "test_loss": loss_sum / float(total),
            "test_precision": float(precision[present].mean()), "test_recall": float(recall[present].mean()),
            "test_f1": float(f1[present].mean())}
