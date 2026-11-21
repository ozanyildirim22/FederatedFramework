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
        correct = torch.zeros((), device=self.device)
        loss = torch.zeros((), device=self.device)
        total = 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=dtype, enabled=prec != "fp32"):
                logits = self.model(x)
            logits = logits.float()
            loss += nn.functional.cross_entropy(logits, y, reduction="sum")
            correct += (logits.argmax(1) == y).sum()
            total += y.numel()
        return {"test_acc": float(correct) / total, "test_loss": float(loss) / total}
