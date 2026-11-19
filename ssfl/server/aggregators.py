"""Aggregation mechanisms.

An aggregator turns the list of client updates (``delta_w_k``) into the new
global state ``W_{t+1}``. Subclasses implement :meth:`combine`, which receives
the stacked float deltas and normalised client weights and returns the
aggregated *step* that is added to ``W_t``. Register new rules with
``@AGGREGATORS.register("name")``.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..client import ClientUpdate
from ..registry import AGGREGATORS

StateDict = Dict[str, torch.Tensor]


class Aggregator:
    def __init__(self, weight_by: str = "num_samples", server_lr: float = 1.0):
        if weight_by not in ("num_samples", "num_labeled", "uniform"):
            raise ValueError(f"Unknown weight_by '{weight_by}'")
        self.weight_by = weight_by
        self.server_lr = server_lr

    def weights(self, updates: List[ClientUpdate]) -> torch.Tensor:
        if self.weight_by == "uniform":
            w = torch.ones(len(updates))
        else:
            w = torch.tensor([float(getattr(u, self.weight_by)) for u in updates])
            if w.sum() == 0:
                w = torch.ones(len(updates))
        return w / w.sum()

    def combine(self, key: str, deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``deltas``: (num_clients, *shape). Return the step for this tensor."""
        raise NotImplementedError

    def aggregate(self, global_state: StateDict, updates: List[ClientUpdate]) -> StateDict:
        w = self.weights(updates)
        new_state = {}
        for k, g in global_state.items():
            if g.is_floating_point():
                stacked = torch.stack([u.delta[k] for u in updates])
                new_state[k] = g + self.combine(k, stacked, w)
            else:  # integer buffers: weighted average, rounded
                vals = torch.stack([u.buffers[k].double() for u in updates])
                avg = (vals * w.double().view(-1, *[1] * (vals.dim() - 1))).sum(0)
                new_state[k] = avg.round().to(g.dtype)
        return new_state


def _weighted_mean(deltas: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (deltas * w.to(deltas.dtype).view(-1, *[1] * (deltas.dim() - 1))).sum(0)


@AGGREGATORS.register("fedavg")
class FedAvg(Aggregator):
    """McMahan et al., 2017 - dataset-size weighted average of client updates."""

    def combine(self, key, deltas, weights):
        return self.server_lr * _weighted_mean(deltas, weights)


@AGGREGATORS.register("fedavgm")
class FedAvgM(Aggregator):
    """Hsu et al., 2019 - FedAvg with server-side momentum."""

    def __init__(self, beta: float = 0.9, **kw):
        super().__init__(**kw)
        self.beta, self.v = beta, {}

    def combine(self, key, deltas, weights):
        d = _weighted_mean(deltas, weights)
        self.v[key] = self.beta * self.v.get(key, torch.zeros_like(d)) + d
        return self.server_lr * self.v[key]


@AGGREGATORS.register("fedadam")
class FedAdam(Aggregator):
    """Reddi et al., 2021 (Adaptive Federated Optimization) - server Adam."""

    def __init__(self, server_lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99,
                 tau: float = 1e-3, **kw):
        super().__init__(server_lr=server_lr, **kw)
        self.b1, self.b2, self.tau = beta1, beta2, tau
        self.m, self.v = {}, {}

    def combine(self, key, deltas, weights):
        d = _weighted_mean(deltas, weights)
        m = self.b1 * self.m.get(key, torch.zeros_like(d)) + (1 - self.b1) * d
        v = self.b2 * self.v.get(key, torch.zeros_like(d)) + (1 - self.b2) * d * d
        self.m[key], self.v[key] = m, v
        return self.server_lr * m / (v.sqrt() + self.tau)


@AGGREGATORS.register("fedmedian")
class FedMedian(Aggregator):
    """Yin et al., 2018 - coordinate-wise median (robust to outlier clients)."""

    def combine(self, key, deltas, weights):
        return self.server_lr * deltas.median(dim=0).values
