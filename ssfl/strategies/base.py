"""Training Protocol Interface.

A :class:`Strategy` owns the local optimisation loop of one client. The base
class already implements the epoch/step iteration, batch cycling, optimiser
creation, back-propagation and an optional FedProx proximal term. Most new
algorithms only override :meth:`compute_loss` (and optionally the round hooks).

Minimal custom strategy::

    from ssfl import STRATEGIES, Strategy

    @STRATEGIES.register("my_algo")
    class MyAlgo(Strategy):
        def compute_loss(self, model, labeled, unlabeled):
            x, y = labeled
            return self.loss(sup=F.cross_entropy(model(x), y))
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from ..losses import CompositeLoss


def _cycle(loader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Strategy:
    #: does the strategy consume the unlabeled loader?
    uses_unlabeled: bool = True
    #: does the strategy need a frozen copy of the received global model?
    needs_global_model: bool = False

    def __init__(self, cfg, num_classes: int, **params):
        self.cfg = cfg
        self.num_classes = num_classes
        self.params = params
        self.loss = CompositeLoss(**params.get("loss_weights", {}))
        self.global_model: Optional[nn.Module] = None
        self._global_params = None
        self.device = torch.device("cpu")
        self.prox_mu: float = float(getattr(cfg, "prox_mu", 0.0) or 0.0)
        self.round: int = 0          # current communication round (1-based), set by the server hook
        self.step: int = 0           # current local step within the round
        self.total_steps: int = 0    # local steps in the current round

    # ---------------------------------------------------------- server hooks
    # Some algorithms (FedMatch, FedTriNet) have a server-side part. In the
    # simulation the same Strategy object plays both roles; these hooks are
    # the server side and run outside any client's local training.
    def setup(self, experiment) -> None:
        """Called once after the data, model and server have been built."""

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Wrap / modify the global architecture (e.g. FedMatch sigma+psi). Called once."""
        return model

    def on_server_round_start(self, server, selected, rnd: int) -> None:
        """Server side, before broadcast. Default: record the round number."""
        self.round = rnd

    def on_server_round_end(self, server, updates, global_state, rnd: int) -> None:
        """Server side, after uploads and before aggregation (``global_state`` = W_t)."""

    # ----------------------------------------------------------- client hooks
    def on_round_start(self, model: nn.Module, client) -> None:
        """Called after the global weights were loaded into ``model``."""

    def on_round_end(self, model: nn.Module, client) -> None:
        """Called after local training, before the update is sent."""

    def compute_loss(self, model: nn.Module, labeled: Optional[Tuple], unlabeled: Optional[Tuple]
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    # ------------------------------------------------------------- optimiser
    def make_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        c = self.cfg
        if c.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=c.lr, momentum=c.momentum,
                                   weight_decay=c.weight_decay, nesterov=c.momentum > 0)
        if c.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        raise ValueError(f"Unknown optimizer '{c.optimizer}'")

    def proximal_term(self, model: nn.Module) -> Optional[torch.Tensor]:
        if self.prox_mu <= 0 or self._global_params is None:
            return None
        sq = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), self._global_params))
        return 0.5 * self.prox_mu * sq

    # ------------------------------------------------------------ main loop
    def num_steps(self, client) -> int:
        if self.cfg.local_steps:
            return self.cfg.local_steps
        n_l = len(client.data.labeled) if client.data.labeled is not None else 0
        n_u = len(client.data.unlabeled) if (self.uses_unlabeled and client.data.unlabeled is not None) else 0
        return self.cfg.local_epochs * max(n_l, n_u, 1)

    def _to_device(self, batch):
        if batch is None:
            return None
        return tuple(t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)

    def local_train(self, model: nn.Module, client, device: torch.device) -> Dict[str, float]:
        self.device = device
        if self.needs_global_model:
            self.global_model = copy.deepcopy(model).eval()
            for p in self.global_model.parameters():
                p.requires_grad_(False)
        if self.prox_mu > 0:
            self._global_params = [p.detach().clone() for p in model.parameters()]

        self.on_round_start(model, client)
        model.train()
        opt = self.make_optimizer(model)
        lab_it = _cycle(client.data.labeled) if client.data.labeled is not None else None
        unl_it = (_cycle(client.data.unlabeled)
                  if self.uses_unlabeled and client.data.unlabeled is not None else None)

        use_amp = bool(getattr(self.cfg, "amp", False)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        sums: Dict[str, float] = defaultdict(float)
        steps = self.num_steps(client)
        self.total_steps = steps
        for step in range(steps):
            self.step = step
            labeled = self._to_device(next(lab_it)) if lab_it else None
            unlabeled = self._to_device(next(unl_it)) if unl_it else None
            with torch.autocast(device.type, enabled=use_amp):
                loss, logs = self.compute_loss(model, labeled, unlabeled)
            prox = self.proximal_term(model)
            if prox is not None:
                loss = loss + prox
                logs["loss_prox"] = float(prox.detach())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                sums[k] += v

        self.on_round_end(model, client)
        self.global_model, self._global_params = None, None
        return {k: v / steps for k, v in sums.items()}
