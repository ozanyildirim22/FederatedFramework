"""FedTriNet (Che et al., IEEE BigData 2021).

"FedTriNet: A Pseudo Labeling Method with Three Players for Federated
Semi-supervised Learning", https://arxiv.org/abs/2109.05612

Three "players" give the pseudo-labels on each client k:

* the **local model** ``l^k``: the model this client trained in its previous
  round (kept on the client between rounds);
* the **global model** ``G = W_t`` (frozen);
* the **combined model** ``c^k``: the first (shallow) layers come from ``G`` and
  the remaining (deep) layers from ``l^k``. ``c^k`` is the model that is trained
  and uploaded.

Training has two phases:

* **Phase I** (``pretrain_rounds`` rounds): plain FedAvg on labeled data.
* **Phase II**: ``c^k`` is first fine-tuned on labeled data for
  ``finetune_steps`` steps. After that, every unlabeled batch gets the
  pseudo-label (Eq. 3)::

      p_bar = (p_l(u) + p_G(u) + p_c(u)) / 3,   y_hat = argmax p_bar

  A sample is kept if ``max p_bar > theta(t)``. The local loss (Eq. 6-7) is
  ``CE(labeled) + lambda_u * CE(pseudo-labeled)``.

Dynamic quality control (Eq. 4-5)
---------------------------------
At the start of each phase-II round, every selected client computes
``theta^k = max_u max_c p_G(c|u)`` on its unlabeled data and uploads that one
scalar. The server averages the scalars into ``theta_bar`` and broadcasts::

    theta(t) = alpha * theta_bar                     t < 10
             = (100 - 2t)/100 * alpha * theta_bar    10 <= t < 35
             = 0.5 * alpha * theta_bar               t >= 35

``t`` is the phase-II round index, starting at 0, and ``alpha = 0.93`` as in
the paper. ``hold_rounds`` (10), ``decay_end`` (35) and ``final_scale`` (0.5)
expose the constants of the schedule.

Implementation choices (not fixed by the paper)
-----------------------------------------------
* ``global_layers``: state-dict key prefixes taken from ``G``. Defaults: the
  first two conv blocks of ``simple_cnn`` (as in the paper's CNN), and the stem
  plus the first residual stage of ``resnet18``. Other models use the first
  ``global_fraction`` of parameter tensors.
* A client seen for the first time has no ``l^k`` yet; ``G`` is used instead.
* ``unsup_view``: the input the pseudo-label loss is applied to. ``"weak"``
  (default) follows the paper; ``"strong"`` gives FixMatch-style consistency.
* Local models are stored in float16 on the CPU (``store_dtype``) to keep
  memory low with 100 cross-device clients.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.manager import _to_pil
from ..losses import masked_ce, supervised_ce
from ..registry import STRATEGIES
from .algorithms import _forward_cat, _pl_stats
from .base import Strategy

StateDict = Dict[str, torch.Tensor]

class _WeakOnly(Dataset):
    """Weak view of an ``UnlabeledView`` (no labels, no strong augmentation)."""

    def __init__(self, view):
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i):
        return self.view.weak(_to_pil(self.view.raw.images[self.view.indices[i]]))


DEFAULT_GLOBAL_LAYERS = {
    "SimpleCNN": ["features.0.", "features.1."],
    "ResNet18": ["stem.", "layers.0.", "layers.1."],
}


@STRATEGIES.register("fedtrinet")
class FedTriNet(Strategy):
    """FedTriNet: local / global / combined models vote by averaging softmax.
    See the module docstring."""

    needs_global_model = True

    def __init__(self, cfg, num_classes, pretrain_rounds: int = 20, alpha: float = 0.93,
                 lambda_u: float = 1.0, finetune_steps: int = 5,
                 global_layers: Optional[List[str]] = None, global_fraction: float = 0.25,
                 hold_rounds: int = 10, decay_end: int = 35, final_scale: float = 0.5,
                 unsup_view: str = "weak", store_dtype: str = "float16", **params):
        params.setdefault("loss_weights", {"sup": 1.0, "unsup": lambda_u})
        super().__init__(cfg, num_classes, **params)
        if unsup_view not in ("weak", "strong"):
            raise ValueError(f"Unknown unsup_view '{unsup_view}'")
        self.pretrain_rounds = int(pretrain_rounds)
        self.alpha = float(alpha)
        self.finetune_steps = int(finetune_steps)
        self.global_layers = list(global_layers) if global_layers else None
        self.global_fraction = float(global_fraction)
        self.hold_rounds, self.decay_end = int(hold_rounds), int(decay_end)
        self.final_scale = float(final_scale)
        self.unsup_view = unsup_view
        self.store_dtype = getattr(torch, store_dtype)
        self._local_states: Dict[int, StateDict] = {}   # client id -> l^k (lives on the client)
        self._global_keys: set = set()
        self._theta: float = 1.0
        self._local_module: Optional[nn.Module] = None
        self._has_local = False

    # ------------------------------------------------------------ phases
    @property
    def phase2(self) -> bool:
        return self.round > self.pretrain_rounds

    @property
    def uses_unlabeled(self) -> bool:   # phase I is labeled-only FedAvg
        return self.phase2

    def threshold_scale(self, t: int) -> float:
        if t < self.hold_rounds:
            return 1.0
        if t < self.decay_end:
            return (100 - 2 * t) / 100
        return self.final_scale

    # ------------------------------------------------------------ setup
    def _resolve_global_keys(self, model: nn.Module) -> set:
        keys = list(model.state_dict().keys())
        prefixes = self.global_layers or DEFAULT_GLOBAL_LAYERS.get(type(model).__name__)
        if prefixes:
            chosen = {k for k in keys if any(k.startswith(p) for p in prefixes)}
        else:
            names = [n for n, _ in model.named_parameters()]
            n = max(1, math.ceil(self.global_fraction * len(names)))
            mods = {name.rsplit(".", 1)[0] for name in names[:n]}
            chosen = {k for k in keys if k.rsplit(".", 1)[0] in mods}
        if not chosen or len(chosen) == len(keys):
            raise ValueError("FedTriNet: global_layers must select some, but not all, layers")
        return chosen

    def setup(self, experiment) -> None:
        self._global_keys = self._resolve_global_keys(experiment.model)
        self._device = experiment.device
        self._probe_model = copy.deepcopy(experiment.model)
        experiment.log(f"[fedtrinet] {len(self._global_keys)} tensors from the global model, "
                       f"{len(experiment.model.state_dict()) - len(self._global_keys)} from the local model")

    # ------------------------------------------------------- server side
    @torch.no_grad()
    def _client_theta(self, model: nn.Module, client) -> Optional[float]:
        """Eq. 4 - computed on the client with the received global model; only the scalar leaves."""
        loader = client.data.unlabeled
        if loader is None:
            return None
        ds = loader.dataset
        # weak view only: skips the (expensive) strong augmentation of the training loader
        weak = DataLoader(_WeakOnly(ds), batch_size=256, shuffle=False,
                          num_workers=loader.num_workers, pin_memory=loader.pin_memory)
        best = 0.0
        for u_w in weak:
            p = F.softmax(model(u_w.to(self._device, non_blocking=True)), 1)
            best = max(best, float(p.max()))
        return best

    def on_server_round_start(self, server, selected: Sequence, rnd: int) -> None:
        super().on_server_round_start(server, selected, rnd)
        if not self.phase2:
            return
        m = self._probe_model
        m.load_state_dict(server.global_state)
        m.to(self._device).eval()
        thetas = [t for t in (self._client_theta(m, c) for c in selected) if t is not None]
        theta_bar = sum(thetas) / len(thetas) if thetas else 1.0
        t = rnd - self.pretrain_rounds - 1
        self._theta = self.threshold_scale(t) * self.alpha * theta_bar

    # ------------------------------------------------------- client side
    def on_round_start(self, model: nn.Module, client) -> None:
        local = self._local_states.get(client.id)
        self._has_local = local is not None
        if not self.phase2:
            return
        if self._local_module is None:
            self._local_module = copy.deepcopy(model)
        lm = self._local_module
        if local is not None:
            g = model.state_dict()
            lm.load_state_dict({k: v.to(g[k].dtype) for k, v in local.items()})
            # combined model c^k = G[shallow] + l^k[deep]
            ls = lm.state_dict()
            model.load_state_dict({k: (g[k] if k in self._global_keys else ls[k]) for k in g})
        else:
            lm.load_state_dict(model.state_dict())
        lm.to(self.device).eval()
        for p in lm.parameters():
            p.requires_grad_(False)

    def on_round_end(self, model: nn.Module, client) -> None:
        self._local_states[client.id] = {
            k: (v.detach().to("cpu", self.store_dtype) if v.is_floating_point() else v.detach().cpu().clone())
            for k, v in model.state_dict().items()}

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled if labeled is not None else (None, None)
        if not self.phase2 or unlabeled is None or (self.step < self.finetune_steps and x is not None):
            if x is None:
                raise RuntimeError("FedTriNet needs labeled data on the client")
            return self.loss(sup=supervised_ce(model(x), y))

        u_w, u_s, u_true = unlabeled
        views = [x, u_w] + ([u_s] if self.unsup_view == "strong" else [])
        logits = _forward_cat(model, *views)
        terms, logs = {}, {}
        if x is not None:
            terms["sup"] = supervised_ce(logits.pop(0), y)
        lw = logits.pop(0)
        with torch.no_grad():
            p_bar = (F.softmax(lw.detach(), 1) + F.softmax(self.global_model(u_w), 1)
                     + F.softmax(self._local_module(u_w), 1)) / 3
            conf, pseudo = p_bar.max(1)
            mask = conf > self._theta
        target_logits = logits.pop(0) if self.unsup_view == "strong" else lw
        terms["unsup"] = masked_ce(target_logits, pseudo, mask)
        logs.update(_pl_stats(pseudo, mask, u_true))
        logs["threshold"] = self._theta
        logs["has_local_model"] = float(self._has_local)
        total, l = self.loss(**terms)
        return total, {**l, **logs}
