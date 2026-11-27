"""FedMatch (Jeong et al., ICLR 2021) - labels-at-client scenario.

"Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint
Learning", https://arxiv.org/abs/2006.12097

Components
----------
1. **Parameter decomposition** ``theta = sigma + psi``. ``sigma`` (dense) is
   learned from labeled data with ``psi`` frozen (Eq. 4); ``psi`` (sparse) is
   learned from unlabeled data with ``sigma`` frozen (Eq. 5)::

       L_s = lambda_s    * CE(y, p_{sigma + psi*}(x))
       L_u = lambda_iccs * Phi_{sigma* + psi}(u) + lambda_l2 * ||sigma* - psi||^2
             + lambda_l1 * ||psi||_1

   Both losses are summed in one backward pass. Because of the ``detach``
   calls, ``sigma`` receives only the gradient of ``L_s`` and ``psi`` only the
   gradient of ``L_u``. This is equivalent to the paper's two disjoint updates
   applied at the same time.

2. **Inter-client consistency** (Eq. 1-2)::

       Phi = CE(y_hat, p_local(strong(u))) + 1/H * sum_j KL(p_helper_j(u) || p_local(u))

3. **Agreement-based pseudo-label** (Eq. 3): the local model and the H helpers
   each vote with their arg-max on the weak view. The class with the most votes
   becomes ``y_hat``. Ties go to the local model's class. Samples whose local
   confidence is below ``tau`` are discarded.

4. **Helper selection (server side)**. Each uploaded client model is embedded by
   its softmax output on a fixed Gaussian-noise input held by the server. For
   every selected client, the H nearest client models (Euclidean distance on
   the embeddings) are sent as frozen helpers. The helper bank is refreshed
   every ``helper_every`` rounds (paper: 10) from the models uploaded in the
   latest round. The paper uses a KD-tree; with at most a few hundred clients
   an exact brute-force search gives the same neighbours.

Implementation choices (not fixed by the paper)
-----------------------------------------------
* ``psi`` starts at zero, so the first global model equals the plain model.
* ``reg_reduction="mean"`` averages the L1/L2 terms over parameters instead of
  summing them. With ResNet-18 (11M weights) the summed version with
  ``lambda_l2=10`` dominates every other term. Use ``"sum"`` for the literal form.
* A client that has never uploaded has no embedding yet. Its helpers are then
  chosen with the embedding of the global model (the paper skips helpers here).
* ``decompose=False`` trains a single set of weights (ablation without
  parameter decomposition).

Default loss weights follow the paper's reported values (tau=0.85, lambda_s=10,
lambda_iccs=1e-2, lambda_l2=10, lambda_l1=1e-4, H=2). Check them against
Table 4 / Appendix A.3 of the paper before quoting exact reproductions.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from ..losses import kl_consistency, masked_ce, supervised_ce
from ..registry import STRATEGIES
from .algorithms import _pl_stats
from .base import Strategy

StateDict = Dict[str, torch.Tensor]


class DecomposedModel(nn.Module):
    """Wraps a model so that its weights are ``sigma + psi``.

    ``self.base`` holds ``sigma`` (its ordinary parameters); ``self.psi`` holds one
    tensor per parameter of the same shape. ``forward(x, mode)``:

    * ``"full"``  - ``sigma + psi`` (evaluation / inference)
    * ``"sup"``   - ``sigma + psi.detach()`` (gradient flows to sigma only)
    * ``"unsup"`` - ``sigma.detach() + psi`` (gradient flows to psi only)
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.param_names: List[str] = [n for n, _ in base.named_parameters()]
        self.psi = nn.ParameterList([nn.Parameter(torch.zeros_like(p)) for p in base.parameters()])

    def sigma(self) -> List[torch.Tensor]:
        return [p for p in self.base.parameters()]

    def _params(self, mode: str) -> Dict[str, torch.Tensor]:
        out = {}
        for name, s, p in zip(self.param_names, self.base.parameters(), self.psi):
            if mode == "sup":
                out[name] = s + p.detach()
            elif mode == "unsup":
                out[name] = s.detach() + p
            else:
                out[name] = s + p
        return out

    def forward(self, x: torch.Tensor, mode: str = "full") -> torch.Tensor:
        return functional_call(self.base, self._params(mode), (x,))

    def merged_state(self, state: StateDict) -> StateDict:
        """Turn a state dict of this wrapper into a plain ``base`` state dict (theta = sigma + psi)."""
        index = {n: i for i, n in enumerate(self.param_names)}
        out = {}
        for k in self.base.state_dict().keys():
            v = state[f"base.{k}"]
            if k in index:
                v = v + state[f"psi.{index[k]}"].to(v.dtype)
            out[k] = v
        return out


@STRATEGIES.register("fedmatch")
class FedMatch(Strategy):
    """FedMatch: inter-client consistency + disjoint (sigma/psi) learning.
    See the module docstring for the equations."""

    # helper bank and client embeddings live on the server across rounds
    persistent_state = ("_embeddings", "_recent", "_bank")

    def __init__(self, cfg, num_classes, tau: float = 0.85, lambda_s: float = 10.0,
                 lambda_iccs: float = 1e-2, lambda_l2: float = 10.0, lambda_l1: float = 1e-4,
                 num_helpers: int = 2, helper_every: int = 10, decompose: bool = True,
                 reg_reduction: str = "mean", embed_samples: int = 8, **params):
        params.setdefault("loss_weights", {"sup": lambda_s, "unsup": lambda_iccs,
                                           "iccs": lambda_iccs, "l2": lambda_l2, "l1": lambda_l1})
        super().__init__(cfg, num_classes, **params)
        if reg_reduction not in ("mean", "sum"):
            raise ValueError(f"Unknown reg_reduction '{reg_reduction}'")
        self.tau = tau
        self.num_helpers = int(num_helpers)
        self.helper_every = max(1, int(helper_every))
        self.decompose = bool(decompose)
        self.reg_reduction = reg_reduction
        self.embed_samples = int(embed_samples)
        # server-side state (kept in checkpoints: see Strategy.persistent_state)
        self._embeddings: Dict[int, torch.Tensor] = {}   # client id -> model embedding
        self._recent: Dict[int, StateDict] = {}          # models uploaded in the latest round
        self._bank: Dict[int, StateDict] = {}            # helper models currently on offer
        self._assign: Dict[int, List[int]] = {}          # client id -> helper ids this round
        # client-side state
        self._helper_modules: List[nn.Module] = []
        self._helpers: List[nn.Module] = []

    # ------------------------------------------------------------ model
    def prepare_model(self, model: nn.Module) -> nn.Module:
        return DecomposedModel(model) if self.decompose else model

    def _plain(self, model: nn.Module) -> nn.Module:
        return model.base if isinstance(model, DecomposedModel) else model

    def _merge(self, model: nn.Module, state: StateDict) -> StateDict:
        return model.merged_state(state) if isinstance(model, DecomposedModel) else dict(state)

    # ----------------------------------------------------- server side
    def setup(self, experiment) -> None:
        raw = experiment.data.train
        g = torch.Generator().manual_seed(int(experiment.cfg.seed))
        self._probe = torch.randn(self.embed_samples, raw.in_channels, raw.image_size, raw.image_size,
                                  generator=g)
        self._model = experiment.model
        self._template = copy.deepcopy(self._plain(experiment.model)).eval()
        self._device = experiment.device

    @torch.no_grad()
    def _embed(self, merged: StateDict) -> torch.Tensor:
        self._template.load_state_dict(merged)
        self._template.to(self._device).eval()
        p = F.softmax(self._template(self._probe.to(self._device)), dim=1)
        return p.flatten().cpu()

    def on_server_round_start(self, server, selected, rnd: int) -> None:
        super().on_server_round_start(server, selected, rnd)
        self._assign = {}
        if self.num_helpers <= 0:
            return
        if self._recent and ((rnd - 1) % self.helper_every == 0 or not self._bank):
            self._bank = dict(self._recent)
        if not self._bank:
            return
        global_emb = None
        for c in selected:
            emb = self._embeddings.get(c.id)
            if emb is None:
                if global_emb is None:
                    global_emb = self._embed(self._merge(self._model, server.global_state))
                emb = global_emb
            cands = [cid for cid in self._bank if cid != c.id]
            if not cands:
                continue
            dist = torch.stack([(self._embeddings[cid] - emb).norm() for cid in cands])
            order = dist.argsort()[: self.num_helpers].tolist()
            self._assign[c.id] = [cands[i] for i in order]

    def on_server_round_end(self, server, updates, global_state, rnd: int) -> None:
        if self.num_helpers <= 0:
            return
        self._recent = {}
        for u in updates:
            local = {k: (v + u.delta[k]) if v.is_floating_point() else u.buffers[k]
                     for k, v in global_state.items()}
            merged = self._merge(self._model, local)
            self._embeddings[u.client_id] = self._embed(merged)
            self._recent[u.client_id] = {k: v.half() if v.is_floating_point() else v.clone()
                                         for k, v in merged.items()}

    # ----------------------------------------------------- client side
    def on_round_start(self, model, client) -> None:
        self._helpers = []
        ids = self._assign.get(client.id, [])
        while len(self._helper_modules) < len(ids):
            self._helper_modules.append(copy.deepcopy(self._plain(model)))
        for module, cid in zip(self._helper_modules, ids):
            module.load_state_dict({k: v.float() if v.is_floating_point() else v
                                    for k, v in self._bank[cid].items()})
            module.to(self.device).eval()
            for p in module.parameters():
                p.requires_grad_(False)
            self._helpers.append(module)

    def on_round_end(self, model, client) -> None:
        self._helpers = []

    def _forward(self, model, x, mode):
        # with strategy.compile the model arrives wrapped by torch.compile (``_orig_mod``)
        plain = getattr(model, "_orig_mod", model)
        return model(x, mode) if isinstance(plain, DecomposedModel) else model(x)

    def compute_loss(self, model, labeled, unlabeled):
        terms, logs = {}, {}
        if labeled is not None:
            x, y = labeled
            terms["sup"] = supervised_ce(self._forward(model, x, "sup"), y)
        if unlabeled is not None:
            u_w, u_s, u_true = unlabeled
            lw, ls = self._forward(model, torch.cat([u_w, u_s]), "unsup").split([len(u_w), len(u_s)])
            with torch.no_grad():
                p_local = F.softmax(lw.detach(), 1)
                conf, local_cls = p_local.max(1)
                votes = F.one_hot(local_cls, self.num_classes).float() * (1.0 + 1e-3)  # ties -> local
                h_logits = [h(u_w) for h in self._helpers]
                for hl in h_logits:
                    votes += F.one_hot(hl.argmax(1), self.num_classes).float()
                pseudo = votes.argmax(1)
                mask = conf >= self.tau
            terms["unsup"] = masked_ce(ls, pseudo, mask)
            if h_logits:
                terms["iccs"] = sum(kl_consistency(lw, hl) for hl in h_logits) / len(h_logits)
                logs["helper_agree"] = (pseudo == local_cls).float().mean()
            logs["num_helpers"] = float(len(h_logits))
            logs.update(_pl_stats(pseudo, mask, u_true))
            plain = getattr(model, "_orig_mod", model)       # unwrap torch.compile
            if isinstance(plain, DecomposedModel):
                sig, psi = plain.sigma(), list(plain.psi)
                if self.reg_reduction == "mean":
                    n = sum(p.numel() for p in psi)
                    terms["l2"] = sum(((s.detach() - p) ** 2).sum() for s, p in zip(sig, psi)) / n
                    terms["l1"] = sum(p.abs().sum() for p in psi) / n
                else:
                    terms["l2"] = sum(((s.detach() - p) ** 2).sum() for s, p in zip(sig, psi))
                    terms["l1"] = sum(p.abs().sum() for p in psi)
        total, l = self.loss(**terms)
        return total, {**l, **logs}
