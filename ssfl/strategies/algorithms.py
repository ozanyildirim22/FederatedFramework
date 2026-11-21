"""Built-in client strategies (pseudo-labeling engines)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..losses import kl_consistency, masked_ce, supervised_ce
from ..registry import STRATEGIES
from .base import Strategy


def _forward_cat(model, *xs):
    """Single forward pass over concatenated inputs (keeps BatchNorm statistics
    consistent across labeled / unlabeled views) and split the logits back."""
    xs = [x for x in xs if x is not None]
    logits = model(torch.cat(xs))
    return list(logits.split([x.shape[0] for x in xs]))


def _pl_stats(pseudo, mask, true) -> Dict[str, torch.Tensor]:
    """Pseudo-label diagnostics, kept on the device (no GPU sync per step).
    ``Strategy.local_train`` turns ``_pl_correct`` / ``_pl_count`` into ``pl_acc``:
    the accuracy over all pseudo-labeled samples of the round."""
    m = mask.float()
    return {"mask_rate": m.mean(), "_pl_correct": ((pseudo == true).float() * m).sum(), "_pl_count": m.sum()}


def confidence_score(probs: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-sample confidence of a softmax distribution.

    * ``max_prob``  - highest class probability.
    * ``variance``  - variance of the probability vector: a peaked (confident)
                      distribution has high variance, a flat one ~0.
    * ``entropy``   - negative Shannon entropy.
    """
    if kind == "max_prob":
        return probs.max(dim=1).values
    if kind == "variance":
        return probs.var(dim=1, unbiased=False)
    if kind == "entropy":
        return (probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    raise ValueError(f"Unknown confidence '{kind}'")


# --------------------------------------------------------------------------
@STRATEGIES.register("supervised")
class Supervised(Strategy):
    """FedAvg on the labeled subset only (lower-bound baseline).
    Combine with ``labeled_ratio: 1.0`` for the fully-supervised upper bound."""

    uses_unlabeled = False

    def compute_loss(self, model, labeled, unlabeled):
        if labeled is None:
            raise RuntimeError("Supervised strategy needs labeled data on every client")
        x, y = labeled
        return self.loss(sup=supervised_ce(model(x), y))


# --------------------------------------------------------------------------
@STRATEGIES.register("pseudolabel")
class PseudoLabel(Strategy):
    """Classic hard pseudo-labeling (Lee, 2013) with a confidence threshold."""

    def __init__(self, cfg, num_classes, tau: float = 0.95, lambda_u: float = 1.0, **params):
        params.setdefault("loss_weights", {"sup": 1.0, "unsup": lambda_u})
        super().__init__(cfg, num_classes, **params)
        self.tau = tau

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled if labeled is not None else (None, None)
        u_w, _, u_true = unlabeled if unlabeled is not None else (None, None, None)
        logits = _forward_cat(model, x, u_w)
        terms, logs = {}, {}
        if x is not None:
            terms["sup"] = supervised_ce(logits.pop(0), y)
        if u_w is not None:
            lu = logits.pop(0)
            conf, pseudo = F.softmax(lu.detach(), 1).max(1)
            mask = conf >= self.tau
            terms["unsup"] = masked_ce(lu, pseudo, mask)
            logs.update(_pl_stats(pseudo, mask, u_true))
        total, l = self.loss(**terms)
        return total, {**l, **logs}


# --------------------------------------------------------------------------
@STRATEGIES.register("fixmatch")
class FixMatch(Strategy):
    """FedAvg + FixMatch (Sohn et al., 2020): pseudo-label from the weak view,
    enforce it on the strong view."""

    def __init__(self, cfg, num_classes, tau: float = 0.95, lambda_u: float = 1.0, **params):
        params.setdefault("loss_weights", {"sup": 1.0, "unsup": lambda_u})
        super().__init__(cfg, num_classes, **params)
        self.tau = tau

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled if labeled is not None else (None, None)
        u_w, u_s, u_true = unlabeled if unlabeled is not None else (None, None, None)
        logits = _forward_cat(model, x, u_w, u_s)
        terms, logs = {}, {}
        if x is not None:
            terms["sup"] = supervised_ce(logits.pop(0), y)
        if u_w is not None:
            lw, ls = logits
            conf, pseudo = F.softmax(lw.detach(), 1).max(1)
            mask = conf >= self.tau
            terms["unsup"] = masked_ce(ls, pseudo, mask)
            logs.update(_pl_stats(pseudo, mask, u_true))
        total, l = self.loss(**terms)
        return total, {**l, **logs}


# --------------------------------------------------------------------------
@STRATEGIES.register("fedlabel")
class FedLabel(Strategy):
    """FedLabel (Cho et al., ICCV 2023) - selective local/global pseudo-labeling.

    For every unlabeled sample both the *local* model (being fine-tuned on the
    client's data) and the *frozen global* model ``W_t`` predict on the weak
    view. Whichever is more confident (``confidence`` score) provides the
    pseudo-label; it is kept if its max-probability exceeds ``tau`` and then
    enforced on the strong view. A global-local KL consistency term keeps the
    local model from drifting away from the global knowledge.

    ``selection`` = ``adaptive`` (FedLabel) | ``local`` | ``global`` (ablations).
    ``local_warmup_steps`` = supervised-only steps that let the local model
    specialise before it is trusted as a pseudo-labeler.
    """

    needs_global_model = True

    def __init__(self, cfg, num_classes, tau: float = 0.95, lambda_u: float = 1.0,
                 lambda_kl: float = 1.0, temperature: float = 1.0,
                 confidence: str = "variance", selection: str = "adaptive",
                 local_warmup_steps: int = 0, **params):
        params.setdefault("loss_weights", {"sup": 1.0, "unsup": lambda_u, "kl": lambda_kl})
        super().__init__(cfg, num_classes, **params)
        if selection not in ("adaptive", "local", "global"):
            raise ValueError(f"Unknown selection '{selection}'")
        self.tau, self.temperature = tau, temperature
        self.confidence, self.selection = confidence, selection
        self.local_warmup_steps = local_warmup_steps

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled if labeled is not None else (None, None)
        warmup = self.step < self.local_warmup_steps and x is not None
        if warmup or unlabeled is None:
            total, l = self.loss(sup=supervised_ce(model(x), y))
            return total, l

        u_w, u_s, u_true = unlabeled
        logits = _forward_cat(model, x, u_w, u_s)
        terms, logs = {}, {}
        if x is not None:
            terms["sup"] = supervised_ce(logits.pop(0), y)
        lw, ls = logits

        with torch.no_grad():
            g_logits = self.global_model(u_w)
            p_local = F.softmax(lw.detach(), 1)
            p_global = F.softmax(g_logits, 1)
            if self.selection == "local":
                use_local = torch.ones(len(u_w), dtype=torch.bool, device=u_w.device)
            elif self.selection == "global":
                use_local = torch.zeros(len(u_w), dtype=torch.bool, device=u_w.device)
            else:
                use_local = (confidence_score(p_local, self.confidence)
                             >= confidence_score(p_global, self.confidence))
            p_sel = torch.where(use_local[:, None], p_local, p_global)
            conf, pseudo = p_sel.max(1)
            mask = conf >= self.tau

        terms["unsup"] = masked_ce(ls, pseudo, mask)
        terms["kl"] = kl_consistency(lw, g_logits, self.temperature)
        logs.update(_pl_stats(pseudo, mask, u_true))
        logs["local_selected"] = use_local.float().mean()
        total, l = self.loss(**terms)
        return total, {**l, **logs}
