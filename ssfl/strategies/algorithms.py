"""Built-in client strategies (pseudo-labeling engines)."""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..losses import masked_ce, supervised_ce
from ..registry import STRATEGIES
from .base import Strategy, round_batches


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


def confidence_score(logits: torch.Tensor, kind: str = "variance") -> torch.Tensor:
    """Per-sample confidence h(.) of a prediction, computed from its logits. All scores
    are positive, so ratios of them (FedLabel's adaptive KL weight) are meaningful.

    * ``variance``      - variance of the **logit** vector (FedLabel / the ADD): a peaked
                          prediction has a high dispersion, a flat one ~0.
    * ``prob_variance`` - variance of the softmax probabilities.
    * ``max_prob``      - highest softmax probability.
    * ``entropy``       - exp(-entropy) of the softmax distribution.
    """
    logits = logits.float()
    if kind == "variance":
        return logits.var(dim=1, unbiased=False)
    probs = F.softmax(logits, dim=1)
    if kind == "prob_variance":
        return probs.var(dim=1, unbiased=False)
    if kind == "max_prob":
        return probs.max(dim=1).values
    if kind == "entropy":
        return torch.exp((probs * probs.clamp_min(1e-12).log()).sum(dim=1))
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
@STRATEGIES.register("uda")
class UDA(Strategy):
    """FedAvg + UDA (Xie et al., NeurIPS 2020), a baseline of the FedLabel paper.

    Supervised cross-entropy on labeled data plus unsupervised consistency: the weak-view
    prediction, sharpened with ``temperature``, is the target for the strong-view prediction
    (KL divergence), only for samples whose weak-view top probability exceeds ``threshold``
    (UDA's confidence-based masking). Loss averaged over the whole unlabeled batch.
    FedProx + UDA = this strategy with ``strategy.prox_mu > 0``.
    Defaults are UDA's: temperature 0.4, threshold 0.8, lambda_u 1.
    """

    def __init__(self, cfg, num_classes, lambda_u: float = 1.0, temperature: float = 0.4,
                 threshold: float = 0.8, **params):
        params.setdefault("loss_weights", {"sup": 1.0, "unsup": lambda_u})
        super().__init__(cfg, num_classes, **params)
        self.temperature, self.threshold = float(temperature), float(threshold)

    def compute_loss(self, model, labeled, unlabeled):
        x, y = labeled if labeled is not None else (None, None)
        u_w, u_s, u_true = unlabeled if unlabeled is not None else (None, None, None)
        logits = _forward_cat(model, x, u_w, u_s)
        terms, logs = {}, {}
        if x is not None:
            terms["sup"] = supervised_ce(logits.pop(0), y)
        if u_w is not None:
            lw, ls = logits
            with torch.no_grad():
                p_w = F.softmax(lw.detach().float(), dim=1)
                conf, pseudo = p_w.max(dim=1)
                mask = conf > self.threshold
                target = F.softmax(lw.detach().float() / self.temperature, dim=1)
            kl_each = (target * (target.clamp_min(1e-12).log() - F.log_softmax(ls.float(), dim=1))).sum(dim=1)
            terms["unsup"] = (kl_each * mask.float()).mean()
            logs.update(_pl_stats(pseudo, mask, u_true))
        total, l = self.loss(**terms)
        return total, {**l, **logs}


# --------------------------------------------------------------------------
def fedlabel_confidence(logits: torch.Tensor, kind: str = "variance") -> torch.Tensor:
    """FedLabel's confidence h(.) >= 0 (as in the test1 implementation). The logits are
    clamped to [-40, 40] first.

    * ``variance`` - variance of the logit vector (ADD / paper).
    * ``entropy``  - log(K) - entropy of the softmax distribution.
    * ``max_prob`` - highest softmax probability.
    """
    safe = logits.float().clamp(-40.0, 40.0)
    if kind == "variance":
        return safe.var(dim=-1, unbiased=False)
    probs = F.softmax(safe, dim=-1)
    if kind == "entropy":
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return (math.log(max(safe.size(-1), 2)) - ent).clamp_min(0.0)
    if kind == "max_prob":
        return probs.max(dim=-1)[0]
    raise ValueError(f"Unknown confidence metric '{kind}' (variance | entropy | max_prob)")


@STRATEGIES.register("fedlabel")
class FedLabel(Strategy):
    """FedLabel (Cho et al., ICCV 2023). The client algorithm is the one of the ``test1``
    implementation (``fedlabel/client.py``); the data (partition, labeled / unlabeled split,
    weak and strong views, batch sizes) comes from the framework's own preprocessing.

    **Stage 1 - supervised local update.** ``w_L`` starts from ``W_t`` and takes tau SGD
    steps of cross-entropy on labeled batches (a batch of one sample, or a non-finite loss,
    is skipped). ``dw_L = w_L - W_t``.

    **Stage 2 - semi-supervised update.** ``w_U`` starts again from ``W_t``. For each of the
    tau' unlabeled batches the frozen global model ``W_t`` and the frozen local model ``w_L``
    give logits on the weak view and are scored with ``h(.)`` (``confidence``). The global
    model is the pseudo-labeller ``s*`` only if it is strictly more confident
    (``h(s_G) > h(s_L)``, ties go to the local model); the other is ``s_bar``. Samples with
    ``max softmax(s*) > beta`` are kept. If fewer than two samples are kept, the step is
    skipped (no update). Otherwise ``w_U`` sees only the kept samples' strong view:

        loss = lambda_u * CE(w_U(strong), y_hat)                    (mean over kept samples)
             + mean over kept samples with argmax s_bar = y_hat of
               lambda_k * KL(softmax(s_bar) || softmax(w_U(strong)))

    with ``lambda_k = lambda0 * clamp(h(s_bar) / h(s*), 0, 1)``.

    **Stage 3 - merging.** Parameters: ``dw = dw_L + dw_U``. Floating-point buffers (BatchNorm
    running statistics): the mean of the two deltas if ``w_U`` was updated, else ``dw_L``.
    Integer buffers keep the value of ``W_t``. Aggregation weight
    ``r_k = |labeled set| + number of kept unlabeled samples`` (``weight_by_used``).

    Parameters
    ----------
    beta          pseudo-label threshold (strict ``>``).
    lambda0       upper bound of the consistency weight lambda_k.
    lambda_u      weight of the pseudo-label cross-entropy (1 in test1).
    sup_steps     tau: SGD steps on labeled data. If unset, ``sup_epochs``.
    unsup_steps   tau': unlabeled batches. If unset, ``unsup_epochs``.
    sup_epochs    epochs over the labeled data when ``sup_steps`` is unset; default ``strategy.local_epochs``.
    unsup_epochs  epochs over the unlabeled data when ``unsup_steps`` is unset; default ``strategy.local_epochs``.
                  ``strategy.local_steps``, if set and no ``*_steps`` is given, gives both phases that many steps.
    weight_by_used  r_k as above; False = the server's ``weight_by`` (dataset size).
    confidence    h(.): ``variance`` | ``entropy`` | ``max_prob`` (see :func:`fedlabel_confidence`).
    selection     ``adaptive`` (FedLabel) | ``local`` | ``global`` (ablations: always one model).

    Older parameter names (``tau``, ``lambda_kl``) are still accepted; ``kl_direction``,
    ``local_warmup_steps`` and ``temperature`` are ignored.
    """

    needs_global_model = True
    param_aliases = {"tau": "beta", "lambda_kl": "lambda0"}            # names before the ADD alignment
    retired_params = ("local_warmup_steps", "temperature", "kl_direction")

    def __init__(self, cfg, num_classes, beta: float = 0.5, lambda0: float = 1.0, lambda_u: float = 1.0,
                 sup_epochs: Optional[int] = None, unsup_epochs: Optional[int] = None,
                 confidence: str = "variance", selection: str = "adaptive",
                 sup_steps: Optional[int] = None, unsup_steps: Optional[int] = None,
                 weight_by_used: bool = True, **params):
        beta = params.pop("tau", beta)                       # older configs
        lambda0 = params.pop("lambda_kl", lambda0)
        for old in self.retired_params:
            params.pop(old, None)
        super().__init__(cfg, num_classes, **params)
        if selection not in ("adaptive", "local", "global"):
            raise ValueError(f"Unknown selection '{selection}'")
        fedlabel_confidence(torch.zeros(1, 2), confidence)   # validates the name
        self.beta, self.lambda0, self.lambda_u = float(beta), float(lambda0), float(lambda_u)
        self.sup_epochs, self.unsup_epochs = sup_epochs, unsup_epochs
        self.sup_steps, self.unsup_steps = sup_steps, unsup_steps
        self.weight_by_used = bool(weight_by_used)
        self.confidence, self.selection = confidence, selection
        self.local_model: Optional[torch.nn.Module] = None

    # ------------------------------------------------------------ steps
    def _phase_steps(self, loader, epochs: Optional[int], steps: Optional[int] = None) -> int:
        if loader is None:
            return 0
        if steps:                                            # tau / tau' given as SGD steps (paper)
            return int(steps)
        if self.cfg.local_steps:
            return int(self.cfg.local_steps)
        return int(epochs if epochs is not None else self.cfg.local_epochs) * max(1, len(loader))

    def num_steps(self, client) -> int:
        return (self._phase_steps(client.data.labeled, self.sup_epochs, self.sup_steps)
                + self._phase_steps(client.data.unlabeled, self.unsup_epochs, self.unsup_steps))

    # ------------------------------------------------------------ losses
    def _sup_loss(self, model, labeled, _):
        x, y = labeled
        if x.size(0) <= 1:
            return None, {}
        loss = F.cross_entropy(model(x), y.long().view(-1))
        if not torch.isfinite(loss):
            return None, {}
        return loss, {"loss_sup": loss.detach()}

    def _unsup_loss(self, model, _, unlabeled):
        u_w, u_s, u_true = unlabeled
        zero = u_w.new_zeros(())
        logs = {"loss_unsup": zero, "loss_kl": zero, "mask_rate": zero, "local_selected": zero,
                "kl_rate": zero, "lambda_k": zero, "_pl_correct": zero, "_pl_count": zero,
                "_used_unlabeled": zero}
        if u_w.size(0) <= 1:
            return None, logs
        with torch.no_grad():
            s_g, s_l = self.global_model(u_w).float(), self.local_model(u_w).float()
            if not (torch.isfinite(s_g).all() and torch.isfinite(s_l).all()):
                return None, logs
            h_g = fedlabel_confidence(s_g, self.confidence)
            h_l = fedlabel_confidence(s_l, self.confidence)
            if self.selection == "local":
                g_better = torch.zeros_like(h_g, dtype=torch.bool)
            elif self.selection == "global":
                g_better = torch.ones_like(h_g, dtype=torch.bool)
            else:                                            # test1: global only if strictly more confident
                g_better = h_g > h_l
            s_sel = torch.where(g_better[:, None], s_g, s_l)
            s_bar = torch.where(g_better[:, None], s_l, s_g)
            h_sel = torch.where(g_better, h_g, h_l)
            h_bar = torch.where(g_better, h_l, h_g)
            max_probs, pseudo = F.softmax(s_sel, dim=-1).max(dim=-1)
            kept = max_probs > self.beta
            logs.update(mask_rate=kept.float().mean(), local_selected=(~g_better).float().mean(),
                        _pl_correct=((pseudo == u_true) & kept).float().sum(), _pl_count=kept.float().sum())
        n_kept = int(kept.sum())
        if n_kept <= 1:                                      # test1: too few samples -> no update
            return None, logs

        pseudo_k, s_bar_k = pseudo[kept], s_bar[kept]
        logits_u = model(u_s[kept]).float()                  # strong view of the kept samples only
        loss_ce = F.cross_entropy(logits_u, pseudo_k)

        agree = s_bar_k.argmax(dim=-1) == pseudo_k
        loss_kl, lam_mean = zero, zero
        if agree.any():
            log_p_u = F.log_softmax(logits_u[agree], dim=-1)
            log_p_bar = F.log_softmax(s_bar_k[agree], dim=-1)
            ratio = h_bar[kept][agree] / h_sel[kept][agree].clamp_min(1e-6)
            lam = self.lambda0 * ratio.clamp(0.0, 1.0)
            kl_each = F.kl_div(log_p_u, log_p_bar, reduction="none", log_target=True).sum(dim=-1)
            loss_kl = (lam * kl_each).mean()
            lam_mean = lam.mean()
        loss = self.lambda_u * loss_ce + loss_kl
        if not torch.isfinite(loss):
            return None, logs
        logs.update(loss_unsup=loss_ce.detach(), loss_kl=loss_kl.detach(),
                    kl_rate=agree.float().sum() / max(1, u_w.size(0)), lambda_k=lam_mean.detach(),
                    _used_unlabeled=float(n_kept))
        self._unlabeled_updated = True
        return loss, logs

    # ------------------------------------------------------------ round
    def local_train(self, model, client, device):
        self.prepare(model, device)                          # frozen copy of W_t
        self.on_round_start(model, client)
        start = {k: v.detach().clone() for k, v in model.state_dict().items()}
        lab, unl = client.data.labeled, client.data.unlabeled
        tau = self._phase_steps(lab, self.sup_epochs, self.sup_steps)
        tau_u = self._phase_steps(unl, self.unsup_epochs, self.unsup_steps)
        self.total_steps = tau + tau_u
        sums: Dict[str, object] = defaultdict(float)

        # Stage 1: supervised local update w_L (tau steps from W_t)
        if tau:
            self.optimize(model, tau, round_batches(lab, tau), None, self._sup_loss, sums)
        after_l = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.local_model = copy.deepcopy(model).eval()
        for p in self.local_model.parameters():
            p.requires_grad_(False)

        # Stage 2: semi-supervised update w_U (tau' unlabeled batches, again from W_t)
        model.load_state_dict(start)
        self._unlabeled_updated = False
        if tau_u:
            self.optimize(model, tau_u, None, round_batches(unl, tau_u), self._unsup_loss, sums)
        after_u = {k: v.detach() for k, v in model.state_dict().items()}

        # Stage 3: merge. Parameters dw_L + dw_U; float buffers averaged (keeps BatchNorm
        # running_var positive) if w_U was updated, else dw_L; integer buffers stay at W_t.
        param_keys = {k for k, _ in model.named_parameters()}
        merged = {}
        for k, v0 in start.items():
            if not v0.is_floating_point():
                merged[k] = v0
                continue
            d_l, d_u = after_l[k] - v0, after_u[k] - v0
            if k in param_keys:
                d = d_l + d_u
            elif self._unlabeled_updated:
                d = 0.5 * (d_l + d_u)
            else:
                d = d_l
            merged[k] = v0 + torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        model.load_state_dict(merged)
        self.on_round_end(model, client)
        self.global_model, self.local_model, self._global_params = None, None, None

        # r_k = labeled samples of the client + unlabeled samples that passed the threshold
        used_u = float(sums.pop("_used_unlabeled", 0.0))
        used_l = float(len(lab.dataset)) if lab is not None else 0.0
        if self.weight_by_used:
            self.aggregation_weight = max(1.0, used_l + used_u)
        metrics = self.summarize(sums, {"loss_sup": tau, "loss_prox": tau + tau_u, "": tau_u})
        metrics["used_samples"] = used_l + used_u
        metrics["loss"] = metrics.get("loss_sup", 0.0) + metrics.get("loss_unsup", 0.0) + metrics.get("loss_kl", 0.0)
        return metrics
