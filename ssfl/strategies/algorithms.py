"""Built-in client strategies (pseudo-labeling engines)."""
from __future__ import annotations

import copy
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
@STRATEGIES.register("fedlabel")
class FedLabel(Strategy):
    """FedLabel (Cho et al., ICCV 2023), as specified in the project ADD (Section 3.1,
    Algorithm 2). Each client round has two updates that both start from W_t:

    **Step 1 - supervised local update.** ``w_L = W_t``; tau SGD steps of cross-entropy
    on the labeled data give the local model ``w_L`` and ``dw_L = w_L - W_t``.

    **Step 2 - confidence-based model selection.** For every unlabeled sample both the
    frozen global model ``W_t`` and the frozen local model ``w_L`` give logits on the weak
    view. ``h(.)`` (``confidence``; default: variance of the logits) scores them; the more
    confident one (ties go to the global model) is the pseudo-labeller ``s*``, the other is
    ``s_bar``. The hard label ``argmax s*`` is kept if ``max softmax(s*) >= beta`` and then
    trained on the strong view with cross-entropy.

    **Step 3 - global-local consistency.** For kept samples where ``argmax s_bar`` gives the
    same label, a KL term pulls the training model's weak-view output towards ``s_bar``,
    weighted by ``lambda_k = lambda0 * h(s_bar) / h(s*)``.

    The semi-supervised model ``w_U`` starts again from ``W_t`` and takes tau' SGD steps on

        F_U = mean over the unlabeled batch of
              [ lambda_u * CE(w_U, strong(xi), y_hat) + lambda_k * KL(.) * 1[argmax s_bar = y_hat] ] * 1[kept]

    and ``dw_U = w_U - W_t``. The client sends ``dw = dw_L + dw_U`` (Eq. 4); the server's
    weighted average of ``W_t + dw_k`` is Eq. 5.

    Parameters
    ----------
    beta          pseudo-label threshold (ADD: beta). Default 0.5: the paper's appendix A (Fig. 10)
                  finds 0.3-0.5 best and 0.9 the worst (strong augmentation lowers the confidence).
    lambda0       global scale of the consistency weight (ADD: lambda_0).
    lambda_u      weight of the pseudo-label cross-entropy (1 in the ADD).
    sup_steps     tau: SGD steps on labeled data in Step 1 (the paper counts tau in local
                  iterations; its ablation found tau = 5 the worst). If unset, ``sup_epochs``.
    unsup_steps   tau': SGD steps on unlabeled data in Steps 2-3. If unset, ``unsup_epochs``.
    sup_epochs    epochs over the labeled data when ``sup_steps`` is unset; default ``strategy.local_epochs``.
    unsup_epochs  epochs over the unlabeled data when ``unsup_steps`` is unset; default ``strategy.local_epochs``.
                  ``strategy.local_steps``, if set and no ``*_steps`` is given, gives both phases that many steps.
    weight_by_used  aggregation weight r_k = samples used this round (labeled samples processed +
                  unlabeled samples that passed the threshold), as in the paper's Alg. 1; False =
                  the server's ``weight_by`` (dataset size).
    confidence    h(.): ``variance`` (logits, ADD) | ``prob_variance`` | ``max_prob`` | ``entropy``.
    selection     ``adaptive`` (FedLabel) | ``local`` | ``global`` (ablations: always one model).
    kl_direction  ``add``: KL(s(w_U) || s_bar) exactly as written in the ADD (Eq. 2);
                  ``teacher``: KL(s_bar || s(w_U)), the usual distillation direction.

    The names used before this version (``tau``, ``lambda_kl``, ``local_warmup_steps``) are
    still accepted, so older configs keep loading.
    """

    needs_global_model = True
    param_aliases = {"tau": "beta", "lambda_kl": "lambda0"}            # names before the ADD alignment
    retired_params = ("local_warmup_steps", "temperature")

    def __init__(self, cfg, num_classes, beta: float = 0.5, lambda0: float = 1.0, lambda_u: float = 1.0,
                 sup_epochs: Optional[int] = None, unsup_epochs: Optional[int] = None,
                 confidence: str = "variance", selection: str = "adaptive", kl_direction: str = "add",
                 sup_steps: Optional[int] = None, unsup_steps: Optional[int] = None,
                 weight_by_used: bool = True, **params):
        beta = params.pop("tau", beta)                       # older configs
        lambda0 = params.pop("lambda_kl", lambda0)
        for old in ("local_warmup_steps", "temperature"):
            params.pop(old, None)
        super().__init__(cfg, num_classes, **params)
        if selection not in ("adaptive", "local", "global"):
            raise ValueError(f"Unknown selection '{selection}'")
        if kl_direction not in ("add", "teacher"):
            raise ValueError(f"Unknown kl_direction '{kl_direction}'")
        confidence_score(torch.zeros(1, 2), confidence)      # validates the name
        self.beta, self.lambda0, self.lambda_u = float(beta), float(lambda0), float(lambda_u)
        self.sup_epochs, self.unsup_epochs = sup_epochs, unsup_epochs
        self.sup_steps, self.unsup_steps = sup_steps, unsup_steps
        self.weight_by_used = bool(weight_by_used)
        self.confidence, self.selection, self.kl_direction = confidence, selection, kl_direction
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
        loss = supervised_ce(model(x), y)
        return loss, {"loss_sup": loss.detach()}

    def _unsup_loss(self, model, _, unlabeled):
        u_w, u_s, u_true = unlabeled
        lw, ls = model(torch.cat([u_w, u_s])).split([len(u_w), len(u_s)])
        with torch.no_grad():
            s_g, s_l = self.global_model(u_w).float(), self.local_model(u_w).float()
            h_g, h_l = confidence_score(s_g, self.confidence), confidence_score(s_l, self.confidence)
            if self.selection == "local":
                use_local = torch.ones_like(h_l, dtype=torch.bool)
            elif self.selection == "global":
                use_local = torch.zeros_like(h_l, dtype=torch.bool)
            else:                                            # Alg. 2: if h(s_G) >= h(s_L) -> global
                use_local = h_l > h_g
            s_sel = torch.where(use_local[:, None], s_l, s_g)
            s_bar = torch.where(use_local[:, None], s_g, s_l)
            h_sel = torch.where(use_local, h_l, h_g)
            h_bar = torch.where(use_local, h_g, h_l)
            conf, pseudo = F.softmax(s_sel, dim=1).max(dim=1)
            kept = conf > self.beta                          # Eq. 4: max s*(xi) > beta (strict)
            agree = kept & (s_bar.argmax(dim=1) == pseudo)
            # lambda_k = lambda0 * h(s_bar) / h(s*); <= lambda0 when s* is the more confident
            # one (adaptive); capped there for the forced local / global ablations
            lam = self.lambda0 * (h_bar / h_sel.clamp_min(1e-12)).clamp(max=1.0)
            weight = lam * agree.float()
        ce = masked_ce(ls, pseudo, kept)
        log_p_u = F.log_softmax(lw.float(), dim=1)
        log_p_bar = F.log_softmax(s_bar, dim=1)
        if self.kl_direction == "add":                       # KL(s(w_U) || s_bar)
            kl_each = (log_p_u.exp() * (log_p_u - log_p_bar)).sum(dim=1)
        else:                                                # KL(s_bar || s(w_U))
            kl_each = (log_p_bar.exp() * (log_p_bar - log_p_u)).sum(dim=1)
        kl = (kl_each * weight).mean()                       # averaged over the whole batch (Eq. 3)
        loss = self.lambda_u * ce + kl
        logs = {"loss_unsup": ce.detach(), "loss_kl": kl.detach(),
                "local_selected": use_local.float().mean(), "kl_rate": agree.float().mean(),
                "lambda_k": (lam * agree).sum() / agree.sum().clamp_min(1)}
        logs.update(_pl_stats(pseudo, kept, u_true))
        logs["_used_unlabeled"] = kept.float().sum()          # for the aggregation weight r_k
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

        # Step 1: supervised local update w_L (tau steps from W_t)
        if tau:
            self.optimize(model, tau, round_batches(lab, tau), None, self._sup_loss, sums)
        after_l = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.local_model = copy.deepcopy(model).eval()
        for p in self.local_model.parameters():
            p.requires_grad_(False)

        # Steps 2-3: semi-supervised update w_U (tau' steps, again from W_t)
        model.load_state_dict(start)
        if tau_u:
            self.optimize(model, tau_u, None, round_batches(unl, tau_u), self._unsup_loss, sums)
        after_u = model.state_dict()

        # Eq. 4: dw = dw_L + dw_U  ->  the model sent back is W_t + dw_L + dw_U
        model.load_state_dict({k: (v0 + (after_l[k] - v0) + (after_u[k] - v0)) if v0.is_floating_point()
                               else after_u[k] for k, v0 in start.items()})
        self.on_round_end(model, client)
        self.global_model, self.local_model, self._global_params = None, None, None
        # Alg. 1, line 8: r_k = number of samples used for dw_L and dw_U this round (labeled
        # samples processed + unlabeled samples that passed the threshold)
        used_u = float(sums.pop("_used_unlabeled", 0.0))
        used_l = float(tau * lab.batch_size) if tau else 0.0
        if self.weight_by_used:
            self.aggregation_weight = used_l + used_u
        metrics = self.summarize(sums, {"loss_sup": tau, "loss_prox": tau + tau_u, "": tau_u})
        metrics["used_samples"] = used_l + used_u
        metrics["loss"] = metrics.get("loss_sup", 0.0) + metrics.get("loss_unsup", 0.0) + metrics.get("loss_kl", 0.0)
        return metrics
