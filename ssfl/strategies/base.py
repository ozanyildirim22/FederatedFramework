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


def _native_bf16() -> bool:
    """bf16 in hardware (Ampere / compute capability 8.0 and newer), not emulated."""
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:                                # older PyTorch: no emulation flag
        return torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8


def round_batches(loader, steps: int) -> Iterator:
    """Exactly ``steps`` shuffled batches from ``loader`` for one local round.

    Without worker processes this simply cycles the loader. With workers, cycling
    would start a fresh set of worker processes every time the loader runs out,
    and a client's labeled set is often only a few batches long, so a round
    started them many times. Here one loader covers the whole round, re-shuffling
    after every pass over the data, so its workers start once per round.
    """
    if loader.num_workers == 0:
        return _cycle(loader)
    from torch.utils.data import DataLoader, RandomSampler

    ds, bs = loader.dataset, loader.batch_size
    sampler = RandomSampler(ds, replacement=False, num_samples=steps * bs)  # repeats permutations
    return iter(DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=loader.num_workers,
                           pin_memory=loader.pin_memory, drop_last=False))


COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune")


def compile_mode(cfg) -> Optional[str]:
    """``strategy.compile`` as a torch.compile mode, or None when off (``true`` = ``default``)."""
    v = getattr(cfg, "compile", None)
    if v is None or v is False or str(v).lower() in ("", "none", "off", "false", "0"):
        return None
    if v is True or str(v).lower() in ("true", "on", "1"):
        return "default"
    if v not in COMPILE_MODES:
        raise ValueError(f"Unknown strategy.compile '{v}' (none | {' | '.join(COMPILE_MODES)})")
    return str(v)


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
        self._channels_last = False
        self.aggregation_weight: Optional[float] = None   # set by strategies that define r_k
        self._compile_mode = compile_mode(cfg)
        self._compiled: Optional[Tuple[nn.Module, nn.Module]] = None   # (model, torch.compile(model))

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
        out = []
        for t in batch:
            if torch.is_tensor(t):
                t = t.to(self.device, non_blocking=True)
                if self._channels_last and t.dim() == 4:
                    t = t.contiguous(memory_format=torch.channels_last)
            out.append(t)
        return tuple(out)

    # ------------------------------------------------------------ precision
    def precision(self, device: torch.device) -> str:
        """Effective precision: ``fp32`` | ``bf16`` | ``fp16`` (``amp: true`` = fp16).
        bf16 on a GPU without bf16 support and fp16 on the CPU fall back to fp32."""
        p = str(getattr(self.cfg, "precision", "fp32") or "fp32").lower()
        if p == "fp32" and getattr(self.cfg, "amp", False):
            p = "fp16"
        if p not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"Unknown precision '{p}' (fp32 | bf16 | fp16)")
        if p == "fp16" and device.type != "cuda":
            return "fp32"
        if p == "bf16" and device.type == "cuda" and not _native_bf16():
            return "fp32"            # e.g. T4: bf16 would only be emulated (slower than fp32)
        return p

    def training_forward(self, model: nn.Module) -> nn.Module:
        """The module the loss functions call: ``model`` itself, or with ``strategy.compile`` a
        ``torch.compile`` wrapper that shares its parameters (built once and reused: the
        experiment trains the same model object every round). Only the trained model is
        compiled; frozen copies (global / local / helper models) and evaluation stay eager."""
        if self._compile_mode is None:
            return model
        if self._compiled is None or self._compiled[0] is not model:
            self._compiled = (model, torch.compile(model, mode=self._compile_mode))
        return self._compiled[1]

    # -------------------------------------------------------- loop pieces
    def prepare(self, model: nn.Module, device: torch.device) -> None:
        """Device / memory format, frozen copy of W_t and the FedProx anchor."""
        self.device = device
        self._channels_last = bool(getattr(self.cfg, "channels_last", False)) and device.type == "cuda"
        if self._channels_last:
            model.to(memory_format=torch.channels_last)
        if self.needs_global_model:
            self.global_model = copy.deepcopy(model).eval()
            for p in self.global_model.parameters():
                p.requires_grad_(False)
        if self.prox_mu > 0:
            self._global_params = [p.detach().clone() for p in model.parameters()]

    def optimize(self, model: nn.Module, steps: int, lab_it, unl_it, loss_fn=None,
                 sums: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """``steps`` SGD steps of ``loss_fn(model, labeled, unlabeled)`` (default
        ``compute_loss``) with a fresh optimiser. Handles precision, the FedProx term
        and gradient clipping. A loss function may return ``(None, logs)`` to skip the
        update of that step (its logs still count). Logged values are summed into ``sums`` on the device:
        a float() per value per step would make the CPU wait for the GPU every step."""
        loss_fn = loss_fn or self.compute_loss
        model.train()
        opt = self.make_optimizer(model)
        # bf16 has the float32 range, so it needs no loss scaling; fp16 does
        prec = self.precision(self.device)
        use_amp = prec != "fp32"
        amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
        sums = sums if sums is not None else defaultdict(float)
        fwd = self.training_forward(model)
        cudagraphs = self._compile_mode == "reduce-overhead"
        for step in range(steps):
            self.step = step
            labeled = self._to_device(next(lab_it)) if lab_it else None
            unlabeled = self._to_device(next(unl_it)) if unl_it else None
            if cudagraphs:                        # a new training iteration for CUDA graph trees
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(self.device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = loss_fn(fwd, labeled, unlabeled)
            if loss is None:                      # the loss function skipped this step (no update)
                for k, v in logs.items():
                    sums[k] = sums[k] + (v.detach().float() if torch.is_tensor(v) else v)
                continue
            prox = self.proximal_term(model)
            if prox is not None:
                loss = loss + prox
                logs["loss_prox"] = prox.detach()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in logs.items():
                sums[k] = sums[k] + (v.detach().float() if torch.is_tensor(v) else v)
        return sums

    @staticmethod
    def summarize(sums: Dict[str, object], steps: Dict[str, int] | int) -> Dict[str, float]:
        """Per-step means of the summed logs (``steps`` may map a key prefix to its own
        step count, e.g. for two-phase strategies); ``_pl_correct / _pl_count`` become
        ``pl_acc``: the accuracy over all pseudo-labeled samples of the round."""
        out = {k: float(v) for k, v in sums.items()}
        correct, count = out.pop("_pl_correct", None), out.pop("_pl_count", None)
        metrics = {}
        for k, v in out.items():
            n = steps if isinstance(steps, int) else next(
                (s for p, s in steps.items() if k.startswith(p)), steps.get("", 1))
            metrics[k] = v / max(1, n)
        if count:
            metrics["pl_acc"] = correct / count
        return metrics

    def local_train(self, model: nn.Module, client, device: torch.device) -> Dict[str, float]:
        self.prepare(model, device)
        self.on_round_start(model, client)
        steps = self.num_steps(client)
        self.total_steps = steps
        lab_it = round_batches(client.data.labeled, steps) if client.data.labeled is not None else None
        unl_it = (round_batches(client.data.unlabeled, steps)
                  if self.uses_unlabeled and client.data.unlabeled is not None else None)
        sums = self.optimize(model, steps, lab_it, unl_it)
        self.on_round_end(model, client)
        self.global_model, self._global_params = None, None
        return self.summarize(sums, steps)
