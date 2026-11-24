"""Speed check: time a few training rounds of one config under different speed settings.

    python -m ssfl -c configs/cifar10_fedlabel.yaml --bench 5 --set data.augment_device=gpu strategy.precision=bf16

Each setting builds a fresh experiment, runs two warm-up rounds (torch.compile compiles and the
cuDNN benchmark tunes its kernels there), then times ``rounds`` rounds without evaluation.
Nothing is written to disk.
The last column extrapolates to the config's ``server.rounds``, evaluation not included.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, List, Optional, Sequence

import torch

from .config import ExperimentConfig, apply_overrides
from .experiment import Experiment

#: (label, overrides); the compile modes need Triton (Linux / Colab) and are skipped with
#: their error where it is missing
SETTINGS = [
    ("baseline", ["strategy.channels_last=false", "strategy.cudnn_benchmark=false", "strategy.compile=none"]),
    ("cudnn_benchmark", ["strategy.channels_last=false", "strategy.cudnn_benchmark=true", "strategy.compile=none"]),
    ("+ channels_last", ["strategy.channels_last=true", "strategy.cudnn_benchmark=true", "strategy.compile=none"]),
    ("+ compile", ["strategy.channels_last=false", "strategy.cudnn_benchmark=true", "strategy.compile=default"]),
    ("+ compile cudagraphs", ["strategy.channels_last=false", "strategy.cudnn_benchmark=true",
                              "strategy.compile=reduce-overhead"]),
]
WARMUP_ROUNDS = 2      # compilation (and cuDNN tuning) happens here, mostly in the first round


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_setting(cfg: ExperimentConfig, rounds: int, overrides: Sequence[str]) -> dict:
    # evaluation off: the timed rounds measure training only
    run_cfg = ExperimentConfig.from_dict(apply_overrides(cfg.to_dict(), list(overrides)
                                                         + ["server.eval_every=1000000"]))
    exp = Experiment(run_cfg, verbose=False)
    exp.cfg.server.rounds = 10 ** 9                      # no "last round" evaluation either
    try:
        t0 = time.perf_counter()
        for r in range(1, WARMUP_ROUNDS + 1):
            exp.run_round(r)
        _sync(exp.device)
        warmup = time.perf_counter() - t0
        t0 = time.perf_counter()
        client_s = [exp.run_round(r)["client_s"] for r in range(WARMUP_ROUNDS + 1, WARMUP_ROUNDS + rounds + 1)]
        _sync(exp.device)
        return {"s_per_round": (time.perf_counter() - t0) / rounds,
                "client_s": sum(client_s) / len(client_s), "warmup_s": warmup}
    finally:
        del exp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch._dynamo.reset()                            # no compiled code carried into the next setting


def _error_text(e: Exception) -> str:
    lines = [l.strip() for l in str(e).splitlines() if l.strip()] or [""]
    # Inductor puts the real cause (e.g. "Cannot find a working triton installation") on a later line
    why = next((l for l in lines[1:] if "Error" in l), lines[0])
    return f"{type(e).__name__}: {why[:160]}"


def bench_config(cfg: ExperimentConfig, rounds: int = 5,
                 on_update: Optional[Callable[[List[dict]], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> List[dict]:
    """Time every entry of :data:`SETTINGS` on ``cfg``. One row per setting, in order:
    ``status`` is queued | running | ok | failed | skipped; ok rows carry ``s_per_round``,
    ``client_s``, ``warmup_s``, ``hours`` (extrapolated to ``cfg.server.rounds``) and
    ``speedup`` (vs. the baseline); failed rows carry ``error``. ``on_update(rows)`` is
    called whenever a row changes; ``should_stop()`` is checked before each setting."""
    rows = [{"setting": label, "status": "queued"} for label, _ in SETTINGS]

    def update():
        if on_update:
            on_update([dict(r) for r in rows])

    for row, (_, overrides) in zip(rows, SETTINGS):
        if should_stop and should_stop():
            row["status"] = "skipped"
            continue
        row["status"] = "running"
        update()
        try:
            row.update(_time_setting(cfg, rounds, overrides), status="ok")
            row["hours"] = row["s_per_round"] * cfg.server.rounds / 3600
            base = rows[0].get("s_per_round")
            row["speedup"] = base / row["s_per_round"] if base else None
        except Exception as e:                           # e.g. no Triton for torch.compile
            row.update(status="failed", error=_error_text(e))
        update()
    update()
    return rows


def bench(config: str, rounds: int = 5, extra: Optional[Sequence[str]] = None) -> List[dict]:
    """``python -m ssfl -c CONFIG --bench ROUNDS``: prints one line per setting."""
    cfg = ExperimentConfig.from_yaml(config, list(extra or []))
    printed = set()

    def show(rows):
        for r in rows:
            if r["status"] in ("ok", "failed") and r["setting"] not in printed:
                printed.add(r["setting"])
                if r["status"] == "failed":
                    print(f"{r['setting']:22s} failed: {r['error']}", flush=True)
                else:
                    print(f"{r['setting']:22s} {r['s_per_round']:7.2f} s/round   {r['client_s']:6.2f} s/client   "
                          f"~{r['hours']:5.1f} h for {cfg.server.rounds} rounds   "
                          f"(warm-up {r['warmup_s']:.0f} s)", flush=True)

    rows = bench_config(cfg, rounds, on_update=show)
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["s_per_round"])
        if best.get("speedup"):
            print(f"fastest: {best['setting']} ({best['speedup']:.2f}x the baseline)")
        else:
            print(f"fastest: {best['setting']}")
    return rows
