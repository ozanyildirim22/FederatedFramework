"""SSFL Studio - graphical user interface (Gradio).

Launch locally:   python -m ssfl --gui           (opens http://127.0.0.1:7860)
Launch in Colab:  from ssfl.gui import launch; launch(share=True)

Tabs
  * Experiment  - pick a scenario and one or more algorithms, adjust the key
                  settings, preview the non-IID split, run (several algorithms
                  run one after another on the same split) and watch a live
                  dashboard. Refreshing the page re-attaches to a running job.
  * Thesis grid - run a whole grid file (configs/grids/*.yaml) with live status.
  * Results     - leaderboard, curves of any metric, thesis table, CSV/LaTeX export.
  * Guide       - what every algorithm does and how to use the studio.

Every list is filled from the registries, so custom strategies, aggregators,
datasets and models registered before ``launch()`` appear automatically.
Their constructor arguments become form fields.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .runinfo import code_version, default_params  # noqa: F401 - re-exported for plug-ins / tests
from .registry import AGGREGATORS, DATASETS, MODELS, PARTITIONERS, STRATEGIES
from .strategies.base import COMPILE_MODES, compile_mode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
GRID_DIR = CONFIG_DIR / "grids"

# ------------------------------------------------------------ catalogue
#: name -> (display name, venue, one-line description, kind)
ALGO_INFO: Dict[str, Tuple[str, str, str, str]] = {
    "fedavg": ("FedAvg", "Baseline · AISTATS 2017",
               "Trains on labeled data only; the server averages updates weighted by data size.", "baseline"),
    "fedprox": ("FedProx", "Baseline · MLSys 2020",
                "FedAvg plus a proximal term μ that keeps local models close to the global model.", "baseline"),
    "fedmatch": ("FedMatch", "SSFL · ICLR 2021",
                 "Weights split into σ (labeled) + ψ (unlabeled); pseudo-labels from a vote with "
                 "helper models of similar clients.", "ssl"),
    "fedtrinet": ("FedTriNet", "SSFL · BigData 2021",
                  "FedAvg warm-up, then local, global and combined models average their predictions; "
                  "dynamic confidence threshold.", "ssl"),
    "fedlabel": ("FedLabel", "SSFL · ICCV 2023",
                 "Supervised local update, then a semi-supervised update from W_t: per sample, the more "
                 "confident of the local and global model gives the pseudo-label; KL to the other model "
                 "where both agree. Uploads Δw_L + Δw_U.", "featured"),
    "fixmatch": ("FixMatch", "SSL baseline · NeurIPS 2020",
                 "Weak-view pseudo-label above τ is enforced on the strong view.", "ssl"),
    "uda": ("UDA", "SSL baseline · NeurIPS 2020",
            "Sharpened weak-view prediction is the KL target for the strong view (FedLabel paper baseline).", "ssl"),
    "pseudolabel": ("Pseudo-Label", "SSL baseline · 2013",
                    "Hard pseudo-labels above a confidence threshold τ.", "ssl"),
    "supervised": ("Supervised", "Baseline",
                   "Same as FedAvg on labeled data; with 100% labels it is the upper bound.", "baseline"),
}
ALGO_ORDER = ["fedavg", "fedprox", "fedmatch", "fedtrinet", "fedlabel", "fixmatch", "uda", "pseudolabel",
              "supervised"]

#: preset file -> (icon, title, subtitle)
PRESET_INFO = {
    "quick_test.yaml": ("⚡", "Quick demo", "Synthetic data · runs in seconds"),
    "organamnist_fedlabel.yaml": ("🫁", "OrganAMNIST", "Cross-silo · 10 hospitals · 100 rounds"),
    "pathmnist_fedlabel.yaml": ("🔬", "PathMNIST", "Cross-silo · 10 hospitals · 100 rounds"),
    "cifar10_fedlabel.yaml": ("🖼️", "CIFAR-10", "Cross-device · 100 clients · ResNet-34"),
    "organamnist_fedlabel_test1.yaml": ("🫁", "OrganAMNIST (test1)", "test1 settings · 3 of 10 clients · fast"),
    "cifar10_fedlabel_test1.yaml": ("🖼️", "CIFAR-10 (test1)", "test1 settings · batch 20 · fast"),
}

#: parameter -> (help text, choices or None)
PARAM_HELP: Dict[str, Tuple[str, Optional[List[str]]]] = {
    "tau": ("Confidence threshold to keep a pseudo-label", None),
    "lambda_u": ("Weight of the unlabeled (pseudo-label) loss", None),
    "beta": ("β: confidence threshold to keep a pseudo-label (paper appendix: 0.3–0.5 best)", None),
    "sup_steps": ("τ: SGD steps on labeled data (Step 1); empty = use sup_epochs", None),
    "unsup_steps": ("τ′: SGD steps on unlabeled data (Steps 2–3); empty = use unsup_epochs", None),
    "weight_by_used": ("Aggregation weight r_k = samples used this round (paper's Alg. 1)", None),
    "threshold": ("UDA confidence mask: keep if top probability > threshold", None),
    "lambda0": ("λ₀: scale of the global–local consistency weight", None),
    "sup_epochs": ("τ: epochs on labeled data (Step 1); empty = local epochs", None),
    "unsup_epochs": ("τ′: epochs on unlabeled data (Steps 2–3); empty = local epochs", None),
    "confidence": ("h(·): variance of the logits (ADD), log K − entropy, or max probability",
                   ["variance", "entropy", "max_prob"]),
    "selection": ("Pseudo-label source: adaptive = FedLabel, local/global = ablations",
                  ["adaptive", "local", "global"]),
    "mu": ("Proximal coefficient μ", None),
    "lambda_s": ("Weight of the supervised loss (on σ)", None),
    "lambda_iccs": ("Weight of the inter-client consistency (on ψ)", None),
    "lambda_l2": ("Weight of ‖σ − ψ‖²", None),
    "lambda_l1": ("Weight of the sparsity term ‖ψ‖₁", None),
    "num_helpers": ("Helper models sent to each client (H)", None),
    "helper_every": ("Refresh the helper models every k rounds", None),
    "decompose": ("Use the σ + ψ parameter decomposition", None),
    "reg_reduction": ("Average (mean) or sum the L1/L2 terms", ["mean", "sum"]),
    "embed_samples": ("Noise images used to embed client models", None),
    "pretrain_rounds": ("Rounds of labeled-only FedAvg before pseudo-labeling (phase I)", None),
    "alpha": ("Scale of the dynamic threshold (paper: 0.93)", None),
    "finetune_steps": ("Supervised steps on the combined model each round", None),
    "global_layers": ('Key prefixes taken from the global model, e.g. ["stem.", "layers.0."]; '
                      "empty = default split", None),
    "global_fraction": ("Fraction of layers from the global model (unknown models)", None),
    "hold_rounds": ("Phase-II rounds with the full threshold", None),
    "decay_end": ("Phase-II round where the threshold decay stops", None),
    "final_scale": ("Threshold scale after the decay", None),
    "unsup_view": ("View the pseudo-label loss is applied to", ["weak", "strong"]),
    "store_dtype": ("Precision of stored local models", ["float16", "float32"]),
}

METRIC_LABELS = {
    "test_acc": "Test accuracy", "test_f1": "Test macro F1", "test_precision": "Test macro precision",
    "test_recall": "Test macro recall", "test_loss": "Test loss", "train_loss": "Train loss",
    "train_mask_rate": "Pseudo-label mask rate", "train_pl_acc": "Pseudo-label accuracy",
    "train_loss_sup": "Supervised loss", "train_loss_unsup": "Unlabeled loss",
    "train_local_selected": "Local model chosen (FedLabel)", "train_kl_rate": "KL applied (FedLabel)",
    "train_lambda_k": "Mean λ_k (FedLabel)", "train_threshold": "Threshold (FedTriNet)",
    "time_s": "Seconds per round",
}


def algo_label(name: str) -> str:
    return ALGO_INFO.get(name, (name,))[0]


def algo_choices() -> List[Tuple[str, str]]:
    names = STRATEGIES.names()
    ordered = [n for n in ALGO_ORDER if n in names] + [n for n in names if n not in ALGO_ORDER]
    return [(algo_label(n), n) for n in ordered]


def preset_choices(presets: Sequence[str]) -> List[Tuple[str, str]]:
    known = [p for p in PRESET_INFO if p in presets]
    out = []
    for p in known + [p for p in presets if p not in PRESET_INFO]:
        icon, title, sub = PRESET_INFO.get(p, ("📄", Path(p).stem, "custom preset"))
        out.append((f"{icon}  {title} — {sub}", p))
    return out


# --------------------------------------------------------------- helpers
def list_presets() -> List[str]:
    return sorted(p.name for p in CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else []


def list_grids() -> List[str]:
    return sorted(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in GRID_DIR.glob("*.yaml")) \
        if GRID_DIR.exists() else []


def load_preset(name: Optional[str]) -> ExperimentConfig:
    if name and (CONFIG_DIR / name).exists():
        return ExperimentConfig.from_yaml(CONFIG_DIR / name)
    return ExperimentConfig()


# Form fields (dotted config keys). The algorithm(s) and their parameters live
# outside this list: the algorithm checkbox group and the parameter-form state.
FIELDS = [
    "name", "seed", "device", "output_dir",
    "data.dataset", "data.root", "data.image_size", "data.partitioner", "data.alpha",
    "data.labeled_ratio", "data.batch_size", "data.unlabeled_batch_ratio", "data.num_workers",
    "data.strong_aug", "data.augment_device", "data.randaugment_n", "data.randaugment_m", "data.cutout",
    "data.split", "data.hflip", "data.holdout", "data.val_ratio", "data.test_ratio",
    "model.name", "model.norm", "model.params",
    "strategy.local_epochs", "strategy.local_steps", "strategy.optimizer",
    "strategy.lr", "strategy.momentum", "strategy.weight_decay", "strategy.prox_mu",
    "strategy.grad_clip", "strategy.precision", "strategy.channels_last", "strategy.cudnn_benchmark",
    "strategy.compile",
    "server.aggregator", "server.rounds", "server.num_clients", "server.client_fraction",
    "server.eval_every", "server.params",
]
JSON_FIELDS = {"model.params", "server.params"}
INT_FIELDS = {"seed", "data.image_size", "data.batch_size", "data.unlabeled_batch_ratio",
              "data.num_workers", "data.randaugment_n", "data.randaugment_m",
              "strategy.local_epochs", "strategy.local_steps",
              "server.rounds", "server.num_clients", "server.eval_every"}


#: data.hflip in the form: None = auto (the dataset decides; off for OrganAMNIST)
HFLIP_CHOICES = {None: "auto", True: "on", False: "off"}


def _get(d: Dict[str, Any], key: str):
    for k in key.split("."):
        d = d[k]
    return d


def _set(d: Dict[str, Any], key: str, value) -> None:
    parts = key.split(".")
    for k in parts[:-1]:
        d = d.setdefault(k, {})
    d[parts[-1]] = value


def cfg_to_values(cfg: ExperimentConfig) -> List[Any]:
    d = cfg.to_dict()
    vals = []
    for key in FIELDS:
        v = _get(d, key)
        if key in JSON_FIELDS:
            v = json.dumps(v or {}, indent=2)
        elif key == "strategy.local_steps":
            v = v or 0
        elif key == "strategy.grad_clip":
            v = v or 0.0
        elif key == "data.hflip":
            v = HFLIP_CHOICES.get(v, "auto")
        elif key == "strategy.compile":
            v = compile_mode(SimpleNamespace(compile=v)) or "none"
        vals.append(v)
    return vals


def values_to_cfg(*vals, algorithm: str = "fedlabel", params: Optional[Dict[str, Any]] = None
                  ) -> ExperimentConfig:
    raw: Dict[str, Any] = {}
    for key, v in zip(FIELDS, vals):
        if key in JSON_FIELDS:
            try:
                v = json.loads(v) if (v or "").strip() else {}
            except json.JSONDecodeError as e:
                raise ValueError(f"'{key}' is not valid JSON: {e}") from e
        elif key in INT_FIELDS:
            v = int(v) if v not in (None, "") else 0
        if key == "strategy.local_steps" and not v:
            v = None
        if key == "strategy.grad_clip" and not v:
            v = None
        if key == "data.hflip":
            v = {"on": True, "off": False}.get(v)
        if key == "strategy.compile" and v in (None, "", "none"):
            v = None
        _set(raw, key, v)
    raw["strategy"]["name"] = algorithm
    raw["strategy"]["params"] = copy.deepcopy(params or {})
    return ExperimentConfig.from_dict(raw)


def _ratio_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")


def build_configs(vals: Sequence[Any], algos: Sequence[str], params_state: Dict[str, Dict[str, Any]]
                  ) -> List[ExperimentConfig]:
    """One config per selected algorithm, all on the same data split (same seed).
    An empty run name becomes ``<dataset>_<algorithm>_l<labels>``."""
    if not algos:
        raise ValueError("Select at least one algorithm.")
    out = []
    for algo in algos:
        cfg = values_to_cfg(*vals, algorithm=algo, params=(params_state or {}).get(algo, {}))
        base = (cfg.name or "").strip()
        if not base:
            cfg.name = (f"{cfg.data.dataset}_{algo}_l{_ratio_tag(cfg.data.labeled_ratio)}"
                        f"_a{_ratio_tag(cfg.data.alpha)}")
        elif len(algos) > 1:
            cfg.name = f"{base}_{algo}"
        out.append(cfg)
    return out


def _fmt_secs(s: float) -> str:
    s = int(max(0, s))
    if s >= 3600:
        return f"{s // 3600}h {s % 3600 // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


#: Environment defaults (set by ``launch(data_dir=..., runs_dir=..., num_workers=...)``, e.g. the
#: Google Drive folders on Colab). They are applied on top of every preset the user loads.
ENV_DEFAULTS: Dict[str, Any] = {}


def apply_env(cfg: ExperimentConfig) -> ExperimentConfig:
    if ENV_DEFAULTS.get("data_dir"):
        cfg.data.root = ENV_DEFAULTS["data_dir"]
    if ENV_DEFAULTS.get("runs_dir"):
        cfg.output_dir = ENV_DEFAULTS["runs_dir"]
    if ENV_DEFAULTS.get("num_workers") is not None:
        cfg.data.num_workers = int(ENV_DEFAULTS["num_workers"])
    return cfg


def system_info() -> Dict[str, str]:
    import torch
    info = {"torch": torch.__version__.split("+")[0]}
    if torch.cuda.is_available():
        info["device"] = torch.cuda.get_device_name(0)
        info["mem"] = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB"
    else:
        info["device"] = "CPU only"
    return info


# --------------------------------------------------------------- runner
class _Stopped(Exception):
    pass


class SpeedCheck:
    """The Experiment tab's speed check (:func:`ssfl.bench.bench_config`) in a background
    thread. It shares the GPU with training, so it and the RunController never run together."""

    ROUNDS = 3                                     # timed rounds per setting (after 2 warm-up rounds)

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.rows: List[Dict[str, Any]] = []
        self.name, self.total_rounds, self.error, self.version = "", 0, None, 0

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: ExperimentConfig) -> None:
        from .bench import SETTINGS
        with self.lock:
            self.rows = [{"setting": label, "status": "queued"} for label, _ in SETTINGS]
            self.name, self.total_rounds, self.error = cfg.name, cfg.server.rounds, None
            self.version += 1
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._work, args=(cfg,), daemon=True)
        self.thread.start()

    def stop(self) -> bool:
        if not self.running:
            return False
        self.stop_event.set()
        return True

    def _set(self, rows) -> None:
        with self.lock:
            self.rows, self.version = rows, self.version + 1

    def _work(self, cfg: ExperimentConfig) -> None:
        from .bench import bench_config
        try:
            bench_config(cfg, self.ROUNDS, on_update=self._set, should_stop=self.stop_event.is_set)
        except Exception as e:  # noqa: BLE001 - e.g. the dataset failed to load
            with self.lock:
                self.error, self.version = f"{type(e).__name__}: {e}", self.version + 1

    def table(self) -> pd.DataFrame:
        with self.lock:
            rows, total = [dict(r) for r in self.rows], self.total_rounds
        out = []
        for r in rows:
            ok = r["status"] == "ok"
            out.append({
                "setting": r["setting"],
                "status": {"running": "⏳ running", "ok": "✅ done", "failed": "❌ failed",
                           "skipped": "⏭ skipped"}.get(r["status"], r["status"]),
                "s / round": f"{r['s_per_round']:.2f}" if ok else "",
                "s / client": f"{r['client_s']:.2f}" if ok else "",
                f"time for {total} rounds": _fmt_secs(r["s_per_round"] * total) if ok else "",
                "speed-up": f"{r['speedup']:.2f}×" if ok and r.get("speedup") else "",
                "warm-up": _fmt_secs(r["warmup_s"]) if ok else "",
                "note": r.get("error", ""),
            })
        return pd.DataFrame(out)

    def summary(self) -> str:
        with self.lock:
            rows, name, error = [dict(r) for r in self.rows], self.name, self.error
        if error:
            return f'<div class="algo-info empty">⚠️ Speed check failed: {html.escape(error)}</div>'
        ok = [r for r in rows if r["status"] == "ok"]
        if self.running:
            cur = next((r["setting"] for r in rows if r["status"] == "running"), "…")
            return (f'<div class="algo-info empty">⏳ Timing <b>{html.escape(name)}</b>: '
                    f'{len([r for r in rows if r["status"] in ("ok", "failed")])}/{len(rows)} settings done, '
                    f'now <b>{html.escape(cur)}</b>. Each setting builds the run, trains 2 warm-up rounds '
                    f'and {self.ROUNDS} timed rounds; the compile settings compile in the first round.</div>')
        if not ok:
            return '<div class="algo-info empty">No setting finished.</div>'
        best = min(ok, key=lambda r: r["s_per_round"])
        if best["setting"] == "baseline" or (best.get("speedup") or 0) < 1.05:
            return (f'<div class="algo-info">For <b>{html.escape(name)}</b> no option is clearly faster than '
                    f'the baseline (within 5%): keep the settings as they are.</div>')
        return (f'<div class="algo-info">Fastest for <b>{html.escape(name)}</b>: '
                f'<b>{html.escape(best["setting"])}</b> — {best["speedup"]:.2f}× the baseline. Set it under '
                f'<i>Model &amp; local training</i> (cuDNN benchmark, channels_last, torch.compile) '
                f'before you launch.</div>')


class RunController:
    """Runs a queue of experiments in one background thread.

    The state lives here (not in the browser), so any page, including one
    opened after a refresh, can show the live progress via :meth:`snapshot`.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._reset([], "experiment")

    def _reset(self, plan, kind: str) -> None:
        self.kind = kind
        self.plan = [{"name": n, "cfg": c, "status": s, "history": [], "best": None,
                      "out_dir": None, "error": None, "started": None, "ended": None}
                     for n, c, s in plan]
        self.current = -1
        self.logs: List[str] = []
        self.state = "idle"                 # idle | running | stopping | done | stopped | error
        self.message = "Ready."
        self.started = None
        self.version = 0

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, plan: Sequence[Tuple[str, ExperimentConfig, str]], kind: str) -> None:
        if self.running:
            raise RuntimeError("An experiment is already running - stop it first.")
        with self.lock:
            self._reset(plan, kind)
            self.state, self.started = "running", time.time()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def stop(self) -> bool:
        if not self.running:
            return False
        self.stop_event.set()
        with self.lock:
            self.state = "stopping"
            self.message = "Stopping after the current round …"
            self.version += 1
        return True

    def _log(self, msg: str) -> None:
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.logs = self.logs[-2000:]
            self.version += 1

    def _work(self) -> None:
        from .experiment import Experiment

        errors = 0
        for i, run in enumerate(self.plan):
            if run["status"] != "queued":
                continue
            if self.stop_event.is_set():
                break
            root = run["cfg"].data.root
            size0 = _dir_size(root)
            with self.lock:
                self.current = i
                run["status"], run["started"] = "running", time.time()
                run["root"], run["root_size0"] = root, size0
                self.message = (f"Loading data and building clients for {run['name']} … "
                                "(the first run of a dataset also downloads it)")
                self.version += 1
            self._log(f"▶ {run['name']} ({i + 1}/{len(self.plan)})")

            def on_round(rnd, rec, exp, run=run):
                with self.lock:
                    run["history"].append(dict(rec))
                    if "test_acc" in rec:
                        run["best"] = max(run["best"] or 0.0, rec["test_acc"])
                    self.message = f"Training {run['name']}"
                    self.version += 1
                self._log(_fmt_round(rec))
                if self.stop_event.is_set():
                    raise _Stopped()

            try:
                exp = Experiment(run["cfg"], callbacks=[on_round], verbose=False)
                with self.lock:
                    run["out_dir"] = str(exp.out_dir)
                    run["data_ready"] = True
                    self.message = f"Training {run['name']}"
                    self.version += 1
                self._log(exp.data.summary())
                self._log(f"strategy={run['cfg'].strategy.name} aggregator={run['cfg'].server.aggregator} "
                          f"model={run['cfg'].model.name} device={exp.device} "
                          f"workers={exp.num_workers}{' (auto)' if run['cfg'].data.num_workers < 0 else ''} "
                          f"augment={run['cfg'].data.augment_device} precision={exp.server.precision} "
                          f"-> {exp.out_dir}")
                exp.run()
                with self.lock:
                    run["status"], run["ended"] = "done", time.time()
                    self.version += 1
                self._log(f"✓ {run['name']} finished - best test accuracy {run['best'] or 0:.4f}")
            except _Stopped:
                with self.lock:
                    run["status"], run["ended"] = "stopped", time.time()
                self._log("■ Stopped by user (metrics so far are saved in the run folder).")
                break
            except Exception:  # noqa: BLE001 - show any error in the UI
                errors += 1
                tb = traceback.format_exc()
                with self.lock:
                    run["status"], run["error"], run["ended"] = "error", tb, time.time()
                self._log(f"✗ {run['name']} failed:\n{tb}")
        with self.lock:
            if self.stop_event.is_set():
                self.state, self.message = "stopped", "Stopped by user."
            elif errors:
                self.state, self.message = "error", f"Finished with {errors} failed run(s) - see the log."
            else:
                self.state, self.message = "done", "All runs finished."
            for run in self.plan:
                if run["status"] == "queued":
                    run["status"] = "cancelled"
            self.version += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "kind": self.kind, "state": self.state, "message": self.message,
                "current": self.current, "started": self.started, "version": self.version,
                "logs": list(self.logs),
                "plan": [{**{k: v for k, v in r.items() if k != "history"},
                          "history": list(r["history"])} for r in self.plan],
            }


def _fmt_round(rec: Dict[str, float]) -> str:
    parts = [f"round {int(rec['round']):4d}", f"loss {rec.get('train_loss', float('nan')):.4f}"]
    for k, label in [("train_mask_rate", "mask"), ("train_pl_acc", "pl_acc"), ("test_acc", "test_acc")]:
        if k in rec and rec[k] == rec[k]:
            parts.append(f"{label} {rec[k]:.4f}")
    parts.append(f"{rec.get('time_s', 0):.1f}s")
    return " | ".join(parts)


# ------------------------------------------------------- HTML renderers
STATUS_ICON = {"queued": "⏳", "running": "▶", "done": "✓", "stopped": "■", "error": "✗",
               "cancelled": "–", "finished": "✓"}


def _eta(snap) -> Optional[float]:
    plan, cur = snap["plan"], snap["current"]
    if cur < 0 or snap["state"] not in ("running", "stopping"):
        return None
    run = plan[cur]
    times = [r.get("time_s", 0) for r in run["history"][-5:]]
    if not times:
        return None
    per_round = float(np.mean(times))
    remaining = run["cfg"].server.rounds - len(run["history"])
    remaining += sum(r["cfg"].server.rounds for r in plan[cur + 1:] if r["status"] == "queued")
    return per_round * remaining


def render_status(snap) -> str:
    state = snap["state"]
    plan, cur = snap["plan"], snap["current"]
    active = [r for r in plan if r["status"] != "finished"]
    if state == "idle" and not plan:
        return ('<div class="status status-idle"><div class="status-head"><span class="dot"></span>'
                '<b>Ready</b><span class="muted">Pick a scenario and algorithms, then press Launch.</span>'
                '</div></div>')
    total_rounds = sum(r["cfg"].server.rounds for r in active) or 1
    done_rounds = sum(r["cfg"].server.rounds if r["status"] == "done" else len(r["history"]) for r in active)
    pct = 100.0 * done_rounds / total_rounds
    label = {"running": "Running", "stopping": "Stopping", "done": "Finished", "stopped": "Stopped",
             "error": "Finished with errors", "idle": "Ready"}[state]
    head = f'<span class="dot"></span><b>{label}</b>'
    if 0 <= cur < len(plan):
        r = plan[cur]
        pos = active.index(r) + 1 if r in active else cur + 1
        head += f'<span class="muted">{html.escape(r["name"])} · run {pos}/{len(active)}</span>'
    eta = _eta(snap)
    foot = [html.escape(snap["message"])]
    if (0 <= cur < len(plan) and plan[cur]["status"] == "running" and not plan[cur].get("data_ready")
            and plan[cur].get("root")):
        mb = (_dir_size(plan[cur]["root"]) - plan[cur].get("root_size0", 0)) / 2**20
        if mb > 0.05:
            foot.append(f"downloaded {mb:.1f} MB")
    if 0 <= cur < len(plan) and (plan[cur]["history"] or plan[cur].get("data_ready")):
        done, total = len(plan[cur]["history"]), plan[cur]["cfg"].server.rounds
        if state in ("running", "stopping") and plan[cur]["status"] == "running" and done < total:
            foot.append(f"round {done + 1}/{total} in progress")
        else:
            foot.append(f"round {done}/{total}")
    if snap["started"]:
        foot.append(f"elapsed {_fmt_secs(time.time() - snap['started'])}" if state in ("running", "stopping")
                    else "")
    if eta is not None:
        foot.append(f"ETA {_fmt_secs(eta)}")
    if state in ("running", "stopping"):
        foot.append(gpu_usage())
    chips = ""
    if 1 < len(active) and snap["kind"] == "experiment":     # the grid tab has its own table
        chips = '<div class="queue">' + "".join(
            f'<span class="qchip q-{r["status"]}">{STATUS_ICON.get(r["status"], "")} '
            f'{html.escape(algo_label(r["cfg"].strategy.name))}'
            + (f' · {100 * r["best"]:.1f}%' if r["best"] is not None else "") + "</span>"
            for r in active) + "</div>"
    return (f'<div class="status status-{state}"><div class="status-head">{head}</div>'
            f'<div class="bar"><div style="width:{pct:.1f}%"></div></div>'
            f'<div class="status-foot">{" · ".join(f for f in foot if f)}</div>{chips}</div>')


def _kpi(label: str, value: str, sub: str = "", accent: str = "") -> str:
    return (f'<div class="kpi {accent}"><div class="kpi-label">{label.upper()}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def render_kpis(snap) -> str:
    plan, cur = snap["plan"], snap["current"]
    hist = plan[cur]["history"] if 0 <= cur < len(plan) else []
    last_eval = next((r for r in reversed(hist) if "test_acc" in r), None)
    best = plan[cur]["best"] if 0 <= cur < len(plan) else None
    last = hist[-1] if hist else {}
    pct = lambda v: f"{100 * v:.1f}<span>%</span>" if v is not None and v == v else "—"  # noqa: E731
    best_round = ""
    if best is not None:
        br = next((int(r["round"]) for r in hist if r.get("test_acc") == best), None)
        best_round = f"at round {br}" if br else ""
    cards = [
        _kpi("Test accuracy", pct(last_eval["test_acc"] if last_eval else None),
             f"round {int(last_eval['round'])}" if last_eval else "waiting for evaluation", "a1"),
        _kpi("Best accuracy", pct(best), best_round, "a2"),
        _kpi("Pseudo-label mask", pct(last.get("train_mask_rate")),
             "share of unlabeled samples used", "a3"),
        _kpi("Pseudo-label accuracy", pct(last.get("train_pl_acc")), "vs hidden true labels", "a4"),
        _kpi("Train loss", f"{last['train_loss']:.3f}" if "train_loss" in last else "—",
             f"{last.get('time_s', 0):.1f}s / round" if last else "", "a5"),
    ]
    return '<div class="kpi-grid">' + "".join(cards) + "</div>"


PALETTE = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#0ea5e9", "#8b5cf6", "#ef4444", "#14b8a6",
           "#84cc16", "#f97316", "#64748b", "#a855f7", "#06b6d4", "#e11d48", "#22c55e", "#eab308",
           "#3b82f6", "#d946ef", "#78716c", "#0d9488"]
ALGO_COLORS = {"FedAvg": "#64748b", "FedProx": "#0ea5e9", "FedMatch": "#10b981", "FedTriNet": "#f59e0b",
               "FedLabel": "#8b5cf6", "FixMatch": "#ec4899", "UDA": "#e11d48", "Pseudo-Label": "#14b8a6", "Supervised": "#78716c"}


def color_map(df: pd.DataFrame, col: str) -> Dict[str, str]:
    """Stable colour per series: algorithms keep their own colour, other series
    take the palette in order of appearance."""
    out, i = {}, 0
    for v in dict.fromkeys(df[col].tolist()) if col in df else []:
        if v in ALGO_COLORS:
            out[v] = ALGO_COLORS[v]
        else:
            out[v] = PALETTE[i % len(PALETTE)]
            i += 1
    return out


def plot_update(df: pd.DataFrame, col: str = "run", **extra):
    """Update a native Gradio plot. Passing ``color_map`` makes the chart rebuild
    its colour scale; without it the series added after the first draw stay
    uncoloured (the scale domain is fixed when the chart is first drawn)."""
    import gradio as gr
    return gr.update(value=df, color_map=color_map(df, col) or None, **extra)


def _typed(rows: List[Dict[str, Any]], group: str = "run") -> pd.DataFrame:
    """Long-format frame with explicit dtypes (native Gradio plots infer the axis
    type from the dtype; an object column would make the y-axis categorical)."""
    df = pd.DataFrame(rows, columns=["round", "value", group])
    return df.astype({"round": "int64", "value": "float64", group: "str"})


def _long_frame(snap, metric: str, label_by_algo: bool = True) -> pd.DataFrame:
    rows = []
    for r in snap["plan"]:
        name = algo_label(r["cfg"].strategy.name) if label_by_algo else r["name"]
        for rec in r["history"]:
            if metric in rec and rec[metric] == rec[metric]:
                rows.append({"round": int(rec["round"]), "value": float(rec[metric]), "run": name})
    return _typed(rows)


def _pl_frame(snap) -> pd.DataFrame:
    rows = []
    multi = len([r for r in snap["plan"] if r["history"]]) > 1
    for r in snap["plan"]:
        a = algo_label(r["cfg"].strategy.name)
        for rec in r["history"]:
            for k, lab in (("train_mask_rate", "mask rate"), ("train_pl_acc", "PL accuracy")):
                if k in rec and rec[k] == rec[k]:
                    rows.append({"round": int(rec["round"]), "value": float(rec[k]),
                                 "series": f"{a} · {lab}" if multi else lab})
    return _typed(rows, "series")


def _metrics_table(snap) -> pd.DataFrame:
    plan, cur = snap["plan"], snap["current"]
    if not (0 <= cur < len(plan)) or not plan[cur]["history"]:
        return pd.DataFrame()
    df = pd.DataFrame(plan[cur]["history"])
    cols = ["round"] + [c for c in df.columns if c != "round"]
    return df[cols].round(4).iloc[::-1]


def render_algo_cards(selected: Sequence[str]) -> str:
    if not selected:
        return '<div class="algo-info empty">Select one or more algorithms. Several algorithms run one ' \
               'after another on the <b>same</b> data split so you can compare them.</div>'
    cards = []
    for name in selected:
        title, venue, desc, kind = ALGO_INFO.get(
            name, (name, "Custom", (STRATEGIES.get(name).__doc__ or "Registered strategy.").strip()
                   .splitlines()[0], "custom"))
        cards.append(f'<div class="algo-card k-{kind}"><div class="algo-top"><b>{html.escape(title)}</b>'
                     f'<span class="tag t-{kind}">{html.escape(venue)}</span></div>'
                     f'<div class="algo-desc">{html.escape(desc)}</div></div>')
    note = ""
    if len(selected) > 1:
        note = f'<div class="algo-note">{len(selected)} runs will be queued on the same split.</div>'
    return '<div class="algo-info">' + "".join(cards) + note + "</div>"


def render_hero() -> str:
    try:
        info = system_info()
    except Exception:  # noqa: BLE001
        info = {"device": "unknown", "torch": "?"}
    dev = info["device"] + (f" · {info['mem']}" if "mem" in info else "")
    chips = [f'<span class="chip"><i class="led {"on" if "CPU" not in dev else ""}"></i>{html.escape(dev)}</span>',
             f'<span class="chip">PyTorch {html.escape(info["torch"])}</span>',
             f'<span class="chip">{len(STRATEGIES.names())} algorithms</span>',
             f'<span class="chip">{len(DATASETS.names())} datasets</span>',
             f'<span class="chip" title="git commit of the running code">build {html.escape(code_version())}'
             '</span>']
    logo = """<svg viewBox="0 0 64 64" class="logo" aria-hidden="true">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#fff" stop-opacity=".95"/><stop offset="1" stop-color="#fff" stop-opacity=".6"/>
      </linearGradient></defs>
      <g stroke="url(#lg)" stroke-width="2.2" stroke-linecap="round" opacity=".8">
        <line x1="32" y1="32" x2="12" y2="14"/><line x1="32" y1="32" x2="52" y2="14"/>
        <line x1="32" y1="32" x2="8" y2="40"/><line x1="32" y1="32" x2="56" y2="40"/>
        <line x1="32" y1="32" x2="32" y2="58"/></g>
      <circle cx="32" cy="32" r="9" fill="#fff"/>
      <g fill="url(#lg)"><circle cx="12" cy="14" r="5"/><circle cx="52" cy="14" r="5"/>
        <circle cx="8" cy="40" r="5"/><circle cx="56" cy="40" r="5"/><circle cx="32" cy="58" r="5"/></g>
    </svg>"""
    return (f'<div class="hero"><div class="hero-main">{logo}<div><h1>SSFL Studio</h1>'
            '<p>Semi-supervised federated learning — configure, run and compare FedAvg, FedProx, '
            'FedMatch, FedTriNet and FedLabel on CIFAR-10, OrganAMNIST and PathMNIST.</p></div></div>'
            f'<div class="chips">{"".join(chips)}</div></div>')


def render_guide() -> str:
    cards = []
    for name in [n for n in ALGO_ORDER if n in STRATEGIES]:
        title, venue, desc, kind = ALGO_INFO[name]
        cards.append(f'<div class="algo-card k-{kind}"><div class="algo-top"><b>{title}</b>'
                     f'<span class="tag t-{kind}">{venue}</span></div><div class="algo-desc">{desc}</div>'
                     f'<code>strategy.name: {name}</code></div>')
    steps = [
        ("1", "Pick a scenario", "A preset loads the thesis settings: dataset, clients, rounds, model."),
        ("2", "Choose algorithms", "Tick one or several; several run in a queue on the same data split."),
        ("3", "Tune", "Labeled ratio, rounds and non-IID α are on top; everything else is in the panels."),
        ("4", "Preview", "<b>Preview split</b> shows how the classes are spread across clients."),
        ("5", "Launch", "Watch accuracy and pseudo-label quality live. Refreshing the page re-attaches."),
        ("6", "Compare", "The <b>Results</b> tab builds the leaderboard and the thesis table (LaTeX)."),
    ]
    step_html = "".join(f'<div class="step"><div class="step-n">{n}</div><div><b>{t}</b><p>{d}</p></div></div>'
                        for n, t, d in steps)
    return (f'<div class="guide"><h2>How it works</h2><div class="steps">{step_html}</div>'
            f'<h2>Algorithms</h2><div class="algo-grid">{"".join(cards)}</div>'
            '<h2>Tips</h2><ul class="tips">'
            '<li>On Colab set <b>DataLoader workers</b> to 2 and the output/download folders to Google Drive.</li>'
            '<li>For a fast check use the <b>Quick demo</b> scenario or set the model to <code>simple_cnn</code>.</li>'
            '<li>Every run writes <code>config.yaml</code>, <code>metrics.csv</code>, <code>summary.json</code> '
            'and <code>global_model.pt</code> to its run folder.</li>'
            '<li>The <b>Thesis grid</b> tab skips finished runs, so a stopped grid continues where it ended.</li>'
            '</ul></div>')


# ------------------------------------------------------------ results
#: Timestamp suffix of a run folder: ``_YYYYmmdd-HHMMSS`` plus ``-2``, ``-3`` ... when two
#: runs with the same name start within the same second.
RUN_STAMP = r"_\d{8}-\d{6}(?:-\d+)?$"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - missing / half-written
        return {}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _started_from_name(name: str) -> str:
    m = re.search(r"_(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:-\d+)?$", name)
    return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}" if m else ""


def _variant(algo: str, strat: Dict[str, Any], tag: Optional[str]) -> str:
    """Row label in the Results tables. FixMatch / UDA read like the FedLabel paper's
    baselines: FedAvg+FixMatch, or FedProx+FixMatch when the FedProx term was on."""
    label = algo_label(algo)
    if algo in ("fixmatch", "uda"):
        label = ("FedProx+" if float(strat.get("prox_mu") or 0) > 0 else "FedAvg+") + label
        if tag == "prox":
            tag = None
    return label + (f" ({tag})" if tag else "")


def _run_meta(run_dir: Path) -> Dict[str, Any]:
    cfg = _read_yaml(run_dir / "config.yaml")
    summ = _read_json(run_dir / "summary.json")
    base = re.sub(RUN_STAMP, "", run_dir.name)
    seed = cfg.get("seed", 0)
    group = re.sub(rf"_s{seed}$", "", base) if seed is not None else base   # grid seed repeats
    m = re.search(r"_l[0-9p]+(?:_a[0-9p]+)?(?:_(.+))?$", group)   # tag after labels / alpha
    algo = cfg.get("strategy", {}).get("name", "?")
    data, server, strat = cfg.get("data", {}), cfg.get("server", {}), cfg.get("strategy", {})
    return {"run": run_dir.name, "base": base, "group": group, "seed": seed, "dataset": data.get("dataset", "?"),
            "labels": data.get("labeled_ratio"), "alpha": data.get("alpha"),
            "clients": server.get("num_clients"), "fraction": server.get("client_fraction"),
            "algorithm": algo, "variant": _variant(algo, strat, m.group(1) if m else None),
            "aggregator": server.get("aggregator"), "model": cfg.get("model", {}).get("name"),
            "planned_rounds": server.get("rounds"),
            "fast": data.get("augment_device") == "gpu" or strat.get("precision") in ("bf16", "fp16"),
            "started": (summ.get("started_at") or "")[:16] or _started_from_name(run_dir.name),
            "summary": summ, "config": cfg}


def _status(meta: Dict[str, Any]) -> str:
    """finished | stopped | error | running | interrupted (a "running" run that stopped
    writing, e.g. a Colab session that disconnected) | partial (old runs without status)."""
    summ = meta["summary"]
    if not summ:
        return "partial"
    st = summ.get("status", "finished")
    if st == "running":
        per_round = summ.get("mean_round_s") or 60
        if time.time() - meta["mtime"] > max(600, 5 * per_round):
            return "interrupted"
    return st


STATUS_LABEL = {"finished": "✓ finished", "running": "▶ running", "stopped": "■ stopped",
                "error": "✗ error", "interrupted": "⚠ interrupted", "partial": "… partial"}


def scan_runs(runs_dir: str) -> List[Dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    out = [m for m in (load_run(p.parent) for p in root.glob("*/metrics.csv")) if m is not None]
    return sorted(out, key=lambda m: m["mtime"], reverse=True)


def load_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Everything the Results tab shows about one run folder (None if unreadable)."""
    csv_path = Path(run_dir) / "metrics.csv"
    meta = _run_meta(Path(run_dir))
    try:
        df = pd.read_csv(csv_path)
        meta["mtime"] = csv_path.stat().st_mtime
    except Exception:  # noqa: BLE001 - missing / half-written file
        return None
    meta["rounds"] = int(df["round"].max()) if len(df) else 0
    acc = df["test_acc"].dropna() if "test_acc" in df else pd.Series(dtype=float)
    meta["best_acc"] = float(acc.max()) if len(acc) else None
    meta["final_acc"] = float(acc.iloc[-1]) if len(acc) else None
    f1 = df["test_f1"].dropna() if "test_f1" in df else pd.Series(dtype=float)
    meta["best_f1"] = float(f1.max()) if len(f1) else None
    # model selection on the validation set: test accuracy at the best validation round
    meta["best_val"], meta["test_at_val"] = None, None
    if "val_acc" in df and df["val_acc"].notna().any():
        i = df["val_acc"].idxmax()
        meta["best_val"] = float(df.loc[i, "val_acc"])
        meta["test_at_val"] = float(df.loc[i, "test_acc"]) if "test_acc" in df and df.loc[i, "test_acc"] == df.loc[i, "test_acc"] else None
    cs = df["client_s"].dropna() if "client_s" in df else pd.Series(dtype=float)
    meta["client_s"] = float(cs.mean()) if len(cs) else None
    last = df.iloc[-1] if len(df) else {}
    meta["mask"] = float(last["train_mask_rate"]) if "train_mask_rate" in df and last["train_mask_rate"] == last["train_mask_rate"] else None
    meta["pl_acc"] = float(df["train_pl_acc"].dropna().iloc[-1]) if "train_pl_acc" in df and df["train_pl_acc"].notna().any() else None
    times = df["time_s"].dropna() if "time_s" in df else pd.Series(dtype=float)
    summ = meta["summary"]
    meta["time_s"] = summ.get("elapsed_s") or (float(times.sum()) if len(times) else None)
    meta["round_s"] = summ.get("mean_round_s") or (float(times.mean()) if len(times) else None)
    meta["status"] = _status(meta)
    meta["finished"] = meta["status"] == "finished"
    return meta


def _pct(v) -> str:
    return f"{100 * v:.2f}" if v is not None and v == v else "—"


def leaderboard(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["#", "run", "started", "dataset", "labels", "α", "clients", "algorithm", "best acc",
            "final acc", "best F1", "rounds", "time", "s/round", "status"]
    if not runs:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(runs).sort_values("best_acc", ascending=False, na_position="last", kind="mergesort")
    out = pd.DataFrame({
        "run": df["base"], "started": df["started"], "dataset": df["dataset"],
        "labels": df["labels"].map(lambda v: f"{100 * v:g}%" if v is not None and v == v else ""),
        "α": df["alpha"].map(lambda v: f"{v:g}" if v is not None and v == v else ""),
        "clients": [f"{n}×{c:g}" if n and c else (str(n) if n else "") for n, c in zip(df["clients"], df["fraction"])],
        "algorithm": df["variant"] + df["fast"].map(lambda f: " ⚡" if f else ""),
        "best acc": df["best_acc"].map(_pct), "final acc": df["final_acc"].map(_pct),
        "best F1": df["best_f1"].map(_pct) if "best_f1" in df else "—",
        "rounds": [f"{r}/{p}" if p else str(r) for r, p in zip(df["rounds"], df["planned_rounds"])],
        "time": df["time_s"].map(lambda v: _fmt_secs(v) if v is not None and v == v else "—"),
        "s/round": df["round_s"].map(lambda v: f"{v:.1f}" if v is not None and v == v else "—"),
        "status": df["status"].map(lambda s: STATUS_LABEL.get(s, s)),
    })
    out.insert(0, "#", range(1, len(out) + 1))
    return out[cols].reset_index(drop=True)


def board_order(runs: List[Dict[str, Any]]) -> List[str]:
    """Run folders in the leaderboard's row order (same stable sort)."""
    return [r["run"] for r in sorted(runs, key=lambda r: (r["best_acc"] is None, -(r["best_acc"] or 0)))]


def clear_choices(runs_dir: str) -> List[Tuple[str, str]]:
    """Every run folder (also ones without metrics yet) for the Clear runs picker, newest first."""
    from .cleanup import list_run_dirs, run_status
    out = []
    for p in list_run_dirs(runs_dir):
        st = run_status(p)
        out.append((f"{p.name} · {STATUS_LABEL.get(st, st)}", p.name))
    return out


def run_choices(runs: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return [(f"{r['base']} · {r['started']} · best {_pct(r['best_acc'])}%", r["run"]) for r in runs]


def _fmt_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if v is None:
        return "—"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def _rows(pairs: Sequence[Tuple[str, Any]]) -> str:
    return "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt_val(v))}</td></tr>" for k, v in pairs)


def _param_rows(cls, given: Dict[str, Any]) -> str:
    """Every parameter with its value; values that differ from the default are highlighted."""
    defaults = default_params(cls) if cls is not None else {}
    keys = list(defaults) + [k for k in (given or {}) if k not in defaults]
    out = []
    for k in keys:
        val = (given or {}).get(k, defaults.get(k))
        changed = k in (given or {}) and k in defaults and given[k] != defaults[k]
        note = f'<span class="dflt">default {html.escape(_fmt_val(defaults[k]))}</span>' if changed else ""
        out.append(f'<tr class="{"chg" if changed else ""}"><td>{html.escape(k)}</td>'
                   f"<td>{html.escape(_fmt_val(val))}{note}</td></tr>")
    return "".join(out) or '<tr><td colspan="2">no parameters</td></tr>'


def render_run_details(meta: Dict[str, Any], run_dir: Path) -> str:
    cfg, summ = meta["config"], meta["summary"]
    data, model, strat, server = (cfg.get(k, {}) for k in ("data", "model", "strategy", "server"))
    env = summ.get("environment", {})
    status = meta["status"]
    best_round = summ.get("best_round")
    head = (f'<div class="dhead"><div><h3>{html.escape(meta["base"])}</h3>'
            f'<div class="muted">{html.escape(meta["variant"])} · {html.escape(meta["dataset"])} · '
            f'started {html.escape(meta["started"] or "?")}'
            + (f' · last update {html.escape(summ["updated_at"][:16])}' if summ.get("updated_at") else "")
            + f'</div></div><span class="badge b-{status}">{STATUS_LABEL.get(status, status)}</span></div>')
    kpis = ('<div class="kpi-grid">'
            + _kpi("Best accuracy", f"{_pct(meta['best_acc'])}<span>%</span>",
                   f"at round {best_round}" if best_round else "", "a1")
            + _kpi("Final accuracy", f"{_pct(meta['final_acc'])}<span>%</span>",
                   f"round {meta['rounds']} of {meta['planned_rounds']}", "a2")
            + _kpi("Total time", _fmt_secs(meta["time_s"]) if meta["time_s"] else "—",
                   f"setup {_fmt_secs(summ['setup_s'])}" if summ.get("setup_s") is not None else "", "a3")
            + _kpi("Per round", f"{meta['round_s']:.1f}<span>s</span>" if meta["round_s"] else "—",
                   "training + evaluation", "a4")
            + _kpi("Pseudo-labels", f"{_pct(meta['mask'])}<span>%</span>" if meta["mask"] is not None else "—",
                   f"used · {_pct(meta['pl_acc'])}% correct" if meta["pl_acc"] is not None else "used (last round)",
                   "a5") + "</div>")
    lr = strat.get("local_steps") or f"{strat.get('local_epochs', 1)} epoch(s)"
    cards = [
        ("🗂️ Data", _rows([("dataset", data.get("dataset")), ("labeled per client", data.get("labeled_ratio")),
                           ("partitioner", data.get("partitioner")), ("Dirichlet α", data.get("alpha")),
                           ("image size", data.get("image_size")), ("labeled batch", data.get("batch_size")),
                           ("unlabeled ratio μ", data.get("unlabeled_batch_ratio")),
                           ("strong augmentation", f"{data.get('strong_aug')} (N={data.get('randaugment_n', 2)}, "
                                                   f"M={data.get('randaugment_m', 10)}"
                                                   f"{', + Cutout' if data.get('cutout', 'randaugment_n' not in data) else ''})"),
                           ("labeled split", data.get("split", "stratified")),
                           ("train / val / test", "re-split " + f"{100 * (1 - data.get('val_ratio', 0) - data.get('test_ratio', 0.15)):g}"
                            f" / {100 * data.get('val_ratio', 0):g} / {100 * data.get('test_ratio', 0.15):g}%"
                            if data.get("holdout") == "resplit" else
                            "official split" + (f", {100 * data['val_ratio']:g}% of train held out for validation"
                                                if data.get("val_ratio") else "")),
                           ("horizontal flip", {True: "on", False: "off"}.get(summ.get("hflip"), "on (not recorded)")),
                           ("augmentation on", data.get("augment_device", "cpu")),
                           ("DataLoader workers", summ.get("num_workers", data.get("num_workers")))])),
        ("🌐 Federation", _rows([("clients N", server.get("num_clients")), ("clients per round C", server.get("client_fraction")),
                                 ("rounds", f"{meta['rounds']} / {server.get('rounds')}"),
                                 ("aggregator", server.get("aggregator")), ("evaluate every", server.get("eval_every")),
                                 ("seed", cfg.get("seed")),
                                 ("local training / client / round", f"{summ['mean_client_s']:.2f} s"
                                  if summ.get("mean_client_s") is not None else "not recorded"),
                                 ("server work / round", f"{summ['mean_server_s']:.2f} s"
                                  if summ.get("mean_server_s") is not None else "not recorded")]
                                + ([("best validation acc", f"{_pct(summ['best_val_acc'])}% at round {summ.get('best_val_round')}"),
                                    ("test acc at best validation", f"{_pct(summ.get('test_acc_at_best_val'))}%")]
                                   if summ.get("best_val_acc") is not None else []))),
        ("🧠 Local training", _rows([("model", model.get("name")), ("normalisation", model.get("norm")),
                                     ("optimizer", strat.get("optimizer")), ("learning rate", strat.get("lr")),
                                     ("momentum", strat.get("momentum")), ("weight decay", strat.get("weight_decay")),
                                     ("local work", lr), ("grad clip", strat.get("grad_clip")),
                                     ("extra FedProx μ", strat.get("prox_mu")),
                                     ("precision", summ.get("precision", strat.get("precision", "fp32"))),
                                     ("channels_last", strat.get("channels_last", False)),
                                     ("cuDNN benchmark", strat.get("cudnn_benchmark", False)),
                                     ("torch.compile", strat.get("compile") or "off")])),
        (f"🎛️ {html.escape(algo_label(meta['algorithm']))} parameters",
         _param_rows(STRATEGIES.get(meta["algorithm"]) if meta["algorithm"] in STRATEGIES else None,
                     strat.get("params", {}))),
        (f"🖥️ Aggregator ({html.escape(str(server.get('aggregator')))})",
         _param_rows(AGGREGATORS.get(server["aggregator"]) if server.get("aggregator") in AGGREGATORS else None,
                     server.get("params", {}))),
        ("💻 Environment", _rows([("GPU", f"{env['gpu']} · {env.get('gpu_memory_gb', '?')} GB" if env.get("gpu")
                                   else env.get("device", "not recorded")),
                                  ("PyTorch", env.get("torch")), ("Python", env.get("python")),
                                  ("SSFL build", env.get("code")), ("system", env.get("platform")),
                                  ("folder", str(run_dir))])),
    ]
    body = "".join(f'<div class="dcard"><div class="dtitle">{t}</div><table>{rows}</table></div>' for t, rows in cards)
    note = ""
    if not summ:
        note = ('<div class="muted small">This run was made with an older version: timing, hardware and '
                'default parameters were not recorded.</div>')
    return f'<div class="details">{head}{kpis}<div class="dgrid">{body}</div>{note}</div>'


def downloadable(run_dir: Path, names: Sequence[str]) -> List[Optional[str]]:
    """Copies of a run's files in the temp folder, for the download buttons.

    Gradio only serves files from the working directory or the temp folder. Runs on
    Google Drive (/content/drive/...) are outside both, and returning their paths
    failed the whole update ("Error" on every panel)."""
    import shutil
    dest = Path(tempfile.gettempdir()) / "ssfl_studio" / "run_files" / Path(run_dir).name
    dest.mkdir(parents=True, exist_ok=True)
    out: List[Optional[str]] = []
    for n in names:
        src = Path(run_dir) / n
        if src.exists():
            shutil.copy2(src, dest / n)
            out.append(str(dest / n))
        else:
            out.append(None)
    return out


def detail_frames(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = pd.read_csv(run_dir / "metrics.csv")
    except Exception:  # noqa: BLE001
        return _typed([], "series"), _typed([], "series")

    def frame(cols):
        rows = []
        for col, label in cols:
            if col in df:
                d = df.dropna(subset=[col])
                rows += [{"round": int(r), "value": float(v), "series": label} for r, v in zip(d["round"], d[col])]
        return _typed(rows, "series")

    acc = frame([("test_acc", "test accuracy"), ("train_pl_acc", "pseudo-label accuracy"),
                 ("train_mask_rate", "pseudo-label mask")])
    loss = frame([(c, c.replace("train_loss_", "").replace("train_", "").replace("_", " "))
                  for c in df.columns if c.startswith("train_loss")] + [("test_loss", "test loss")])
    return acc, loss


THESIS_VALUES = {"acc": "Best test accuracy (%)", "f1": "Best macro F1 (%)",
                 "val": "Test accuracy at the best validation round (%)",
                 "ratio": "Relative accuracy ratio (method / FedAvg, same setting)"}
THESIS_COLUMN = {"acc": "best_acc", "f1": "best_f1", "val": "test_at_val", "ratio": "best_acc"}


def thesis_table(runs: List[Dict[str, Any]], value: str = "acc") -> pd.DataFrame:
    """Rows = algorithm (variant), columns = dataset · labels · α.

    ``value``: ``acc`` best test accuracy (%), ``f1`` best macro F1 (%), ``val`` test accuracy
    at the round with the best validation accuracy (runs with ``data.val_ratio > 0``), ``ratio``
    best accuracy divided by FedAvg's best accuracy in the same column (the ADD's Eq. 7; FedAvg
    trained on the same labeled data, i.e. the low baseline). Only finished runs count; for
    repeated runs of the same name the most recent one is used.

    Runs that differ only in the seed (grid ``seeds``) are averaged, as in the FedLabel paper:
    the table holds the mean; ``table.attrs["std"]`` / ``table.attrs["n"]`` hold the standard
    deviation and the number of seeds per cell (see :func:`thesis_display`)."""
    rows = [r for r in runs if r["best_acc"] is not None and r["finished"]]
    if not rows:
        return pd.DataFrame()
    latest: Dict[str, Dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r["mtime"]):
        latest[r["base"]] = r
    df = pd.DataFrame(latest.values())

    def setting(r):
        out = r["dataset"]
        if r["labels"] is not None and r["labels"] == r["labels"]:
            out += f" {100 * r['labels']:g}%"
        if r.get("alpha") is not None and r["alpha"] == r["alpha"]:
            out += f" α{r['alpha']:g}"
        return out

    col = THESIS_COLUMN.get(value, "best_acc")
    if col not in df or df[col].isna().all():
        return pd.DataFrame()
    df["setting"] = df.apply(setting, axis=1)
    df["seed"] = df["seed"].fillna(0) if "seed" in df else 0
    # one value per seed (same-seed duplicates, e.g. a GUI and a grid run: the best), then over seeds
    per_seed = df.dropna(subset=[col]).groupby(["variant", "setting", "seed"])[col].max()
    cells = per_seed.groupby(["variant", "setting"])
    piv = cells.mean().unstack("setting")
    std = cells.std(ddof=0).unstack("setting").reindex_like(piv)
    n = cells.count().unstack("setting").reindex_like(piv)
    if value == "ratio":
        base = piv.loc["FedAvg"] if "FedAvg" in piv.index else None
        piv = piv / base if base is not None else piv * float("nan")
        std = std * float("nan")
    else:
        piv, std = piv * 100, std * 100
    order = [algo_label(a) for a in ALGO_ORDER]
    idx = sorted(piv.index, key=lambda v: (order.index(v.split(" (")[0]) if v.split(" (")[0] in order else 99, v))
    digits = 3 if value == "ratio" else 2

    def tidy(t):
        t = t.reindex(idx).round(digits).reset_index().rename(columns={"variant": "algorithm"})
        t.columns.name = None
        return t

    out = tidy(piv)
    out.attrs["std"], out.attrs["n"] = tidy(std), tidy(n)
    return out


def thesis_display(table: pd.DataFrame) -> pd.DataFrame:
    """The thesis table as text: ``mean ± std`` where a cell averages several seeds."""
    if table is None or not len(table) or "std" not in table.attrs:
        return table
    std, n = table.attrs["std"], table.attrs["n"]
    out = table.copy().astype(object)
    for c in table.columns[1:]:
        for i in range(len(table)):
            v = table.at[i, c]
            if v != v:
                out.at[i, c] = ""
            elif n.at[i, c] == n.at[i, c] and n.at[i, c] > 1 and std.at[i, c] == std.at[i, c]:
                out.at[i, c] = f"{v:g} ± {std.at[i, c]:.2f}"
            else:
                out.at[i, c] = f"{v:g}"
    return out


def _tex(text: str) -> str:
    """Escape LaTeX special characters (``%`` would otherwise start a comment)."""
    out = str(text).replace("\\", r"\textbackslash{}")
    for ch in "%_&#$":
        out = out.replace(ch, "\\" + ch)
    return out


def to_latex(table: pd.DataFrame, caption: str = "Best test accuracy (%)", digits: int = 2) -> str:
    if table is None or not len(table):
        return "% no results yet\n"
    cols = list(table.columns)
    num = table[cols[1:]].apply(pd.to_numeric, errors="coerce")
    best = num.max(axis=0)
    lines = [r"% requires \usepackage{booktabs}", r"\begin{table}[t]", r"\centering",
             f"\\caption{{{_tex(caption)}}}",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_tex(c) for c in cols) + r" \\", r"\midrule"]
    std, n = table.attrs.get("std"), table.attrs.get("n")
    for i, row in table.iterrows():
        cells = [_tex(row[cols[0]])]
        for c in cols[1:]:
            v = num.loc[i, c]
            if v != v:
                cells.append("--")
            else:
                s = f"{v:.{digits}f}"
                s = f"\\textbf{{{s}}}" if v == best[c] else s
                if std is not None and n.at[i, c] > 1 and std.at[i, c] == std.at[i, c]:
                    s += f" {{\\scriptsize$\\pm${std.at[i, c]:.2f}}}"     # mean (± std) over seeds
                cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run_curves(runs_dir: str, names: Sequence[str], metric: str) -> pd.DataFrame:
    rows = []
    root = Path(runs_dir)
    bases = [re.sub(RUN_STAMP, "", n) for n in names or []]
    for name in names or []:
        p = root / name / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if metric not in df:
            continue
        d = df.dropna(subset=[metric])
        label = re.sub(RUN_STAMP, "", name)
        if bases.count(label) > 1:          # repeated setting: keep the timestamp to tell them apart
            label = name
        rows += [{"round": int(r), "value": float(v), "run": label} for r, v in zip(d["round"], d[metric])]
    return _typed(rows)


# ------------------------------------------------------------- partition
#: Last dataset loaded for a preview, keyed by (name, root, image size): previews are instant
#: after the first one.
RAW_CACHE: Dict[Tuple[str, str, int], Any] = {}
#: Latest preview request per browser session (an older, still-loading preview stops updating).
PREVIEW_SEQ: Dict[str, int] = {}
LOAD_LOCKS: Dict[Tuple[str, str, int], threading.Lock] = {}


def _dir_size(root: str) -> int:
    """Bytes under ``root`` (used to show download progress)."""
    import os
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _busy_html(msg: str) -> str:
    return f'<div class="busy"><span class="spinner"></span><span>{msg}</span></div>'


def _error_html(msg: str, details: str = "") -> str:
    more = (f'<details><summary>Details</summary><pre>{html.escape(details)}</pre></details>'
            if details else "")
    return f'<div class="busy err"><b>⚠️ {html.escape(msg)}</b>{more}</div>'


def partition_frame(manager, max_clients: int = 40) -> pd.DataFrame:
    rows = []
    for cid, part in enumerate(manager.partitions[:max_clients]):
        counts = np.bincount(manager.train.labels[part], minlength=manager.num_classes)
        for c, n in enumerate(counts):
            if n:
                rows.append({"client": f"C{cid:02d}", "samples": int(n), "class": f"class {c}"})
    return pd.DataFrame(rows, columns=["client", "samples", "class"]).astype(
        {"client": "str", "samples": "int64", "class": "str"})


def partition_stats(manager) -> str:
    sizes = np.array([len(p) for p in manager.partitions])
    labs = np.array([len(s[0]) for s in manager.splits])
    cls_per_client = np.array([(np.bincount(manager.train.labels[p], minlength=manager.num_classes) > 0).sum()
                               for p in manager.partitions])
    cards = [
        _kpi("Train / test", f"{len(manager.train):,}", f"{len(manager.test):,} test · "
                                                       f"{manager.num_classes} classes", "a1"),
        _kpi("Clients", str(len(sizes)), f"{sizes.min()}–{sizes.max()} samples each", "a2"),
        _kpi("Labeled", f"{100 * labs.sum() / sizes.sum():.1f}<span>%</span>",
             f"{labs.sum():,} labeled samples", "a3"),
        _kpi("Classes / client", f"{cls_per_client.mean():.1f}", f"min {cls_per_client.min()} · "
                                                                 f"max {cls_per_client.max()}", "a4"),
    ]
    note = ""
    if len(sizes) > 40:
        note = '<div class="muted small">Chart shows the first 40 clients.</div>'
    return '<div class="kpi-grid four">' + "".join(cards) + "</div>" + note


# ------------------------------------------------------------------ app
THEME_ARGS = dict(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size="lg")

CSS = r"""
:root { --ss-a:#6366f1; --ss-b:#8b5cf6; --ss-c:#d946ef; --ss-ok:#10b981; --ss-warn:#f59e0b; --ss-err:#f43f5e; }
.gradio-container { width: 100% !important; max-width: 1480px !important; margin: 0 auto !important; }
html, body { overflow-x: hidden; }
footer { display:none !important; }

/* hero */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:26px 30px; margin-bottom:6px;
  color:#fff; display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(255,255,255,.18), transparent 60%),
              linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #c026d3 100%);
  box-shadow: 0 18px 40px -18px rgba(79,70,229,.55); }
.hero::after { content:""; position:absolute; inset:0; pointer-events:none; opacity:.18;
  background-image: radial-gradient(rgba(255,255,255,.9) 1px, transparent 1px); background-size: 22px 22px;
  mask-image: linear-gradient(90deg, transparent 30%, #000 100%); }
.hero-main { display:flex; align-items:center; gap:18px; position:relative; z-index:1; }
.hero h1 { margin:0 !important; font-size:30px !important; font-weight:800 !important; letter-spacing:-.02em;
  color:#fff !important; }
.hero p { margin:4px 0 0 !important; opacity:.9; max-width:720px; font-size:14.5px; color:#fff !important; }
.logo { width:58px; height:58px; flex:none; filter: drop-shadow(0 6px 12px rgba(0,0,0,.25)); }
.chips { display:flex; gap:8px; flex-wrap:wrap; position:relative; z-index:1; }
.chip { background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px);
  padding:6px 12px; border-radius:999px; font-size:12.5px; font-weight:600; display:inline-flex;
  align-items:center; gap:7px; white-space:nowrap; }
@media (max-width: 640px) {
  .hero { padding:18px; border-radius:16px; }
  .hero-main { flex-direction:column; align-items:flex-start; gap:10px; }
  .hero h1 { font-size:24px !important; }
  .hero p { font-size:13.5px; }
  .logo { width:42px; height:42px; }
  .chip { white-space:normal; }
}
.led { width:8px; height:8px; border-radius:50%; background:#fbbf24; display:inline-block; }
.led.on { background:#34d399; box-shadow:0 0 0 3px rgba(52,211,153,.3); }

/* panels and section titles */
.panel { border-radius:16px !important; }
.sec-title { display:flex; align-items:center; gap:10px; margin:2px 0 2px; }
.sec-title .n { width:26px; height:26px; border-radius:8px; display:grid; place-items:center; color:#fff;
  font-weight:700; font-size:13px; background:linear-gradient(135deg,var(--ss-a),var(--ss-c)); }
.sec-title b { font-size:15px; }
.sec-title span.muted { font-size:12.5px; }
.muted { opacity:.65; font-weight:400; margin-left:8px; }
.small { font-size:12px; margin-top:6px; }

/* scenario + algorithm cards (radio / checkbox groups) */
.cards .wrap { display:grid !important; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap:10px; }
.cards .wrap label { margin:0 !important; padding:12px 14px !important; border-radius:12px !important;
  border:1.5px solid var(--border-color-primary) !important; background:var(--block-background-fill) !important;
  transition: all .15s ease; cursor:pointer; min-height:46px; font-weight:600; box-shadow:none !important; }
.cards .wrap label:hover { border-color:var(--ss-a) !important; transform: translateY(-1px);
  box-shadow:0 6px 16px -10px rgba(99,102,241,.6) !important; }
.cards .wrap label.selected, .cards .wrap label:has(input:checked) { border-color:var(--ss-a) !important;
  background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(217,70,239,.10)) !important;
  color: var(--body-text-color) !important; box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important; }
.cards .wrap label.selected span, .cards .wrap label:has(input:checked) span { color: var(--body-text-color) !important; }
.cards .wrap label span { line-height:1.35; }
.algos .wrap { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }

/* algorithm info */
.algo-info { display:grid; grid-template-columns: repeat(auto-fill, minmax(240px,1fr)); gap:10px; }
.algo-info.empty { display:block; opacity:.75; font-size:13.5px; padding:4px 2px; }
.clear-note { font-size:13.5px; padding:4px 2px; line-height:1.5; }
.clear-note ul { margin:6px 0 0 18px; }
.algo-card { border-radius:14px; padding:12px 14px; border:1px solid var(--border-color-primary);
  background:var(--block-background-fill); position:relative; overflow:hidden; }
.algo-card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--ss-a); }
.algo-card.k-baseline::before { background:#64748b; }
.algo-card.k-featured::before { background:linear-gradient(var(--ss-b),var(--ss-c)); }
.algo-card.k-ssl::before { background:var(--ss-ok); }
.algo-top { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:5px; }
.algo-top b { font-size:15px; }
.algo-desc { font-size:13px; opacity:.85; line-height:1.45; }
.algo-card code { display:inline-block; margin-top:8px; font-size:11.5px; }
.algo-note { grid-column:1/-1; font-size:12.5px; opacity:.75; }
.tag { font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; white-space:nowrap; }
.t-baseline { background:rgba(100,116,139,.15); color:#64748b; }
.t-ssl { background:rgba(16,185,129,.14); color:#059669; }
.t-featured { background:linear-gradient(135deg, rgba(139,92,246,.18), rgba(217,70,239,.18)); color:#a21caf; }
.t-custom { background:rgba(245,158,11,.15); color:#b45309; }

/* buttons */
.launch button, button.launch { font-size:16px !important; font-weight:700 !important; min-height:48px;
  background: linear-gradient(120deg, #4f46e5, #7c3aed 55%, #c026d3) !important; color:#fff !important;
  border:none !important; box-shadow: 0 10px 24px -12px rgba(124,58,237,.8) !important; }
.launch:hover { filter: brightness(1.07); }

/* status banner */
.status { border-radius:16px; padding:14px 18px; border:1px solid var(--border-color-primary);
  background: var(--block-background-fill); }
.status-head { display:flex; align-items:center; gap:4px; font-size:15px; flex-wrap:wrap; }
.status .dot { width:10px; height:10px; border-radius:50%; background:#94a3b8; margin-right:8px; }
.status-running .dot, .status-stopping .dot { background:var(--ss-a); animation: pulse 1.4s infinite; }
.status-done .dot { background:var(--ss-ok); } .status-stopped .dot { background:var(--ss-warn); }
.status-error .dot { background:var(--ss-err); }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(99,102,241,.6);} 70% { box-shadow:0 0 0 10px rgba(99,102,241,0);}
  100% { box-shadow:0 0 0 0 rgba(99,102,241,0);} }
.bar { height:8px; border-radius:99px; background:rgba(148,163,184,.25); margin:12px 0 8px; overflow:hidden; }
.bar > div { height:100%; border-radius:99px; transition: width .6s ease;
  background: linear-gradient(90deg, var(--ss-a), var(--ss-b), var(--ss-c)); background-size:200% 100%;
  animation: flow 3s linear infinite; }
.status-done .bar > div { background: linear-gradient(90deg, #10b981, #34d399); animation:none; }
@keyframes flow { 0% { background-position:0 0; } 100% { background-position:200% 0; } }
.status-foot { font-size:12.5px; opacity:.75; }
.queue { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
.qchip { font-size:12px; font-weight:600; padding:4px 10px; border-radius:999px;
  border:1px solid var(--border-color-primary); }
.q-running { border-color:var(--ss-a); color:var(--ss-a); background:rgba(99,102,241,.08); }
.q-done { border-color:var(--ss-ok); color:#059669; background:rgba(16,185,129,.08); }
.q-error { border-color:var(--ss-err); color:var(--ss-err); }
.q-stopped, .q-cancelled { opacity:.6; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:10px; }
.kpi-grid.four { grid-template-columns: repeat(4, minmax(0,1fr)); }
@media (max-width: 900px) { .kpi-grid, .kpi-grid.four { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.kpi { border-radius:14px; padding:12px 14px; border:1px solid var(--border-color-primary);
  background: var(--block-background-fill); position:relative; overflow:hidden; }
.kpi::after { content:""; position:absolute; right:-30px; top:-30px; width:90px; height:90px; border-radius:50%;
  opacity:.14; background: var(--kc, var(--ss-a)); }
.kpi.a1 { --kc:#6366f1; } .kpi.a2 { --kc:#d946ef; } .kpi.a3 { --kc:#10b981; } .kpi.a4 { --kc:#f59e0b; }
.kpi.a5 { --kc:#0ea5e9; }
.kpi-label { font-size:11.5px; font-weight:700; letter-spacing:.05em; opacity:.6; }
.kpi-value { font-size:26px; font-weight:800; letter-spacing:-.02em; margin-top:2px; color: var(--kc); }
.kpi-value span { font-size:15px; margin-left:1px; opacity:.8; }
.kpi-sub { font-size:12px; opacity:.6; min-height:16px; }

/* guide */
.guide h2 { font-size:20px !important; margin:18px 0 10px !important; }
.steps { display:grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap:10px; }
.step { display:flex; gap:12px; border-radius:14px; padding:12px 14px; border:1px solid var(--border-color-primary);
  background:var(--block-background-fill); }
.step-n { width:30px; height:30px; flex:none; border-radius:10px; display:grid; place-items:center; color:#fff;
  font-weight:800; background:linear-gradient(135deg,var(--ss-a),var(--ss-c)); }
.step p { margin:3px 0 0 !important; font-size:13px; opacity:.8; }
.algo-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px,1fr)); gap:10px; }
.tips li { margin:4px 0; font-size:14px; }
.param-group { border-radius:14px !important; }
/* run details */
.details .dhead { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin:4px 0 12px; }
.details h3 { margin:0 !important; font-size:18px !important; word-break:break-all; }
.badge { font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px; white-space:nowrap;
  border:1px solid var(--border-color-primary); }
.b-finished { color:#059669; border-color:#10b981; background:rgba(16,185,129,.08); }
.b-running { color:var(--ss-a); border-color:var(--ss-a); }
.b-stopped, .b-interrupted, .b-partial { color:#b45309; border-color:#f59e0b; background:rgba(245,158,11,.08); }
.b-error { color:var(--ss-err); border-color:var(--ss-err); }
.dgrid { display:grid; grid-template-columns: repeat(auto-fill, minmax(300px,1fr)); gap:10px; margin-top:12px; }
.dcard { border-radius:14px; padding:12px 14px; border:1px solid var(--border-color-primary);
  background:var(--block-background-fill); }
.dtitle { font-weight:700; margin-bottom:6px; }
.dcard table { width:100%; border-collapse:collapse; font-size:13px; }
.dcard td { padding:4px 4px; border-top:1px solid var(--border-color-primary); vertical-align:top; }
.dcard td:first-child { opacity:.7; width:46%; }
.dcard td:last-child { font-family: var(--font-mono); word-break:break-all; }
.dcard tr.chg td:last-child { color:var(--ss-a); font-weight:700; }
.dflt { display:block; font-weight:400; opacity:.6; font-size:11px; }
.busy { display:flex; align-items:center; gap:12px; flex-wrap:wrap; border-radius:14px; padding:14px 16px;
  border:1px solid var(--border-color-primary); background:var(--block-background-fill); font-size:14px; }
.busy.err { border-color:var(--ss-err); display:block; }
.busy pre { white-space:pre-wrap; font-size:11.5px; max-height:220px; overflow:auto; }
.spinner { width:18px; height:18px; border-radius:50%; flex:none; border:3px solid rgba(99,102,241,.25);
  border-top-color:var(--ss-a); animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
"""


FAST_MODE = {"data.augment_device": "gpu", "strategy.precision": "bf16", "strategy.cudnn_benchmark": True}


def _is_fast(values: Dict[str, Any]) -> bool:
    return all(values.get(k) == val for k, val in FAST_MODE.items())


def gpu_usage() -> str:
    """"GPU 63% · 5.1/96 GB" for the status bar (utilisation needs nvidia-ml-py; omitted without it)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        free, total = torch.cuda.mem_get_info()
        text = f"{(total - free) / 2**30:.1f}/{total / 2**30:.0f} GB"
        try:
            text = f"{torch.cuda.utilization()}% · " + text
        except Exception:  # noqa: BLE001 - pynvml missing
            pass
        return "GPU " + text
    except Exception:  # noqa: BLE001
        return ""


def _section(n: str, title: str, hint: str = "") -> str:
    return (f'<div class="sec-title"><div class="n">{n}</div><b>{title}</b>'
            + (f'<span class="muted">{hint}</span>' if hint else "") + "</div>")


def build_app():
    import gradio as gr

    ctrl = RunController()
    speed = SpeedCheck()
    presets = list_presets()
    start_preset = "organamnist_fedlabel.yaml" if "organamnist_fedlabel.yaml" in presets else (
        presets[0] if presets else None)
    start_cfg = apply_env(load_preset(start_preset))
    v = dict(zip(FIELDS, cfg_to_values(start_cfg)))
    v["name"] = ""
    start_algos = [start_cfg.strategy.name]
    start_params = {start_cfg.strategy.name: dict(start_cfg.strategy.params)}

    with gr.Blocks(title="SSFL Studio") as app:
        gr.HTML(render_hero())
        params_state = gr.State(start_params)

        with gr.Tabs() as main_tabs:
            # ======================================================== EXPERIMENT
            with gr.Tab("🧪 Experiment", id="experiment"):
                with gr.Row(equal_height=False):
                    # ---------------------------------------- left: configuration
                    with gr.Column(scale=5, min_width=380):
                        with gr.Group(elem_classes="panel"):
                            gr.HTML(_section("1", "Scenario", "loads the thesis settings"))
                            preset = gr.Radio(preset_choices(presets), value=start_preset, show_label=False,
                                              elem_classes="cards", container=False)
                        with gr.Group(elem_classes="panel"):
                            gr.HTML(_section("2", "Algorithms", "tick several to compare them"))
                            algos = gr.CheckboxGroup(algo_choices(), value=start_algos, show_label=False,
                                                     elem_classes=["cards", "algos"], container=False)
                            algo_info = gr.HTML(render_algo_cards(start_algos))
                        with gr.Group(elem_classes="panel"):
                            gr.HTML(_section("3", "Key settings"))
                            labeled_ratio = gr.Slider(0.01, 1.0, value=v["data.labeled_ratio"], step=0.01,
                                                      label="Labeled data per client",
                                                      info="Thesis: 20% / 50% (CIFAR-10), 5% / 20% (MedMNIST)")
                            rounds = gr.Slider(1, 2000, value=v["server.rounds"], step=1,   # presets: 1,000 (ADD)
                                               label="Communication rounds")
                            alpha = gr.Slider(0.01, 10, value=v["data.alpha"], step=0.01,
                                              label="Non-IID level — Dirichlet α",
                                              info="Smaller α = more skewed class mix per client")
                            with gr.Row():
                                num_clients = gr.Number(v["server.num_clients"], label="Clients N", precision=0,
                                                        minimum=1)
                                client_fraction = gr.Slider(0.01, 1.0, value=v["server.client_fraction"],
                                                            step=0.01, label="Clients per round C")
                            fast = gr.Checkbox(
                                _is_fast(v), label="⚡ Fast GPU mode",
                                info="Augmentation on the GPU + bf16 precision + cuDNN benchmark, for large GPUs (A100 / H100 / "
                                     "RTX PRO class), where CPU augmentation is the bottleneck. On small GPUs "
                                     "(T4, laptops) it gains little. Same augmentations, but not bit-identical "
                                     "to the CPU pipeline: keep it off to reproduce earlier runs.")
                        with gr.Accordion("🎛️  Algorithm parameters", open=False):
                            gr.Markdown("Defaults come from the papers. Each run's `config.yaml` records "
                                        "every value it used, defaults included.", elem_classes="muted")

                            # One static form per registered algorithm, shown only when the algorithm
                            # is ticked. (Static components are robust behind proxies such as Colab /
                            # gradio.live; dynamically rendered forms lost edits there.)
                            param_groups: Dict[str, Any] = {}
                            param_fields: List[Tuple[str, str, type, Any]] = []
                            for _, algo in algo_choices():
                                defaults = default_params(STRATEGIES.get(algo))
                                current = {**defaults, **start_params.get(algo, {})}
                                with gr.Group(visible=algo in start_algos, elem_classes="param-group") as grp:
                                    gr.HTML(f'<div class="sec-title"><b>{html.escape(algo_label(algo))}</b>'
                                            f'<span class="muted">{len(defaults)} parameters</span></div>')
                                    if not defaults:
                                        gr.Markdown("*No parameters.*")
                                    with gr.Row(equal_height=True):
                                        for key, default in defaults.items():
                                            comp, kind = _param_field(gr, key, current.get(key), default)
                                            param_fields.append((algo, key, kind, comp))
                                param_groups[algo] = grp

                        with gr.Accordion("🗂️  Data & augmentation", open=False):
                            with gr.Row():
                                dataset = gr.Dropdown(DATASETS.names(), value=v["data.dataset"], label="Dataset")
                                partitioner = gr.Dropdown(PARTITIONERS.names(), value=v["data.partitioner"],
                                                          label="Partitioner")
                                image_size = gr.Dropdown([28, 64, 128, 224], value=v["data.image_size"],
                                                         label="Image size (MedMNIST)")
                            with gr.Row():
                                batch_size = gr.Number(v["data.batch_size"], label="Labeled batch", precision=0)
                                unl_ratio = gr.Number(v["data.unlabeled_batch_ratio"], precision=0,
                                                      label="Unlabeled batch ratio μ")
                                num_workers = gr.Number(v["data.num_workers"], precision=0,
                                                        label="DataLoader workers", minimum=-1,
                                                        info="-1 = auto: 0 on Windows or for small rounds "
                                                             "(quick demo), up to 4 on Colab / Linux")
                            with gr.Row():
                                strong_aug = gr.Dropdown(["randaugment", "none"], value=v["data.strong_aug"],
                                                         label="Strong augmentation")
                                augment_device = gr.Dropdown(["cpu", "gpu"], value=v["data.augment_device"],
                                                             label="Augmentation on",
                                                             info="gpu: dataset kept on the GPU, batched")
                                root = gr.Textbox(v["data.root"], label="Download folder")
                            with gr.Row():
                                ra_n = gr.Number(v["data.randaugment_n"], precision=0, label="RandAugment N",
                                                 info="ops per image (ADD: 1)")
                                ra_m = gr.Number(v["data.randaugment_m"], precision=0, label="RandAugment M",
                                                 info="magnitude (ADD: 10)")
                                cutout = gr.Checkbox(v["data.cutout"], label="Cutout",
                                                     info="extra erasing on the strong view (not in the ADD)")
                                split = gr.Dropdown(["random", "stratified"], value=v["data.split"],
                                                    label="Labeled split", info="random = ADD")
                                hflip = gr.Dropdown(["auto", "on", "off"], value=v["data.hflip"],
                                                    label="Horizontal flip",
                                                    info="auto = off for OrganAMNIST (left/right organ classes)")
                            with gr.Row():
                                holdout = gr.Dropdown(["official", "resplit"], value=v["data.holdout"],
                                                      label="Test set",
                                                      info="official = dataset's test split (ADD) · resplit = "
                                                           "pool all and re-divide (paper 80/5/15)")
                                val_ratio = gr.Number(v["data.val_ratio"], label="Validation share",
                                                      info="0 = none · paper 0.05 · official: share of train")
                                test_ratio = gr.Number(v["data.test_ratio"], label="Test share (resplit)",
                                                       info="paper 0.15")
                        with gr.Accordion("🧠  Model & local training", open=False):
                            with gr.Row():
                                model_name = gr.Dropdown(MODELS.names(), value=v["model.name"], label="Model")
                                norm = gr.Dropdown(["gn", "bn", "none"], value=v["model.norm"],
                                                   label="Normalisation", info="GroupNorm is robust to non-IID")
                                optimizer = gr.Dropdown(["sgd", "adam"], value=v["strategy.optimizer"],
                                                        label="Optimizer")
                            with gr.Row():
                                lr = gr.Number(v["strategy.lr"], label="Learning rate")
                                momentum = gr.Number(v["strategy.momentum"], label="Momentum")
                                weight_decay = gr.Number(v["strategy.weight_decay"], label="Weight decay")
                            with gr.Row():
                                local_epochs = gr.Number(v["strategy.local_epochs"], label="Local epochs",
                                                         precision=0)
                                local_steps = gr.Number(v["strategy.local_steps"], precision=0,
                                                        label="Local steps", info="0 = use epochs")
                                grad_clip = gr.Number(v["strategy.grad_clip"], label="Grad clip",
                                                      info="0 = off")
                            with gr.Row():
                                prox_mu = gr.Number(v["strategy.prox_mu"], label="Extra FedProx μ",
                                                    info="adds a proximal term to any algorithm")
                            with gr.Row():
                                precision = gr.Dropdown(["fp32", "bf16", "fp16"], value=v["strategy.precision"],
                                                        label="Precision",
                                                        info="bf16: faster on modern GPUs, no loss scaling")
                                channels_last = gr.Checkbox(v["strategy.channels_last"], label="channels_last",
                                                            info="NHWC memory layout for convolutions")
                                cudnn_benchmark = gr.Checkbox(v["strategy.cudnn_benchmark"], label="cuDNN benchmark",
                                                              info="fastest conv kernels per input shape")
                                compile_ = gr.Dropdown(["none", *COMPILE_MODES], value=v["strategy.compile"],
                                                       label="torch.compile",
                                                       info="fewer GPU kernel launches; needs Triton (Colab / "
                                                            "Linux). Check with --bench first")
                            model_params = gr.Code(v["model.params"], language="json",
                                                   label="Model parameters (JSON)", lines=2)
                        with gr.Accordion("🖥️  Server & run", open=False):
                            with gr.Row():
                                aggregator = gr.Dropdown(AGGREGATORS.names(), value=v["server.aggregator"],
                                                         label="Aggregator")
                                eval_every = gr.Number(v["server.eval_every"], precision=0,
                                                       label="Evaluate every k rounds")
                            server_params = gr.Code(v["server.params"], language="json",
                                                    label="Aggregator parameters (JSON)", lines=2)
                            with gr.Row():
                                name = gr.Textbox(v["name"], label="Run name",
                                                  placeholder="auto: dataset_algorithm_labels")
                                seed = gr.Number(v["seed"], label="Seed", precision=0)
                                device = gr.Dropdown(["auto", "cuda", "cpu"], value=v["device"], label="Device",
                                                     allow_custom_value=True)
                            output_dir = gr.Textbox(v["output_dir"], label="Output folder")

                    # ---------------------------------------- right: dashboard
                    with gr.Column(scale=7, min_width=480):
                        with gr.Row():
                            run_btn = gr.Button("🚀  Launch", variant="primary", elem_classes="launch", scale=3)
                            stop_btn = gr.Button("■  Stop", variant="stop", scale=1)
                            preview_btn = gr.Button("🧩  Preview split", scale=1)
                            speed_btn = gr.Button("⏱️  Speed check", scale=1)
                            save_btn = gr.Button("💾  Save preset", scale=1)
                        notice = gr.Markdown("", visible=False)
                        status = gr.HTML(render_status(ctrl.snapshot()))
                        kpis = gr.HTML(render_kpis(ctrl.snapshot()))
                        with gr.Tabs() as dash_tabs:
                            with gr.Tab("📈  Live charts", id="charts"):
                                acc_plot = gr.LinePlot(_long_frame(ctrl.snapshot(), "test_acc"), x="round",
                                                       y="value", color="run", title="Test accuracy",
                                                       y_title="accuracy", y_lim=[0, 1], height=300,
                                                       color_title="algorithm")
                                with gr.Row():
                                    loss_plot = gr.LinePlot(_long_frame(ctrl.snapshot(), "train_loss"),
                                                            x="round", y="value", color="run",
                                                            title="Train loss", y_title="loss", height=250,
                                                            color_title="algorithm")
                                    pl_plot = gr.LinePlot(_pl_frame(ctrl.snapshot()), x="round", y="value",
                                                          color="series", title="Pseudo-label quality",
                                                          y_title="rate", y_lim=[0, 1], height=250,
                                                          color_title=" ")
                            # render_children: mount the chart at page load. A chart first mounted
                            # while data is arriving drew its bars without colours.
                            with gr.Tab("🧩  Data split", id="partition", render_children=True):
                                part_stats = gr.HTML('<div class="algo-info empty">Press <b>Preview split</b> '
                                                     'to see how the data is divided between clients.</div>')
                                part_plot = gr.BarPlot(pd.DataFrame({"client": pd.Series(dtype="str"),
                                                                     "samples": pd.Series(dtype="int64"),
                                                                     "class": pd.Series(dtype="str")}),
                                                       x="client", y="samples", color="class",
                                                       title="Samples per client, coloured by class",
                                                       height=360, sort="x")
                            with gr.Tab("⏱️  Speed check", id="speed"):
                                speed_info = gr.HTML(
                                    '<div class="algo-info empty">Press <b>Speed check</b> to time a few rounds of '
                                    'the current settings with each speed option (baseline, cuDNN benchmark, '
                                    'channels_last, torch.compile, torch.compile + CUDA graphs). Results are '
                                    'not saved. With several algorithms selected, FedLabel (or the first one) '
                                    'is timed.</div>')
                                speed_table = gr.Dataframe(pd.DataFrame(), label="Speed per setting",
                                                           wrap=True, interactive=False)
                            with gr.Tab("📜  Log", id="log"):
                                log = gr.Code("", language=None, label="Log", lines=22, max_lines=22)
                            with gr.Tab("🔢  Metrics", id="metrics"):
                                metrics = gr.Dataframe(pd.DataFrame(), label="Per-round metrics (newest first)",
                                                       wrap=True, max_height=480)
                            with gr.Tab("🧾  Config", id="config"):
                                yaml_view = gr.Code(language="yaml", label="Generated YAML (one document per run)",
                                                    lines=24)

            # ======================================================== GRID
            with gr.Tab("🗺️ Thesis grid", id="grid"):
                gr.HTML('<div class="algo-info empty">Run every dataset × label ratio × algorithm from a grid '
                        'file. Finished runs are skipped, so a stopped grid continues where it ended.</div>')
                with gr.Row():
                    grids = list_grids()
                    grid_file = gr.Dropdown(grids, value=grids[0] if grids else None, label="Grid file", scale=2)
                    grid_only = gr.Textbox("", label="Only runs containing",
                                           placeholder="e.g. organamnist fedmatch", scale=2)
                    env_extra = " ".join(
                        f"{k}={ENV_DEFAULTS[e]}" for e, k in (("data_dir", "data.root"), ("runs_dir", "output_dir"),
                                                             ("num_workers", "data.num_workers"))
                        if ENV_DEFAULTS.get(e) not in (None, ""))
                    grid_extra = gr.Textbox(env_extra, label="Extra overrides", scale=3,
                                            placeholder="e.g. server.rounds=50 output_dir=/content/drive/MyDrive/ssfl/runs")
                    grid_skip = gr.Checkbox(True, label="Skip finished", scale=1)
                with gr.Row():
                    grid_preview_btn = gr.Button("🔍  Preview runs", scale=1)
                    grid_run_btn = gr.Button("🚀  Run grid", variant="primary", elem_classes="launch", scale=2)
                    grid_stop_btn = gr.Button("■  Stop", variant="stop", scale=1)
                grid_status = gr.HTML(render_status(ctrl.snapshot()))
                grid_table = gr.Dataframe(pd.DataFrame(), label="Runs", wrap=True, max_height=420)
                grid_plot = gr.LinePlot(_typed([]), x="round", y="value",
                                        color="run", title="Test accuracy of the grid runs", y_lim=[0, 1],
                                        y_title="accuracy",
                                        height=320)
                with gr.Accordion("📜  Log", open=False):
                    grid_log = gr.Code("", language=None, lines=16, max_lines=16)

            # ======================================================== RESULTS
            with gr.Tab("🏆 Results", id="results"):
                with gr.Row():
                    runs_dir = gr.Textbox(v["output_dir"], label="Runs folder", scale=4)
                    refresh_btn = gr.Button("🔄  Refresh", scale=1)
                    clear_open_btn = gr.Button("🗑️  Clear runs", variant="stop", scale=1)
                with gr.Row():
                    f_dataset = gr.Dropdown([], value=[], multiselect=True, label="Dataset filter", scale=2)
                    f_algo = gr.Dropdown([], value=[], multiselect=True, label="Algorithm filter", scale=2)
                    metric = gr.Dropdown([(lab, k) for k, lab in METRIC_LABELS.items()], value="test_acc",
                                         label="Metric to plot", scale=2)
                res_kpis = gr.HTML("")
                board_runs = gr.State([])                  # run folders in leaderboard row order
                with gr.Tabs() as res_tabs:
                    with gr.Tab("🥇  Leaderboard", id="board"):
                        gr.Markdown("Click a row to open its details. ⚡ = Fast GPU mode.", elem_classes="muted")
                        board = gr.Dataframe(leaderboard([]), wrap=False, max_height=460, interactive=False)
                    with gr.Tab("🔍  Run details", id="details", render_children=True):
                        detail_pick = gr.Dropdown([], value=None, label="Run", info="newest first")
                        detail_html = gr.HTML('<div class="algo-info empty">Pick a run, or click a row in the '
                                              'leaderboard.</div>')
                        with gr.Row():
                            detail_acc = gr.LinePlot(_typed([], "series"), x="round", y="value", color="series",
                                                     title="Accuracy and pseudo-labels", y_lim=[0, 1],
                                                     y_title="rate", color_title=" ", height=300)
                            detail_loss = gr.LinePlot(_typed([], "series"), x="round", y="value", color="series",
                                                      title="Losses", y_title="loss", color_title=" ", height=300)
                        with gr.Row():
                            load_btn = gr.Button("↻  Load these settings into Experiment", variant="primary")
                            dl_run_cfg = gr.DownloadButton("⬇  config.yaml", value=None)
                            dl_run_metrics = gr.DownloadButton("⬇  metrics.csv", value=None)
                            dl_run_summary = gr.DownloadButton("⬇  summary.json", value=None)
                    with gr.Tab("📈  Curves", id="curves"):
                        run_select = gr.Dropdown([], value=[], multiselect=True,
                                                 label="Runs to plot (top 5 are pre-selected)")
                        curve_plot = gr.LinePlot(_typed([]), x="round",
                                                 y="value", color="run", height=380)
                    with gr.Tab("🎓  Thesis table", id="thesis"):
                        gr.Markdown("One row per algorithm, one column per dataset · labels · α, from finished "
                                    "runs (latest run of each name). Runs repeated with other seeds "
                                    "(grid `seeds`) are averaged: **mean ± std**, as in the FedLabel paper. "
                                    "The ratio divides by FedAvg's best accuracy in the same column (ADD "
                                    "Eq. 7). *Test acc. at best validation* needs runs with a validation "
                                    "set (`data.val_ratio`). Bold in LaTeX = best in column; the table "
                                    "uses the `booktabs` package.")
                        thesis_value = gr.Radio([(lab, k) for k, lab in THESIS_VALUES.items()], value="acc",
                                                label="Value")
                        thesis = gr.Dataframe(pd.DataFrame(), wrap=True)
                        with gr.Row():
                            dl_csv = gr.DownloadButton("⬇  Leaderboard CSV", value=None)
                            dl_tex = gr.DownloadButton("⬇  Thesis table LaTeX", value=None)
                        latex_view = gr.Code("", language="latex", label="LaTeX", lines=12)
                    with gr.Tab("🗑️  Clear runs", id="clear"):
                        gr.Markdown("Delete previous run folders from the runs folder above. Deleting is "
                                    "**permanent**. A run that is still running is never deleted. The "
                                    "grid runs a deleted run again the next time.", elem_classes="muted")
                        clear_pick = gr.Dropdown([], value=[], multiselect=True, label="Runs to delete",
                                                 info="newest first")
                        with gr.Row():
                            clear_shown_btn = gr.Button("Select runs in the leaderboard (filters)", size="sm")
                            clear_unfin_btn = gr.Button("Select unfinished", size="sm")
                            clear_all_btn = gr.Button("Select all", size="sm")
                            clear_none_btn = gr.Button("Clear selection", size="sm")
                        clear_btn = gr.Button("🗑️  Delete selected runs", variant="stop")
                        clear_msg = gr.HTML("")
                        with gr.Row(visible=False) as clear_confirm_row:
                            clear_yes = gr.Button("Yes, delete permanently", variant="stop")
                            clear_no = gr.Button("Cancel")

            # ======================================================== GUIDE
            with gr.Tab("📘 Guide", id="guide"):
                gr.HTML(render_guide())

        # ============================================================ wiring
        components = [name, seed, device, output_dir,
                      dataset, root, image_size, partitioner, alpha, labeled_ratio, batch_size,
                      unl_ratio, num_workers, strong_aug, augment_device, ra_n, ra_m, cutout, split, hflip,
                      holdout, val_ratio, test_ratio,
                      model_name, norm, model_params,
                      local_epochs, local_steps, optimizer, lr, momentum, weight_decay,
                      prox_mu, grad_clip, precision, channels_last, cudnn_benchmark, compile_,
                      aggregator, rounds, num_clients, client_fraction, eval_every, server_params]
        assert len(components) == len(FIELDS)
        cfg_inputs = components + [algos, params_state]

        def _configs(*vals):
            *fields, sel, state = vals
            return build_configs(fields, sel, state)

        def show_yaml(*vals):
            try:
                return "\n---\n".join(yaml.safe_dump(c.to_dict(), sort_keys=False) for c in _configs(*vals))
            except Exception as e:  # noqa: BLE001
                return f"# invalid configuration: {e}"

        def param_values(state) -> List[Any]:
            """Form values for every parameter field: the state's value, else the default."""
            out = []
            for algo, key, kind, _ in param_fields:
                defaults = default_params(STRATEGIES.get(algo))
                value = (state or {}).get(algo, {}).get(key, defaults.get(key))
                out.append(_display_value(kind, value))
            return out

        param_comps = [f[3] for f in param_fields]
        for algo, key, kind, comp in param_fields:
            # Every edit rewrites the whole parameter dict, so the updates must run one at a
            # time and in order (one shared queue); otherwise a slower event overwrites a
            # newer edit. Free-text JSON fields save on blur / Enter, not on every keystroke.
            events = [comp.blur, comp.submit] if kind not in (bool, int, float, str) else [comp.input]
            gr.on(events, _param_updater(algo, key, kind), [comp, params_state], params_state,
                  show_progress="hidden", concurrency_limit=1, concurrency_id="param-state",
                  trigger_mode="always_last" if kind in (int, float) else "multiple")

        def form_from_cfg(cfg):
            vals = cfg_to_values(cfg)
            vals[0] = ""                                  # auto run name
            state = {cfg.strategy.name: dict(cfg.strategy.params)}
            return (*vals, [cfg.strategy.name], state, render_algo_cards([cfg.strategy.name]),
                    _is_fast(dict(zip(FIELDS, vals))), *param_values(state))

        def on_preset(p):
            return form_from_cfg(apply_env(load_preset(p)))

        # `.input` fires only on user edits, so loading a preset keeps its tuned params.
        preset.input(on_preset, preset, components + [algos, params_state, algo_info, fast] + param_comps,
                     concurrency_limit=1, concurrency_id="param-state").then(
            show_yaml, cfg_inputs, yaml_view)

        # fast GPU mode <-> the three fields it sets
        fast.input(lambda on: ("gpu", "bf16", True) if on else ("cpu", "fp32", False), fast,
                   [augment_device, precision, cudnn_benchmark]).then(show_yaml, cfg_inputs, yaml_view)
        for comp in (augment_device, precision, cudnn_benchmark):
            comp.input(lambda a, p, b: _is_fast({"data.augment_device": a, "strategy.precision": p,
                                                 "strategy.cudnn_benchmark": b}),
                       [augment_device, precision, cudnn_benchmark], fast)

        def on_algos(sel):
            sel = sel or []
            return (render_algo_cards(sel), *[gr.update(visible=a in sel) for a in param_groups])

        algos.change(on_algos, algos, [algo_info] + list(param_groups.values()), trigger_mode="always_last",
                     show_progress="hidden").then(show_yaml, cfg_inputs, yaml_view)
        aggregator.input(lambda a: json.dumps(default_params(AGGREGATORS.get(a)), indent=2),
                         aggregator, server_params).then(show_yaml, cfg_inputs, yaml_view)
        for c in components:
            if c is not aggregator:
                c.input(show_yaml, cfg_inputs, yaml_view)
        params_state.change(show_yaml, cfg_inputs, yaml_view)

        def warn(msg):
            return gr.update(value=f"⚠️ {msg}", visible=True)

        def on_save(*vals):
            try:
                cfg = _configs(*vals)[0]
            except Exception as e:  # noqa: BLE001
                return warn(e), gr.update()
            path = CONFIG_DIR / f"{cfg.name}.yaml"
            CONFIG_DIR.mkdir(exist_ok=True)
            cfg.save(path)
            return (gr.update(value=f"💾 Saved preset **{path.name}** (first selected algorithm).", visible=True),
                    gr.update(choices=preset_choices(list_presets())))

        save_btn.click(on_save, cfg_inputs, [notice, preset])

        def on_preview(*vals, request: gr.Request = None):
            """Streams progress: the first use of a dataset downloads it (can take minutes
            on Colab / Drive), so show elapsed time and downloaded MB instead of nothing.
            A newer click in the same browser session takes over the panel."""
            from .data import DataManager
            session = getattr(request, "session_hash", None) or "default"
            my_id = PREVIEW_SEQ[session] = PREVIEW_SEQ.get(session, 0) + 1

            def superseded() -> bool:
                return PREVIEW_SEQ.get(session) != my_id

            try:
                cfg = _configs(*vals)[0]
            except Exception as e:  # noqa: BLE001
                yield gr.update(), gr.update(), warn(e), gr.update()
                return
            key = (cfg.data.dataset, str(Path(cfg.data.root).resolve()), cfg.data.image_size)
            result: Dict[str, Any] = {}

            def work():
                try:
                    # one loader per dataset at a time: a second click waits for the running
                    # download instead of writing the same file twice
                    with LOAD_LOCKS.setdefault(key, threading.Lock()):
                        raw = RAW_CACHE.get(key)
                        if raw is None:
                            raw = DATASETS.get(cfg.data.dataset)(cfg.data)
                            RAW_CACHE.clear()              # keep one dataset in memory
                            RAW_CACHE[key] = raw
                    result["dm"] = DataManager(cfg.data, cfg.server.num_clients, seed=cfg.seed, raw=raw)
                except Exception:  # noqa: BLE001 - shown in the panel
                    result["error"] = traceback.format_exc()

            t0, size0 = time.time(), _dir_size(cfg.data.root)
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            first = True
            while worker.is_alive():
                if superseded():                   # a newer preview owns the panel now
                    return
                mb = (_dir_size(cfg.data.root) - size0) / 2**20
                msg = (f"Loading <b>{html.escape(cfg.data.dataset)}</b> from "
                       f"<code>{html.escape(cfg.data.root)}</code> · elapsed {_fmt_secs(time.time() - t0)}"
                       + (f" · downloaded {mb:.1f} MB" if mb > 0.05 else
                          " · the first use of a dataset downloads it"))
                yield (gr.update(), _busy_html(msg), gr.update(visible=False),
                       gr.Tabs(selected="partition") if first else gr.update())
                first = False
                worker.join(timeout=1.0)
            if superseded():
                return
            if "error" in result:
                err = result["error"].strip().splitlines()[-1]
                yield (gr.update(), _error_html(f"Could not load the data: {err}", result["error"]),
                       warn(err), gr.Tabs(selected="partition"))
                return
            dm = result["dm"]
            frame = partition_frame(dm)
            yield (plot_update(frame, "class"), partition_stats(dm), gr.update(visible=False),
                   gr.Tabs(selected="partition"))
            # The bar chart does not rebuild its colour scale when data and colours arrive in
            # the same update (bars came out uncoloured). Re-sending a changed colour map once
            # the data is in place forces that rebuild; the extra key matches no bar.
            time.sleep(0.5)
            yield (gr.update(color_map={**color_map(frame, "class"), "​": "#000000"}),
                   gr.update(), gr.update(), gr.update())

        on_preview.__annotations__["request"] = gr.Request
        preview_btn.click(on_preview, cfg_inputs, [part_plot, part_stats, notice, dash_tabs],
                          show_progress="hidden", trigger_mode="multiple")

        # ---------------------------------------------------- live dashboard
        def exp_view(snap):
            return (render_status(snap), render_kpis(snap), plot_update(_long_frame(snap, "test_acc")),
                    plot_update(_long_frame(snap, "train_loss")), plot_update(_pl_frame(snap), "series"),
                    "\n".join(snap["logs"][-400:]), _metrics_table(snap))

        exp_outputs = [status, kpis, acc_plot, loss_plot, pl_plot, log, metrics]

        def stream(view, kind):
            """Push a full update whenever the job changes, and refresh only the status
            banner (first output) once a second in between, so the elapsed time keeps
            counting while nothing else happens, e.g. while data loads."""
            last, n_out, last_tick = -1, None, 0.0
            series: Dict[int, frozenset] = {}         # output index -> series names drawn so far

            def full(snap):
                """The view, then (if a chart got a new series) a colour-map re-send: the
                charts fix their colour scale when they are first drawn, so a series that
                arrives later stays uncoloured until the chart is redrawn."""
                out = view(snap)
                yield out
                changed = {}
                for i, o in enumerate(out):
                    cmap = o.get("color_map") if isinstance(o, dict) else None
                    if cmap and frozenset(cmap) != series.get(i):
                        series[i] = frozenset(cmap)
                        changed[i] = gr.update(color_map={**cmap, "​": "#000000"})
                if changed:
                    time.sleep(0.4)
                    yield tuple(changed.get(i, gr.update()) for i in range(len(out)))

            while True:
                snap = ctrl.snapshot()
                if snap["kind"] == kind and snap["version"] != last:
                    last, last_tick = snap["version"], time.time()
                    for out in full(snap):
                        n_out = len(out)
                        yield out
                elif snap["kind"] == kind and n_out and time.time() - last_tick >= 1.0:
                    last_tick = time.time()
                    yield (render_status(snap), *[gr.update()] * (n_out - 1))
                if not ctrl.running:
                    snap = ctrl.snapshot()
                    if snap["kind"] == kind and snap["version"] != last:
                        yield from full(snap)
                    return
                time.sleep(0.5)

        def on_run(*vals):
            if ctrl.running or speed.running:
                yield (warn("A job or speed check is already running — stop it first."), gr.update(),
                       *[gr.update()] * 7)
                return
            try:
                cfgs = _configs(*vals)
            except Exception as e:  # noqa: BLE001
                yield (warn(e), gr.update(), *[gr.update()] * 7)
                return
            ctrl.start([(c.name, c, "queued") for c in cfgs], "experiment")
            first = True
            for out in stream(exp_view, "experiment"):
                # hide an old notice once; later updates leave it alone so a warning from a
                # second click (e.g. "already running") stays visible
                yield (gr.update(visible=False) if first else gr.update(),
                       gr.Tabs(selected="charts") if first else gr.update(), *out)
                first = False

        run_btn.click(on_run, cfg_inputs, [notice, dash_tabs] + exp_outputs, show_progress="hidden")

        def on_stop():
            if speed.stop():
                return gr.update(value="⏹ Stopping the speed check after the current setting …", visible=True)
            if ctrl.stop():
                return gr.update(value="⏹ Stopping after the current round …", visible=True)
            return gr.update(value="Nothing is running.", visible=True)

        stop_btn.click(on_stop, None, notice)

        def on_speed(*vals):
            if ctrl.running or speed.running:
                yield warn("A job or speed check is already running — stop it first."), gr.update(), \
                    gr.update(), gr.update()
                return
            try:
                cfgs = _configs(*vals)
            except Exception as e:  # noqa: BLE001
                yield warn(e), gr.update(), gr.update(), gr.update()
                return
            cfg = next((c for c in cfgs if c.strategy.name == "fedlabel"), cfgs[0])
            speed.start(cfg)
            last = -1
            yield gr.update(visible=False), gr.Tabs(selected="speed"), speed.summary(), speed.table()
            while True:
                running = speed.running
                if speed.version != last:
                    last = speed.version
                    yield gr.update(), gr.update(), speed.summary(), speed.table()
                if not running:
                    yield gr.update(), gr.update(), speed.summary(), speed.table()
                    return
                time.sleep(0.5)

        speed_btn.click(on_speed, cfg_inputs, [notice, dash_tabs, speed_info, speed_table],
                        show_progress="hidden")

        def reattach_experiment():
            snap = ctrl.snapshot()
            if snap["kind"] != "experiment" or not snap["plan"]:
                yield tuple(gr.update() for _ in exp_outputs)
                return
            yield from stream(exp_view, "experiment")     # one update if finished, live if running

        # --------------------------------------------------------------- grid
        def grid_plan(path, only, extra, skip):
            from .grid import _finished, expand_grid
            if not path:
                raise ValueError("No grid file selected.")
            runs = expand_grid(PROJECT_ROOT / path if not Path(path).exists() else path,
                               (extra or "").split())
            words = (only or "").split()
            runs = [(n, c) for n, c in runs if all(w in n for w in words)]
            return [(n, c, "finished" if skip and _finished(c) else "queued") for n, c in runs]

        def grid_frame(plan_rows) -> pd.DataFrame:
            return pd.DataFrame([{
                "status": f"{STATUS_ICON.get(r['status'], '')} {r['status']}",
                "run": r["name"], "dataset": r["cfg"].data.dataset,
                "labels": f"{100 * r['cfg'].data.labeled_ratio:g}%",
                "algorithm": algo_label(r["cfg"].strategy.name),
                "progress": f"{len(r['history'])}/{r['cfg'].server.rounds}" if r["status"] != "finished" else "✓",
                "best acc": f"{100 * r['best']:.2f}" if r.get("best") is not None else "",
            } for r in plan_rows])

        def grid_view(snap):
            return (render_status(snap), grid_frame(snap["plan"]),
                    plot_update(_long_frame(snap, "test_acc", label_by_algo=False)),
                    "\n".join(snap["logs"][-300:]))

        grid_outputs = [grid_status, grid_table, grid_plot, grid_log]

        def on_grid_preview(path, only, extra, skip):
            try:
                plan = grid_plan(path, only, extra, skip)
            except Exception as e:  # noqa: BLE001
                return (f'<div class="status status-error"><div class="status-head"><span class="dot"></span>'
                        f'<b>{html.escape(str(e))}</b></div></div>', gr.update())
            rows = [{"name": n, "cfg": c, "status": s, "history": [], "best": None} for n, c, s in plan]
            todo = sum(s == "queued" for *_, s in plan)
            rounds = sum(c.server.rounds for _, c, s in plan if s == "queued")
            msg = (f'<div class="status status-idle"><div class="status-head"><span class="dot"></span>'
                   f'<b>{len(plan)} runs</b><span class="muted">{todo} to run · {len(plan) - todo} finished · '
                   f'{rounds:,} rounds in total</span></div></div>')
            return msg, grid_frame(rows)

        grid_preview_btn.click(on_grid_preview, [grid_file, grid_only, grid_extra, grid_skip],
                               [grid_status, grid_table])

        def on_grid_run(path, only, extra, skip):
            if ctrl.running or speed.running:
                yield ('<div class="status status-error"><div class="status-head"><span class="dot"></span>'
                       '<b>A job is already running — stop it first.</b></div></div>', *[gr.update()] * 3)
                return
            try:
                plan = grid_plan(path, only, extra, skip)
            except Exception as e:  # noqa: BLE001
                yield (f'<div class="status status-error"><div class="status-head"><span class="dot"></span>'
                       f'<b>{html.escape(str(e))}</b></div></div>', *[gr.update()] * 3)
                return
            ctrl.start(plan, "grid")
            yield from stream(grid_view, "grid")

        grid_run_btn.click(on_grid_run, [grid_file, grid_only, grid_extra, grid_skip], grid_outputs,
                           show_progress="hidden")
        grid_stop_btn.click(lambda: ctrl.stop() and None, None, None)

        def reattach_grid():
            snap = ctrl.snapshot()
            if snap["kind"] != "grid" or not snap["plan"]:
                yield tuple(gr.update() for _ in grid_outputs)
                return
            yield from stream(grid_view, "grid")

        # ------------------------------------------------------------ results
        def on_results(d, ds_filter, algo_filter, metric_key, value_key="acc", keep_selection=None):
            runs = scan_runs(d)
            datasets = sorted({r["dataset"] for r in runs})
            algos_found = sorted({r["variant"] for r in runs})
            sel = [r for r in runs if (not ds_filter or r["dataset"] in ds_filter)
                   and (not algo_filter or r["variant"] in algo_filter)]
            board_df = leaderboard(sel)
            thesis_df = thesis_table(sel, value_key)
            tmp = Path(tempfile.gettempdir()) / "ssfl_studio"
            tmp.mkdir(exist_ok=True)
            csv_path, tex_path = tmp / "leaderboard.csv", tmp / "thesis_table.tex"
            board_df.to_csv(csv_path, index=False)
            latex = to_latex(thesis_df, THESIS_VALUES[value_key], digits=3 if value_key == "ratio" else 2)
            tex_path.write_text(latex, encoding="utf-8")
            names = [r["run"] for r in sel]
            ranked = [r["run"] for r in sorted(sel, key=lambda r: -(r["best_acc"] or 0))]
            chosen = [n for n in (keep_selection or []) if n in names] or ranked[:5]
            best = max((r["best_acc"] or 0 for r in sel), default=0)
            n_fin = sum(r["finished"] for r in sel)
            kp = ('<div class="kpi-grid four">'
                  + _kpi("Runs", str(len(sel)), f"{n_fin} finished · {len(sel) - n_fin} partial", "a1")
                  + _kpi("Best accuracy", f"{100 * best:.1f}<span>%</span>" if sel else "—",
                         next((r["base"] for r in sel if r["best_acc"] == best), ""), "a2")
                  + _kpi("Datasets", str(len({r['dataset'] for r in sel})),
                         ", ".join(sorted({r['dataset'] for r in sel}))[:60], "a3")
                  + _kpi("Algorithms", str(len({r['variant'] for r in sel})), "", "a4") + "</div>")
            return (gr.update(choices=datasets), gr.update(choices=algos_found),
                    gr.update(choices=names, value=chosen), board_df, thesis_display(thesis_df),
                    str(csv_path), str(tex_path), latex, kp, plot_update(run_curves(d, chosen, metric_key),
                                                                 y_title=METRIC_LABELS.get(metric_key, metric_key)),
                    board_order(sel), gr.update(choices=run_choices(sel)),
                    gr.update(choices=clear_choices(d)))

        res_inputs = [runs_dir, f_dataset, f_algo, metric, thesis_value]
        res_outputs = [f_dataset, f_algo, run_select, board, thesis, dl_csv, dl_tex, latex_view, res_kpis,
                       curve_plot, board_runs, detail_pick, clear_pick]

        # -------------------------------------------------------- clear runs
        def on_clear_select(kind, d, shown):
            from .cleanup import list_run_dirs, select_runs
            if kind == "shown":
                names = list(shown or [])
            elif kind == "unfinished":
                names = [p.name for p in select_runs(d, statuses=["unfinished"])]
            elif kind == "all":
                names = [p.name for p in list_run_dirs(d)]
            else:
                names = []
            return gr.update(choices=clear_choices(d), value=names), "", gr.update(visible=False)

        for btn, kind in ((clear_shown_btn, "shown"), (clear_unfin_btn, "unfinished"),
                          (clear_all_btn, "all"), (clear_none_btn, "none")):
            btn.click(lambda d, shown, k=kind: on_clear_select(k, d, shown), [runs_dir, board_runs],
                      [clear_pick, clear_msg, clear_confirm_row])

        def on_clear_ask(picked, d):
            if not picked:
                return '<div class="algo-info empty">Select one or more runs first.</div>', gr.update(visible=False)
            listing = "".join(f"<li><code>{html.escape(n)}</code></li>" for n in picked[:12])
            more = f"<li>… and {len(picked) - 12} more</li>" if len(picked) > 12 else ""
            return (f'<div class="clear-note">⚠️ Permanently delete <b>{len(picked)}</b> run folder(s) from '
                    f'<code>{html.escape(str(Path(d).resolve()))}</code>? This cannot be undone.'
                    f'<ul>{listing}{more}</ul></div>', gr.update(visible=True))

        def on_clear_confirm(picked, d):
            from .cleanup import delete_runs
            deleted, skipped = delete_runs(d, picked or [])
            kept = "".join(f"<li><code>{html.escape(n)}</code>: {html.escape(why)}</li>" for n, why in skipped.items())
            msg = (f'<div class="clear-note">Deleted <b>{len(deleted)}</b> run folder(s).'
                   + (f" Kept {len(skipped)}:<ul>{kept}</ul>" if skipped else "") + "</div>")
            return gr.update(choices=clear_choices(d), value=[]), msg, gr.update(visible=False)

        clear_open_btn.click(lambda d: (gr.Tabs(selected="clear"), gr.update(choices=clear_choices(d))),
                             runs_dir, [res_tabs, clear_pick])
        clear_btn.click(on_clear_ask, [clear_pick, runs_dir], [clear_msg, clear_confirm_row])
        clear_no.click(lambda: ("", gr.update(visible=False)), None, [clear_msg, clear_confirm_row])
        clear_yes.click(on_clear_confirm, [clear_pick, runs_dir], [clear_pick, clear_msg, clear_confirm_row]).then(
            on_results, res_inputs, res_outputs)

        # ------------------------------------------------------- run details
        detail_outputs = [detail_html, detail_acc, detail_loss, dl_run_cfg, dl_run_metrics, dl_run_summary]

        def on_detail(run, d):
            if not run:
                yield tuple(gr.update() for _ in detail_outputs)
                return
            run_dir = Path(d) / run
            meta = load_run(run_dir)
            if meta is None:
                yield (_error_html(f"Cannot read {run_dir}"), *[gr.update()] * 5)
                return
            try:
                acc, loss = detail_frames(run_dir)
                files = downloadable(run_dir, ("config.yaml", "metrics.csv", "summary.json"))
                page = render_run_details(meta, run_dir)
            except Exception as e:  # noqa: BLE001 - show it in the panel, not as a failed update
                yield (_error_html(f"Cannot show {run}: {e}", traceback.format_exc()), *[gr.update()] * 5)
                return
            yield (page, plot_update(acc, "series"), plot_update(loss, "series"), *files)
            # colour scales are fixed when a chart is first drawn: re-send them now the data is in
            time.sleep(0.4)
            yield (gr.update(), *[gr.update(color_map={**color_map(f, "series"), "​": "#000000"})
                                  for f in (acc, loss)], gr.update(), gr.update(), gr.update())

        detail_pick.input(on_detail, [detail_pick, runs_dir], detail_outputs)

        def on_board_select(runs, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            run = runs[row] if runs and 0 <= row < len(runs) else None
            return gr.update(value=run), gr.Tabs(selected="details")

        # string annotations (from __future__) cannot see the local `gr`: give Gradio the class
        on_board_select.__annotations__["evt"] = gr.SelectData
        board.select(on_board_select, board_runs, [detail_pick, res_tabs]).then(
            on_detail, [detail_pick, runs_dir], detail_outputs)

        def on_load_run(run, d):
            if not run or not (Path(d) / run / "config.yaml").exists():
                return (*[gr.update()] * (len(components) + 4 + len(param_comps)), gr.update())
            cfg = ExperimentConfig.from_yaml(Path(d) / run / "config.yaml")
            return (*form_from_cfg(cfg), gr.Tabs(selected="experiment"))

        load_btn.click(on_load_run, [detail_pick, runs_dir],
                       components + [algos, params_state, algo_info, fast] + param_comps + [main_tabs]).then(
            show_yaml, cfg_inputs, yaml_view)
        refresh_btn.click(on_results, res_inputs, res_outputs)
        f_dataset.input(on_results, res_inputs, res_outputs)
        f_algo.input(on_results, res_inputs, res_outputs)
        thesis_value.input(on_results, res_inputs, res_outputs)
        def on_curves(d, names, m):
            return plot_update(run_curves(d, names, m), y_title=METRIC_LABELS.get(m, m))

        run_select.input(on_curves, [runs_dir, run_select, metric], curve_plot)
        metric.input(on_curves, [runs_dir, run_select, metric], curve_plot)

        # --------------------------------------------------------------- load
        app.load(show_yaml, cfg_inputs, yaml_view)
        app.load(on_results, res_inputs, res_outputs)
        app.load(reattach_experiment, None, exp_outputs, show_progress="hidden")
        app.load(reattach_grid, None, grid_outputs, show_progress="hidden")

    return app


#: Optional integer parameters (default None): shown as number fields, empty = default.
INT_PARAMS = {"sup_epochs", "unsup_epochs", "sup_steps", "unsup_steps"}


def _display_value(kind: type, value: Any) -> Any:
    if kind in (bool, int, float, str):
        return value
    return "" if value is None else json.dumps(value)          # JSON text field


def _param_field(gr, key: str, value: Any, default: Any):
    """One form field for an algorithm parameter. Returns ``(component, kind)``."""
    help_text, choices = PARAM_HELP.get(key, ("", None))
    info = help_text + (f" (default {default})" if default is not None and not isinstance(default, bool) else "")
    kind = int if key in INT_PARAMS else type(default if default is not None else value)
    if choices:
        comp = gr.Dropdown(choices, value=value, label=key, info=info, min_width=200)
    elif kind is bool:
        comp = gr.Checkbox(bool(value), label=key, info=help_text, min_width=200)
    elif kind is int:
        comp = gr.Number(value, label=key, info=info, precision=0, min_width=160)
    elif kind is float:
        # arrows / mouse wheel move by a tenth of the default's order of magnitude
        # (tau 0.95 -> 0.01, lambda 10 -> 1, lambda_l1 1e-4 -> 1e-5) instead of the default 1
        ref = abs(default if isinstance(default, (int, float)) and default else 0.1)
        step = 10 ** (math.floor(math.log10(ref)) - 1)
        comp = gr.Number(value, label=key, info=info, min_width=160, step=float(f"{step:.0e}"))
    elif kind is str:
        comp = gr.Textbox(value, label=key, info=info, min_width=200)
    else:                                      # None / list / dict -> JSON
        comp = gr.Textbox(_display_value(kind, value), label=key, info=info + " — JSON", min_width=240)
    return comp, kind


def _param_updater(algo: str, key: str, kind: type):
    """Event handler that writes one edited parameter into the parameter state."""

    def update(new, state, algo=algo, key=key, kind=kind):
        state = copy.deepcopy(state or {})
        p = state.setdefault(algo, {})
        if kind not in (bool, int, float, str) and not isinstance(new, (bool, int, float)):
            text = (new or "").strip()
            if not text:
                p.pop(key, None)
                return state
            try:
                new = json.loads(text)
            except json.JSONDecodeError:
                return state                    # keep the last valid value while typing
        if new is None or new == "":
            p.pop(key, None)
        else:
            p[key] = int(new) if kind is int else new
        return state

    return update


def launch(data_dir: Optional[str] = None, runs_dir: Optional[str] = None,
           num_workers: Optional[int] = None, **kwargs):
    """Build and launch the GUI.

    ``data_dir`` / ``runs_dir`` / ``num_workers`` become the defaults of the download
    folder, output/runs folder and DataLoader workers, also after loading a preset
    (on Colab: the Google Drive folders and 2 workers).
    """
    import gradio as gr

    ENV_DEFAULTS.update(data_dir=data_dir, runs_dir=runs_dir, num_workers=num_workers)
    print(f"[ssfl] SSFL Studio, build {code_version()}", flush=True)
    app = build_app()
    app.queue(default_concurrency_limit=8)
    kwargs.setdefault("theme", gr.themes.Soft(**THEME_ARGS, font=[gr.themes.GoogleFont("Inter"),
                                                                  "ui-sans-serif", "system-ui", "sans-serif"]))
    kwargs.setdefault("css", CSS)
    return app.launch(**kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ssfl.gui", description="SSFL Studio - graphical interface")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--data-dir", help="default download folder")
    p.add_argument("--runs-dir", help="default output / runs folder")
    p.add_argument("--workers", type=int, help="default DataLoader workers")
    args = p.parse_args(argv)
    launch(data_dir=args.data_dir, runs_dir=args.runs_dir, num_workers=args.workers,
           server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
