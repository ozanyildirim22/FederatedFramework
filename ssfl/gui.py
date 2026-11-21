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
import inspect
import json
import re
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .registry import AGGREGATORS, DATASETS, MODELS, PARTITIONERS, STRATEGIES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
GRID_DIR = CONFIG_DIR / "grids"
SKIP_ARGS = {"self", "cfg", "num_classes"}

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
                 "Per sample, the more confident of the local and global model gives the pseudo-label; "
                 "plus global–local KL consistency.", "featured"),
    "fixmatch": ("FixMatch", "SSL baseline · NeurIPS 2020",
                 "Weak-view pseudo-label above τ is enforced on the strong view.", "ssl"),
    "pseudolabel": ("Pseudo-Label", "SSL baseline · 2013",
                    "Hard pseudo-labels above a confidence threshold τ.", "ssl"),
    "supervised": ("Supervised", "Baseline",
                   "Same as FedAvg on labeled data; with 100% labels it is the upper bound.", "baseline"),
}
ALGO_ORDER = ["fedavg", "fedprox", "fedmatch", "fedtrinet", "fedlabel", "fixmatch", "pseudolabel", "supervised"]

#: preset file -> (icon, title, subtitle)
PRESET_INFO = {
    "quick_test.yaml": ("⚡", "Quick demo", "Synthetic data · runs in seconds"),
    "organamnist_fedlabel.yaml": ("🫁", "OrganAMNIST", "Cross-silo · 10 hospitals · 100 rounds"),
    "pathmnist_fedlabel.yaml": ("🔬", "PathMNIST", "Cross-silo · 10 hospitals · 100 rounds"),
    "cifar10_fedlabel.yaml": ("🖼️", "CIFAR-10", "Cross-device · 100 clients · 200 rounds"),
}

#: parameter -> (help text, choices or None)
PARAM_HELP: Dict[str, Tuple[str, Optional[List[str]]]] = {
    "tau": ("Confidence threshold to keep a pseudo-label", None),
    "lambda_u": ("Weight of the unlabeled (pseudo-label) loss", None),
    "lambda_kl": ("Weight of the global–local KL consistency", None),
    "temperature": ("Softmax temperature of the KL term", None),
    "confidence": ("Confidence score that decides local vs global", ["variance", "max_prob", "entropy"]),
    "selection": ("Pseudo-label source: adaptive = FedLabel, local/global = ablations",
                  ["adaptive", "local", "global"]),
    "local_warmup_steps": ("Supervised-only steps before pseudo-labeling starts", None),
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
    "test_acc": "Test accuracy", "test_loss": "Test loss", "train_loss": "Train loss",
    "train_mask_rate": "Pseudo-label mask rate", "train_pl_acc": "Pseudo-label accuracy",
    "train_loss_sup": "Supervised loss", "train_loss_unsup": "Unlabeled loss",
    "train_local_selected": "Local model chosen (FedLabel)", "train_threshold": "Threshold (FedTriNet)",
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
def default_params(cls) -> Dict[str, Any]:
    """Collect keyword defaults from a class' ``__init__`` along its MRO."""
    out: Dict[str, Any] = {}
    for klass in getattr(cls, "__mro__", [cls]):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, p in inspect.signature(init).parameters.items():
            if name in SKIP_ARGS or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.default is not inspect.Parameter.empty and name not in out:
                out[name] = p.default
    return out


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
    "data.strong_aug",
    "model.name", "model.norm", "model.params",
    "strategy.local_epochs", "strategy.local_steps", "strategy.optimizer",
    "strategy.lr", "strategy.momentum", "strategy.weight_decay", "strategy.prox_mu",
    "strategy.grad_clip", "strategy.amp",
    "server.aggregator", "server.rounds", "server.num_clients", "server.client_fraction",
    "server.eval_every", "server.params",
]
JSON_FIELDS = {"model.params", "server.params"}
INT_FIELDS = {"seed", "data.image_size", "data.batch_size", "data.unlabeled_batch_ratio",
              "data.num_workers", "strategy.local_epochs", "strategy.local_steps",
              "server.rounds", "server.num_clients", "server.eval_every"}


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
            cfg.name = f"{cfg.data.dataset}_{algo}_l{_ratio_tag(cfg.data.labeled_ratio)}"
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
            with self.lock:
                self.current = i
                run["status"], run["started"] = "running", time.time()
                self.message = f"Loading data and building clients for {run['name']} …"
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
                self._log(exp.data.summary())
                self._log(f"strategy={run['cfg'].strategy.name} aggregator={run['cfg'].server.aggregator} "
                          f"model={run['cfg'].model.name} device={exp.device} -> {exp.out_dir}")
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
    if 0 <= cur < len(plan) and plan[cur]["history"]:
        foot.append(f"round {len(plan[cur]['history'])}/{plan[cur]['cfg'].server.rounds}")
    if snap["started"]:
        foot.append(f"elapsed {_fmt_secs(time.time() - snap['started'])}" if state in ("running", "stopping")
                    else "")
    if eta is not None:
        foot.append(f"ETA {_fmt_secs(eta)}")
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
    return (f'<div class="kpi {accent}"><div class="kpi-label">{label}</div>'
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
               "FedLabel": "#8b5cf6", "FixMatch": "#ec4899", "Pseudo-Label": "#14b8a6", "Supervised": "#78716c"}


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
             f'<span class="chip">{len(DATASETS.names())} datasets</span>']
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
def _run_meta(run_dir: Path) -> Dict[str, Any]:
    cfg = {}
    if (run_dir / "config.yaml").exists():
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    base = re.sub(r"_\d{8}-\d{6}$", "", run_dir.name)
    m = re.search(r"_l[0-9p]+_(.+)$", base)
    algo = cfg.get("strategy", {}).get("name", "?")
    return {"run": run_dir.name, "base": base, "dataset": cfg.get("data", {}).get("dataset", "?"),
            "labels": cfg.get("data", {}).get("labeled_ratio"), "alpha": cfg.get("data", {}).get("alpha"),
            "algorithm": algo, "variant": algo_label(algo) + (f" ({m.group(1)})" if m else ""),
            "aggregator": cfg.get("server", {}).get("aggregator"),
            "planned_rounds": cfg.get("server", {}).get("rounds")}


def scan_runs(runs_dir: str) -> List[Dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    out = []
    for csv_path in root.glob("*/metrics.csv"):
        meta = _run_meta(csv_path.parent)
        try:
            df = pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001 - half-written file
            continue
        meta["rounds"] = int(df["round"].max()) if len(df) else 0
        acc = df["test_acc"].dropna() if "test_acc" in df else pd.Series(dtype=float)
        meta["best_acc"] = float(acc.max()) if len(acc) else None
        meta["final_acc"] = float(acc.iloc[-1]) if len(acc) else None
        meta["finished"] = (csv_path.parent / "summary.json").exists()
        meta["mtime"] = csv_path.stat().st_mtime
        out.append(meta)
    return sorted(out, key=lambda m: m["mtime"], reverse=True)


def leaderboard(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame(columns=["#", "run", "dataset", "labels", "algorithm", "best acc", "final acc",
                                     "rounds", "status"])
    df = pd.DataFrame(runs).sort_values("best_acc", ascending=False, na_position="last")
    out = pd.DataFrame({
        "run": df["run"], "dataset": df["dataset"],
        "labels": df["labels"].map(lambda v: f"{100 * v:g}%" if v is not None and v == v else ""),
        "algorithm": df["variant"],
        "best acc": df["best_acc"].map(lambda v: f"{100 * v:.2f}" if v is not None and v == v else "—"),
        "final acc": df["final_acc"].map(lambda v: f"{100 * v:.2f}" if v is not None and v == v else "—"),
        "rounds": [f"{r}/{p}" if p else str(r) for r, p in zip(df["rounds"], df["planned_rounds"])],
        "status": df["finished"].map(lambda f: "✓ finished" if f else "… partial"),
    })
    out.insert(0, "#", range(1, len(out) + 1))
    return out


def thesis_table(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Rows = algorithm (variant), columns = dataset @ label ratio, cells = best test accuracy (%).
    For repeated runs of the same setting, the most recent one is used."""
    rows = [r for r in runs if r["best_acc"] is not None]
    if not rows:
        return pd.DataFrame()
    latest: Dict[str, Dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: r["mtime"]):
        latest[r["base"]] = r
    df = pd.DataFrame(latest.values())
    df["setting"] = df.apply(lambda r: f"{r['dataset']} {100 * r['labels']:g}%" if r["labels"] is not None
                             else r["dataset"], axis=1)
    piv = df.pivot_table(index="variant", columns="setting", values="best_acc", aggfunc="max") * 100
    order = [algo_label(a) for a in ALGO_ORDER]
    piv = piv.reindex(sorted(piv.index, key=lambda v: (order.index(v.split(" (")[0])
                                                       if v.split(" (")[0] in order else 99, v)))
    piv = piv.round(2).reset_index().rename(columns={"variant": "algorithm"})
    piv.columns.name = None
    return piv


def to_latex(table: pd.DataFrame, caption: str = "Best test accuracy (%)") -> str:
    if table is None or not len(table):
        return "% no results yet\n"
    cols = list(table.columns)
    num = table[cols[1:]].apply(pd.to_numeric, errors="coerce")
    best = num.max(axis=0)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(c.replace("%", r"\%").replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for i, row in table.iterrows():
        cells = [str(row[cols[0]]).replace("_", r"\_")]
        for c in cols[1:]:
            v = num.loc[i, c]
            if v != v:
                cells.append("--")
            else:
                s = f"{v:.2f}"
                cells.append(f"\\textbf{{{s}}}" if v == best[c] else s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def run_curves(runs_dir: str, names: Sequence[str], metric: str) -> pd.DataFrame:
    rows = []
    root = Path(runs_dir)
    bases = [re.sub(r"_\d{8}-\d{6}$", "", n) for n in names or []]
    for name in names or []:
        p = root / name / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if metric not in df:
            continue
        d = df.dropna(subset=[metric])
        label = re.sub(r"_\d{8}-\d{6}$", "", name)
        if bases.count(label) > 1:          # repeated setting: keep the timestamp to tell them apart
            label = name
        rows += [{"round": int(r), "value": float(v), "run": label} for r, v in zip(d["round"], d[metric])]
    return _typed(rows)


# ------------------------------------------------------------- partition
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
.kpi-label { font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; opacity:.6; }
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
"""


def _section(n: str, title: str, hint: str = "") -> str:
    return (f'<div class="sec-title"><div class="n">{n}</div><b>{title}</b>'
            + (f'<span class="muted">{hint}</span>' if hint else "") + "</div>")


def build_app():
    import gradio as gr

    ctrl = RunController()
    presets = list_presets()
    start_preset = "organamnist_fedlabel.yaml" if "organamnist_fedlabel.yaml" in presets else (
        presets[0] if presets else None)
    start_cfg = load_preset(start_preset)
    v = dict(zip(FIELDS, cfg_to_values(start_cfg)))
    v["name"] = ""
    start_algos = [start_cfg.strategy.name]
    start_params = {start_cfg.strategy.name: dict(start_cfg.strategy.params)}

    with gr.Blocks(title="SSFL Studio") as app:
        gr.HTML(render_hero())
        params_state = gr.State(start_params)
        render_tick = gr.Number(0, visible=False)

        with gr.Tabs():
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
                            rounds = gr.Slider(1, 500, value=v["server.rounds"], step=1,
                                               label="Communication rounds")
                            alpha = gr.Slider(0.01, 10, value=v["data.alpha"], step=0.01,
                                              label="Non-IID level — Dirichlet α",
                                              info="Smaller α = more skewed class mix per client")
                            with gr.Row():
                                num_clients = gr.Number(v["server.num_clients"], label="Clients N", precision=0,
                                                        minimum=1)
                                client_fraction = gr.Slider(0.01, 1.0, value=v["server.client_fraction"],
                                                            step=0.01, label="Clients per round C")
                        with gr.Accordion("🎛️  Algorithm parameters", open=False):
                            gr.Markdown("Defaults come from the papers. Only the values you change are "
                                        "written to the config.", elem_classes="muted")

                            @gr.render(inputs=[algos, params_state], triggers=[algos.change, render_tick.change,
                                                                                app.load])
                            def _param_forms(selected, state):
                                if not selected:
                                    gr.Markdown("*No algorithm selected.*")
                                for algo in selected or []:
                                    defaults = default_params(STRATEGIES.get(algo))
                                    current = {**defaults, **(state or {}).get(algo, {})}
                                    keys = [k for k in defaults] + [k for k in current
                                                                    if k not in defaults and k != "loss_weights"]
                                    with gr.Group(elem_classes="param-group"):
                                        gr.HTML(f'<div class="sec-title"><b>{html.escape(algo_label(algo))}'
                                                f'</b><span class="muted">{len(keys)} parameters</span></div>')
                                        if not keys:
                                            gr.Markdown("*No parameters.*")
                                        with gr.Row(equal_height=True):
                                            for key in keys:
                                                _param_field(gr, algo, key, current.get(key), defaults.get(key),
                                                             params_state)

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
                                                        label="DataLoader workers", info="2 on Colab / Linux")
                            with gr.Row():
                                strong_aug = gr.Dropdown(["randaugment", "none"], value=v["data.strong_aug"],
                                                         label="Strong augmentation")
                                root = gr.Textbox(v["data.root"], label="Download folder")
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
                                amp = gr.Checkbox(v["strategy.amp"], label="Mixed precision (AMP)")
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
                            with gr.Tab("🧩  Data split", id="partition"):
                                part_stats = gr.HTML('<div class="algo-info empty">Press <b>Preview split</b> '
                                                     'to see how the data is divided between clients.</div>')
                                part_plot = gr.BarPlot(pd.DataFrame({"client": pd.Series(dtype="str"),
                                                                     "samples": pd.Series(dtype="int64"),
                                                                     "class": pd.Series(dtype="str")}),
                                                       x="client", y="samples", color="class",
                                                       title="Samples per client, coloured by class",
                                                       height=360, sort="x")
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
                    grid_extra = gr.Textbox("", label="Extra overrides", scale=3,
                                            placeholder="e.g. data.num_workers=2 output_dir=/content/drive/MyDrive/ssfl/runs")
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
                with gr.Row():
                    f_dataset = gr.Dropdown([], value=[], multiselect=True, label="Dataset filter", scale=2)
                    f_algo = gr.Dropdown([], value=[], multiselect=True, label="Algorithm filter", scale=2)
                    metric = gr.Dropdown([(lab, k) for k, lab in METRIC_LABELS.items()], value="test_acc",
                                         label="Metric to plot", scale=2)
                res_kpis = gr.HTML("")
                with gr.Tabs():
                    with gr.Tab("🥇  Leaderboard"):
                        board = gr.Dataframe(leaderboard([]), wrap=True, max_height=420)
                    with gr.Tab("📈  Curves"):
                        run_select = gr.Dropdown([], value=[], multiselect=True,
                                                 label="Runs to plot (top 5 are pre-selected)")
                        curve_plot = gr.LinePlot(_typed([]), x="round",
                                                 y="value", color="run", height=380)
                    with gr.Tab("🎓  Thesis table"):
                        gr.Markdown("Best test accuracy (%) per algorithm and setting; latest run of each "
                                    "setting. Bold in LaTeX = best in column.")
                        thesis = gr.Dataframe(pd.DataFrame(), wrap=True)
                        with gr.Row():
                            dl_csv = gr.DownloadButton("⬇  Leaderboard CSV", value=None)
                            dl_tex = gr.DownloadButton("⬇  Thesis table LaTeX", value=None)
                        latex_view = gr.Code("", language="latex", label="LaTeX", lines=12)

            # ======================================================== GUIDE
            with gr.Tab("📘 Guide", id="guide"):
                gr.HTML(render_guide())

        # ============================================================ wiring
        components = [name, seed, device, output_dir,
                      dataset, root, image_size, partitioner, alpha, labeled_ratio, batch_size,
                      unl_ratio, num_workers, strong_aug,
                      model_name, norm, model_params,
                      local_epochs, local_steps, optimizer, lr, momentum, weight_decay,
                      prox_mu, grad_clip, amp,
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

        def on_preset(p, tick):
            cfg = load_preset(p)
            vals = cfg_to_values(cfg)
            vals[0] = ""                                  # auto run name
            return (*vals, [cfg.strategy.name], {cfg.strategy.name: dict(cfg.strategy.params)},
                    (tick or 0) + 1, render_algo_cards([cfg.strategy.name]))

        # `.input` fires only on user edits, so loading a preset keeps its tuned params.
        preset.input(on_preset, [preset, render_tick],
                     components + [algos, params_state, render_tick, algo_info]).then(
            show_yaml, cfg_inputs, yaml_view)
        algos.change(render_algo_cards, algos, algo_info, trigger_mode="always_last",
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

        def on_preview(*vals):
            from .data import DataManager
            try:
                cfg = _configs(*vals)[0]
                dm = DataManager(cfg.data, cfg.server.num_clients, seed=cfg.seed)
                return (plot_update(partition_frame(dm), "class"), partition_stats(dm), gr.update(visible=False),
                        gr.Tabs(selected="partition"))
            except Exception as e:  # noqa: BLE001
                return gr.update(), gr.update(), warn(e), gr.update()

        preview_btn.click(on_preview, cfg_inputs, [part_plot, part_stats, notice, dash_tabs])

        # ---------------------------------------------------- live dashboard
        def exp_view(snap):
            return (render_status(snap), render_kpis(snap), plot_update(_long_frame(snap, "test_acc")),
                    plot_update(_long_frame(snap, "train_loss")), plot_update(_pl_frame(snap), "series"),
                    "\n".join(snap["logs"][-400:]), _metrics_table(snap))

        exp_outputs = [status, kpis, acc_plot, loss_plot, pl_plot, log, metrics]

        def stream(view, kind):
            last = -1
            while True:
                snap = ctrl.snapshot()
                if snap["kind"] == kind and snap["version"] != last:
                    last = snap["version"]
                    yield view(snap)
                if not ctrl.running:
                    snap = ctrl.snapshot()
                    if snap["kind"] == kind and snap["version"] != last:
                        yield view(snap)
                    return
                time.sleep(0.5)

        def on_run(*vals):
            if ctrl.running:
                yield (warn("A job is already running — stop it first."), gr.update(), *[gr.update()] * 7)
                return
            try:
                cfgs = _configs(*vals)
            except Exception as e:  # noqa: BLE001
                yield (warn(e), gr.update(), *[gr.update()] * 7)
                return
            ctrl.start([(c.name, c, "queued") for c in cfgs], "experiment")
            first = True
            for out in stream(exp_view, "experiment"):
                yield (gr.update(visible=False), gr.Tabs(selected="charts") if first else gr.update(), *out)
                first = False

        run_btn.click(on_run, cfg_inputs, [notice, dash_tabs] + exp_outputs, show_progress="hidden")

        def on_stop():
            if ctrl.stop():
                return gr.update(value="⏹ Stopping after the current round …", visible=True)
            return gr.update(value="Nothing is running.", visible=True)

        stop_btn.click(on_stop, None, notice)

        def reattach_experiment():
            snap = ctrl.snapshot()
            if snap["kind"] != "experiment" or not snap["plan"]:
                yield tuple(gr.update() for _ in exp_outputs)
                return
            yield exp_view(snap)
            if ctrl.running:
                yield from stream(exp_view, "experiment")

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
            if ctrl.running:
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
            yield grid_view(snap)
            if ctrl.running:
                yield from stream(grid_view, "grid")

        # ------------------------------------------------------------ results
        def on_results(d, ds_filter, algo_filter, metric_key, keep_selection=None):
            runs = scan_runs(d)
            datasets = sorted({r["dataset"] for r in runs})
            algos_found = sorted({r["variant"] for r in runs})
            sel = [r for r in runs if (not ds_filter or r["dataset"] in ds_filter)
                   and (not algo_filter or r["variant"] in algo_filter)]
            board_df = leaderboard(sel)
            thesis_df = thesis_table(sel)
            tmp = Path(tempfile.gettempdir()) / "ssfl_studio"
            tmp.mkdir(exist_ok=True)
            csv_path, tex_path = tmp / "leaderboard.csv", tmp / "thesis_table.tex"
            board_df.to_csv(csv_path, index=False)
            latex = to_latex(thesis_df)
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
                  + _kpi("Datasets", str(len({r['dataset'] for r in sel})), ", ".join(datasets)[:60], "a3")
                  + _kpi("Algorithms", str(len({r['variant'] for r in sel})), "", "a4") + "</div>")
            return (gr.update(choices=datasets), gr.update(choices=algos_found),
                    gr.update(choices=names, value=chosen), board_df, thesis_df,
                    str(csv_path), str(tex_path), latex, kp, plot_update(run_curves(d, chosen, metric_key),
                                                                 y_title=METRIC_LABELS.get(metric_key, metric_key)))

        res_inputs = [runs_dir, f_dataset, f_algo, metric]
        res_outputs = [f_dataset, f_algo, run_select, board, thesis, dl_csv, dl_tex, latex_view, res_kpis,
                       curve_plot]
        refresh_btn.click(on_results, res_inputs, res_outputs)
        f_dataset.input(on_results, res_inputs, res_outputs)
        f_algo.input(on_results, res_inputs, res_outputs)
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


def _param_field(gr, algo: str, key: str, value: Any, default: Any, params_state) -> None:
    """One form field for an algorithm parameter; edits are written into ``params_state``."""
    help_text, choices = PARAM_HELP.get(key, ("", None))
    info = help_text + (f" (default {default})" if default is not None and not isinstance(default, bool) else "")
    kind = type(default if default is not None else value)
    if choices:
        comp = gr.Dropdown(choices, value=value, label=key, info=info, min_width=200)
    elif kind is bool:
        comp = gr.Checkbox(bool(value), label=key, info=help_text, min_width=200)
    elif kind is int:
        comp = gr.Number(value, label=key, info=info, precision=0, min_width=160)
    elif kind is float:
        comp = gr.Number(value, label=key, info=info, min_width=160)
    elif kind is str:
        comp = gr.Textbox(value, label=key, info=info, min_width=200)
    else:                                      # None / list / dict -> JSON
        comp = gr.Textbox("" if value is None else json.dumps(value), label=key, info=info + " — JSON",
                          min_width=240)

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

    comp.input(update, [comp, params_state], params_state)


def launch(**kwargs):
    """Build and launch the GUI. In Colab, Gradio shows it inline in the cell."""
    import gradio as gr

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
    args = p.parse_args(argv)
    launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
