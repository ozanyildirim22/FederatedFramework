"""Graphical user interface (Gradio).

Launch locally:   python -m ssfl.gui            (opens http://127.0.0.1:7860)
Launch in Colab:  from ssfl.gui import launch; launch()

Tabs
  * Experiment  - pick a preset, edit every setting, preview the non-IID
                  partition, run with live progress, stop, save the config.
  * Results     - compare finished runs (curves + summary table).

Every dropdown is filled from the registries, so custom strategies /
aggregators / datasets registered before ``launch()`` appear automatically,
and their constructor arguments are pre-filled as editable JSON.
"""
from __future__ import annotations

import argparse
import inspect
import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .config import ExperimentConfig  # noqa: E402
from .registry import AGGREGATORS, DATASETS, MODELS, PARTITIONERS, STRATEGIES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
SKIP_ARGS = {"self", "cfg", "num_classes"}


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


def load_preset(name: Optional[str]) -> ExperimentConfig:
    if name and (CONFIG_DIR / name).exists():
        return ExperimentConfig.from_yaml(CONFIG_DIR / name)
    return ExperimentConfig()


# Ordered list of (dotted config key, label). Drives both form -> config and config -> form.
FIELDS = [
    "name", "seed", "device", "output_dir",
    "data.dataset", "data.root", "data.image_size", "data.partitioner", "data.alpha",
    "data.labeled_ratio", "data.batch_size", "data.unlabeled_batch_ratio", "data.num_workers",
    "data.strong_aug",
    "model.name", "model.norm", "model.params",
    "strategy.name", "strategy.local_epochs", "strategy.local_steps", "strategy.optimizer",
    "strategy.lr", "strategy.momentum", "strategy.weight_decay", "strategy.prox_mu",
    "strategy.grad_clip", "strategy.amp", "strategy.params",
    "server.aggregator", "server.rounds", "server.num_clients", "server.client_fraction",
    "server.eval_every", "server.params",
]
JSON_FIELDS = {"model.params", "strategy.params", "server.params"}
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


def values_to_cfg(*vals) -> ExperimentConfig:
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
    return ExperimentConfig.from_dict(raw)


# ------------------------------------------------------------- plotting
def plot_history(history: List[Dict[str, float]]):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    df = pd.DataFrame(history)
    panels = [("test_acc", "Test accuracy"), ("train_loss", "Train loss"),
              ("train_mask_rate", "Pseudo-label mask rate / accuracy")]
    for ax, (col, title) in zip(axes, panels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("round")
        ax.grid(alpha=.3)
        if col in df:
            d = df.dropna(subset=[col])
            ax.plot(d["round"], d[col], marker="o", ms=3, label=col.replace("train_", ""))
        if col == "train_mask_rate" and "train_pl_acc" in df:
            d = df.dropna(subset=["train_pl_acc"])
            ax.plot(d["round"], d["train_pl_acc"], marker="o", ms=3, label="pl_acc")
            ax.legend(fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_partition(manager, max_clients: int = 30):
    counts = np.stack([np.bincount(manager.train.labels[p], minlength=manager.num_classes)
                       for p in manager.partitions[:max_clients]])
    lab = np.array([len(s[0]) for s in manager.splits[:max_clients]])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    bottom = np.zeros(len(counts))
    cmap = plt.get_cmap("tab20")
    for c in range(manager.num_classes):
        ax.bar(range(len(counts)), counts[:, c], bottom=bottom, color=cmap(c % 20), label=str(c), width=.85)
        bottom += counts[:, c]
    ax.scatter(range(len(counts)), lab, color="black", marker="_", s=120, label="labeled", zorder=3)
    ax.set_xlabel("client")
    ax.set_ylabel("samples")
    shown = f"first {max_clients} of {len(manager.partitions)}" if len(manager.partitions) > max_clients else "all"
    ax.set_title(f"Class distribution per client ({shown}); black tick = # labeled samples", fontsize=10)
    ax.legend(title="class", fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


# --------------------------------------------------------------- runner
class _Stopped(Exception):
    pass


class RunController:
    """Runs one experiment in a background thread and streams its progress."""

    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.events: "queue.Queue[tuple]" = queue.Queue()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: ExperimentConfig) -> None:
        self.stop_event.clear()
        self.events = queue.Queue()
        self.thread = threading.Thread(target=self._work, args=(cfg,), daemon=True)
        self.thread.start()

    def _work(self, cfg: ExperimentConfig) -> None:
        from .experiment import Experiment

        def on_round(rnd, rec, exp):
            self.events.put(("round", rec))
            if self.stop_event.is_set():
                raise _Stopped()

        try:
            self.events.put(("log", "Loading data and building clients ..."))
            exp = Experiment(cfg, callbacks=[on_round], verbose=False)
            self.events.put(("log", exp.data.summary()))
            self.events.put(("log", f"strategy={cfg.strategy.name}  aggregator={cfg.server.aggregator}  "
                                    f"model={cfg.model.name}  device={exp.device}"))
            self.events.put(("log", f"output: {exp.out_dir}"))
            exp.run()
            best = max((r.get("test_acc", 0) for r in exp.history), default=0)
            self.events.put(("done", f"Finished - best test accuracy {best:.4f}  ->  {exp.out_dir}"))
        except _Stopped:
            self.events.put(("stopped", "Stopped by user (metrics so far are saved in the run folder)."))
        except Exception:  # noqa: BLE001 - show any error in the UI
            self.events.put(("error", traceback.format_exc()))


def _fmt_round(rec: Dict[str, float]) -> str:
    parts = [f"round {int(rec['round']):4d}", f"loss {rec.get('train_loss', float('nan')):.4f}"]
    for k, label in [("train_mask_rate", "mask"), ("train_pl_acc", "pl_acc"), ("test_acc", "test_acc")]:
        if k in rec and rec[k] == rec[k]:
            parts.append(f"{label} {rec[k]:.4f}")
    parts.append(f"{rec.get('time_s', 0):.1f}s")
    return " | ".join(parts)


# ----------------------------------------------------------- results tab
def scan_runs(runs_dir: str) -> List[str]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted((p.parent.name for p in root.glob("*/metrics.csv")), reverse=True)


def compare_runs(runs_dir: str, names: List[str]):
    root = Path(runs_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    rows = []
    for name in names or []:
        df = pd.read_csv(root / name / "metrics.csv")
        cfg = {}
        if (root / name / "config.yaml").exists():
            cfg = yaml.safe_load((root / name / "config.yaml").read_text()) or {}
        if "test_acc" in df:
            d = df.dropna(subset=["test_acc"])
            axes[0].plot(d["round"], d["test_acc"], label=name)
        if "train_loss" in df:
            axes[1].plot(df["round"], df["train_loss"], label=name)
        rows.append({
            "run": name,
            "dataset": cfg.get("data", {}).get("dataset"),
            "labeled_ratio": cfg.get("data", {}).get("labeled_ratio"),
            "alpha": cfg.get("data", {}).get("alpha"),
            "strategy": cfg.get("strategy", {}).get("name"),
            "aggregator": cfg.get("server", {}).get("aggregator"),
            "rounds": int(df["round"].max()),
            "best_acc": round(float(df["test_acc"].max()), 4) if "test_acc" in df else None,
            "final_acc": round(float(df["test_acc"].dropna().iloc[-1]), 4) if "test_acc" in df else None,
        })
    for ax, t in zip(axes, ["Test accuracy", "Train loss"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("round")
        ax.grid(alpha=.3)
    if names:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    plt.close(fig)
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("best_acc", ascending=False)
    return fig, table


# ------------------------------------------------------------------ app
def build_app():
    import gradio as gr

    ctrl = RunController()
    presets = list_presets()
    start_preset = "organamnist_fedlabel.yaml" if "organamnist_fedlabel.yaml" in presets else (
        presets[0] if presets else None)
    init = cfg_to_values(load_preset(start_preset))
    v = dict(zip(FIELDS, init))

    with gr.Blocks(title="SSFL - Semi-Supervised Federated Learning") as app:
        gr.Markdown("# SSFL — Semi-Supervised Federated Learning\n"
                    "Configure an experiment, preview how the data is split across clients, "
                    "and watch training live. Finished runs can be compared in **Results**.")

        with gr.Tab("Experiment"):
            with gr.Row():
                preset = gr.Dropdown(presets, value=start_preset, label="Start from preset", scale=3)
                name = gr.Textbox(v["name"], label="Run name", scale=3)
                seed = gr.Number(v["seed"], label="Seed", precision=0, scale=1)
                device = gr.Dropdown(["auto", "cuda", "cpu"], value=v["device"], label="Device",
                                     allow_custom_value=True, scale=1)
                output_dir = gr.Textbox(v["output_dir"], label="Output folder", scale=2)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1 · Data")
                    dataset = gr.Dropdown(DATASETS.names(), value=v["data.dataset"], label="Dataset")
                    root = gr.Textbox(v["data.root"], label="Download folder")
                    image_size = gr.Dropdown([28, 64, 128, 224], value=v["data.image_size"],
                                             label="Image size (MedMNIST)")
                    partitioner = gr.Dropdown(PARTITIONERS.names(), value=v["data.partitioner"],
                                              label="Partitioner")
                    alpha = gr.Slider(0.01, 10, value=v["data.alpha"], step=0.01,
                                      label="Dirichlet α (smaller = more non-IID)")
                    labeled_ratio = gr.Slider(0.01, 1.0, value=v["data.labeled_ratio"], step=0.01,
                                              label="Labeled ratio")
                    batch_size = gr.Number(v["data.batch_size"], label="Labeled batch size", precision=0)
                    unl_ratio = gr.Number(v["data.unlabeled_batch_ratio"], precision=0,
                                          label="Unlabeled batch ratio (μ)")
                    num_workers = gr.Number(v["data.num_workers"], label="DataLoader workers", precision=0)
                    strong_aug = gr.Dropdown(["randaugment", "none"], value=v["data.strong_aug"],
                                             label="Strong augmentation")
                with gr.Column():
                    gr.Markdown("### 2 · Client")
                    model_name = gr.Dropdown(MODELS.names(), value=v["model.name"], label="Model")
                    norm = gr.Dropdown(["gn", "bn", "none"], value=v["model.norm"], label="Normalisation")
                    model_params = gr.Code(v["model.params"], language="json", label="Model parameters",
                                           lines=1)
                    strategy = gr.Dropdown(STRATEGIES.names(), value=v["strategy.name"],
                                           label="Strategy (pseudo-labeling algorithm)")
                    with gr.Row():
                        local_epochs = gr.Number(v["strategy.local_epochs"], label="Local epochs", precision=0)
                        local_steps = gr.Number(v["strategy.local_steps"], precision=0,
                                                label="Local steps (0 = use epochs)")
                    with gr.Row():
                        optimizer = gr.Dropdown(["sgd", "adam"], value=v["strategy.optimizer"], label="Optimizer")
                        lr = gr.Number(v["strategy.lr"], label="Learning rate")
                    with gr.Row():
                        momentum = gr.Number(v["strategy.momentum"], label="Momentum")
                        weight_decay = gr.Number(v["strategy.weight_decay"], label="Weight decay")
                    with gr.Row():
                        prox_mu = gr.Number(v["strategy.prox_mu"], label="FedProx μ (0 = off)")
                        grad_clip = gr.Number(v["strategy.grad_clip"], label="Grad clip (0 = off)")
                    amp = gr.Checkbox(v["strategy.amp"], label="Mixed precision (AMP)")
                    strategy_params = gr.Code(v["strategy.params"], language="json",
                                              label="Strategy parameters")
                with gr.Column():
                    gr.Markdown("### 3 · Server")
                    aggregator = gr.Dropdown(AGGREGATORS.names(), value=v["server.aggregator"],
                                             label="Aggregator")
                    rounds = gr.Number(v["server.rounds"], label="Communication rounds", precision=0)
                    num_clients = gr.Number(v["server.num_clients"], label="Number of clients N", precision=0)
                    client_fraction = gr.Slider(0.01, 1.0, value=v["server.client_fraction"], step=0.01,
                                                label="Client fraction C per round")
                    eval_every = gr.Number(v["server.eval_every"], label="Evaluate every k rounds", precision=0)
                    server_params = gr.Code(v["server.params"], language="json",
                                            label="Aggregator parameters")
                    gr.Markdown("Switching strategy or aggregator fills in its parameters with defaults.")

            components = [name, seed, device, output_dir,
                          dataset, root, image_size, partitioner, alpha, labeled_ratio, batch_size,
                          unl_ratio, num_workers, strong_aug,
                          model_name, norm, model_params,
                          strategy, local_epochs, local_steps, optimizer, lr, momentum, weight_decay,
                          prox_mu, grad_clip, amp, strategy_params,
                          aggregator, rounds, num_clients, client_fraction, eval_every, server_params]
            assert len(components) == len(FIELDS)

            with gr.Row():
                run_btn = gr.Button("▶ Run experiment", variant="primary")
                stop_btn = gr.Button("■ Stop")
                preview_btn = gr.Button("Preview data partition")
                save_btn = gr.Button("Save as preset")
            status = gr.Markdown("Ready.")
            with gr.Row():
                progress_plot = gr.Plot(label="Training progress")
            with gr.Row():
                log = gr.Textbox(label="Log", lines=12, max_lines=20, autoscroll=True)
                metrics = gr.Dataframe(value=pd.DataFrame(), label="Per-round metrics", wrap=True)
            partition_plot = gr.Plot(label="Data partition", visible=False)
            with gr.Accordion("Resulting YAML config", open=False):
                yaml_view = gr.Code(language="yaml")

        with gr.Tab("Results"):
            with gr.Row():
                runs_dir = gr.Textbox(v["output_dir"], label="Runs folder", scale=4)
                refresh_btn = gr.Button("Refresh", scale=1)
            run_select = gr.CheckboxGroup([], label="Runs to compare")
            compare_plot = gr.Plot()
            compare_table = gr.Dataframe(value=pd.DataFrame(), label="Summary")

        # ---------------------------------------------------- callbacks
        def on_preset(p):
            return cfg_to_values(load_preset(p))

        def show_yaml(*vals):
            try:
                return yaml.safe_dump(values_to_cfg(*vals).to_dict(), sort_keys=False)
            except Exception as e:  # noqa: BLE001
                return f"# invalid configuration: {e}"

        # `.input` fires only on user edits, so loading a preset keeps its tuned params.
        preset.input(on_preset, preset, components).then(show_yaml, components, yaml_view)
        strategy.input(lambda s: json.dumps(default_params(STRATEGIES.get(s)), indent=2),
                       strategy, strategy_params).then(show_yaml, components, yaml_view)
        aggregator.input(lambda a: json.dumps(default_params(AGGREGATORS.get(a)), indent=2),
                         aggregator, server_params).then(show_yaml, components, yaml_view)
        for c in components:
            if c not in (strategy, aggregator):
                c.input(show_yaml, components, yaml_view)

        def on_save(*vals):
            try:
                cfg = values_to_cfg(*vals)
            except Exception as e:  # noqa: BLE001
                return f"⚠️ {e}", gr.update()
            path = CONFIG_DIR / f"{cfg.name}.yaml"
            CONFIG_DIR.mkdir(exist_ok=True)
            cfg.save(path)
            return f"Saved preset **{path.name}**.", gr.update(choices=list_presets())

        save_btn.click(on_save, components, [status, preset])

        def on_preview(*vals):
            from .data import DataManager
            try:
                cfg = values_to_cfg(*vals)
                dm = DataManager(cfg.data, cfg.server.num_clients, seed=cfg.seed)
                return gr.update(value=plot_partition(dm), visible=True), dm.summary()
            except Exception as e:  # noqa: BLE001
                return gr.update(visible=False), f"⚠️ {e}"

        preview_btn.click(on_preview, components, [partition_plot, status])

        def on_run(*vals):
            if ctrl.running:
                yield gr.update(), "⚠️ An experiment is already running — stop it first.", gr.update(), gr.update()
                return
            try:
                cfg = values_to_cfg(*vals)
            except Exception as e:  # noqa: BLE001
                yield gr.update(), f"⚠️ {e}", gr.update(), gr.update()
                return
            ctrl.start(cfg)
            lines, history = [], []
            status_msg = f"Running **{cfg.name}** …"
            yield "", status_msg, None, None
            while True:
                try:
                    kind, payload = ctrl.events.get(timeout=0.5)
                except queue.Empty:
                    if not ctrl.running and ctrl.events.empty():
                        break
                    continue
                if kind == "log":
                    lines.append(payload)
                elif kind == "round":
                    history.append(payload)
                    lines.append(_fmt_round(payload))
                    status_msg = f"Running **{cfg.name}** — round {int(payload['round'])}/{cfg.server.rounds}"
                elif kind == "done":
                    lines.append(payload)
                    status_msg = f"✅ {payload}"
                elif kind == "stopped":
                    lines.append(payload)
                    status_msg = f"⏹ {payload}"
                elif kind == "error":
                    lines.append(payload)
                    status_msg = "❌ Error — see log."
                table = pd.DataFrame(history).round(4) if history else None
                yield ("\n".join(lines[-300:]), status_msg,
                       plot_history(history) if history else None, table)
                if kind in ("done", "stopped", "error"):
                    break

        run_btn.click(on_run, components, [log, status, progress_plot, metrics])

        def on_stop():
            if ctrl.running:
                ctrl.stop_event.set()
                return "Stopping after the current round …"
            return "Nothing is running."

        stop_btn.click(on_stop, None, status)

        def on_refresh(d):
            return gr.update(choices=scan_runs(d), value=[])

        refresh_btn.click(on_refresh, runs_dir, run_select)
        run_select.change(compare_runs, [runs_dir, run_select], [compare_plot, compare_table])
        app.load(show_yaml, components, yaml_view)
        app.load(on_refresh, runs_dir, run_select)

    return app


def launch(**kwargs):
    """Build and launch the GUI. In Colab, Gradio shows it inline in the cell."""
    app = build_app()
    app.queue()
    return app.launch(**kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ssfl.gui", description="SSFL graphical interface")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = p.parse_args(argv)
    launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
