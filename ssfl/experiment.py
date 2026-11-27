"""Experiment: wires the three tiers together and runs the federated loop.

Pause / resume: :meth:`Experiment.request_pause` stops a run after the current round and
writes ``checkpoint.pt`` into the run folder (also every ``server.checkpoint_every`` rounds,
so a crashed or disconnected run can be picked up too). :meth:`Experiment.resume` rebuilds
the run from its folder and continues with the next round, in the same folder::

    exp = Experiment.resume("runs/cifar10_fedlabel_20261127-150500")
    exp.run()                      # rounds k+1 ... R; metrics.csv keeps the earlier rounds
"""
from __future__ import annotations

import copy
import csv
import json
import random
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .client import Client
from .config import ExperimentConfig
from .data import DataManager
from .models import build_model
from .registry import AGGREGATORS, STRATEGIES
from .runinfo import environment, full_params
from .server import Server

Callback = Callable[[int, Dict[str, float], "Experiment"], None]

#: file in the run folder that holds everything needed to continue a run
CHECKPOINT = "checkpoint.pt"
CHECKPOINT_INFO = "checkpoint.json"
#: control files another process (the GUI / a parallel runner) creates in the run folder:
#: the run pauses / stops after its current round and deletes the file
PAUSE_FILE = "PAUSE"
STOP_FILE = "STOP"


class StopRequested(Exception):
    """Raised after a round when a STOP control file was found (run status: stopped)."""
CHECKPOINT_VERSION = 1
#: config entries that may differ when a run is resumed (where and how long it runs,
#: not what it trains)
RESUME_FREE_KEYS = ("name", "output_dir", "device", "server.rounds", "server.eval_every",
                    "server.checkpoint_every", "data.root", "data.num_workers")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> Dict[str, object]:
    """Every global random generator the run draws from (client sampling has its own)."""
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, object]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        cuda = st["cuda"]
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        else:                                       # resumed on a machine with other GPUs
            torch.cuda.set_rng_state(cuda[0])


def _flat(d: Dict, prefix: str = "") -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "params":
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def new_run_dir(cfg: ExperimentConfig) -> Path:
    """``<output_dir>/<name>_<timestamp>`` that does not exist yet."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(cfg.output_dir) / f"{cfg.name}_{stamp}"
    n = 2
    while out.exists():                     # same name started within the same second
        out = Path(cfg.output_dir) / f"{cfg.name}_{stamp}-{n}"
        n += 1
    return out


def resolved_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """The config with every strategy / aggregator parameter written out, defaults
    included, so ``config.yaml`` alone reproduces the run."""
    cfg = copy.deepcopy(cfg)
    cfg.strategy.params = full_params(STRATEGIES.get(cfg.strategy.name), cfg.strategy.params)
    cfg.server.params = full_params(AGGREGATORS.get(cfg.server.aggregator), cfg.server.params)
    return cfg


def write_queued_run(cfg: ExperimentConfig) -> Path:
    """A run folder for a run that has not started yet (``config.yaml`` and a ``queued``
    summary), e.g. the runs still waiting when a queue is paused. Resuming it starts the
    run from round 1 in that folder."""
    out = new_run_dir(cfg)
    out.mkdir(parents=True)
    resolved_config(cfg).save(out / "config.yaml")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "summary.json").write_text(json.dumps({
        "name": cfg.name, "status": "queued", "started_at": None, "updated_at": now,
        "queued_at": time.time(),                       # keeps the queue order for Resume
        "rounds_done": 0, "rounds_planned": cfg.server.rounds}, indent=2))
    return out


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Experiment:
    """High-level entry point.

    >>> exp = Experiment.from_yaml("configs/cifar10_fedlabel.yaml", ["server.rounds=5"])
    >>> history = exp.run()
    """

    def __init__(self, cfg: ExperimentConfig, callbacks: Optional[List[Callback]] = None,
                 verbose: bool = True, out_dir: Optional[str | Path] = None):
        """``out_dir``: write into this run folder instead of a new ``<name>_<timestamp>``
        (used by :meth:`resume`)."""
        t_setup = time.time()
        self.cfg = cfg
        self.callbacks = list(callbacks or [])
        self.verbose = verbose
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        if self.device.type == "cuda":
            # tries the conv algorithms once per new input shape and keeps the fastest;
            # same results up to floating-point rounding
            torch.backends.cudnn.benchmark = bool(cfg.strategy.cudnn_benchmark)

        # Tier 1 - data
        self.data = DataManager(cfg.data, cfg.server.num_clients, seed=cfg.seed)
        self.data.set_device(self.device)
        self.num_workers = self.data.resolve_workers(cfg.strategy.local_steps, cfg.strategy.local_epochs)
        # Tier 2 - clients + strategy
        self.clients = [Client(self.data.client_data(i)) for i in range(cfg.server.num_clients)]
        self.strategy = STRATEGIES.build(cfg.strategy.name, cfg.strategy,
                                         self.data.num_classes, **cfg.strategy.params)
        # Tier 3 - server
        self.model = self.strategy.prepare_model(
            build_model(cfg.model, self.data.num_classes, self.data.in_channels))
        aggregator = AGGREGATORS.build(cfg.server.aggregator, **cfg.server.params)
        self.server = Server(self.model, aggregator, self.clients, cfg.server.client_fraction,
                             self.device, seed=cfg.seed)
        self.server.precision = self.strategy.precision(self.device)
        self.test_loader = self.data.test_loader()
        self.val_loader = self.data.val_loader()     # None unless data.val_ratio > 0
        self.strategy.setup(self)
        self.history: List[Dict[str, float]] = []
        self.setup_s = time.time() - t_setup           # data loading, partitioning, model building

        self.out_dir = Path(out_dir) if out_dir is not None else new_run_dir(cfg)
        # pause / resume
        self.pause_requested = False           # set by request_pause(); checked after each round
        self.paused = False                    # True once run() returned because of a pause
        self.resumed_from = 0                  # rounds restored from a checkpoint
        self._started_at: Optional[float] = None
        self._elapsed_before = 0.0             # wall time of the earlier sessions of this run
        self._resumes = 0

    @classmethod
    def from_yaml(cls, path, overrides=None, **kw) -> "Experiment":
        return cls(ExperimentConfig.from_yaml(path, overrides), **kw)

    # -------------------------------------------------------- pause / resume
    def request_pause(self) -> None:
        """Finish the current round, save a checkpoint and return from :meth:`run`
        (thread-safe: it only sets a flag)."""
        self.pause_requested = True

    @classmethod
    def resume(cls, run_dir: str | Path, overrides=None, **kw) -> "Experiment":
        """Rebuild a paused, stopped or interrupted run from its folder; :meth:`run` then
        continues with the round after the last checkpoint, in the same folder.
        ``overrides`` may change where / how long it runs (e.g. ``server.rounds=1500``);
        changing what it trains is allowed but logged as a warning. A ``queued`` run
        (never started) starts from round 1."""
        run_dir = Path(run_dir)
        if not (run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"{run_dir} is not a run folder (no config.yaml)")
        cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml", overrides)
        cfg.output_dir = str(run_dir.parent)
        ckpt = run_dir / CHECKPOINT
        status = None
        try:
            status = json.loads((run_dir / "summary.json").read_text()).get("status")
        except (OSError, ValueError):
            pass
        if not ckpt.exists() and status != "queued":
            if status == "finished":
                raise ValueError(f"{run_dir.name} has already finished")
            raise FileNotFoundError(
                f"{run_dir.name} has no {CHECKPOINT}: it stopped before its first checkpoint "
                f"(every server.checkpoint_every rounds, or on pause), so it can only be started again")
        exp = cls(cfg, out_dir=run_dir, **kw)
        if ckpt.exists():
            exp.load_checkpoint(ckpt)
        return exp

    def save_checkpoint(self) -> Path:
        """Everything needed to continue after the last finished round: global model,
        history, client-sampling and global random states, aggregator and strategy state."""
        state = {
            "version": CHECKPOINT_VERSION,
            "round": len(self.history),
            "global_state": self.server.global_state,
            "history": self.history,
            "server_rng": self.server.rng.bit_generator.state,
            "aggregator": self.server.aggregator.state_dict(),
            "strategy": self.strategy.state_dict(),
            "rng": rng_state(),
            "started_at": self._started_at,
            "elapsed_s": self._elapsed(),
            "resumes": self._resumes,
            "config": resolved_config(self.cfg).to_dict(),
        }
        path = self.out_dir / CHECKPOINT
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)                       # never a half-written checkpoint
        # small sidecar, so run lists can show the checkpoint's round without loading it
        (self.out_dir / CHECKPOINT_INFO).write_text(json.dumps({
            "round": len(self.history), "rounds_planned": self.cfg.server.rounds,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path}: unsupported checkpoint version {ck.get('version')}")
        old, new = _flat(ck.get("config", {})), _flat(resolved_config(self.cfg).to_dict())
        changed = [k for k in sorted(set(old) | set(new))
                   if old.get(k) != new.get(k) and not any(k == f or k.startswith(f + ".")
                                                           for f in RESUME_FREE_KEYS)]
        if changed:
            self.log(f"[ssfl] warning: resuming with a different config for "
                     f"{', '.join(f'{k} ({old.get(k)!r} -> {new.get(k)!r})' for k in changed)}")
        self.server.global_state = ck["global_state"]
        self.server.rng.bit_generator.state = ck["server_rng"]
        self.server.aggregator.load_state_dict(ck["aggregator"])
        self.strategy.load_state_dict(ck["strategy"])
        self.history = list(ck["history"])
        self.resumed_from = int(ck["round"])
        self._started_at = ck.get("started_at")
        self._elapsed_before = float(ck.get("elapsed_s") or 0.0)
        self._resumes = int(ck.get("resumes", 0)) + 1
        set_rng_state(ck["rng"])                # last: nothing else may draw from them now

    def _check_control_files(self) -> None:
        """PAUSE / STOP files written into the run folder by another process."""
        stop, pause = self.out_dir / STOP_FILE, self.out_dir / PAUSE_FILE
        if stop.exists():
            stop.unlink(missing_ok=True)
            raise StopRequested(f"{STOP_FILE} file in {self.out_dir}")
        if pause.exists():
            pause.unlink(missing_ok=True)
            self.request_pause()

    def _elapsed(self) -> float:
        return self._elapsed_before + (time.time() - self._t_start) + self.setup_s

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ run
    def run_round(self, rnd: int) -> Dict[str, float]:
        t0 = time.time()
        selected = self.server.select_clients()                 # orchestration
        self.strategy.on_server_round_start(self.server, selected, rnd)
        global_state = self.server.broadcast()                  # downlink
        server_s = time.time() - t0
        updates, client_s = [], []
        for c in selected:                                      # local execution + uplink
            t = time.time()
            updates.append(c.fit(self.model, global_state, self.strategy, self.device))
            client_s.append(time.time() - t)
        t = time.time()
        self.strategy.on_server_round_end(self.server, updates, global_state, rnd)
        self.server.aggregate(updates)                          # global aggregation
        server_s += time.time() - t

        # computation cost as in the FedLabel paper's Table 7: local training time per client
        # and the server's own work (selection, helpers / clustering, thresholds, aggregation)
        rec: Dict[str, float] = {"round": rnd, "num_clients": len(selected),
                                 "client_s": float(np.mean(client_s)), "server_s": server_s}
        agg: Dict[str, List[float]] = defaultdict(list)
        for u in updates:
            for k, v in u.metrics.items():
                agg[k].append(v)
        rec.update({f"train_{k}": float(np.mean(v)) for k, v in agg.items()})
        if rnd % self.cfg.server.eval_every == 0 or rnd == self.cfg.server.rounds:
            rec.update(self.server.evaluate(self.test_loader))
            if self.val_loader is not None:
                rec.update({"val_" + k[len("test_"):]: v
                            for k, v in self.server.evaluate(self.val_loader).items()})
        rec["time_s"] = time.time() - t0
        return rec

    # ---------------------------------------------------------- bookkeeping
    def resolved_config(self) -> ExperimentConfig:
        """The config with every strategy / aggregator parameter written out, defaults
        included, so ``config.yaml`` alone reproduces the run."""
        return resolved_config(self.cfg)

    def _write_summary(self, status: str) -> None:
        """``summary.json``: status, timing, best / final metrics and environment. Written
        after every round, so stopped or crashed runs keep their timing too."""
        now = time.time()
        acc = [(r["test_acc"], int(r["round"])) for r in self.history if "test_acc" in r]
        best = max(acc, default=(None, None))
        times = [r["time_s"] for r in self.history if "time_s" in r]
        val = [(r["val_acc"], int(r["round"]), r.get("test_acc")) for r in self.history if "val_acc" in r]
        best_val = max(val, key=lambda v: v[0], default=(None, None, None))   # first round on ties

        def mean(key):
            v = [r[key] for r in self.history if key in r]
            return round(float(np.mean(v)), 4) if v else None
        summary = {
            "name": self.cfg.name,
            "status": status,                         # running | paused | finished | stopped | error
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._started_at)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "rounds_done": len(self.history),
            "rounds_planned": self.cfg.server.rounds,
            "setup_s": round(self.setup_s, 2),
            "train_s": round(sum(times), 2),
            "elapsed_s": round(self._elapsed(), 2),   # wall time, pauses not counted
            "mean_round_s": round(sum(times) / len(times), 3) if times else None,
            "best_test_acc": best[0],
            "best_round": best[1],
            "best_test_f1": max((r["test_f1"] for r in self.history if "test_f1" in r), default=None),
            # model selection on the validation set (data.val_ratio > 0)
            "best_val_acc": best_val[0],
            "best_val_round": best_val[1],
            "test_acc_at_best_val": best_val[2],
            "mean_client_s": mean("client_s"),        # local training per client per round
            "mean_server_s": mean("server_s"),
            "final": self.history[-1] if self.history else {},
            "num_workers": self.num_workers,
            "precision": self.server.precision,
            "augment_device": self.cfg.data.augment_device,
            "hflip": self.data.hflip,                 # resolved (auto = off for OrganAMNIST)
            "data_summary": self.data.summary(),
            "environment": self._env,
            "resumes": self._resumes,                 # times this run was continued from a checkpoint
            "checkpoint": (self.out_dir / CHECKPOINT).exists(),
        }
        tmp = self.out_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(self.out_dir / "summary.json")

    def run(self) -> List[Dict[str, float]]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config().save(self.out_dir / "config.yaml")
        self._t_start = time.time()
        if self._started_at is None:
            self._started_at = self._t_start
        self.paused = False
        self._env = environment(self.device)
        try:
            return self._run()
        except BaseException as e:                     # stopped from the GUI, Ctrl+C, crash
            if self.history:
                self._write_summary("stopped" if type(e).__name__ in ("_Stopped", "KeyboardInterrupt",
                                                                      "StopRequested")
                                    else "error")
            raise

    def _run(self) -> List[Dict[str, float]]:
        self.log(f"[ssfl] {self.data.summary()}")
        self.log(f"[ssfl] strategy={self.cfg.strategy.name} aggregator={self.cfg.server.aggregator} "
                 f"model={self.cfg.model.name} device={self.device} "
                 f"workers={self.num_workers}{' (auto)' if self.cfg.data.num_workers < 0 else ''} "
                 f"augment={self.cfg.data.augment_device} precision={self.server.precision} "
                 f"-> {self.out_dir}")
        if self.resumed_from:
            self.log(f"[ssfl] resumed after round {self.resumed_from} of {self.cfg.server.rounds}")

        rounds = self.cfg.server.rounds
        every = int(getattr(self.cfg.server, "checkpoint_every", 0) or 0)
        for rnd in range(len(self.history) + 1, rounds + 1):
            rec = self.run_round(rnd)
            self.history.append(rec)
            self._write_csv()
            msg = f"round {rnd:4d} | loss {rec.get('train_loss', float('nan')):.4f}"
            if "train_mask_rate" in rec:
                msg += f" | mask {rec['train_mask_rate']:.2f}"
            if "train_pl_acc" in rec:
                msg += f" | pl_acc {rec['train_pl_acc']:.3f}"
            if "test_acc" in rec:
                msg += f" | test_acc {rec['test_acc']:.4f}"
            if "val_acc" in rec:
                msg += f" | val_acc {rec['val_acc']:.4f}"
            self.log(msg + f" | {rec['time_s']:.1f}s")
            self._write_summary("running")
            for cb in self.callbacks:
                cb(rnd, rec, self)              # may call request_pause() (GUI) or raise (stop)
            self._check_control_files()
            if rnd < rounds and (self.pause_requested or (every and rnd % every == 0)):
                self.save_checkpoint()
            if self.pause_requested and rnd < rounds:
                self.paused = True
                self._write_summary("paused")
                self.log(f"[ssfl] paused after round {rnd}/{rounds}; continue with "
                         f"python -m ssfl --resume \"{self.out_dir}\"")
                return self.history

        torch.save(self.server.global_state, self.out_dir / "global_model.pt")
        for f in (CHECKPOINT, CHECKPOINT_INFO):         # finished: global_model.pt is enough
            (self.out_dir / f).unlink(missing_ok=True)
        self._write_summary("finished")
        best = max((r["test_acc"] for r in self.history if "test_acc" in r), default=None)
        msg = f"[ssfl] done. best test_acc={best}"
        if self.val_loader is not None:
            val = [(r["val_acc"], r.get("test_acc")) for r in self.history if "val_acc" in r]
            if val:
                msg += f" | test_acc at best val_acc={max(val, key=lambda v: v[0])[1]}"
        self.log(msg + f" | {time.time() - self._t_start:.0f}s")
        return self.history

    def _write_csv(self) -> None:
        keys = sorted({k for r in self.history for k in r}, key=lambda k: (k != "round", k))
        with open(self.out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)


def run_pausable(exp: "Experiment") -> List[Dict[str, float]]:
    """Run with Ctrl+C = pause after the current round (checkpoint saved), Ctrl+C again = stop now."""
    def on_sigint(signum, frame):
        if exp.pause_requested:
            raise KeyboardInterrupt
        exp.request_pause()
        print("\n[ssfl] pausing after the current round (Ctrl+C again to stop now) ...", flush=True)

    try:
        old = signal.signal(signal.SIGINT, on_sigint)
    except ValueError:                      # not the main thread: plain run
        return exp.run()
    try:
        return exp.run()
    finally:
        signal.signal(signal.SIGINT, old)

