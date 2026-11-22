"""Experiment: wires the three tiers together and runs the federated loop."""
from __future__ import annotations

import copy
import csv
import json
import random
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                 verbose: bool = True):
        t_setup = time.time()
        self.cfg = cfg
        self.callbacks = list(callbacks or [])
        self.verbose = verbose
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)

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
        self.strategy.setup(self)
        self.history: List[Dict[str, float]] = []
        self.setup_s = time.time() - t_setup           # data loading, partitioning, model building

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out_dir = Path(cfg.output_dir) / f"{cfg.name}_{stamp}"
        n = 2
        while self.out_dir.exists():           # same name started within the same second
            self.out_dir = Path(cfg.output_dir) / f"{cfg.name}_{stamp}-{n}"
            n += 1

    @classmethod
    def from_yaml(cls, path, overrides=None, **kw) -> "Experiment":
        return cls(ExperimentConfig.from_yaml(path, overrides), **kw)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ run
    def run_round(self, rnd: int) -> Dict[str, float]:
        t0 = time.time()
        selected = self.server.select_clients()                 # orchestration
        self.strategy.on_server_round_start(self.server, selected, rnd)
        global_state = self.server.broadcast()                  # downlink
        updates = [c.fit(self.model, global_state, self.strategy, self.device)
                   for c in selected]                           # local execution + uplink
        self.strategy.on_server_round_end(self.server, updates, global_state, rnd)
        self.server.aggregate(updates)                          # global aggregation

        rec: Dict[str, float] = {"round": rnd, "num_clients": len(selected)}
        agg: Dict[str, List[float]] = defaultdict(list)
        for u in updates:
            for k, v in u.metrics.items():
                agg[k].append(v)
        rec.update({f"train_{k}": float(np.mean(v)) for k, v in agg.items()})
        if rnd % self.cfg.server.eval_every == 0 or rnd == self.cfg.server.rounds:
            rec.update(self.server.evaluate(self.test_loader))
        rec["time_s"] = time.time() - t0
        return rec

    # ---------------------------------------------------------- bookkeeping
    def resolved_config(self) -> ExperimentConfig:
        """The config with every strategy / aggregator parameter written out, defaults
        included, so ``config.yaml`` alone reproduces the run."""
        cfg = copy.deepcopy(self.cfg)
        cfg.strategy.params = full_params(STRATEGIES.get(cfg.strategy.name), cfg.strategy.params)
        cfg.server.params = full_params(AGGREGATORS.get(cfg.server.aggregator), cfg.server.params)
        return cfg

    def _write_summary(self, status: str) -> None:
        """``summary.json``: status, timing, best / final metrics and environment. Written
        after every round, so stopped or crashed runs keep their timing too."""
        now = time.time()
        acc = [(r["test_acc"], int(r["round"])) for r in self.history if "test_acc" in r]
        best = max(acc, default=(None, None))
        times = [r["time_s"] for r in self.history if "time_s" in r]
        summary = {
            "name": self.cfg.name,
            "status": status,                         # running | finished | stopped | error
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._t_start)),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "rounds_done": len(self.history),
            "rounds_planned": self.cfg.server.rounds,
            "setup_s": round(self.setup_s, 2),
            "train_s": round(sum(times), 2),
            "elapsed_s": round(now - self._t_start + self.setup_s, 2),
            "mean_round_s": round(sum(times) / len(times), 3) if times else None,
            "best_test_acc": best[0],
            "best_round": best[1],
            "best_test_f1": max((r["test_f1"] for r in self.history if "test_f1" in r), default=None),
            "final": self.history[-1] if self.history else {},
            "num_workers": self.num_workers,
            "precision": self.server.precision,
            "augment_device": self.cfg.data.augment_device,
            "hflip": self.data.hflip,                 # resolved (auto = off for OrganAMNIST)
            "data_summary": self.data.summary(),
            "environment": self._env,
        }
        tmp = self.out_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(self.out_dir / "summary.json")

    def run(self) -> List[Dict[str, float]]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config().save(self.out_dir / "config.yaml")
        self._t_start = time.time()
        self._env = environment(self.device)
        try:
            return self._run()
        except BaseException as e:                     # stopped from the GUI, Ctrl+C, crash
            if self.history:
                self._write_summary("stopped" if type(e).__name__ in ("_Stopped", "KeyboardInterrupt")
                                    else "error")
            raise

    def _run(self) -> List[Dict[str, float]]:
        self.log(f"[ssfl] {self.data.summary()}")
        self.log(f"[ssfl] strategy={self.cfg.strategy.name} aggregator={self.cfg.server.aggregator} "
                 f"model={self.cfg.model.name} device={self.device} "
                 f"workers={self.num_workers}{' (auto)' if self.cfg.data.num_workers < 0 else ''} "
                 f"augment={self.cfg.data.augment_device} precision={self.server.precision} "
                 f"-> {self.out_dir}")

        for rnd in range(1, self.cfg.server.rounds + 1):
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
            self.log(msg + f" | {rec['time_s']:.1f}s")
            self._write_summary("running")
            for cb in self.callbacks:
                cb(rnd, rec, self)

        torch.save(self.server.global_state, self.out_dir / "global_model.pt")
        self._write_summary("finished")
        best = max((r["test_acc"] for r in self.history if "test_acc" in r), default=None)
        self.log(f"[ssfl] done. best test_acc={best} | {time.time() - self._t_start:.0f}s")
        return self.history

    def _write_csv(self) -> None:
        keys = sorted({k for r in self.history for k in r}, key=lambda k: (k != "round", k))
        with open(self.out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
