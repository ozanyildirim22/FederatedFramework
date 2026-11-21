"""Experiment: wires the three tiers together and runs the federated loop."""
from __future__ import annotations

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
        self.cfg = cfg
        self.callbacks = list(callbacks or [])
        self.verbose = verbose
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)

        # Tier 1 - data
        self.data = DataManager(cfg.data, cfg.server.num_clients, seed=cfg.seed)
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
        self.test_loader = self.data.test_loader()
        self.strategy.setup(self)
        self.history: List[Dict[str, float]] = []

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.out_dir = Path(cfg.output_dir) / f"{cfg.name}_{stamp}"

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

    def run(self) -> List[Dict[str, float]]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(self.out_dir / "config.yaml")
        self.log(f"[ssfl] {self.data.summary()}")
        self.log(f"[ssfl] strategy={self.cfg.strategy.name} aggregator={self.cfg.server.aggregator} "
                 f"model={self.cfg.model.name} device={self.device} -> {self.out_dir}")

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
            for cb in self.callbacks:
                cb(rnd, rec, self)

        torch.save(self.server.global_state, self.out_dir / "global_model.pt")
        best = max((r["test_acc"] for r in self.history if "test_acc" in r), default=None)
        summary = {"final": self.history[-1], "best_test_acc": best}
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.log(f"[ssfl] done. best test_acc={best}")
        return self.history

    def _write_csv(self) -> None:
        keys = sorted({k for r in self.history for k in r}, key=lambda k: (k != "round", k))
        with open(self.out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
