"""Run a grid of experiments (datasets x label ratios x algorithms) from one YAML file.

    python -m ssfl --grid configs/grids/thesis.yaml               # run everything
    python -m ssfl --grid configs/grids/thesis.yaml --dry-run     # list the runs only
    python -m ssfl --grid configs/grids/thesis.yaml --only organamnist fedmatch

Grid file layout::

    common: {server.eval_every: 5}          # overrides for every run (optional)
    algorithms:                             # name -> strategy.params (replaces the preset's params)
      fedavg: {}
      fedlabel: {tau: 0.95, lambda_kl: 1.0}
    experiments:
      - config: configs/organamnist_fedlabel.yaml
        labeled_ratio: [0.05, 0.2]
        overrides: {server.rounds: 100}     # optional, for this experiment only
        algorithm_params: {fedtrinet: {pretrain_rounds: 20}}   # optional, merged into the params
        algorithms: [fedavg, fedlabel]      # optional subset (default: all)

Every run is named ``<dataset>_<algorithm>_l<ratio>[_<tag>]``. A run whose name
already has a ``summary.json`` in ``output_dir`` is skipped, so an interrupted
grid (for example a Colab session that timed out) continues where it stopped.
Use ``--no-resume`` to run everything again.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .config import ExperimentConfig, apply_overrides


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(config: str, grid_file: str | Path) -> Path:
    """A preset path in a grid file may be relative to the working directory,
    to the grid file, or to the project root (in that order)."""
    for base in (Path.cwd(), Path(grid_file).resolve().parent, PROJECT_ROOT):
        p = Path(config) if Path(config).is_absolute() else base / config
        if p.exists():
            return p
    raise FileNotFoundError(f"Preset '{config}' not found (from grid {grid_file})")


def _ratio_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")


def expand_grid(path: str | Path, extra: Optional[Sequence[str]] = None
                ) -> List[Tuple[str, ExperimentConfig]]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    algorithms: Dict[str, Dict[str, Any]] = spec.get("algorithms") or {}
    common: Dict[str, Any] = spec.get("common") or {}
    runs: List[Tuple[str, ExperimentConfig]] = []
    for exp in spec.get("experiments") or []:
        raw = yaml.safe_load(_resolve(exp["config"], path).read_text(encoding="utf-8")) or {}
        raw = apply_overrides(raw, common)
        raw = apply_overrides(raw, exp.get("overrides") or {})
        ratios = exp.get("labeled_ratio") or [raw.get("data", {}).get("labeled_ratio", 0.2)]
        ratios = ratios if isinstance(ratios, list) else [ratios]
        names = exp.get("algorithms") or list(algorithms)
        for ratio in ratios:
            for algo in names:
                if algo not in algorithms:
                    raise KeyError(f"Algorithm '{algo}' is not defined under 'algorithms'")
                params = copy.deepcopy(algorithms[algo] or {})
                params.update((exp.get("algorithm_params") or {}).get(algo) or {})
                r = copy.deepcopy(raw)
                r.setdefault("data", {})["labeled_ratio"] = ratio
                strat = r.setdefault("strategy", {})
                strat["name"] = algo
                strat["params"] = params
                dataset = r["data"].get("dataset", "data")
                name = f"{dataset}_{algo}_l{_ratio_tag(ratio)}"
                if exp.get("tag"):
                    name += f"_{exp['tag']}"
                r["name"] = name
                r = apply_overrides(r, list(extra or []))
                runs.append((name, ExperimentConfig.from_dict(r)))
    return runs


def _finished(cfg: ExperimentConfig) -> bool:
    out = Path(cfg.output_dir)
    return any(out.glob(f"{cfg.name}_*/summary.json")) if out.exists() else False


def run_grid(path: str | Path, only: Sequence[str] = (), dry_run: bool = False,
             resume: bool = True, extra: Optional[Sequence[str]] = None) -> List[str]:
    from .experiment import Experiment

    runs = [(n, c) for n, c in expand_grid(path, extra) if all(o in n for o in only)]
    done: List[str] = []
    for i, (name, cfg) in enumerate(runs, 1):
        skip = resume and _finished(cfg)
        status = "skip (finished)" if skip else ("planned" if dry_run else "run")
        print(f"[grid {i}/{len(runs)}] {name}: {status}", flush=True)
        if skip or dry_run:
            continue
        Experiment(cfg).run()
        done.append(name)
    return done
