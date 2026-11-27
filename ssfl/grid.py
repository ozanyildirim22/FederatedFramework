"""Run a grid of experiments (datasets x label ratios x alpha x algorithms) from one YAML file.

    python -m ssfl --grid configs/grids/thesis.yaml               # run everything
    python -m ssfl --grid configs/grids/thesis.yaml --dry-run     # list the runs only
    python -m ssfl --grid configs/grids/thesis.yaml --only organamnist fedmatch

Grid file layout::

    common: {server.eval_every: 5}          # overrides for every run (optional)
    algorithms:                             # name -> strategy.params (replaces the preset's params)
      fedavg: {}
      fedlabel: {beta: 0.5, lambda0: 1.0}
    experiments:
      - config: configs/organamnist_fedlabel.yaml
        labeled_ratio: [0.05, 0.2]
        alpha: [0.1, 1.0]                   # optional Dirichlet alpha values (default: the preset's)
        overrides: {server.rounds: 100}     # optional, for this experiment only
        algorithm_params: {fedtrinet: {pretrain_rounds: 20}}   # optional, merged into the params
        algorithms: [fedavg, fedlabel]      # optional subset (default: all)
        seeds: [0, 1, 2]                    # optional, per experiment (overrides the top-level list)
    seeds: [0]                              # optional: repeat every run with these seeds

Every run is named ``<dataset>_<algorithm>_l<ratio>_a<alpha>[_<tag>][_s<seed>]``; the
``_s<seed>`` suffix is left out for the preset's own seed, so adding seeds to a grid keeps
the runs already done. The FedLabel paper averages 3 seeds (appendix A); the Results tab
shows the mean and standard deviation over seeds. A run whose name
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
    top_seeds = spec.get("seeds")
    runs: List[Tuple[int, str, ExperimentConfig]] = []
    for exp in spec.get("experiments") or []:
        raw = yaml.safe_load(_resolve(exp["config"], path).read_text(encoding="utf-8")) or {}
        raw = apply_overrides(raw, common)
        raw = apply_overrides(raw, exp.get("overrides") or {})
        ratios = exp.get("labeled_ratio") or [raw.get("data", {}).get("labeled_ratio", 0.2)]
        ratios = ratios if isinstance(ratios, list) else [ratios]
        alphas = exp.get("alpha") or [raw.get("data", {}).get("alpha", 0.5)]
        alphas = alphas if isinstance(alphas, list) else [alphas]
        names = exp.get("algorithms") or list(algorithms)
        base_seed = int(raw.get("seed", ExperimentConfig().seed))
        seeds = exp.get("seeds", top_seeds)
        seeds = [base_seed] if seeds is None else (seeds if isinstance(seeds, list) else [seeds])
        for ratio, alpha, algo, seed in ((r_, a_, n_, s_) for r_ in ratios for a_ in alphas
                                         for n_ in names for s_ in seeds):
            if algo not in algorithms:
                raise KeyError(f"Algorithm '{algo}' is not defined under 'algorithms'")
            params = copy.deepcopy(algorithms[algo] or {})
            params.update((exp.get("algorithm_params") or {}).get(algo) or {})
            r = copy.deepcopy(raw)
            r.setdefault("data", {})["labeled_ratio"] = ratio
            r["data"]["alpha"] = alpha
            strat = r.setdefault("strategy", {})
            strat["name"] = algo
            strat["params"] = params
            dataset = r["data"].get("dataset", "data")
            name = f"{dataset}_{algo}_l{_ratio_tag(ratio)}_a{_ratio_tag(alpha)}"
            if exp.get("tag"):
                name += f"_{exp['tag']}"
            r["seed"] = int(seed)
            if int(seed) != base_seed:
                name += f"_s{int(seed)}"
            r["name"] = name
            r = apply_overrides(r, list(extra or []))
            runs.append((seeds.index(seed), name, ExperimentConfig.from_dict(r)))
    # the whole grid with the first seed, then again with the second, ...: an interrupted
    # grid has complete tables for the seeds it finished
    return [(n, c) for _, n, c in sorted(runs, key=lambda t: t[0])]


def run_finished(summary_path: Path) -> bool:
    """A run folder is finished when its summary says so. Summaries from before the
    ``status`` field were only written at the end, so they count as finished."""
    import json
    try:
        return json.loads(summary_path.read_text()).get("status", "finished") == "finished"
    except (OSError, ValueError):
        return False


def _finished(cfg: ExperimentConfig) -> bool:
    out = Path(cfg.output_dir)
    if not out.exists():
        return False
    return any(run_finished(p) for p in out.glob(f"{cfg.name}_*/summary.json"))


def resumable_dir(cfg: ExperimentConfig) -> Optional[Path]:
    """The newest folder of this run that can be continued (paused / stopped / interrupted
    with a checkpoint, or queued), or None."""
    from .cleanup import resumable_runs
    out = Path(cfg.output_dir)
    if not out.exists():
        return None
    for r in resumable_runs(out):                       # newest first
        if r["name"].startswith(f"{cfg.name}_") and r["name"][len(cfg.name) + 1:][:1].isdigit():
            return Path(r["path"])
    return None


def run_grid(path: str | Path, only: Sequence[str] = (), dry_run: bool = False,
             resume: bool = True, extra: Optional[Sequence[str]] = None) -> List[str]:
    """Run every experiment of the grid. With ``resume`` (default) finished runs are skipped
    and paused / interrupted ones continue from their checkpoint. Ctrl+C pauses the current
    run after its round and stops the grid; run the same command again to continue."""
    from .experiment import Experiment, run_pausable

    runs = [(n, c) for n, c in expand_grid(path, extra) if all(o in n for o in only)]
    done: List[str] = []
    for i, (name, cfg) in enumerate(runs, 1):
        skip = resume and _finished(cfg)
        cont = resumable_dir(cfg) if resume and not skip else None
        status = ("skip (finished)" if skip else f"resume {cont.name}" if cont
                  else ("planned" if dry_run else "run"))
        print(f"[grid {i}/{len(runs)}] {name}: {status}", flush=True)
        if skip or dry_run:
            continue
        exp = Experiment.resume(cont) if cont else Experiment(cfg)
        run_pausable(exp)
        if exp.paused:
            print(f"[grid] paused at {name}; run the same command again to continue.", flush=True)
            break
        done.append(name)
    return done
