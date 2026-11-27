"""Command-line interface.

    python -m ssfl --config configs/cifar10_fedlabel.yaml
    python -m ssfl --config configs/cifar10_fedlabel.yaml --set strategy.name=fixmatch server.rounds=50
    python -m ssfl --list
    python -m ssfl --gui
    python -m ssfl --grid configs/grids/thesis.yaml [--only organamnist] [--dry-run]
    python -m ssfl --config configs/cifar10_fedlabel.yaml --bench 5   # speed settings compared
    python -m ssfl --clear-runs [--runs-dir DIR] [--only TEXT] [--algorithm NAME] [--status unfinished] [--dry-run] [--yes]
"""
from __future__ import annotations

import argparse

from . import AGGREGATORS, DATASETS, MODELS, PARTITIONERS, STRATEGIES
from .experiment import Experiment


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ssfl", description="Semi-Supervised Federated Learning")
    p.add_argument("--config", "-c", help="YAML experiment config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. data.labeled_ratio=0.05")
    p.add_argument("--list", action="store_true", help="list registered components and exit")
    p.add_argument("--gui", action="store_true", help="open the graphical interface")
    p.add_argument("--share", action="store_true", help="with --gui: public gradio.live link")
    p.add_argument("--data-dir", help="with --gui: default download folder (e.g. on Google Drive)")
    p.add_argument("--runs-dir", help="with --gui: default output / runs folder; with --clear-runs: the "
                                      "folder to clear (default ./runs)")
    p.add_argument("--workers", type=int, help="with --gui: default DataLoader workers")
    p.add_argument("--grid", metavar="YAML", help="run a grid of experiments (see ssfl/grid.py)")
    p.add_argument("--only", nargs="*", default=[], metavar="TEXT",
                   help="with --grid / --clear-runs: only runs whose name contains every TEXT")
    p.add_argument("--dry-run", action="store_true",
                   help="with --grid: list the runs, train nothing; with --clear-runs: list, delete nothing")
    p.add_argument("--clear-runs", action="store_true",
                   help="delete previous run folders (asks first; running runs are never deleted)")
    p.add_argument("--algorithm", nargs="*", default=[], metavar="NAME",
                   help="with --clear-runs: only runs of these algorithms, e.g. fedlabel")
    p.add_argument("--status", nargs="*", default=[], metavar="STATUS",
                   help="with --clear-runs: finished | stopped | error | interrupted | partial | unfinished")
    p.add_argument("--yes", "-y", action="store_true", help="with --clear-runs: do not ask")
    p.add_argument("--no-resume", action="store_true",
                   help="with --grid: also re-run experiments that already finished")
    p.add_argument("--bench", type=int, metavar="ROUNDS",
                   help="with --config: time ROUNDS rounds with and without channels_last, the cuDNN "
                        "benchmark and torch.compile; nothing is written to disk")
    args = p.parse_args(argv)

    if args.list:
        for reg in (DATASETS, PARTITIONERS, MODELS, STRATEGIES, AGGREGATORS):
            print(f"{reg.kind + 's':14s}: {', '.join(reg.names())}")
        return
    if args.gui:
        from .gui import launch
        launch(share=args.share, data_dir=args.data_dir, runs_dir=args.runs_dir, num_workers=args.workers)
        return
    if args.clear_runs:
        from .cleanup import clear_runs_cli
        clear_runs_cli(args.runs_dir or "./runs", only=args.only, algorithms=args.algorithm,
                       statuses=args.status, dry_run=args.dry_run, yes=args.yes)
        return
    if args.grid:
        from .grid import run_grid
        run_grid(args.grid, only=args.only, dry_run=args.dry_run, resume=not args.no_resume,
                 extra=args.set)
        return
    if not args.config:
        p.error("--config is required")
    if args.bench:
        from .bench import bench
        bench(args.config, args.bench, args.set)
        return
    Experiment.from_yaml(args.config, args.set).run()


if __name__ == "__main__":
    main()
