"""Command-line interface.

    python -m ssfl --config configs/cifar10_fedlabel.yaml
    python -m ssfl --config configs/cifar10_fedlabel.yaml --set strategy.name=fixmatch server.rounds=50
    python -m ssfl --list
    python -m ssfl --gui
    python -m ssfl --grid configs/grids/thesis.yaml [--only organamnist] [--dry-run]
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
    p.add_argument("--grid", metavar="YAML", help="run a grid of experiments (see ssfl/grid.py)")
    p.add_argument("--only", nargs="*", default=[], metavar="TEXT",
                   help="with --grid: only runs whose name contains every TEXT")
    p.add_argument("--dry-run", action="store_true", help="with --grid: list the runs, train nothing")
    p.add_argument("--no-resume", action="store_true",
                   help="with --grid: also re-run experiments that already finished")
    args = p.parse_args(argv)

    if args.list:
        for reg in (DATASETS, PARTITIONERS, MODELS, STRATEGIES, AGGREGATORS):
            print(f"{reg.kind + 's':14s}: {', '.join(reg.names())}")
        return
    if args.gui:
        from .gui import launch
        launch(share=args.share)
        return
    if args.grid:
        from .grid import run_grid
        run_grid(args.grid, only=args.only, dry_run=args.dry_run, resume=not args.no_resume,
                 extra=args.set)
        return
    if not args.config:
        p.error("--config is required")
    Experiment.from_yaml(args.config, args.set).run()


if __name__ == "__main__":
    main()
