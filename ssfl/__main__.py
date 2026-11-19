"""Command-line interface.

    python -m ssfl --config configs/cifar10_fedlabel.yaml
    python -m ssfl --config configs/cifar10_fedlabel.yaml --set strategy.name=fixmatch server.rounds=50
    python -m ssfl --list
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
    args = p.parse_args(argv)

    if args.list:
        for reg in (DATASETS, PARTITIONERS, MODELS, STRATEGIES, AGGREGATORS):
            print(f"{reg.kind + 's':14s}: {', '.join(reg.names())}")
        return
    if not args.config:
        p.error("--config is required")
    Experiment.from_yaml(args.config, args.set).run()


if __name__ == "__main__":
    main()
