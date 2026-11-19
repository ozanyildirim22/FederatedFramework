"""Tiny registry system that makes every component plug-and-play.

Each component family (datasets, partitioners, models, strategies,
aggregators) has its own :class:`Registry`. New components are added with a
decorator and are then selectable by name from a YAML config::

    from ssfl import STRATEGIES, Strategy

    @STRATEGIES.register("my_algo")
    class MyAlgo(Strategy):
        ...
"""
from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        def deco(obj: Any) -> Any:
            key = (name or obj.__name__).lower()
            if key in self._items:
                raise KeyError(f"{self.kind} '{key}' is already registered")
            self._items[key] = obj
            return obj

        return deco

    def get(self, name: str) -> Any:
        key = name.lower()
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __repr__(self) -> str:
        return f"Registry({self.kind}: {self.names()})"


DATASETS = Registry("dataset")
PARTITIONERS = Registry("partitioner")
MODELS = Registry("model")
STRATEGIES = Registry("strategy")
AGGREGATORS = Registry("aggregator")
