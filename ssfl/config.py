"""Experiment configuration: typed dataclasses + YAML loading + CLI overrides."""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "cifar10"            # cifar10 | organamnist | pathmnist | synthetic | <registered>
    root: str = "./data"
    image_size: int = 28                # only used by MedMNIST datasets (28/64/128/224)
    partitioner: str = "dirichlet"      # iid | dirichlet | <registered>
    alpha: float = 0.5                  # Dirichlet concentration (smaller = more non-IID)
    min_samples_per_client: int = 10
    labeled_ratio: float = 0.2          # fraction of each client's data that keeps its labels
    batch_size: int = 64                # labeled batch size
    unlabeled_batch_ratio: int = 7      # unlabeled batch = ratio * batch_size (FixMatch "mu")
    num_workers: int = -1               # DataLoader workers; -1 = auto (0 on Windows / small rounds)
    strong_aug: str = "randaugment"     # randaugment | none
    augment_device: str = "cpu"         # cpu (PIL, exact) | gpu (dataset on the GPU, batched augmentation)
    randaugment_n: int = 1              # RandAugment ops per image (ADD: N = 1)
    randaugment_m: int = 10             # RandAugment magnitude (ADD: 10)
    cutout: bool = False                # add Cutout (RandomErasing) to the strong view (not in the ADD)
    hflip: Optional[bool] = None        # random horizontal flip in weak/strong views; None = auto
                                        # (off for OrganAMNIST: left/right organ classes)
    split: str = "random"               # labeled/unlabeled split per client: random (ADD) | stratified
    holdout: str = "official"           # official: the dataset's train/test (ADD) | resplit: pool all splits
                                        # and re-divide (FedLabel paper: 80/5/15 with val 0.05, test 0.15)
    val_ratio: float = 0.0              # validation share (official: of the training set); 0 = none
    test_ratio: float = 0.15            # test share of the pooled data (resplit only)


@dataclass
class ModelConfig:
    name: str = "resnet18"              # simple_cnn | resnet18 | resnet34 | <registered>
    norm: str = "gn"                    # bn | gn (GroupNorm is more robust to non-IID data)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Client-side training protocol / pseudo-labeling strategy."""
    name: str = "fedlabel"              # supervised | pseudolabel | fixmatch | fedlabel | <registered>
    local_epochs: int = 1
    local_steps: Optional[int] = None   # if set, overrides local_epochs
    optimizer: str = "sgd"              # sgd | adam
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    prox_mu: float = 0.0                # > 0 adds a FedProx proximal term to any strategy
    amp: bool = False                   # legacy switch: true = precision fp16
    precision: str = "fp32"             # fp32 | bf16 (fast on Ampere+ GPUs, no loss scaling) | fp16
    channels_last: bool = False         # NHWC memory layout for convolutions (CUDA only)
    cudnn_benchmark: bool = False       # let cuDNN pick the fastest conv kernels per input shape (CUDA only)
    compile: Optional[str] = None       # torch.compile the training forward: default | reduce-overhead |
                                        # max-autotune-no-cudagraphs (None / false = off; needs Triton)
    grad_clip: Optional[float] = 5.0    # max gradient L2 norm (None disables)
    params: Dict[str, Any] = field(default_factory=dict)  # strategy-specific kwargs


@dataclass
class ServerConfig:
    aggregator: str = "fedavg"          # fedavg | fedavgm | fedadam | fedmedian | <registered>
    rounds: int = 100
    num_clients: int = 100
    client_fraction: float = 0.1        # C: fraction of clients sampled per round
    eval_every: int = 1
    params: Dict[str, Any] = field(default_factory=dict)  # aggregator-specific kwargs


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    device: str = "auto"                # auto | cpu | cuda | cuda:0 ...
    output_dir: str = "./runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        return _from_dict(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[List[str] | Dict[str, Any]] = None
                  ) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = apply_overrides(raw, overrides)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _from_dict(cls, d: Dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in d:
            continue
        value = d[name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def apply_overrides(raw: Dict[str, Any], overrides: Optional[List[str] | Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """Apply ``key.sub=value`` overrides, e.g. ``strategy.params.tau=0.9``."""
    if not overrides:
        return raw
    raw = copy.deepcopy(raw)
    items = overrides.items() if isinstance(overrides, dict) else (
        (o.split("=", 1)[0], _parse_value(o.split("=", 1)[1])) for o in overrides)
    for key, value in items:
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return raw
