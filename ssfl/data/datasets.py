"""Dataset loaders.

Every loader returns a :class:`RawDataset`: uint8 images in ``(N, H, W, C)``
layout plus integer labels. Keeping raw arrays (instead of torch Datasets)
lets the partitioner and the augmentation factory stay dataset-agnostic.

Register a new dataset with::

    @DATASETS.register("my_dataset")
    def load_my_dataset(cfg: DataConfig) -> tuple[RawDataset, RawDataset]:
        ...  # return (train, test)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ..registry import DATASETS


@dataclass
class RawDataset:
    images: np.ndarray            # uint8, (N, H, W, C)
    labels: np.ndarray            # int64, (N,)
    num_classes: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    #: False when a horizontal flip changes the meaning of an image, e.g. OrganAMNIST's
    #: left / right organ classes (femur, kidney, lung): a flipped left kidney looks like a
    #: right kidney but would keep the label "left". ``data.hflip`` overrides it.
    hflip_safe: bool = True

    def __post_init__(self) -> None:
        if self.images.ndim == 3:                       # grayscale -> add channel dim
            self.images = self.images[..., None]
        self.images = np.ascontiguousarray(self.images, dtype=np.uint8)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def in_channels(self) -> int:
        return self.images.shape[-1]

    @property
    def image_size(self) -> int:
        return self.images.shape[1]


def _channel_stats(images: np.ndarray) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    x = images.reshape(-1, images.shape[-1] if images.ndim == 4 else 1).astype(np.float64) / 255.0
    return tuple(x.mean(0).round(4).tolist()), tuple(x.std(0).round(4).tolist())


@DATASETS.register("cifar10")
def load_cifar10(cfg) -> Tuple[RawDataset, RawDataset]:
    from torchvision.datasets import CIFAR10

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    out = []
    for train in (True, False):
        ds = CIFAR10(cfg.root, train=train, download=True)
        out.append(RawDataset(ds.data, np.array(ds.targets), 10, mean, std))
    return out[0], out[1]


def _load_medmnist(cls_name: str, cfg, hflip_safe: bool = True) -> Tuple[RawDataset, RawDataset]:
    import medmnist

    cls = getattr(medmnist, cls_name)
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    kwargs = dict(root=str(root), download=True)
    if cfg.image_size != 28:
        kwargs["size"] = cfg.image_size
    train = cls(split="train", **kwargs)
    test = cls(split="test", **kwargs)
    num_classes = len(train.info["label"])
    mean, std = _channel_stats(train.imgs)
    return (RawDataset(train.imgs, train.labels, num_classes, mean, std, hflip_safe),
            RawDataset(test.imgs, test.labels, num_classes, mean, std, hflip_safe))


@DATASETS.register("organamnist")
def load_organamnist(cfg):
    # classes include femur / kidney / lung, each as left and right: no horizontal flips
    return _load_medmnist("OrganAMNIST", cfg, hflip_safe=False)


@DATASETS.register("pathmnist")
def load_pathmnist(cfg):
    return _load_medmnist("PathMNIST", cfg)


@DATASETS.register("synthetic")
def load_synthetic(cfg, n_train: int = 2000, n_test: int = 400, num_classes: int = 10
                   ) -> Tuple[RawDataset, RawDataset]:
    """Small random dataset with class-dependent colour, for tests / debugging."""
    rng = np.random.default_rng(0)
    size = 32

    def make(n):
        y = rng.integers(0, num_classes, n)
        base = (y[:, None] * (255 // num_classes)).astype(np.float64)
        x = rng.normal(0, 30, (n, size, size, 3)) + base[:, None, None, :]
        return np.clip(x, 0, 255).astype(np.uint8), y

    (xtr, ytr), (xte, yte) = make(n_train), make(n_test)
    mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    return (RawDataset(xtr, ytr, num_classes, mean, std),
            RawDataset(xte, yte, num_classes, mean, std))
