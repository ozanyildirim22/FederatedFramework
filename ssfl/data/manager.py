"""Data Manager: ingestion, partitioning and per-client loader construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..registry import DATASETS, PARTITIONERS
from .augment import AugmentationFactory
from .datasets import RawDataset
from .partition import split_labeled_unlabeled


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img[..., 0] if img.shape[-1] == 1 else img)


class LabeledView(Dataset):
    """Returns ``(transform(x), y)``."""

    def __init__(self, raw: RawDataset, indices: np.ndarray, transform):
        self.raw, self.indices, self.transform = raw, np.asarray(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(_to_pil(self.raw.images[j])), int(self.raw.labels[j])


class UnlabeledView(Dataset):
    """Returns ``(weak(x), strong(x))``. Labels are never exposed to the client.

    The true label is kept only for diagnostics (pseudo-label accuracy) and is
    returned as a third element that strategies must not train on.
    """

    def __init__(self, raw: RawDataset, indices: np.ndarray, weak, strong):
        self.raw, self.indices = raw, np.asarray(indices)
        self.weak, self.strong = weak, strong

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = _to_pil(self.raw.images[j])
        return self.weak(img), self.strong(img), int(self.raw.labels[j])


@dataclass
class ClientData:
    client_id: int
    labeled: Optional[DataLoader]
    unlabeled: Optional[DataLoader]
    num_labeled: int
    num_unlabeled: int
    label_histogram: np.ndarray

    @property
    def num_samples(self) -> int:
        return self.num_labeled + self.num_unlabeled


class DataManager:
    """Loads a dataset, partitions it across clients and builds DataLoaders."""

    def __init__(self, cfg, num_clients: int, seed: int = 0):
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

        self.train, self.test = DATASETS.get(cfg.dataset)(cfg)
        self.num_classes = self.train.num_classes
        self.in_channels = self.train.in_channels
        self.aug = AugmentationFactory(self.train.image_size, self.train.mean, self.train.std,
                                       cfg.strong_aug, cfg.randaugment_n, cfg.randaugment_m)

        self.partitions: List[np.ndarray] = PARTITIONERS.get(cfg.partitioner)(
            self.train.labels, num_clients, self.rng, alpha=cfg.alpha,
            min_samples_per_client=cfg.min_samples_per_client)
        self.splits = [split_labeled_unlabeled(p, self.train.labels, cfg.labeled_ratio, self.rng)
                       for p in self.partitions]

    # -------------------------------------------------------------- loaders
    def _loader(self, ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=shuffle,
                          drop_last=shuffle and len(ds) > batch_size,
                          num_workers=self.cfg.num_workers,
                          pin_memory=torch.cuda.is_available())

    def client_data(self, cid: int) -> ClientData:
        lab_idx, unl_idx = self.splits[cid]
        labeled = (self._loader(LabeledView(self.train, lab_idx, self.aug.weak()),
                                self.cfg.batch_size, True) if len(lab_idx) else None)
        unlabeled = (self._loader(UnlabeledView(self.train, unl_idx, self.aug.weak(), self.aug.strong()),
                                  self.cfg.batch_size * self.cfg.unlabeled_batch_ratio, True)
                     if len(unl_idx) else None)
        hist = np.bincount(self.train.labels[self.partitions[cid]], minlength=self.num_classes)
        return ClientData(cid, labeled, unlabeled, len(lab_idx), len(unl_idx), hist)

    def test_loader(self, batch_size: int = 256) -> DataLoader:
        ds = LabeledView(self.test, np.arange(len(self.test)), self.aug.test())
        return self._loader(ds, batch_size, False)

    def summary(self) -> str:
        sizes = np.array([len(p) for p in self.partitions])
        n_lab = sum(len(s[0]) for s in self.splits)
        return (f"{self.cfg.dataset}: {len(self.train)} train / {len(self.test)} test, "
                f"{self.num_classes} classes | {self.num_clients} clients "
                f"({self.cfg.partitioner}, alpha={self.cfg.alpha}) samples/client "
                f"min={sizes.min()} mean={sizes.mean():.0f} max={sizes.max()} | "
                f"labeled={n_lab} ({100 * n_lab / sizes.sum():.1f}%)")
