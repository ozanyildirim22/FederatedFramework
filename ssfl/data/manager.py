"""Data Manager: ingestion, partitioning and per-client loader construction."""
from __future__ import annotations

import math
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


#: Images a client must load per round (labeled + weak + strong views) before
#: worker processes pay for their start-up, which happens for every client in
#: every round.
AUTO_WORKERS_MIN_IMAGES = 1000


def auto_num_workers(splits, batch_size: int, mu: int, local_steps: Optional[int], local_epochs: int) -> int:
    """DataLoader workers for ``data.num_workers = -1``.

    * Windows: 0. Workers are started with ``spawn``, which re-imports PyTorch and
      costs seconds per worker, more than the augmentation they save.
    * Small rounds (e.g. the synthetic quick demo): 0. Starting workers for every
      client in every round costs more than the augmentation they save.
    * Otherwise (Colab / Linux with real datasets): up to 4, limited by the CPU count.
      Augmentation (RandAugment on 2 views of every unlabeled image) is then done in
      parallel with the GPU instead of before it.
    """
    import os
    import sys

    cpus = os.cpu_count() or 1
    if sys.platform == "win32" or cpus < 2:
        return 0
    per_round = []
    for lab, unl in splits:
        n_lab = math.ceil(len(lab) / batch_size) if len(lab) else 0
        n_unl = math.ceil(len(unl) / (batch_size * mu)) if len(unl) else 0
        steps = local_steps or local_epochs * max(n_lab, n_unl, 1)
        per_round.append(steps * batch_size * (1 + 2 * mu))
    return min(4, cpus) if float(np.median(per_round)) >= AUTO_WORKERS_MIN_IMAGES else 0


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

    def __init__(self, cfg, num_clients: int, seed: int = 0, raw=None):
        """``raw`` = an already loaded ``(train, test)`` pair (skips loading, e.g. a GUI cache)."""
        self.cfg = cfg
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

        self.train, self.test = raw if raw is not None else DATASETS.get(cfg.dataset)(cfg)
        self.num_classes = self.train.num_classes
        self.in_channels = self.train.in_channels
        hflip = getattr(cfg, "hflip", None)
        self.hflip = self.train.hflip_safe if hflip is None else bool(hflip)   # None = dataset decides
        self.aug = AugmentationFactory(self.train.image_size, self.train.mean, self.train.std,
                                       cfg.strong_aug, cfg.randaugment_n, cfg.randaugment_m,
                                       cutout=getattr(cfg, "cutout", False), hflip=self.hflip)

        self.partitions: List[np.ndarray] = PARTITIONERS.get(cfg.partitioner)(
            self.train.labels, num_clients, self.rng, alpha=cfg.alpha,
            min_samples_per_client=cfg.min_samples_per_client)
        self.splits = [split_labeled_unlabeled(p, self.train.labels, cfg.labeled_ratio, self.rng,
                                               mode=getattr(cfg, "split", "random"))
                       for p in self.partitions]

        # data.num_workers = -1 means "auto"; resolve_workers() picks the number
        self.num_workers = max(0, int(cfg.num_workers))
        if getattr(cfg, "augment_device", "cpu") not in ("cpu", "gpu"):
            raise ValueError(f"Unknown augment_device '{cfg.augment_device}' (cpu | gpu)")
        self.device = torch.device("cpu")
        self._gpu = None                              # (DeviceDataset train, test, GPUAugment), lazy

    @property
    def on_gpu(self) -> bool:
        return getattr(self.cfg, "augment_device", "cpu") == "gpu"

    def set_device(self, device: torch.device) -> None:
        """Training device; with ``augment_device: gpu`` the data and augmentation live there."""
        self.device = torch.device(device)

    def _gpu_parts(self):
        if self._gpu is None:
            from .gpu import DeviceDataset, GPUAugment
            aug = GPUAugment(self.train.image_size, self.train.mean, self.train.std, self.cfg.strong_aug,
                             self.cfg.randaugment_n, self.cfg.randaugment_m, device=self.device,
                             cutout=getattr(self.cfg, "cutout", False), hflip=self.hflip)
            self._gpu = (DeviceDataset(self.train, self.device), DeviceDataset(self.test, self.device), aug)
        return self._gpu

    # -------------------------------------------------------------- workers
    def resolve_workers(self, local_steps: Optional[int], local_epochs: int) -> int:
        """Set the DataLoader worker count; ``data.num_workers = -1`` chooses it automatically."""
        if self.on_gpu:                               # no DataLoader, nothing to parallelise
            self.num_workers = 0
        elif self.cfg.num_workers >= 0:
            self.num_workers = int(self.cfg.num_workers)
        else:
            self.num_workers = auto_num_workers(self.splits, self.cfg.batch_size,
                                                self.cfg.unlabeled_batch_ratio, local_steps, local_epochs)
        return self.num_workers

    # -------------------------------------------------------------- loaders
    def _loader(self, ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=min(batch_size, max(1, len(ds))), shuffle=shuffle,
                          drop_last=shuffle and len(ds) > batch_size,
                          num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def client_data(self, cid: int) -> ClientData:
        lab_idx, unl_idx = self.splits[cid]
        hist = np.bincount(self.train.labels[self.partitions[cid]], minlength=self.num_classes)
        if self.on_gpu:
            from .gpu import TensorBatches
            train, _, aug = self._gpu_parts()
            b = self.cfg.batch_size
            labeled = TensorBatches(train, lab_idx, b, True, "labeled", aug) if len(lab_idx) else None
            unlabeled = (TensorBatches(train, unl_idx, b * self.cfg.unlabeled_batch_ratio, True, "unlabeled", aug)
                         if len(unl_idx) else None)
            return ClientData(cid, labeled, unlabeled, len(lab_idx), len(unl_idx), hist)
        labeled = (self._loader(LabeledView(self.train, lab_idx, self.aug.weak()),
                                self.cfg.batch_size, True) if len(lab_idx) else None)
        unlabeled = (self._loader(UnlabeledView(self.train, unl_idx, self.aug.weak(), self.aug.strong()),
                                  self.cfg.batch_size * self.cfg.unlabeled_batch_ratio, True)
                     if len(unl_idx) else None)
        return ClientData(cid, labeled, unlabeled, len(lab_idx), len(unl_idx), hist)

    def test_loader(self, batch_size: int = 256) -> DataLoader:
        if self.on_gpu:
            from .gpu import TensorBatches
            _, test, aug = self._gpu_parts()
            return TensorBatches(test, np.arange(len(self.test)), batch_size, False, "test", aug)
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
