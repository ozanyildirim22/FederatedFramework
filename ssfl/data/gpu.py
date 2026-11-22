"""GPU data pipeline (``data.augment_device: gpu``).

The CPU pipeline decodes every image with PIL and augments it one at a time in
DataLoader workers. With a fast GPU that becomes the bottleneck. Here the whole
dataset lives on the training device as one ``uint8`` tensor (OrganAMNIST
27 MB, CIFAR-10 150 MB) and every batch is augmented with batched tensor ops:

* **weak**   - random horizontal flip (unless ``data.hflip`` / the dataset says no) + random crop with reflect padding
               (``max(2, size // 8)`` pixels), as ``AugmentationFactory.weak``;
* **strong** - weak + RandAugment (+ Cutout if ``data.cutout``), as ``AugmentationFactory.strong``.
  RandAugment uses torchvision's own op table and ``_apply_op`` (same operations
  and magnitudes). Each sample draws its own ops and signs, and samples that drew
  the same op are transformed together. Cutout follows ``RandomErasing(p=1,
  scale=(0.02, 0.15), ratio=(0.5, 2), value=0)`` after normalisation.

The views are statistically the same as the CPU ones, but not bit-identical
(tensor vs PIL backends, and the random draws are made in batches). Use the CPU
pipeline when exact reproducibility of earlier runs matters.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence

import torch
import torchvision.transforms.v2.functional as F2
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment, _apply_op

from .datasets import RawDataset


def _reflect(idx: torch.Tensor, size: int) -> torch.Tensor:
    """Reflect-padding index map (no edge repeat), like ``padding_mode="reflect"``."""
    idx = idx.abs()
    return torch.where(idx >= size, 2 * (size - 1) - idx, idx)


class GPUAugment:
    def __init__(self, image_size: int, mean: Sequence[float], std: Sequence[float],
                 strong_aug: str = "randaugment", randaugment_n: int = 1, randaugment_m: int = 10,
                 device: torch.device = torch.device("cpu"), cutout: bool = False, hflip: bool = True):
        self.use_cutout, self.hflip = cutout, hflip
        if strong_aug not in ("randaugment", "none"):
            raise ValueError(f"Unknown strong_aug '{strong_aug}'")
        self.size, self.pad = image_size, max(2, image_size // 8)
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.strong_aug, self.n, self.m = strong_aug, randaugment_n, randaugment_m
        ra = RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)
        space = ra._augmentation_space(ra.num_magnitude_bins, (image_size, image_size))
        self.ops: List[tuple] = []          # (name, magnitude, signed)
        for name, (mags, signed) in space.items():
            mag = float(mags[randaugment_m].item()) if mags.ndim > 0 else 0.0
            self.ops.append((name, mag, bool(signed)))

    # ------------------------------------------------------------ weak
    def weak(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 (B, C, H, W) -> uint8, random flip + reflect-padded random crop per sample."""
        b, _, h, w = x.shape
        if self.hflip:
            flip = torch.rand(b, device=x.device) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        p = self.pad
        oy = torch.randint(0, 2 * p + 1, (b,), device=x.device) - p
        ox = torch.randint(0, 2 * p + 1, (b,), device=x.device) - p
        rows = _reflect(oy[:, None] + torch.arange(h, device=x.device), h)        # (B, H)
        cols = _reflect(ox[:, None] + torch.arange(w, device=x.device), w)        # (B, W)
        bi = torch.arange(b, device=x.device)[:, None, None]
        out = x.permute(0, 2, 3, 1)[bi, rows[:, :, None], cols[:, None, :]]      # (B, H, W, C)
        return out.permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------- RandAugment
    def randaugment(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 (B, C, H, W) -> uint8; every sample gets its own ``n`` random ops."""
        b = x.shape[0]
        for _ in range(self.n):
            # draws on the CPU: grouping needs the counts, and reading them from the GPU
            # would force a synchronisation for every op group
            op_idx = torch.randint(len(self.ops), (b,))
            neg = torch.randint(2, (b,)).bool()
            out = x.clone()
            for k, (name, mag, signed) in enumerate(self.ops):
                if name == "Identity":
                    continue
                for sign in ((False, True) if signed else (False,)):
                    sel = ((op_idx == k) & (neg == sign)) if signed else (op_idx == k)
                    idx = sel.nonzero().flatten()
                    if idx.numel() == 0:
                        continue
                    idx = idx.to(x.device, non_blocking=True)
                    if name == "Equalize":            # same result, batched (v1 loops per image)
                        out[idx] = F2.equalize(x[idx])
                    else:
                        out[idx] = _apply_op(x[idx], name, -mag if sign else mag,
                                             interpolation=InterpolationMode.NEAREST, fill=None)
            x = out
        return x

    # ---------------------------------------------------------- normalise
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float().div_(255.0) - self.mean) / self.std

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """RandomErasing(p=1, scale=(0.02, 0.15), ratio=(0.5, 2), value=0) per sample."""
        b, _, h, w = x.shape
        dev = x.device
        area = h * w * torch.empty(b, device=dev).uniform_(0.02, 0.15)
        ratio = torch.exp(torch.empty(b, device=dev).uniform_(math.log(0.5), math.log(2.0)))
        eh = torch.sqrt(area * ratio).round().clamp(1, h - 1)
        ew = torch.sqrt(area / ratio).round().clamp(1, w - 1)
        top = (torch.rand(b, device=dev) * (h - eh + 1)).floor()
        left = (torch.rand(b, device=dev) * (w - ew + 1)).floor()
        r = torch.arange(h, device=dev).view(1, h, 1)
        c = torch.arange(w, device=dev).view(1, 1, w)
        mask = ((r >= top.view(-1, 1, 1)) & (r < (top + eh).view(-1, 1, 1))
                & (c >= left.view(-1, 1, 1)) & (c < (left + ew).view(-1, 1, 1)))
        return x.masked_fill(mask.unsqueeze(1), 0.0)

    # ---------------------------------------------------------- views
    def weak_view(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.weak(x))

    def strong_view(self, x: torch.Tensor) -> torch.Tensor:
        x = self.weak(x)
        if self.strong_aug == "randaugment":
            x = self.randaugment(x)
        x = self.normalize(x)
        return self.cutout(x) if self.use_cutout else x

    def test_view(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)


class DeviceDataset:
    """A RawDataset copied to the device once: uint8 images (N, C, H, W) + labels."""

    def __init__(self, raw: RawDataset, device: torch.device):
        self.images = torch.from_numpy(raw.images).to(device).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(raw.labels).to(device)

    def __len__(self) -> int:
        return len(self.labels)


class TensorBatches:
    """DataLoader stand-in that builds augmented batches on the device.

    ``kind``: ``"labeled"`` -> ``(weak, y)``; ``"unlabeled"`` -> ``(weak, strong, true_y)``
    (true labels only for diagnostics, as in the CPU pipeline); ``"test"`` -> ``(x, y)``.
    Batch count and ``drop_last`` follow ``DataManager._loader``.
    """

    num_workers = 0
    pin_memory = False

    def __init__(self, data: DeviceDataset, indices, batch_size: int, shuffle: bool, kind: str,
                 aug: GPUAugment):
        self.data, self.aug, self.kind = data, aug, kind
        self.indices = torch.as_tensor(indices, dtype=torch.long, device=data.labels.device)
        n = len(self.indices)
        self.batch_size = min(batch_size, max(1, n))
        self.shuffle = shuffle
        self.drop_last = shuffle and n > batch_size
        self.dataset = self                       # len(loader.dataset) works like a DataLoader

    def __len__(self) -> int:
        n = len(self.indices)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def _order(self) -> torch.Tensor:
        if not self.shuffle:
            return self.indices
        return self.indices[torch.randperm(len(self.indices), device=self.indices.device)]

    def _batch(self, idx: torch.Tensor):
        x, y = self.data.images[idx], self.data.labels[idx]
        if self.kind == "labeled":
            return self.aug.weak_view(x), y
        if self.kind == "unlabeled":
            return self.aug.weak_view(x), self.aug.strong_view(x), y
        return self.aug.test_view(x), y

    def __iter__(self) -> Iterator:
        order, bs = self._order(), self.batch_size
        for i in range(len(self)):
            yield self._batch(order[i * bs:(i + 1) * bs])

    def weak_only(self, batch_size: int = 256) -> Iterator[torch.Tensor]:
        """Weak views of all samples in order (FedTriNet's threshold pass)."""
        for i in range(0, len(self.indices), batch_size):
            yield self.aug.weak_view(self.data.images[self.indices[i:i + batch_size]])
