"""Augmentation Factory: weak / strong views of the same sample."""
from __future__ import annotations

from typing import Sequence

from torchvision import transforms as T


class AugmentationFactory:
    """Builds the transforms used across the framework.

    * ``weak``   - flip + padded random crop; used for confidence evaluation
                   and for the labeled (supervised) branch.
    * ``strong`` - weak + RandAugment + Cutout; used for the consistency loss.
    * ``test``   - normalisation only.
    """

    def __init__(self, image_size: int, mean: Sequence[float], std: Sequence[float],
                 strong_aug: str = "randaugment", randaugment_n: int = 2, randaugment_m: int = 10):
        self.image_size = image_size
        self.normalize = [T.ToTensor(), T.Normalize(mean, std)]
        pad = max(2, image_size // 8)
        self._weak = [T.RandomHorizontalFlip(),
                      T.RandomCrop(image_size, padding=pad, padding_mode="reflect")]
        self._strong_extra = []
        if strong_aug == "randaugment":
            self._strong_extra = [T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m)]
        elif strong_aug != "none":
            raise ValueError(f"Unknown strong_aug '{strong_aug}'")

    def weak(self) -> T.Compose:
        return T.Compose(self._weak + self.normalize)

    def strong(self) -> T.Compose:
        cutout = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.5, 2.0), value=0)
        return T.Compose(self._weak + self._strong_extra + self.normalize + [cutout])

    def test(self) -> T.Compose:
        return T.Compose(self.normalize)
