"""Model zoo. Register new architectures with ``@MODELS.register("name")``.

A model builder receives ``num_classes``, ``in_channels``, ``norm`` and any
extra ``model.params`` from the config, and returns an ``nn.Module``.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .registry import MODELS


def make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        return nn.GroupNorm(min(32, channels // 2) if channels >= 4 else 1, channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{norm}'")


@MODELS.register("simple_cnn")
class SimpleCNN(nn.Module):
    """Small 4-layer CNN - fast baseline for 28x28 / 32x32 inputs."""

    def __init__(self, num_classes: int, in_channels: int = 3, norm: str = "gn", width: int = 64):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                 make_norm(norm, cout), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            block(in_channels, width), block(width, width), nn.MaxPool2d(2),
            block(width, 2 * width), block(2 * width, 2 * width), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(2 * width, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int, norm: str):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = make_norm(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = make_norm(norm, cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          make_norm(norm, cout))

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


@MODELS.register("resnet18")
class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool) - works for 28px and 32px inputs."""

    def __init__(self, num_classes: int, in_channels: int = 3, norm: str = "gn", width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 3, 1, 1, bias=False),
                                  make_norm(norm, width), nn.ReLU(inplace=True))
        layers, cin = [], width
        for i, (mult, stride) in enumerate([(1, 1), (2, 2), (4, 2), (8, 2)]):
            cout = width * mult
            layers += [BasicBlock(cin, cout, stride, norm), BasicBlock(cout, cout, 1, norm)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)
        # Zero-init the last norm of every residual branch (Goyal et al., 2017):
        # each block starts as identity -> much more stable early training.
        for m in self.modules():
            if isinstance(m, BasicBlock) and getattr(m.n2, "weight", None) is not None:
                nn.init.zeros_(m.n2.weight)

    def forward(self, x):
        x = self.pool(self.layers(self.stem(x))).flatten(1)
        return self.fc(x)


def build_model(cfg, num_classes: int, in_channels: int) -> nn.Module:
    return MODELS.build(cfg.name, num_classes=num_classes, in_channels=in_channels,
                        norm=cfg.norm, **cfg.params)
