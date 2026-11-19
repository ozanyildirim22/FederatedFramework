"""Client-Side Execution Module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from .data.manager import ClientData

StateDict = Dict[str, torch.Tensor]


@dataclass
class ClientUpdate:
    client_id: int
    delta: StateDict            # local - global for floating-point tensors
    buffers: StateDict          # non-float buffers (e.g. BN num_batches_tracked), sent as-is
    num_samples: int
    num_labeled: int
    metrics: Dict[str, float] = field(default_factory=dict)


class Client:
    """A simulated edge device. Holds only its private data; never shares it."""

    def __init__(self, data: ClientData):
        self.data = data
        self.id = data.client_id

    @property
    def num_samples(self) -> int:
        return self.data.num_samples

    def fit(self, model: nn.Module, global_state: StateDict, strategy, device: torch.device
            ) -> ClientUpdate:
        # 1) initialise the local architecture with the received W_t
        model.load_state_dict(global_state)
        model.to(device)
        # 2) local execution (training protocol + pseudo-labeling engine + loss)
        metrics = strategy.local_train(model, self, device)
        # 3) compute the update vector delta_w_k (only numbers leave the device)
        delta, buffers = {}, {}
        for k, v in model.state_dict().items():
            v = v.detach().cpu()
            if v.is_floating_point():
                delta[k] = v - global_state[k]
            else:
                buffers[k] = v.clone()
        return ClientUpdate(self.id, delta, buffers, self.num_samples, self.data.num_labeled, metrics)
