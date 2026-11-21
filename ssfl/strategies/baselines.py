"""Classic federated baselines: FedAvg and FedProx.

Both train on each client's *labeled* data only (the unlabeled part is ignored),
which is how they are used as baselines in semi-supervised FL papers. Use them
with the ``fedavg`` aggregator. With ``data.labeled_ratio=1.0`` they become the
fully supervised upper bound.

To combine FedProx with a semi-supervised strategy instead, set
``strategy.prox_mu`` on that strategy (e.g. ``fixmatch`` + ``prox_mu=0.01``).
"""
from __future__ import annotations

from ..registry import STRATEGIES
from .algorithms import Supervised


@STRATEGIES.register("fedavg")
class FedAvgStrategy(Supervised):
    """FedAvg (McMahan et al., AISTATS 2017): local SGD on labeled data, the server
    averages the updates weighted by dataset size (``server.aggregator: fedavg``)."""


@STRATEGIES.register("fedprox")
class FedProx(Supervised):
    """FedProx (Li et al., MLSys 2020): FedAvg plus a proximal term
    ``mu/2 * ||w - W_t||^2`` that keeps local models close to the global model
    under non-IID data.

    ``mu`` is the proximal coefficient. If ``strategy.prox_mu`` is also set
    in the config, the larger of the two is used.
    """

    def __init__(self, cfg, num_classes, mu: float = 0.01, **params):
        super().__init__(cfg, num_classes, **params)
        self.prox_mu = max(float(mu), self.prox_mu)
