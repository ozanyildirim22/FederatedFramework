"""SSFL - a plug-and-play Semi-Supervised Federated Learning framework."""
from .client import Client, ClientUpdate
from .config import (DataConfig, ExperimentConfig, ModelConfig, ServerConfig,
                     StrategyConfig)
from .data import AugmentationFactory, DataManager, RawDataset
from .experiment import Experiment
from .losses import CompositeLoss, kl_consistency, masked_ce, supervised_ce
from .models import build_model
from .registry import AGGREGATORS, DATASETS, MODELS, PARTITIONERS, STRATEGIES, Registry
from .server import Aggregator, Server
from .strategies import Strategy

__version__ = "0.1.0"

__all__ = [
    "Experiment", "ExperimentConfig", "DataConfig", "ModelConfig", "StrategyConfig", "ServerConfig",
    "Client", "ClientUpdate", "Server", "Strategy", "Aggregator", "DataManager", "RawDataset",
    "AugmentationFactory", "CompositeLoss", "supervised_ce", "masked_ce", "kl_consistency",
    "build_model", "Registry", "DATASETS", "PARTITIONERS", "MODELS", "STRATEGIES", "AGGREGATORS",
]
