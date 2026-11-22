"""Fast end-to-end tests on the synthetic dataset (CPU, no downloads)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ssfl
from ssfl import AGGREGATORS, STRATEGIES, Experiment, ExperimentConfig, Strategy
from ssfl.data.partition import dirichlet_partition, split_labeled_unlabeled


def tiny_cfg(**over) -> ExperimentConfig:
    raw = {
        "name": "test", "device": "cpu", "output_dir": str(over.pop("out")),
        "data": {"dataset": "synthetic", "batch_size": 8, "unlabeled_batch_ratio": 2,
                 "labeled_ratio": 0.2},
        "model": {"name": "simple_cnn", "params": {"width": 8}},
        "strategy": {"name": "fedlabel", "local_steps": 2, "params": {"tau": 0.5}},
        "server": {"rounds": 2, "num_clients": 4, "client_fraction": 0.5},
    }
    from ssfl.config import apply_overrides
    return ExperimentConfig.from_dict(apply_overrides(raw, over))


def test_dirichlet_partition_covers_all_samples():
    labels = np.repeat(np.arange(10), 100)
    parts = dirichlet_partition(labels, 10, np.random.default_rng(0), alpha=0.1)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(labels) and len(np.unique(allidx)) == len(labels)
    lab, unl = split_labeled_unlabeled(parts[0], labels, 0.2, np.random.default_rng(0))
    assert len(np.intersect1d(lab, unl)) == 0 and len(lab) + len(unl) == len(parts[0])


@pytest.mark.parametrize("strategy", ["supervised", "pseudolabel", "fixmatch", "fedlabel"])
@pytest.mark.parametrize("aggregator", ["fedavg", "fedavgm", "fedadam", "fedmedian"])
def test_all_combinations_run(tmp_path, strategy, aggregator):
    cfg = tiny_cfg(out=tmp_path, **{"strategy.name": strategy, "server.aggregator": aggregator,
                                    "strategy.prox_mu": 0.01})
    if strategy == "supervised":
        cfg.strategy.params = {}
    hist = Experiment(cfg, verbose=False).run()
    assert len(hist) == 2 and 0.0 <= hist[-1]["test_acc"] <= 1.0
    assert list(tmp_path.glob("*/metrics.csv"))


def test_fedavg_is_weighted_mean():
    from ssfl.client import ClientUpdate
    g = {"w": torch.zeros(2)}
    ups = [ClientUpdate(0, {"w": torch.tensor([1.0, 1.0])}, {}, 30, 3),
           ClientUpdate(1, {"w": torch.tensor([4.0, 4.0])}, {}, 10, 1)]
    new = AGGREGATORS.build("fedavg").aggregate(g, ups)
    assert torch.allclose(new["w"], torch.tensor([1.75, 1.75]))


def test_custom_strategy_plugs_in(tmp_path):
    @STRATEGIES.register("test_custom_sup")
    class Custom(Strategy):
        uses_unlabeled = False

        def compute_loss(self, model, labeled, unlabeled):
            x, y = labeled
            return self.loss(sup=F.cross_entropy(model(x), y))

    cfg = tiny_cfg(out=tmp_path, **{"strategy.name": "test_custom_sup"})
    cfg.strategy.params = {}
    assert len(Experiment(cfg, verbose=False).run()) == 2


def test_fedlabel_selection_ablation(tmp_path):
    for sel in ("local", "global", "adaptive"):
        cfg = tiny_cfg(out=tmp_path, **{"strategy.params.selection": sel,
                                        "strategy.params.confidence": "max_prob"})
        hist = Experiment(cfg, verbose=False).run()
        assert "train_local_selected" in hist[-1]
        if sel == "local":
            assert hist[-1]["train_local_selected"] == 1.0
        if sel == "global":
            assert hist[-1]["train_local_selected"] == 0.0


def test_gui_config_roundtrip_and_build():
    pytest.importorskip("gradio")
    from ssfl import gui
    for preset in gui.list_presets():
        cfg = gui.load_preset(preset)
        back = gui.values_to_cfg(*gui.cfg_to_values(cfg), algorithm=cfg.strategy.name,
                                 params=cfg.strategy.params)
        assert back.to_dict() == cfg.to_dict()
        multi = gui.build_configs(gui.cfg_to_values(cfg), ["fedavg", "fedmatch"], {"fedmatch": {"tau": 0.7}})
        assert [c.strategy.name for c in multi] == ["fedavg", "fedmatch"]
        assert multi[1].strategy.params == {"tau": 0.7} and multi[0].name != multi[1].name
    assert {"beta", "lambda0", "sup_epochs"} <= set(gui.default_params(STRATEGIES.get("fedlabel")))
    assert "weight_by" in gui.default_params(AGGREGATORS.get("fedadam"))
    gui.build_app()
    assert "Ready" in gui.render_status(gui.RunController().snapshot())
    assert gui.to_latex(gui.thesis_table([])).startswith("%")


# ------------------------------------------------------------ new algorithms
@pytest.mark.parametrize("strategy", ["fedavg", "fedprox", "fedmatch", "fedtrinet"])
def test_baselines_and_ssfl_algorithms_run(tmp_path, strategy):
    over = {"strategy.name": strategy, "server.rounds": 3}
    cfg = tiny_cfg(out=tmp_path, **over)
    cfg.strategy.params = {"pretrain_rounds": 1, "finetune_steps": 1} if strategy == "fedtrinet" else {}
    hist = Experiment(cfg, verbose=False).run()
    assert len(hist) == 3 and 0.0 <= hist[-1]["test_acc"] <= 1.0
    if strategy == "fedtrinet":
        assert "train_threshold" in hist[-1] and "train_loss_unsup" in hist[-1]
        assert "train_threshold" not in hist[0]          # phase I = labeled-only FedAvg
    if strategy == "fedmatch":
        assert hist[-1]["train_num_helpers"] > 0          # helpers were sent from round 2 on


def test_fedprox_uses_mu():
    cfg = tiny_cfg(out="unused")
    s = STRATEGIES.build("fedprox", cfg.strategy, 10, mu=0.3)
    assert s.prox_mu == 0.3 and not s.uses_unlabeled


def test_fedmatch_decomposition_routes_gradients():
    from ssfl.strategies import DecomposedModel
    base = torch.nn.Linear(4, 3)
    m = DecomposedModel(base)
    x = torch.randn(5, 4)
    m(x, "sup").sum().backward()
    assert base.weight.grad is not None and m.psi[0].grad is None
    base.zero_grad()
    m(x, "unsup").sum().backward()
    assert base.weight.grad is None and m.psi[0].grad is not None
    merged = m.merged_state(m.state_dict())
    assert torch.allclose(merged["weight"], base.weight + m.psi[0])


def test_fedtrinet_threshold_schedule_and_split():
    cfg = tiny_cfg(out="unused")
    s = STRATEGIES.build("fedtrinet", cfg.strategy, 10)
    assert s.threshold_scale(0) == 1.0 and s.threshold_scale(9) == 1.0
    assert s.threshold_scale(10) == pytest.approx(0.8) and s.threshold_scale(40) == 0.5
    model = ssfl.build_model(cfg.model, 10, 3)
    keys = s._resolve_global_keys(model)
    assert keys and all(k.startswith("features.") for k in keys)
    assert not any(k.startswith("classifier") for k in keys)


def test_thesis_grid_expands():
    from ssfl.grid import expand_grid
    runs = expand_grid("configs/grids/thesis.yaml")
    names = [n for n, _ in runs]
    assert len(names) == len(set(names)) == 131
    cfg = dict(runs)["organamnist_fedtrinet_l0p05_a0p1"]
    assert cfg.strategy.name == "fedtrinet" and cfg.data.labeled_ratio == 0.05
    assert "lambda0" not in cfg.strategy.params           # preset params are replaced
    assert dict(runs)["cifar10_fedtrinet_l0p2_a1"].strategy.params["pretrain_rounds"] == 200
    for name, cfg in runs:                                 # every param set is accepted
        STRATEGIES.build(cfg.strategy.name, cfg.strategy, 10, **cfg.strategy.params)
