"""Option-level tests for the newer algorithms (synthetic data, CPU)."""
import math
from pathlib import Path

import pytest
import torch

from ssfl import STRATEGIES, Experiment, ExperimentConfig
from ssfl.config import apply_overrides


def cfg_for(tmp_path, **over) -> ExperimentConfig:
    raw = {
        "name": "variant", "device": "cpu", "output_dir": str(tmp_path),
        "data": {"dataset": "synthetic", "batch_size": 8, "unlabeled_batch_ratio": 2, "labeled_ratio": 0.2},
        "model": {"name": "simple_cnn", "params": {"width": 8}},
        "strategy": {"name": "fedmatch", "local_steps": 4, "params": {}},
        "server": {"rounds": 3, "num_clients": 4, "client_fraction": 0.5},
    }
    return ExperimentConfig.from_dict(apply_overrides(raw, over))


def run(cfg):
    hist = Experiment(cfg, verbose=False).run()
    for rec in hist:
        for k, v in rec.items():
            assert not (isinstance(v, float) and math.isnan(v)), f"{k} is NaN in round {rec['round']}"
    return hist


FEDMATCH_VARIANTS = [
    {"decompose": False},
    {"reg_reduction": "sum", "lambda_l2": 1e-4, "lambda_l1": 1e-6},
    {"num_helpers": 0},
    {"helper_every": 1, "num_helpers": 1},
    {"tau": 0.0},
]


@pytest.mark.parametrize("params", FEDMATCH_VARIANTS, ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()))
def test_fedmatch_variants(tmp_path, params):
    hist = run(cfg_for(tmp_path, **{"strategy.params": params}))
    last = hist[-1]
    if params.get("num_helpers") == 0:
        assert last["train_num_helpers"] == 0 and "train_loss_iccs" not in last
    else:
        assert last["train_num_helpers"] > 0
    if params.get("decompose") is False:
        assert "train_loss_l1" not in last
    if params.get("tau") == 0.0:
        assert last["train_mask_rate"] == 1.0


FEDTRINET_VARIANTS = [
    {"unsup_view": "strong"},
    {"store_dtype": "float32"},
    {"global_layers": ["features.0."]},
    {"finetune_steps": 0, "alpha": 0.0},
]


@pytest.mark.parametrize("params", FEDTRINET_VARIANTS, ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()))
def test_fedtrinet_variants(tmp_path, params):
    hist = run(cfg_for(tmp_path, **{"strategy.name": "fedtrinet",
                                    "strategy.params": {"pretrain_rounds": 1, **params}}))
    assert "train_threshold" in hist[-1]
    if params.get("alpha") == 0.0:                       # threshold 0 -> every sample is kept
        assert hist[-1]["train_mask_rate"] == 1.0


@pytest.mark.parametrize("strategy", ["fedavg", "fedprox", "fedmatch", "fedtrinet"])
@pytest.mark.parametrize("aggregator", ["fedavgm", "fedadam", "fedmedian"])
def test_new_strategies_with_other_aggregators(tmp_path, strategy, aggregator):
    params = {"pretrain_rounds": 1} if strategy == "fedtrinet" else {}
    run(cfg_for(tmp_path, **{"strategy.name": strategy, "strategy.params": params,
                             "server.aggregator": aggregator}))


@pytest.mark.parametrize("strategy", ["fedmatch", "fedtrinet", "fedlabel", "fixmatch"])
def test_batchnorm_models(tmp_path, strategy):
    params = {"pretrain_rounds": 1} if strategy == "fedtrinet" else {}
    run(cfg_for(tmp_path, **{"strategy.name": strategy, "strategy.params": params, "model.norm": "bn"}))


@pytest.mark.parametrize("model", ["resnet18", "resnet34"])
@pytest.mark.parametrize("strategy", ["fedmatch", "fedtrinet", "fedlabel"])
def test_resnets(tmp_path, strategy, model):
    params = {"pretrain_rounds": 1} if strategy == "fedtrinet" else {}
    run(cfg_for(tmp_path, **{"strategy.name": strategy, "strategy.params": params, "model.name": model,
                             "model.params": {"width": 8}, "server.rounds": 2}))


def test_resnet_depths():
    from types import SimpleNamespace
    from ssfl.models import build_model
    for name, n_blocks in [("resnet18", 8), ("resnet34", 16)]:
        m = build_model(SimpleNamespace(name=name, norm="gn", params={}), num_classes=10, in_channels=3)
        assert len(m.layers) == n_blocks


def test_fedprox_mu_and_config_prox():
    cfg = cfg_for("unused", **{"strategy.prox_mu": 0.5})
    assert STRATEGIES.build("fedprox", cfg.strategy, 10, mu=0.01).prox_mu == 0.5   # larger wins
    assert STRATEGIES.build("fedprox", cfg.strategy, 10, mu=0.9).prox_mu == 0.9


def test_fedtrinet_combined_model_and_local_memory(tmp_path):
    """Phase II: the trained model starts as G[shallow] + l^k[deep]."""
    exp = Experiment(cfg_for(tmp_path, **{"strategy.name": "fedtrinet",
                                          "strategy.params": {"pretrain_rounds": 1}}), verbose=False)
    s = exp.strategy
    exp.run_round(1)                                     # phase I: every selected client stores l^k
    assert s._local_states
    cid, local = next(iter(s._local_states.items()))
    client = exp.clients[cid]
    s.round = 2
    model = exp.model
    g = {k: v.clone() for k, v in exp.server.global_state.items()}
    model.load_state_dict(g)
    s.device = torch.device("cpu")
    s.on_round_start(model, client)
    state = model.state_dict()
    for k in state:
        expect = g[k] if k in s._global_keys else local[k].to(state[k].dtype)
        assert torch.allclose(state[k].float(), expect.float(), atol=1e-3), k
    assert all(v.dtype == torch.float16 for v in local.values() if v.is_floating_point())


def test_fedmatch_helpers_are_nearest_neighbours(tmp_path):
    exp = Experiment(cfg_for(tmp_path, **{"server.num_clients": 4, "server.client_fraction": 1.0,
                                          "strategy.params": {"num_helpers": 1}}), verbose=False)
    s = exp.strategy
    s._embeddings = {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([0.1, 0.0]),
                     2: torch.tensor([5.0, 5.0]), 3: torch.tensor([5.1, 5.0])}
    s._recent = {i: {} for i in range(4)}
    s.on_server_round_start(exp.server, exp.clients, rnd=1)
    assert s._assign == {0: [1], 1: [0], 2: [3], 3: [2]}


def test_grid_only_resume_and_overrides(tmp_path, capsys):
    from ssfl.__main__ import main
    out = str(tmp_path / "gridruns")
    args = ["--grid", "configs/grids/demo.yaml", "--only", "fedavg", "l0p1",
            "--set", f"output_dir={out}", "server.rounds=1", "device=cpu"]
    main(args)
    first = capsys.readouterr().out
    assert first.count("[grid") == 1 and ": run" in first          # --only keeps one run
    done = list((tmp_path / "gridruns").glob("synthetic_fedavg_l0p1_a0p5_demo_*/summary.json"))
    assert len(done) == 1
    cfg = ExperimentConfig.from_yaml(done[0].parent / "config.yaml")
    assert cfg.server.rounds == 1 and cfg.output_dir == out          # --set reached the run
    main(args)
    assert "skip (finished)" in capsys.readouterr().out             # resume skips it
    main(args + ["--no-resume"])
    assert ": run" in capsys.readouterr().out
    assert len(list((tmp_path / "gridruns").glob("synthetic_fedavg_l0p1_a0p5_demo_*"))) == 2
    main(["--grid", "configs/grids/demo.yaml", "--dry-run", "--set", f"output_dir={out}"])
    dry = capsys.readouterr().out
    assert dry.count("[grid") == 10 and dry.count("skip (finished)") == 1


def test_results_table_latex_and_partial_runs():
    pytest.importorskip("gradio")
    from ssfl import gui
    runs = [
        {"base": "organamnist_fedavg_l0p05", "dataset": "organamnist", "labels": 0.05, "variant": "FedAvg",
         "best_acc": 0.5, "finished": True, "mtime": 1},
        {"base": "organamnist_fedlabel_l0p05", "dataset": "organamnist", "labels": 0.05, "variant": "FedLabel",
         "best_acc": 0.6, "finished": True, "mtime": 2},
        {"base": "organamnist_fedmatch_l0p05", "dataset": "organamnist", "labels": 0.05, "variant": "FedMatch",
         "best_acc": 0.9, "finished": False, "mtime": 3},          # stopped run: not in the table
    ]
    table = gui.thesis_table(runs)
    assert list(table["algorithm"]) == ["FedAvg", "FedLabel"]
    tex = gui.to_latex(table, caption="Best acc (%) & more_info")
    assert r"\caption{Best acc (\%) \& more\_info}" in tex
    assert r"organamnist 5\%" in tex and r"\textbf{60.00}" in tex
    assert tex.count(r"\\") == 3                                   # header + 2 rows


def test_gui_env_defaults_and_version():
    pytest.importorskip("gradio")
    from ssfl import gui
    gui.ENV_DEFAULTS.update(data_dir="/drive/data", runs_dir="/drive/runs", num_workers=2)
    try:
        cfg = gui.apply_env(gui.load_preset("quick_test.yaml"))
        assert (cfg.data.root, cfg.output_dir, cfg.data.num_workers) == ("/drive/data", "/drive/runs", 2)
        gui.build_app()
    finally:
        gui.ENV_DEFAULTS.clear()
    assert gui.code_version()


def test_same_name_same_second_gets_its_own_folder(tmp_path, monkeypatch):
    import ssfl.experiment
    monkeypatch.setattr(ssfl.experiment.time, "strftime", lambda fmt: "20260101-120000")  # same second
    cfg = cfg_for(tmp_path, **{"strategy.name": "fedavg", "server.rounds": 1})
    first = Experiment(cfg, verbose=False)
    first.out_dir.mkdir(parents=True)                        # as if it had just started
    second = Experiment(cfg, verbose=False)
    assert second.out_dir != first.out_dir
    assert second.out_dir.name.startswith(first.out_dir.name)


def test_auto_num_workers(monkeypatch):
    import numpy as np
    import sys
    from ssfl.data import manager
    tiny = [(np.arange(80), np.arange(320))] * 3                      # quick demo: few small batches
    big = [(np.arange(170), np.arange(3200))] * 10                    # OrganAMNIST-like client
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert manager.auto_num_workers(tiny, 16, 2, local_steps=5, local_epochs=1) == 0
    assert manager.auto_num_workers(big, 32, 7, local_steps=None, local_epochs=1) == 2
    monkeypatch.setattr(sys, "platform", "win32")
    assert manager.auto_num_workers(big, 32, 7, local_steps=None, local_epochs=1) == 0


def test_round_batches_with_workers_gives_exact_steps():
    from torch.utils.data import DataLoader, TensorDataset
    from ssfl.strategies.base import round_batches
    loader = DataLoader(TensorDataset(torch.arange(10)), batch_size=4, shuffle=True, num_workers=1)
    batches = list(round_batches(loader, steps=5))                    # one worker start for 2 passes
    assert len(batches) == 5 and all(len(b[0]) == 4 for b in batches)
    seen = torch.cat([b[0] for b in batches])
    assert torch.bincount(seen, minlength=10).tolist() == [2] * 10    # every sample twice, reshuffled


def test_experiment_runs_with_worker_processes(tmp_path):
    cfg = cfg_for(tmp_path, **{"strategy.name": "fedlabel", "data.num_workers": 1, "server.rounds": 1,
                               "strategy.local_steps": 3, "server.num_clients": 2})
    exp = Experiment(cfg, verbose=False)
    assert exp.num_workers == 1 and exp.clients[0].data.unlabeled.num_workers == 1
    assert len(exp.run()) == 1


# ------------------------------------------------------------- GPU data pipeline / precision
@pytest.mark.parametrize("strategy", ["fedlabel", "fedmatch", "fedtrinet", "fedavg"])
def test_gpu_augmentation_pipeline_runs(tmp_path, strategy):
    """augment_device=gpu also works on a CPU device (batched tensor augmentation)."""
    params = {"pretrain_rounds": 1, "finetune_steps": 1} if strategy == "fedtrinet" else {}
    hist = run(cfg_for(tmp_path, **{"strategy.name": strategy, "strategy.params": params,
                                    "data.augment_device": "gpu"}))
    if strategy != "fedavg":
        assert 0.0 <= hist[-1]["train_mask_rate"] <= 1.0
    if "train_pl_acc" in hist[-1]:
        assert 0.0 <= hist[-1]["train_pl_acc"] <= 1.0


def test_gpu_augment_views_match_cpu_definition():
    import torchvision.transforms.functional as TF
    from ssfl.data.gpu import GPUAugment, _reflect
    img = torch.arange(3 * 8 * 8, dtype=torch.uint8).view(1, 3, 8, 8)
    for oy, ox in [(-2, 1), (0, 0), (2, -2)]:               # crop == reflect pad + crop
        ref = TF.crop(TF.pad(img.float(), [2, 2, 2, 2], padding_mode="reflect"), oy + 2, ox + 2, 8, 8)
        mine = img[0][:, _reflect(oy + torch.arange(8), 8)][:, :, _reflect(ox + torch.arange(8), 8)]
        assert torch.equal(ref[0].to(torch.uint8), mine)
    aug = GPUAugment(32, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    x = torch.randint(0, 256, (64, 3, 32, 32), dtype=torch.uint8)
    s = aug.strong_view(x)
    assert s.shape == (64, 3, 32, 32) and s.dtype == torch.float32
    erased = (aug.cutout(torch.ones(256, 3, 32, 32)) == 0).all(1).float().mean((1, 2))
    assert erased.min() >= 0.01 and erased.max() <= 0.16          # RandomErasing scale (0.02, 0.15)
    assert aug.randaugment(x).dtype == torch.uint8


def test_precision_options(tmp_path):
    hist = run(cfg_for(tmp_path, **{"strategy.name": "fedlabel", "strategy.precision": "bf16",
                                    "strategy.channels_last": True}))    # bf16 autocast on CPU
    assert len(hist) == 3
    s = STRATEGIES.build("fedavg", cfg_for(tmp_path, **{"strategy.precision": "fp16"}).strategy, 10)
    assert s.precision(torch.device("cpu")) == "fp32"                      # fp16 needs CUDA
    with pytest.raises(ValueError):
        STRATEGIES.build("fedavg", cfg_for(tmp_path, **{"strategy.precision": "int8"}).strategy, 10
                         ).precision(torch.device("cpu"))


def test_run_records_full_params_timing_and_status(tmp_path):
    import json
    import yaml
    from ssfl.grid import run_finished
    cfg = cfg_for(tmp_path, **{"strategy.name": "fedlabel", "strategy.params": {"tau": 0.7}})
    exp = Experiment(cfg, verbose=False)
    exp.run()
    saved = yaml.safe_load((exp.out_dir / "config.yaml").read_text())
    assert saved["strategy"]["params"]["beta"] == 0.7                       # old name `tau` renamed
    assert "tau" not in saved["strategy"]["params"]
    assert saved["strategy"]["params"]["lambda0"] == 1.0                    # defaults written out too
    assert "weight_by" in saved["server"]["params"]
    assert cfg.strategy.params == {"tau": 0.7}                              # caller's config untouched
    summ = json.loads((exp.out_dir / "summary.json").read_text())
    assert summ["status"] == "finished" and summ["rounds_done"] == 3
    assert summ["elapsed_s"] > 0 and summ["mean_round_s"] > 0 and "environment" in summ
    assert run_finished(exp.out_dir / "summary.json")
    # a run interrupted by a callback keeps a summary with status "stopped"
    class _Stopped(Exception):
        pass

    def stop(rnd, rec, e):
        raise _Stopped()
    exp2 = Experiment(cfg, callbacks=[stop], verbose=False)
    with pytest.raises(_Stopped):
        exp2.run()
    summ2 = json.loads((exp2.out_dir / "summary.json").read_text())
    assert summ2["status"] == "stopped" and summ2["rounds_done"] == 1
    assert not run_finished(exp2.out_dir / "summary.json")


def test_results_tab_helpers(tmp_path):
    pytest.importorskip("gradio")
    from ssfl import gui
    cfg = cfg_for(tmp_path, **{"strategy.name": "fedmatch"})
    exp = Experiment(cfg, verbose=False)
    exp.run()
    runs = gui.scan_runs(str(tmp_path))
    assert len(runs) == 1 and runs[0]["status"] == "finished" and runs[0]["time_s"] > 0
    board = gui.leaderboard(runs)
    assert {"time", "s/round", "started", "α"} <= set(board.columns)
    html_out = gui.render_run_details(runs[0], exp.out_dir)
    assert "FedMatch parameters" in html_out and "lambda_iccs" in html_out and "Environment" in html_out
    acc, loss = gui.detail_frames(exp.out_dir)
    assert len(acc) and len(loss)


# ------------------------------------------------------------- ADD alignment
def test_fedlabel_two_phase_update_is_sum_of_deltas(tmp_path):
    """Eq. 4: the uploaded update is dw_L + dw_U, both computed from W_t."""
    from ssfl.strategies.base import round_batches
    exp = Experiment(cfg_for(tmp_path, **{"strategy.name": "fedlabel", "strategy.local_steps": 3,
                                          "strategy.params": {"beta": 0.0}}), verbose=False)
    s, client, model = exp.strategy, exp.clients[0], exp.model
    w0 = {k: v.clone() for k, v in exp.server.global_state.items()}
    calls = []
    orig = s.optimize

    def spy(m, steps, lab_it, unl_it, loss_fn=None, sums=None):
        start = {k: v.clone() for k, v in m.state_dict().items()}
        out = orig(m, steps, lab_it, unl_it, loss_fn, sums)
        calls.append((start, {k: v.clone() for k, v in m.state_dict().items()}, lab_it is not None))
        return out
    s.optimize = spy
    model.load_state_dict(w0)
    metrics = s.local_train(model, client, torch.device("cpu"))
    assert [c[2] for c in calls] == [True, False]                    # supervised, then unlabeled
    for start, _, _ in calls:                                         # both phases start from W_t
        assert all(torch.equal(start[k], w0[k]) for k in w0 if w0[k].is_floating_point())
    expect = {k: w0[k] + (calls[0][1][k] - w0[k]) + (calls[1][1][k] - w0[k]) for k in w0 if w0[k].is_floating_point()}
    assert all(torch.allclose(model.state_dict()[k], v, atol=1e-6) for k, v in expect.items())
    assert {"loss_sup", "loss_unsup", "loss_kl", "kl_rate", "local_selected"} <= set(metrics)


def test_fedlabel_selection_threshold_and_kl_weight():
    import torch.nn as nn
    from ssfl.config import StrategyConfig

    class Fixed(nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, x):
            return self.out[: len(x)]
    s = STRATEGIES.build("fedlabel", StrategyConfig(), 3, beta=0.6, lambda0=2.0)
    G = torch.tensor([[5., 0, 0], [0, 5., 0], [1., 0.9, 0], [0, 0, 6.]])
    L = torch.tensor([[4., 0, 0], [0, 9., 0], [0, 0, 1.], [8., 0, 0]])
    s.global_model, s.local_model = Fixed(G), Fixed(L)
    student = nn.Linear(1, 3)
    x = torch.zeros(4, 1)
    _, logs = s._unsup_loss(student, None, (x, x, torch.tensor([0, 1, 2, 2])))
    hG, hL = G.var(1, unbiased=False), L.var(1, unbiased=False)       # h = variance of the logits
    assert float(logs["local_selected"]) == 0.75                       # samples 1, 2, 3 pick the local model
    assert float(logs["mask_rate"]) == 0.75                            # sample 2 is below beta
    assert float(logs["kl_rate"]) == 0.5                               # sample 3: the other model disagrees
    lam = (2 * hL[0] / hG[0] + 2 * hG[1] / hL[1]) / 2                  # lambda0 * h(s_bar) / h(s*)
    assert abs(float(logs["lambda_k"]) - float(lam)) < 1e-5


def test_fedlabel_accepts_old_parameter_names():
    from ssfl.config import StrategyConfig
    s = STRATEGIES.build("fedlabel", StrategyConfig(), 10, tau=0.7, lambda_kl=0.3, local_warmup_steps=5)
    assert (s.beta, s.lambda0) == (0.7, 0.3)


def test_macro_metrics_and_random_split():
    import numpy as np
    from ssfl.data.partition import split_labeled_unlabeled
    from ssfl.server.server import classification_metrics
    conf = torch.tensor([[8, 2, 0], [1, 4, 0], [0, 0, 0]])           # class 2 absent from the test set
    m = classification_metrics(conf)
    assert abs(m["test_acc"] - 12 / 15) < 1e-9
    assert abs(m["test_recall"] - (0.8 + 0.8) / 2) < 1e-9               # absent class skipped
    p0, p1 = 8 / 9, 4 / 6
    assert abs(m["test_precision"] - (p0 + p1) / 2) < 1e-9
    labels = np.repeat(np.arange(10), 100)
    lab, unl = split_labeled_unlabeled(np.arange(1000), labels, 0.2, np.random.default_rng(0), mode="random")
    assert len(lab) == 200 and len(np.intersect1d(lab, unl)) == 0


def test_strong_view_follows_add_by_default():
    from ssfl.config import DataConfig
    from ssfl.data.augment import AugmentationFactory
    d = DataConfig()
    assert (d.randaugment_n, d.randaugment_m, d.cutout, d.split) == (1, 10, False, "random")
    strong = AugmentationFactory(32, (0.5,) * 3, (0.25,) * 3, "randaugment", d.randaugment_n, d.randaugment_m).strong()
    names = [type(t).__name__ for t in strong.transforms]
    assert "RandAugment" in names and "RandomErasing" not in names
    assert next(t for t in strong.transforms if type(t).__name__ == "RandAugment").num_ops == 1


def test_thesis_grid_follows_add():
    from ssfl.grid import expand_grid
    runs = dict(expand_grid("configs/grids/thesis.yaml"))
    assert {c.data.alpha for c in runs.values()} == {0.1, 1.0}
    assert {c.server.rounds for c in runs.values()} == {1000}
    assert {"organamnist_uda_l0p05_a0p1", "organamnist_fixmatch_l0p05_a0p1_prox", "organamnist_fedprox_l1_a0p1"} <= set(runs)
    assert "organamnist_fedavg_l1_a0p1" in runs                          # high baseline, 100% labels
    assert runs["cifar10_fedlabel_l0p2_a0p1"].model.name == "resnet34"
    assert {"lambda0_0", "tau_5", "tau_50", "beta_0p8"} <= {n.split("_a0p1_")[-1] for n in runs if "_a0p1_" in n}


# ------------------------------------------------------------- FedLabel paper alignment
def test_fedlabel_aggregation_weight_is_samples_used(tmp_path):
    """Paper Alg. 1: r_k = labeled samples processed + unlabeled samples past the threshold."""
    from ssfl.client import ClientUpdate
    exp = Experiment(cfg_for(tmp_path, **{"strategy.name": "fedlabel", "strategy.params":
                                          {"beta": 0.0, "sup_steps": 2, "unsup_steps": 3}}), verbose=False)
    client = exp.clients[0]
    upd: ClientUpdate = client.fit(exp.model, exp.server.broadcast(), exp.strategy, torch.device("cpu"))
    lab_bs = client.data.labeled.batch_size
    assert upd.metrics["used_samples"] == upd.num_samples
    assert upd.num_samples >= 2 * lab_bs                           # tau = 2 labeled steps counted
    assert upd.num_samples <= 2 * lab_bs + 3 * client.data.unlabeled.batch_size
    exp2 = Experiment(cfg_for(tmp_path, **{"strategy.name": "fedlabel",
                                           "strategy.params": {"weight_by_used": False}}), verbose=False)
    upd2 = exp2.clients[0].fit(exp2.model, exp2.server.broadcast(), exp2.strategy, torch.device("cpu"))
    assert upd2.num_samples == exp2.clients[0].num_samples          # dataset size instead


def test_fedlabel_steps_override_epochs():
    from ssfl.config import StrategyConfig
    s = STRATEGIES.build("fedlabel", StrategyConfig(local_epochs=3), 10, sup_steps=7, unsup_steps=11)
    loader = list(range(4))                                         # 4 batches
    assert s._phase_steps(loader, s.sup_epochs, s.sup_steps) == 7
    assert s._phase_steps(loader, s.unsup_epochs, None) == 12       # 3 epochs x 4 batches


def test_fedlabel_threshold_is_strict():
    import torch.nn as nn
    from ssfl.config import StrategyConfig

    class Fixed(nn.Module):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def forward(self, x):
            return self.out[: len(x)]
    s = STRATEGIES.build("fedlabel", StrategyConfig(), 2, beta=0.5)
    s.global_model = s.local_model = Fixed(torch.tensor([[0.0, 0.0]]))   # max prob exactly 0.5
    _, logs = s._unsup_loss(nn.Linear(1, 2), None, (torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([0])))
    assert float(logs["mask_rate"]) == 0.0                          # 0.5 is not > beta = 0.5


@pytest.mark.parametrize("prox", [0.0, 0.01])
def test_uda_runs(tmp_path, prox):
    hist = run(cfg_for(tmp_path, **{"strategy.name": "uda", "strategy.prox_mu": prox, "strategy.params": {}}))
    assert "train_loss_unsup" in hist[-1] and 0 <= hist[-1]["train_mask_rate"] <= 1


def test_paper_baseline_labels():
    pytest.importorskip("gradio")
    from ssfl import gui
    assert gui._variant("fixmatch", {"prox_mu": 0.01}, "prox") == "FedProx+FixMatch"
    assert gui._variant("uda", {"prox_mu": 0}, None) == "FedAvg+UDA"
    assert gui._variant("fedlabel", {}, "tau_5") == "FedLabel (tau_5)"


# ------------------------------------------------------------- horizontal flips
def test_no_hflip_when_dataset_is_not_flip_safe(tmp_path, monkeypatch):
    import numpy as np
    from ssfl import DATASETS
    from ssfl.config import DataConfig
    from ssfl.data import DataManager
    from ssfl.data.datasets import RawDataset
    from ssfl.data.gpu import GPUAugment

    def unsafe(cfg):                                   # like OrganAMNIST (left/right classes)
        rng = np.random.default_rng(0)
        x = rng.integers(0, 255, (200, 8, 8, 1)).astype(np.uint8)
        y = rng.integers(0, 4, 200)
        return RawDataset(x, y, 4, (0.5,), (0.25,), False), RawDataset(x[:40], y[:40], 4, (0.5,), (0.25,), False)
    monkeypatch.setitem(DATASETS._items, "test_unsafe", unsafe)
    dm = DataManager(DataConfig(dataset="test_unsafe", num_workers=0), 2, seed=0)
    assert dm.hflip is False
    assert "RandomHorizontalFlip" not in [type(t).__name__ for t in dm.aug.weak().transforms]
    assert "RandomHorizontalFlip" not in [type(t).__name__ for t in dm.aug.strong().transforms]
    forced = DataManager(DataConfig(dataset="test_unsafe", num_workers=0, hflip=True), 2, seed=0)
    assert forced.hflip is True                         # explicit setting wins
    syn = DataManager(DataConfig(dataset="synthetic", num_workers=0), 2, seed=0)
    assert syn.hflip is True                            # flip-safe datasets keep the flip
    # GPU pipeline: with hflip off, a left/right-asymmetric image is never mirrored
    aug = GPUAugment(8, (0.5,), (0.25,), hflip=False)
    img = torch.zeros(64, 1, 8, 8, dtype=torch.uint8)
    img[:, :, :, :4] = 255                              # bright left half
    out = aug.weak(img).float()
    assert (out[:, :, :, :2].mean() > out[:, :, :, -2:].mean())   # still brighter on the left


def test_organamnist_loader_marks_flip_unsafe(monkeypatch):
    import sys
    import types
    import numpy as np
    from ssfl.config import DataConfig
    from ssfl.data.datasets import load_organamnist, load_pathmnist

    class Fake:
        info = {"label": {str(i): str(i) for i in range(3)}}

        def __init__(self, split, **kw):
            self.imgs = np.zeros((6, 4, 4), np.uint8)
            self.labels = np.zeros((6, 1), np.int64)
    monkeypatch.setitem(sys.modules, "medmnist", types.SimpleNamespace(OrganAMNIST=Fake, PathMNIST=Fake))
    cfg = DataConfig(root=str(__import__("tempfile").gettempdir()))
    assert load_organamnist(cfg)[0].hflip_safe is False
    assert load_pathmnist(cfg)[0].hflip_safe is True


# ------------------------------------------------------------- FedLabel appendix A
def test_holdout_resplit_and_validation_set():
    import numpy as np
    from ssfl.config import DataConfig
    from ssfl.data.datasets import load_synthetic
    from ssfl.data.manager import DataManager, make_holdout
    tr, te = load_synthetic(None)
    # resplit: pool train + test (2,400) and re-divide 80 / 5 / 15, disjoint and complete
    a, v, t = make_holdout((tr, te), "resplit", 0.05, 0.15, seed=0)
    assert (len(a), len(v), len(t)) == (1920, 120, 360)
    pool = np.concatenate([tr.images, te.images]).reshape(2400, -1)
    seen = {r.tobytes() for r in pool}
    got = [r.tobytes() for d in (a, v, t) for r in d.images.reshape(len(d), -1)]
    assert len(got) == 2400 and set(got) == seen
    a2, _, _ = make_holdout((tr, te), "resplit", 0.05, 0.15, seed=1)
    assert not np.array_equal(a.labels, a2.labels)                         # the seed changes the split
    # official + val_ratio: validation carved from the training set, test untouched
    a, v, t = make_holdout((tr, te), "official", 0.1, 0.15, seed=0)
    assert (len(a), len(v), t is te) == (1800, 200, True)
    # default: no validation set, and the client partition is exactly as before
    dm = DataManager(DataConfig(dataset="synthetic", num_workers=0), 4, seed=0)
    assert dm.val is None and dm.val_loader() is None and len(dm.train) == 2000
    old = [np.array(p) for p in dm.partitions]
    dm2 = DataManager(DataConfig(dataset="synthetic", num_workers=0), 4, seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(old, dm2.partitions))
    with pytest.raises(ValueError):
        make_holdout((tr, te), "nope", 0, 0.15, 0)


def test_validation_metrics_timing_and_summary(tmp_path):
    import json
    cfg = cfg_for(tmp_path, **{"strategy.name": "fedlabel", "data.holdout": "resplit", "data.val_ratio": 0.05,
                               "server.eval_every": 1})
    exp = Experiment(cfg, verbose=False)
    assert "val" in exp.data.summary() and "re-split" in exp.data.summary()
    hist = exp.run()
    assert all({"val_acc", "val_f1", "test_acc", "client_s", "server_s"} <= set(r) for r in hist)
    assert all(r["client_s"] > 0 and r["server_s"] >= 0 for r in hist)
    summ = json.loads((exp.out_dir / "summary.json").read_text())
    best = max(hist, key=lambda r: r["val_acc"])
    assert summ["best_val_acc"] == best["val_acc"] and summ["test_acc_at_best_val"] == best["test_acc"]
    assert summ["mean_client_s"] > 0
    pytest.importorskip("gradio")
    from ssfl import gui
    runs = gui.scan_runs(str(tmp_path))
    assert runs[0]["test_at_val"] == best["test_acc"]
    html_out = gui.render_run_details(runs[0], exp.out_dir)
    assert "re-split 80 / 5 / 15%" in html_out and "test acc at best validation" in html_out
    assert len(gui.thesis_table(runs, "val")) == 1


def test_grid_seeds_names_and_order(tmp_path):
    from ssfl.grid import expand_grid
    grid = tmp_path / "g.yaml"
    grid.write_text(
        "seeds: [0, 1, 2]\n"
        "algorithms: {fedavg: {}, fedlabel: {}}\n"
        "experiments:\n"
        f"  - config: {Path('configs/quick_test.yaml').resolve().as_posix()}\n"
        "    labeled_ratio: [0.1]\n"
        f"  - config: {Path('configs/quick_test.yaml').resolve().as_posix()}\n"
        "    labeled_ratio: [0.1]\n"
        "    algorithms: [fedlabel]\n"
        "    tag: abl\n"
        "    seeds: [0]\n")
    runs = expand_grid(grid)
    names = [n for n, _ in runs]
    base = runs[0][1].seed
    assert base == 0 and len(names) == 7
    # all seed-0 runs first (old names, no suffix), then seed 1, then seed 2
    assert names[:3] == ["synthetic_fedavg_l0p1_a0p5", "synthetic_fedlabel_l0p1_a0p5", "synthetic_fedlabel_l0p1_a0p5_abl"]
    assert names[3:5] == ["synthetic_fedavg_l0p1_a0p5_s1", "synthetic_fedlabel_l0p1_a0p5_s1"]
    assert [c.seed for _, c in runs] == [0, 0, 0, 1, 1, 2, 2]


def test_thesis_grid_seeds_and_appendix_ablations():
    from ssfl.config import StrategyConfig
    from ssfl.grid import expand_grid
    runs = dict(expand_grid("configs/grids/thesis.yaml"))
    assert {c.seed for c in runs.values()} == {42, 43, 44}
    assert "cifar10_fedlabel_l0p2_a0p1_s44" in runs and "organamnist_fedlabel_l0p05_a0p1_beta_0p3_s43" not in runs
    assert runs["cifar10_fedlabel_l0p2_a0p1"].seed == 42
    assert runs["organamnist_fedlabel_l0p05_a0p1"].strategy.params["beta"] == 0.5          # appendix A default
    assert runs["organamnist_fedlabel_l0p05_a0p1_beta_0p95"].strategy.params["beta"] == 0.95
    assert runs["organamnist_fedlabel_l0p05_a0p1_conf_entropy"].strategy.params["confidence"] == "entropy"
    assert runs["organamnist_fedlabel_l0p2_a0p1_tau_20"].strategy.params["sup_steps"] == 20
    assert runs["cifar10_fedlabel_l0p2_a0p1_noaug"].data.strong_aug == "none"
    assert runs["cifar10_fedlabel_l0p2_a0p1_noaug_beta_0p95"].strategy.params["beta"] == 0.95
    assert STRATEGIES.get("fedlabel")(StrategyConfig(), 3).beta == 0.5


def test_thesis_table_averages_seeds():
    pytest.importorskip("gradio")
    from ssfl import gui

    def r(variant, acc, seed, base, mtime=1):
        return {"base": base, "group": base, "seed": seed, "dataset": "organamnist", "labels": 0.05, "alpha": 0.1,
                "variant": variant, "best_acc": acc, "finished": True, "mtime": mtime}
    runs = [r("FedAvg", 0.50, 0, "o_fedavg"), r("FedAvg", 0.54, 1, "o_fedavg_s1"),
            r("FedLabel", 0.60, 0, "o_fedlabel"), r("FedLabel", 0.70, 1, "o_fedlabel_s1"),
            r("FedLabel", 0.80, 2, "o_fedlabel_s2"), r("FedMatch", 0.55, 0, "o_fedmatch")]
    table = gui.thesis_table(runs)
    col = "organamnist 5% α0.1"
    assert list(table["algorithm"]) == ["FedAvg", "FedMatch", "FedLabel"]
    assert table.loc[2, col] == 70.0 and table.attrs["n"].loc[2, col] == 3
    assert abs(table.attrs["std"].loc[2, col] - 8.16) < 0.01
    shown = gui.thesis_display(table)
    assert shown.loc[2, col] == "70 ± 8.16" and shown.loc[1, col] == "55"
    tex = gui.to_latex(table)
    assert r"\textbf{70.00} {\scriptsize$\pm$8.16}" in tex and r"55.00 \\" in tex
    ratio = gui.thesis_table(runs, "ratio")
    assert abs(ratio.loc[2, col] - 70 / 52) < 1e-3


def test_run_labels_from_grid_names(tmp_path):
    """The α part of a grid name is not a tag; tags and seed suffixes are parsed apart."""
    pytest.importorskip("gradio")
    import yaml
    from ssfl import gui
    cases = {"organamnist_fedavg_l0p05_a0p1": ("FedAvg", 42),
             "organamnist_fedlabel_l0p05_a1_s43": ("FedLabel", 43),
             "organamnist_fedlabel_l0p05_a0p1_beta_0p5": ("FedLabel (beta_0p5)", 42),
             "organamnist_fixmatch_l0p05_a0p1_prox_s44": ("FedProx+FixMatch", 44),
             "organamnist_fedlabel_l0p05": ("FedLabel", 42)}
    for name, (label, seed) in cases.items():
        d = tmp_path / f"{name}_20261122-120000"
        d.mkdir()
        algo = name.split("_")[1]
        prox = 0.01 if "prox" in name else 0.0
        (d / "config.yaml").write_text(yaml.safe_dump({"seed": seed, "strategy": {"name": algo, "prox_mu": prox}}))
        (d / "metrics.csv").write_text("round,test_acc\n1,0.5\n")
    got = {r["base"]: (r["variant"], r["seed"]) for r in gui.scan_runs(str(tmp_path))}
    assert got == cases


# ------------------------------------------------------------- speed options
@pytest.mark.parametrize("algo", ["fedlabel", "fedmatch", "fixmatch", "fedtrinet"])
def test_compile_trains_through_the_wrapper(tmp_path, monkeypatch, algo):
    """strategy.compile: the loss functions get a torch.compile wrapper that shares the model's
    parameters. The eager backend stands in for Inductor (no Triton needed)."""
    from ssfl.strategies import base
    real = torch.compile
    built = []

    def fake_compile(model, mode=None):
        built.append(mode)
        return real(model, backend="eager")

    monkeypatch.setattr(base.torch, "compile", fake_compile)
    over = {"strategy.name": algo, "strategy.compile": "default"}
    if algo == "fedtrinet":
        over["strategy.params"] = {"pretrain_rounds": 1}
    exp = Experiment(cfg_for(tmp_path, **over), verbose=False)
    w0 = {k: v.clone() for k, v in exp.model.state_dict().items()}
    hist = exp.run()
    assert built == ["default"]                       # one wrapper, reused every round
    assert all(not math.isnan(r["train_loss"]) for r in hist)
    assert any(not torch.equal(w0[k], v) for k, v in exp.model.state_dict().items())
    torch._dynamo.reset()


def test_compile_mode_names():
    from types import SimpleNamespace as NS
    from ssfl.strategies.base import compile_mode
    assert compile_mode(NS(compile=None)) is None and compile_mode(NS(compile=False)) is None
    assert compile_mode(NS(compile="none")) is None and compile_mode(NS(compile=True)) == "default"
    assert compile_mode(NS(compile="reduce-overhead")) == "reduce-overhead"
    with pytest.raises(ValueError):
        compile_mode(NS(compile="fast"))
