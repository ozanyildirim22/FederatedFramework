"""Option-level tests for the newer algorithms (synthetic data, CPU)."""
import math

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
    done = list((tmp_path / "gridruns").glob("synthetic_fedavg_l0p1_demo_*/summary.json"))
    assert len(done) == 1
    cfg = ExperimentConfig.from_yaml(done[0].parent / "config.yaml")
    assert cfg.server.rounds == 1 and cfg.output_dir == out          # --set reached the run
    main(args)
    assert "skip (finished)" in capsys.readouterr().out             # resume skips it
    main(args + ["--no-resume"])
    assert ": run" in capsys.readouterr().out
    assert len(list((tmp_path / "gridruns").glob("synthetic_fedavg_l0p1_demo_*"))) == 2
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
    assert saved["strategy"]["params"]["tau"] == 0.7                        # the given value
    assert saved["strategy"]["params"]["lambda_kl"] == 1.0                  # defaults written out too
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
