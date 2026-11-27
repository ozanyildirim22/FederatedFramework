"""Pause / resume: a run that is paused (or interrupted after a checkpoint) and resumed ends
exactly where the same run without a break ends (synthetic data, CPU)."""
import json
import time
from pathlib import Path

import pytest
import torch

from ssfl import Experiment, ExperimentConfig
from ssfl.cleanup import checkpoint_round, resumable_runs, run_status, select_runs
from ssfl.config import apply_overrides
from ssfl.experiment import CHECKPOINT, CHECKPOINT_INFO, write_queued_run

PARAMS = {"fedlabel": {"beta": 0.5}, "fedmatch": {"num_helpers": 1, "helper_every": 1},
          "fedtrinet": {"pretrain_rounds": 1, "finetune_steps": 1}, "fixmatch": {}}


def cfg_for(out, name="resume", **over) -> ExperimentConfig:
    raw = {
        "name": name, "device": "cpu", "output_dir": str(out),
        "data": {"dataset": "synthetic", "batch_size": 8, "unlabeled_batch_ratio": 2, "labeled_ratio": 0.2},
        "model": {"name": "simple_cnn", "params": {"width": 8}},
        "strategy": {"name": "fedlabel", "local_steps": 2, "params": {"beta": 0.5}},
        "server": {"rounds": 4, "num_clients": 4, "client_fraction": 0.5, "aggregator": "fedavgm",
                   "checkpoint_every": 0},
    }
    return ExperimentConfig.from_dict(apply_overrides(raw, over))


def pause_at(k):
    def cb(rnd, rec, exp):
        if rnd == k:
            exp.request_pause()
    return cb


def same_run(a, b):
    """Two histories / final models that must be identical (CPU, same seed)."""
    ha, hb = a.history, b.history
    assert [int(r["round"]) for r in ha] == [int(r["round"]) for r in hb]
    for ra, rb in zip(ha, hb):
        for k in ra:
            if k.endswith("_s"):                 # timing
                continue
            assert ra[k] == pytest.approx(rb[k], rel=1e-6, abs=1e-7), (ra["round"], k)
    for k, v in a.server.global_state.items():
        assert torch.allclose(v.float(), b.server.global_state[k].float(), atol=1e-6), k


@pytest.mark.parametrize("strategy", ["fedlabel", "fedmatch", "fedtrinet", "fixmatch"])
def test_paused_and_resumed_run_equals_uninterrupted(tmp_path, strategy):
    over = {"strategy.name": strategy, "strategy.params": PARAMS[strategy], "server.eval_every": 1}
    ref = Experiment(cfg_for(tmp_path / "ref", **over), verbose=False)
    ref.run()

    exp = Experiment(cfg_for(tmp_path / "p", **over), callbacks=[pause_at(2)], verbose=False)
    exp.run()
    assert exp.paused and len(exp.history) == 2
    assert run_status(exp.out_dir) == "paused" and checkpoint_round(exp.out_dir) == 2

    res = Experiment.resume(exp.out_dir, verbose=False)
    assert res.resumed_from == 2 and res.out_dir == exp.out_dir
    res.run()
    assert not res.paused
    same_run(ref, res)
    # one folder, full metrics, checkpoint removed, summary counts the resume
    assert len(list((tmp_path / "p").iterdir())) == 1
    summ = json.loads((res.out_dir / "summary.json").read_text())
    assert summ["status"] == "finished" and summ["rounds_done"] == 4 and summ["resumes"] == 1
    assert not (res.out_dir / CHECKPOINT).exists() and not (res.out_dir / CHECKPOINT_INFO).exists()
    assert (res.out_dir / "global_model.pt").exists()
    assert len((res.out_dir / "metrics.csv").read_text().strip().splitlines()) == 1 + 4


def test_interrupted_run_resumes_from_last_checkpoint(tmp_path):
    """A crash after round 3 with checkpoints every 2 rounds: resume repeats round 3."""
    ref = Experiment(cfg_for(tmp_path / "ref"), verbose=False)
    ref.run()

    def crash(rnd, rec, exp):
        if rnd == 3:
            raise RuntimeError("Colab disconnected")
    exp = Experiment(cfg_for(tmp_path / "c", **{"server.checkpoint_every": 2}), callbacks=[crash],
                     verbose=False)
    with pytest.raises(RuntimeError):
        exp.run()
    assert run_status(exp.out_dir) == "error" and checkpoint_round(exp.out_dir) == 2
    assert [r["status"] for r in resumable_runs(tmp_path / "c")] == ["error"]

    res = Experiment.resume(exp.out_dir, verbose=False)
    assert res.resumed_from == 2 and len(res.history) == 2      # round 3 is trained again
    res.run()
    same_run(ref, res)


def test_resume_can_extend_rounds_and_warns_on_changes(tmp_path, capsys):
    exp = Experiment(cfg_for(tmp_path, **{"server.rounds": 3}), callbacks=[pause_at(1)], verbose=False)
    exp.run()
    res = Experiment.resume(exp.out_dir, ["server.rounds=5", "strategy.lr=0.5"])
    assert "different config for strategy.lr" in capsys.readouterr().out    # warned, not refused
    res.run()
    assert len(res.history) == 5 and res.cfg.server.rounds == 5


def test_resume_refuses_finished_and_checkpointless_runs(tmp_path):
    done = Experiment(cfg_for(tmp_path / "f", **{"server.rounds": 1}), verbose=False)
    done.run()
    with pytest.raises(ValueError, match="already finished"):
        Experiment.resume(done.out_dir)

    def stop(rnd, rec, exp):
        raise KeyboardInterrupt
    st = Experiment(cfg_for(tmp_path / "s"), callbacks=[stop], verbose=False)
    with pytest.raises(KeyboardInterrupt):
        st.run()
    assert run_status(st.out_dir) == "stopped" and checkpoint_round(st.out_dir) is None
    with pytest.raises(FileNotFoundError, match="no checkpoint.pt"):
        Experiment.resume(st.out_dir)
    assert resumable_runs(tmp_path / "s") == []


def test_queued_run_folder_starts_from_round_one(tmp_path):
    cfg = cfg_for(tmp_path, **{"server.rounds": 2})
    folder = write_queued_run(cfg)
    assert run_status(folder) == "queued"
    rows = resumable_runs(tmp_path)
    assert [(r["status"], r["rounds_done"], r["rounds_planned"]) for r in rows] == [("queued", 0, 2)]
    assert select_runs(tmp_path, statuses=["unfinished"]) == [folder]
    exp = Experiment.resume(folder, verbose=False)
    assert exp.resumed_from == 0
    exp.run()
    assert run_status(folder) == "finished" and len(list(tmp_path.iterdir())) == 1


def test_cli_resume_and_list(tmp_path, capsys):
    from ssfl.__main__ import main
    exp = Experiment(cfg_for(tmp_path), callbacks=[pause_at(1)], verbose=False)
    exp.run()
    main(["--list-resumable", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "paused" in out and "round 1/4" in out and exp.out_dir.name in out
    main(["--resume", str(exp.out_dir)])
    assert "resumed after round 1 of 4" in capsys.readouterr().out
    assert run_status(exp.out_dir) == "finished"
    main(["--list-resumable", "--runs-dir", str(tmp_path)])
    assert "No run to resume" in capsys.readouterr().out


def test_grid_continues_a_paused_run(tmp_path, capsys):
    from ssfl.__main__ import main
    from ssfl.grid import expand_grid
    out = str(tmp_path / "gridruns")
    extra = [f"output_dir={out}", "server.rounds=2", "device=cpu"]
    (name, cfg), = [(n, c) for n, c in expand_grid("configs/grids/demo.yaml", extra)
                    if "fedavg" in n and "l0p1" in n]
    paused = Experiment(cfg, callbacks=[pause_at(1)], verbose=False)
    paused.run()
    main(["--grid", "configs/grids/demo.yaml", "--only", "fedavg", "l0p1", "--set", *extra])
    assert f"resume {paused.out_dir.name}" in capsys.readouterr().out
    assert run_status(paused.out_dir) == "finished"
    assert len(list(Path(out).iterdir())) == 1                      # continued, not started anew


def test_gui_pause_saves_queue_and_resume_continues(tmp_path):
    gui = pytest.importorskip("ssfl.gui")
    ctrl = gui.RunController()
    cfgs = [cfg_for(tmp_path, name=f"q{i}", **{"server.rounds": 3}) for i in range(2)]
    ctrl.start([(c.name, c, "queued") for c in cfgs], "experiment")
    assert ctrl.pause()                       # pauses after the first round of q0
    ctrl.thread.join(60)
    snap = ctrl.snapshot()
    assert snap["state"] == "paused"
    assert [r["status"] for r in snap["plan"]] == ["paused", "waiting"]
    rows = {r["name"].rsplit("_", 2)[0]: r for r in resumable_runs(tmp_path)}
    assert rows["q0"]["status"] == "paused" and rows["q1"]["status"] == "queued"

    def resume_plan():
        dirs = sorted((Path(r["path"]) for r in resumable_runs(tmp_path)), key=gui.run_start_key)
        return [(d.name, ExperimentConfig.from_yaml(d / "config.yaml"), "queued", str(d)) for d in dirs]

    plan = resume_plan()
    assert [p[0].split("_")[0] for p in plan] == ["q0", "q1"]        # paused run first, then its queue
    # pause again right away: the queued run keeps its folder (no second one)
    ctrl.start(plan, "experiment")
    assert ctrl.pause()
    ctrl.thread.join(60)
    assert ctrl.snapshot()["state"] == "paused" and len(list(tmp_path.iterdir())) == 2
    plan = resume_plan()
    assert [p[0].split("_")[0] for p in plan] == ["q0", "q1"]

    ctrl.start(plan, "experiment")
    ctrl.thread.join(120)
    snap = ctrl.snapshot()
    assert snap["state"] == "done" and [r["status"] for r in snap["plan"]] == ["done", "done"]
    assert len(snap["plan"][0]["history"]) == 3      # earlier rounds shown in the live charts too
    assert len(list(tmp_path.iterdir())) == 2 and not resumable_runs(tmp_path)
