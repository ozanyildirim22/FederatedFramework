"""Parallel runs: several experiments as separate processes (synthetic data, CPU)."""
import time
from pathlib import Path

import pytest

from ssfl import Experiment, ExperimentConfig
from ssfl.cleanup import resumable_runs, run_status
from ssfl.config import apply_overrides
from ssfl.experiment import PAUSE_FILE, STOP_FILE, StopRequested
from ssfl.parallel import Job, ParallelRunner, read_history


def cfg_for(out, name, **over) -> ExperimentConfig:
    raw = {
        "name": name, "device": "cpu", "output_dir": str(out),
        "data": {"dataset": "synthetic", "batch_size": 8, "unlabeled_batch_ratio": 2, "labeled_ratio": 0.2},
        "model": {"name": "simple_cnn", "params": {"width": 8}},
        "strategy": {"name": "fedlabel", "local_steps": 2, "params": {"beta": 0.5}},
        "server": {"rounds": 3, "num_clients": 4, "client_fraction": 0.5, "checkpoint_every": 0},
    }
    return ExperimentConfig.from_dict(apply_overrides(raw, over))


def test_control_files_pause_and_stop_a_run(tmp_path):
    def touch(name):
        def cb(rnd, rec, exp):
            if rnd == 1:
                (exp.out_dir / name).write_text("x")
        return cb
    exp = Experiment(cfg_for(tmp_path / "p", "p"), callbacks=[touch(PAUSE_FILE)], verbose=False)
    exp.run()
    assert exp.paused and len(exp.history) == 1 and not (exp.out_dir / PAUSE_FILE).exists()
    assert run_status(exp.out_dir) == "paused"
    exp = Experiment(cfg_for(tmp_path / "s", "s"), callbacks=[touch(STOP_FILE)], verbose=False)
    with pytest.raises(StopRequested):
        exp.run()
    assert run_status(exp.out_dir) == "stopped" and not (exp.out_dir / STOP_FILE).exists()


def test_parallel_runs_finish_with_the_same_results(tmp_path):
    names = ["a", "b", "c"]
    algos = {"a": "fedlabel", "b": "fedavg", "c": "fixmatch"}
    jobs = [Job(n, cfg_for(tmp_path / "par", n, **{"strategy.name": algos[n]})) for n in names]
    logs = []
    runner = ParallelRunner(jobs, max_parallel=2, log=logs.append, poll_s=0.2)
    peak = []
    runner.run(lambda r: peak.append(len(r.running())))
    assert [j.status for j in jobs] == ["done"] * 3, "\n".join(logs[-30:])
    assert max(peak) == 2                                   # never more than max_parallel at once
    for j in jobs:                                          # same numbers as an in-process run
        ref = Experiment(cfg_for(tmp_path / "ref", j.name, **{"strategy.name": algos[j.name]}),
                         verbose=False)
        ref.run()
        assert [round(h["test_acc"], 6) for h in j.history if "test_acc" in h] == \
               [round(h["test_acc"], 6) for h in ref.history if "test_acc" in h]
        assert run_status(j.folder) == "finished"
    assert any("[a] " in m for m in logs)                   # child console lines reach the log


def test_parallel_pause_then_resume(tmp_path):
    jobs = [Job(n, cfg_for(tmp_path, n, **{"server.rounds": 400})) for n in ("a", "b", "c")]
    runner = ParallelRunner(jobs, max_parallel=2, log=lambda m: None, poll_s=0.2)

    def on_update(r):
        if not r.pausing and any(j.history for j in r.jobs):
            r.pause()
    runner.run(on_update)
    assert sorted(j.status for j in jobs) == ["paused", "paused", "waiting"]
    rows = resumable_runs(tmp_path)
    assert sorted(r["status"] for r in rows) == ["paused", "paused", "queued"]

    again = [Job(j.name, j.cfg, j.folder) for j in jobs]
    runner = ParallelRunner(again, max_parallel=3, log=lambda m: None, poll_s=0.2)

    def stop_soon(r):
        if not r.stopping and all(len(j.history) >= 2 for j in r.jobs):
            r.stop()
    runner.run(stop_soon)
    assert [j.status for j in again] == ["stopped"] * 3
    for j in again:                                         # continued in the same folder
        hist = read_history(j.folder)
        assert [int(h["round"]) for h in hist] == list(range(1, len(hist) + 1))
    assert len(list(tmp_path.iterdir())) == 3


def test_gui_controller_runs_the_queue_in_parallel(tmp_path):
    gui = pytest.importorskip("ssfl.gui")
    ctrl = gui.RunController()
    cfgs = [cfg_for(tmp_path, n) for n in ("x", "y")]
    ctrl.start([(c.name, c, "queued") for c in cfgs], "experiment", parallel=2)
    ctrl.thread.join(180)
    snap = ctrl.snapshot()
    assert snap["state"] == "done", "\n".join(snap["logs"][-20:])
    assert snap["parallel"] == 2
    assert [r["status"] for r in snap["plan"]] == ["done", "done"]
    assert all(len(r["history"]) == 3 and r["best"] is not None for r in snap["plan"])
    assert "Finished" in gui.render_status(snap)
