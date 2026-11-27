"""Running several experiments at the same time.

Each run is its own Python process (``python -m ssfl --resume <run folder>``): one process
per run sidesteps Python's global interpreter lock, which is what limits a single process
when the batches are small and the GPU mostly waits for Python. On a large GPU (A100 /
H100) 2-4 runs side by side do roughly that many times the work per hour; each run gives
the same result as when it runs alone (it has its own seed and random streams).

The parent process creates the run folder (``config.yaml`` + a ``queued`` summary), starts
at most ``max_parallel`` children, follows them through their ``metrics.csv`` /
``summary.json`` and their console output, and pauses or stops them by writing a ``PAUSE``
or ``STOP`` file into their folder (they act on it after their current round).

    python -m ssfl --grid configs/grids/thesis.yaml --parallel 3

Used by the grid command (``--parallel``) and the GUI (*Parallel runs*).
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import ExperimentConfig
from .experiment import PAUSE_FILE, STOP_FILE, write_queued_run

#: job states: queued -> running -> done | paused | stopped | error; a job that was still
#: queued when the runner paused becomes "waiting" (its folder is saved for Resume), one that
#: was queued when it stopped becomes "cancelled"
JOB_STATES = ("queued", "running", "done", "paused", "stopped", "error", "waiting", "cancelled")


def read_history(run_dir: Path) -> List[Dict[str, float]]:
    """``metrics.csv`` of a run as a list of round records (empty while it has none)."""
    try:
        with open(Path(run_dir) / "metrics.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []
    out = []
    for r in rows:
        rec = {}
        for k, v in r.items():
            if v in ("", None):
                continue
            try:
                rec[k] = float(v)
            except ValueError:
                pass
        if "round" in rec:
            out.append(rec)
    return out


def _summary_status(run_dir: Path) -> Optional[str]:
    try:
        return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


class Job:
    """One run of a parallel batch: a new config, or an existing run folder to continue."""

    def __init__(self, name: str, cfg: ExperimentConfig, folder: Optional[str | Path] = None):
        self.name, self.cfg = name, cfg
        self.folder: Optional[Path] = Path(folder) if folder else None
        self.status = "queued"
        self.history: List[Dict[str, float]] = []
        self.error: Optional[str] = None
        self.started: Optional[float] = None
        self.ended: Optional[float] = None
        self.proc: Optional[subprocess.Popen] = None
        self.tail: deque = deque(maxlen=60)          # last console lines (shown on errors)
        self._reader: Optional[threading.Thread] = None
        self._mtime: Optional[float] = None


class ParallelRunner:
    """Runs ``jobs`` as separate processes, at most ``max_parallel`` at a time.

    ``log`` receives every console line of the children, prefixed with the run name.
    :meth:`pause` / :meth:`stop` are thread-safe (they only write control files)."""

    def __init__(self, jobs: List[Job], max_parallel: int = 2, log: Callable[[str], None] = print,
                 poll_s: float = 1.0, python: str = sys.executable, cwd: Optional[str] = None):
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        self.jobs = list(jobs)
        self.max_parallel = int(max_parallel)
        self.log = log
        self.poll_s = poll_s
        self.python = python
        self.cwd = cwd or os.getcwd()
        self.pausing = False
        self.stopping = False
        self._lock = threading.Lock()

    # ----------------------------------------------------------- control
    def running(self) -> List[Job]:
        return [j for j in self.jobs if j.status == "running"]

    def pause(self) -> None:
        """Running jobs pause after their current round (checkpoint saved); no new job starts."""
        with self._lock:
            self.pausing = True
            for j in self.running():
                self._touch(j, PAUSE_FILE)

    def stop(self) -> None:
        """Running jobs stop after their current round; no new job starts."""
        with self._lock:
            self.stopping = True
            for j in self.running():
                self._touch(j, STOP_FILE)

    @staticmethod
    def _touch(job: Job, name: str) -> None:
        try:
            (job.folder / name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        except OSError:
            pass

    # ----------------------------------------------------------- run
    def run(self, on_update: Optional[Callable[["ParallelRunner"], None]] = None) -> List[Job]:
        self._prepare_data()
        while True:
            with self._lock:
                if not (self.pausing or self.stopping):
                    for j in self.jobs:
                        if len(self.running()) >= self.max_parallel:
                            break
                        if j.status == "queued":
                            self._start(j)
            for j in self.running():
                self._poll(j)
            if on_update:
                on_update(self)
            idle = not self.running()
            if idle and (self.pausing or self.stopping or not any(j.status == "queued" for j in self.jobs)):
                break
            time.sleep(self.poll_s)
        for j in self.jobs:                             # never started
            if j.status == "queued":
                if self.pausing:
                    if j.folder is None:
                        j.folder = write_queued_run(j.cfg)
                    j.status = "waiting"
                else:
                    j.status = "cancelled"
        if on_update:
            on_update(self)
        return self.jobs

    def _prepare_data(self) -> None:
        """Download every dataset once here, so parallel children never download the same
        files at the same time."""
        from .registry import DATASETS
        seen = set()
        for j in self.jobs:
            d = j.cfg.data
            key = (d.dataset, str(d.root), getattr(d, "image_size", None))
            if j.status != "queued" or d.dataset == "synthetic" or key in seen:
                continue
            seen.add(key)
            self.log(f"[parallel] preparing {d.dataset} (downloads it the first time) ...")
            try:
                DATASETS.get(d.dataset)(d)
            except Exception as e:  # noqa: BLE001 - the child reports the real error
                self.log(f"[parallel] could not prepare {d.dataset}: {e}")

    def _start(self, job: Job) -> None:
        if job.folder is None:
            job.folder = write_queued_run(job.cfg)
        for f in (PAUSE_FILE, STOP_FILE):
            (job.folder / f).unlink(missing_ok=True)
        job.history = read_history(job.folder)          # earlier rounds of a resumed run
        cmd = [self.python, "-u", "-m", "ssfl", "--resume", str(job.folder)]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        # each child gets its share of the CPU cores: by default every process would start one
        # thread per core, and N processes x all cores slow each other down badly
        share = str(max(1, (os.cpu_count() or 2) // self.max_parallel))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, share)
        job.proc = subprocess.Popen(cmd, cwd=self.cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1)
        job.status, job.started, job.ended, job.error = "running", time.time(), None, None
        job._reader = threading.Thread(target=self._read, args=(job,), daemon=True)
        job._reader.start()
        self.log(f"[parallel] start {job.name} -> {job.folder}")

    def _read(self, job: Job) -> None:
        for line in job.proc.stdout:
            line = line.rstrip()
            if line:
                job.tail.append(line)
                self.log(f"[{job.name}] {line}")

    def _poll(self, job: Job) -> None:
        mpath = job.folder / "metrics.csv"
        try:
            mtime = mpath.stat().st_mtime
        except OSError:
            mtime = None
        if mtime != job._mtime:
            job._mtime = mtime
            job.history = read_history(job.folder)
        rc = job.proc.poll()
        if rc is None:
            return
        if job._reader is not None:
            job._reader.join(timeout=5)
        job.history = read_history(job.folder)
        st = _summary_status(job.folder)
        job.status = {"finished": "done", "paused": "paused", "stopped": "stopped"}.get(st, "error")
        if job.status == "error":
            job.error = f"exit code {rc}\n" + "\n".join(job.tail)
        job.ended = time.time()
        self.log(f"[parallel] {job.name}: {job.status}")


def run_parallel_cli(jobs: List[Job], max_parallel: int) -> List[Job]:
    """Console version: Ctrl+C pauses every running job after its round (checkpoints saved),
    Ctrl+C again stops them. The children share the console, so they get Ctrl+C too and
    act the same way."""
    def log(msg: str) -> None:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)   # e.g. a cp1252 console

    runner = ParallelRunner(jobs, max_parallel, log=log)

    def on_sigint(signum, frame):
        if runner.pausing:
            print("\n[parallel] stopping the running jobs ...", flush=True)
            runner.stop()
        else:
            print("\n[parallel] pausing the running jobs after their current round "
                  "(Ctrl+C again to stop) ...", flush=True)
            runner.pause()

    try:
        old = signal.signal(signal.SIGINT, on_sigint)
    except ValueError:                                   # not the main thread
        return runner.run()
    try:
        return runner.run()
    finally:
        signal.signal(signal.SIGINT, old)
