"""Clearing previous runs.

Every experiment writes one folder ``<output_dir>/<name>_<timestamp>/``. This module
finds those folders, filters them and deletes them. A run that is still running is
never deleted. Used by ``python -m ssfl --clear-runs`` and the GUI's Results tab.

    python -m ssfl --clear-runs --dry-run                      # list every run folder
    python -m ssfl --clear-runs --algorithm fedlabel           # all FedLabel runs (asks first)
    python -m ssfl --clear-runs --status unfinished --yes      # stopped / error / interrupted / partial
    python -m ssfl --clear-runs --runs-dir /content/drive/MyDrive/ssfl/runs --only organamnist
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

#: files that mark a folder as a run folder (anything else in the runs folder is left alone)
RUN_FILES = ("config.yaml", "metrics.csv", "summary.json")
#: a "running" run that has not written anything for this long is treated as interrupted
STALE_S = 600
STATUSES = ("finished", "running", "stopped", "error", "interrupted", "partial")


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / f).exists() for f in RUN_FILES)


def list_run_dirs(runs_dir: str | Path) -> List[Path]:
    """Run folders directly inside ``runs_dir``, newest first."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if is_run_dir(p)), key=_mtime, reverse=True)


def _mtime(run_dir: Path) -> float:
    times = [(run_dir / f).stat().st_mtime for f in RUN_FILES if (run_dir / f).exists()]
    return max(times, default=run_dir.stat().st_mtime)


def run_status(run_dir: Path) -> str:
    """finished | running | stopped | error | interrupted | partial (no summary.json)."""
    try:
        summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "partial"
    st = summ.get("status", "finished")
    if st == "running":
        per_round = summ.get("mean_round_s") or 60
        if time.time() - _mtime(run_dir) > max(STALE_S, 5 * per_round):
            return "interrupted"
    return st


def run_algorithm(run_dir: Path) -> Optional[str]:
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        return (cfg.get("strategy") or {}).get("name")
    except (OSError, yaml.YAMLError):
        return None


def select_runs(runs_dir: str | Path, only: Sequence[str] = (), algorithms: Sequence[str] = (),
                statuses: Sequence[str] = ()) -> List[Path]:
    """Run folders whose name contains every ``only`` text, whose algorithm is in
    ``algorithms`` and whose status is in ``statuses`` (empty = no filter).
    ``unfinished`` in ``statuses`` stands for every status except finished and running."""
    wanted = set(statuses)
    if "unfinished" in wanted:
        wanted |= {"stopped", "error", "interrupted", "partial"}
        wanted.discard("unfinished")
    unknown = wanted - set(STATUSES)
    if unknown:
        raise ValueError(f"Unknown status {sorted(unknown)} ({' | '.join(STATUSES)} | unfinished)")
    out = []
    for p in list_run_dirs(runs_dir):
        if not all(o in p.name for o in only):
            continue
        if algorithms and run_algorithm(p) not in algorithms:
            continue
        if wanted and run_status(p) not in wanted:
            continue
        out.append(p)
    return out


def _force_remove(func, path, _exc):
    """Windows: read-only files cannot be removed until they are made writable."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def delete_runs(runs_dir: str | Path, names: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """Permanently delete the run folders ``names`` (folder names inside ``runs_dir``).
    Returns ``(deleted, skipped)``; ``skipped`` maps a name to the reason. Only run folders
    directly inside ``runs_dir`` are deleted, and never one that is still running."""
    root = Path(runs_dir).resolve()
    deleted: List[str] = []
    skipped: Dict[str, str] = {}
    for name in dict.fromkeys(names):                       # unique, in order
        path = (root / name).resolve()
        if path.parent != root:
            skipped[name] = "not a folder directly inside the runs folder"
        elif not is_run_dir(path):
            skipped[name] = "not a run folder"
        elif run_status(path) == "running":
            skipped[name] = "still running"
        else:
            try:
                if sys.version_info >= (3, 12):
                    shutil.rmtree(path, onexc=_force_remove)
                else:
                    shutil.rmtree(path, onerror=_force_remove)
                deleted.append(name)
            except OSError as e:
                skipped[name] = str(e)
    return deleted, skipped


def clear_runs_cli(runs_dir: str, only: Sequence[str] = (), algorithms: Sequence[str] = (),
                   statuses: Sequence[str] = (), dry_run: bool = False, yes: bool = False) -> List[str]:
    """``python -m ssfl --clear-runs``: list the matching runs, ask, delete."""
    runs = select_runs(runs_dir, only, algorithms, statuses)
    if not runs:
        print(f"No matching run folders in {runs_dir}.")
        return []
    for p in runs:
        print(f"  {p.name:60s} {run_status(p):12s} {run_algorithm(p) or '?'}")
    print(f"{len(runs)} run folder(s) in {Path(runs_dir).resolve()}")
    if dry_run:
        return []
    if not yes:
        try:
            answer = input(f"Delete these {len(runs)} run folder(s) permanently? [y/N] ")
        except EOFError:                                     # no terminal (e.g. a Colab `!` cell)
            print("No answer possible here: add --yes to delete.")
            return []
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return []
    deleted, skipped = delete_runs(runs_dir, [p.name for p in runs])
    for name, why in skipped.items():
        print(f"  kept {name}: {why}")
    print(f"Deleted {len(deleted)} run folder(s).")
    return deleted
