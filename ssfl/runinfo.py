"""Run bookkeeping helpers: full parameter sets, code version and environment.

Every run records these in its folder (``config.yaml`` and ``summary.json``),
so a result can be traced back to the exact settings, code and hardware.
"""
from __future__ import annotations

import inspect
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKIP_ARGS = {"self", "cfg", "num_classes"}


def default_params(cls) -> Dict[str, Any]:
    """Keyword defaults of a class' ``__init__`` along its MRO (strategy / aggregator parameters)."""
    out: Dict[str, Any] = {}
    for klass in getattr(cls, "__mro__", [cls]):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, p in inspect.signature(init).parameters.items():
            if name in SKIP_ARGS or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.default is not inspect.Parameter.empty and name not in out:
                out[name] = p.default
    return out


def full_params(cls, given: Dict[str, Any]) -> Dict[str, Any]:
    """Every parameter the component runs with: its defaults, overridden by ``given``."""
    return {**default_params(cls), **(given or {})}


def code_version() -> str:
    """Short git commit and date of the running code, e.g. ``"616c9e5 2026-11-22 02:31"``."""
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "log", "-1", "--format=%h %cd",
                              "--date=format:%Y-%m-%d %H:%M"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git missing, not a checkout, ...
        pass
    from . import __version__
    return f"v{__version__}"


def environment(device) -> Dict[str, Any]:
    """Hardware / software the run used."""
    import torch
    info: Dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "code": code_version(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
    return info
