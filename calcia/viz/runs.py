"""Discover viewable simulation runs on disk.

A directory is a *viewable run* when it contains the files the viewer needs:
``metadata.json``, ``traces.npz``, ``movies.npz`` and ``cell_footprints.pkl``.
This lets the apps offer an in-program run picker instead of forcing the run
path onto the command line.
"""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REQUIRED = ("metadata.json", "traces.npz", "movies.npz", "cell_footprints.pkl")


def default_root() -> str:
    """Where simulation runs live.

    Order: ``$CALCIA_OUTPUT`` env var, then ``<cwd>/examples/output`` (running
    from the repo), then ``<repo>/examples/output`` relative to this package.
    The demos still write into ``examples/output`` even though the viewer now
    ships inside the ``calcia`` package.
    """
    env = os.environ.get("CALCIA_OUTPUT")
    if env:
        return env
    here = Path(__file__).resolve()               # <repo>/calcia/viz/runs.py
    candidates = [
        Path.cwd() / "examples" / "output",
        here.parents[2] / "examples" / "output",  # parents[2] == repo root
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)
    return str(candidates[0])


@dataclass
class RunInfo:
    path: str
    name: str
    tag: str = ""
    region: str = ""
    n_neur: int = 0
    nt: int = 0
    smoke: bool = False
    mtime: float = 0.0

    def summary(self) -> str:
        kind = "smoke" if self.smoke else "run"
        return (f"{self.name}  [{self.region or '?'} · {kind} · "
                f"{self.n_neur} neurons · {self.nt} frames]")


def _npz_has(path: str, key: str) -> bool:
    """True if an .npz archive contains ``key`` (reads only the zip index)."""
    try:
        with zipfile.ZipFile(path) as z:
            return f"{key}.npy" in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def is_run_dir(path: str) -> bool:
    """A folder is viewable only if the four files exist *and* traces.npz
    actually carries per-neuron traces (``soma_neurons``).  This filters out
    older-format runs that would crash on load."""
    if not all(os.path.exists(os.path.join(path, f)) for f in REQUIRED):
        return False
    return _npz_has(os.path.join(path, "traces.npz"), "soma_neurons")


def _read_info(path: str) -> RunInfo:
    name = os.path.basename(os.path.normpath(path))
    meta_path = os.path.join(path, "metadata.json")
    # sort by metadata.json mtime (set at run creation) -- robust against
    # viz_cache writes bumping the directory's own mtime.
    try:
        mtime = os.path.getmtime(meta_path)
    except OSError:
        mtime = 0.0
    info = RunInfo(path=path, name=name, mtime=mtime)
    try:
        meta = json.load(open(meta_path))
        info.tag = str(meta.get("tag", ""))
        info.region = str(meta.get("region", ""))
        info.n_neur = int(meta.get("N_neur", meta.get("n_soma", 0)) or 0)
        info.nt = int(meta.get("nt", 0) or 0)
        info.smoke = bool(meta.get("smoke", False))
    except Exception:
        pass
    return info


def discover_runs(root: Optional[str] = None, max_depth: int = 2) -> List[RunInfo]:
    """All viewable runs under ``root`` (default ``examples/output``), newest
    first.  Searches up to ``max_depth`` directory levels."""
    root = root or default_root()
    found: List[RunInfo] = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return found

    def walk(d: str, depth: int):
        if is_run_dir(d):
            found.append(_read_info(d))
            return  # don't descend into a run
        if depth >= max_depth:
            return
        try:
            for entry in os.scandir(d):
                if entry.is_dir():
                    walk(entry.path, depth + 1)
        except OSError:
            pass

    walk(root, 0)
    found.sort(key=lambda r: r.mtime, reverse=True)
    return found


def latest_run(root: Optional[str] = None) -> Optional[str]:
    runs = discover_runs(root)
    return runs[0].path if runs else None


def resolve(run_dir: Optional[str], root: Optional[str] = None) -> Optional[str]:
    """Return an explicit ``run_dir`` if given, else the most recent run."""
    if run_dir:
        return run_dir
    return latest_run(root)
