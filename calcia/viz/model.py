"""Load a calcia simulation run into a single in-memory data model.

A "run" is a directory produced by the striatum / full-pipeline demos under
``examples/output/<run_dir>``.  It contains:

===================  ==========================================================
file                 contents used here
===================  ==========================================================
metadata.json        grid size, vres, dt, nt, path to the shared phase-1 cache
traces.npz           soma_neurons (N,T), soma_locs (N,3 um), spikes_neurons
movies.npz           mov_clean / mov_noisy  (T,H,W)
optics.npz           psf, mask, col_mask
cell_footprints.pkl  gp_vals: per-component voxel indices + soma_mask
_shared/phase1_*.pkl NeuralVolumeOutput -> neur_ves (vessels), neur_vol (bg)
===================  ==========================================================

Coordinate conventions (verified against the data):
  * ``neur_num`` / ``gp_vals[k].indices`` are **C-order** linear indices into
    the grid of shape ``grid_shape = (X, Y, Z) = vol_sz * vres``.
  * ``soma_locs`` are in **microns**; a grid/voxel coordinate is ``um * vres``.
  * Alignment: ``soma_neurons[i]`` <-> ``neur_num == i+1`` <-> ``gp_vals[i]``
    <-> ``soma_locs[i]`` for ``i`` in ``0 .. n_neur-1``.

Only per-neuron geometry (soma + dendrites, from ``gp_vals``) needs the small
``cell_footprints.pkl``; the large multi-GB phase-1 pickle is loaded *only* for
the vessels / background volume and is therefore optional.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SimRun:
    """Everything the viewer needs for one simulation run."""

    run_dir: str
    metadata: dict

    # --- grid / timing -----------------------------------------------------
    grid_shape: tuple            # (X, Y, Z) voxels, C-order
    vres: int                    # voxels per micron
    dt: float                    # seconds per frame
    nt: int                      # number of movie frames

    # --- per-neuron traces + geometry sources ------------------------------
    n_neur: int
    soma_neurons: np.ndarray     # (N, T) somatic fluorescence
    soma_locs: np.ndarray        # (N, 3) in microns
    spikes_neurons: Optional[np.ndarray]  # (N, T) or None
    gp_vals: list                # per-component CellFluorescenceData (len >= N)

    # --- 2D imaging --------------------------------------------------------
    mov_clean: np.ndarray        # (T, H, W)
    mov_noisy: Optional[np.ndarray]

    # --- optional heavy 3D volumes (from phase-1 pickle) -------------------
    neur_ves: Optional[np.ndarray] = None   # (X, Y, Z) vessel mask
    neur_vol: Optional[np.ndarray] = None   # (X, Y, Z) fluorescence (bg fog)

    # --- optics (optional, for reference) ----------------------------------
    psf: Optional[np.ndarray] = None
    col_mask: Optional[np.ndarray] = None

    # cached derivatives
    _dff: Optional[np.ndarray] = field(default=None, repr=False)

    # ---------------------------------------------------------------- traces
    def dff(self, baseline_pct: float = 20.0) -> np.ndarray:
        """Per-neuron dF/F0, F0 = ``baseline_pct`` percentile of each trace.

        Used to colour soma glyphs per frame.  Result cached.
        """
        if self._dff is None:
            f = self.soma_neurons.astype(np.float32)
            f0 = np.percentile(f, baseline_pct, axis=1, keepdims=True)
            f0 = np.maximum(f0, 1e-6)
            self._dff = (f - f0) / f0
        return self._dff

    # -------------------------------------------------------------- geometry
    def neuron_voxels(self, i: int, part: str = "all") -> np.ndarray:
        """(M, 3) integer voxel coordinates for neuron ``i``.

        ``part`` in {"all", "soma", "dend"} selects components using the
        ``soma_mask`` stored on the gp_vals entry.
        """
        g = self.gp_vals[i]
        idx = np.asarray(g.indices)
        mask = np.asarray(g.soma_mask, dtype=bool)
        if part == "soma":
            idx = idx[mask]
        elif part == "dend":
            idx = idx[~mask]
        if idx.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        x, y, z = np.unravel_index(idx, self.grid_shape, order="C")
        return np.stack([x, y, z], axis=1)

    def soma_grid_locs(self) -> np.ndarray:
        """(N, 3) soma centres in **grid/voxel** coordinates."""
        return self.soma_locs * self.vres


# --------------------------------------------------------------------------- IO
def _load_npz(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def _orient_movie(mov, axes, t_hint=None):
    """Return a movie as ``(T, H, W)`` regardless of how it was stored.

    Runs saved by different demo versions store the movie either as
    ``(T, H, W)`` (tagged ``axes='THW'``) or, in older runs, as ``(H, W, T)``
    with no axes tag.  We orient using, in order: the axes tag, the axis whose
    length uniquely matches the trace length ``t_hint``, the odd-one-out axis
    when the spatial dims are square, and finally the legacy ``(H, W, T)``
    assumption.
    """
    if mov is None or mov.ndim != 3:
        return mov
    a = axes.upper() if isinstance(axes, str) else ""
    if len(a) == 3 and "T" in a:
        return np.ascontiguousarray(np.moveaxis(mov, a.index("T"), 0))
    if t_hint:
        matches = [i for i, s in enumerate(mov.shape) if s == t_hint]
        if len(matches) == 1:
            return np.ascontiguousarray(np.moveaxis(mov, matches[0], 0))
    from collections import Counter
    cnt = Counter(mov.shape)
    odd = [i for i, s in enumerate(mov.shape) if cnt[s] == 1]
    if len(odd) == 1:
        return np.ascontiguousarray(np.moveaxis(mov, odd[0], 0))
    if not a:  # legacy movies without a tag were saved (H, W, T)
        return np.ascontiguousarray(np.moveaxis(mov, 2, 0))
    return mov


def load(
    run_dir: str,
    *,
    load_vessels: bool = True,
    load_volume: bool = False,
    verbose: bool = True,
) -> SimRun:
    """Load a run directory into a :class:`SimRun`.

    Parameters
    ----------
    load_vessels : also read the (large) phase-1 pickle to get vessel voxels.
    load_volume  : additionally keep the dense fluorescence volume (bg fog).
                   Off by default -- it is the memory-heavy part.
    """
    run_dir = os.path.abspath(run_dir)
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))

    grid = None  # discovered below
    vres = int(meta.get("vres", 1))
    dt = float(meta.get("dt", 0.05))

    # traces ---------------------------------------------------------------
    tr = _load_npz(os.path.join(run_dir, "traces.npz"))
    if tr is None or "soma_neurons" not in tr.files:
        raise ValueError(
            f"{run_dir!r} is not a viewable run: traces.npz lacks "
            "'soma_neurons' (older-format run without per-neuron traces).")
    soma_neurons = tr["soma_neurons"].astype(np.float32)
    soma_locs = tr["soma_locs"].astype(np.float32)
    spikes = tr["spikes_neurons"] if "spikes_neurons" in tr.files else None
    n_neur = soma_neurons.shape[0]
    t_traces = soma_neurons.shape[1]

    # footprints / gp_vals (can be multi-GB -> guard before loading) --------
    from ._memory import guard_load
    fp_path = os.path.join(run_dir, "cell_footprints.pkl")
    guard_load(fp_path)
    with open(fp_path, "rb") as f:
        fp = pickle.load(f)
    gp_vals = fp["gp_vals"]
    grid = tuple(int(v) for v in fp["neur_vol_shape"])

    # movies (oriented to (T, H, W)) ---------------------------------------
    mv = _load_npz(os.path.join(run_dir, "movies.npz"))
    axes = mv["axes"].item() if "axes" in mv.files else ""
    mov_clean = _orient_movie(mv["mov_clean"].astype(np.float32), axes, t_traces)
    mov_noisy = (_orient_movie(mv["mov_noisy"].astype(np.float32), axes, t_traces)
                 if "mov_noisy" in mv.files else None)
    nt = mov_clean.shape[0]

    # optics ---------------------------------------------------------------
    op = _load_npz(os.path.join(run_dir, "optics.npz"))
    psf = op["psf"] if op is not None and "psf" in op.files else None
    col_mask = op["col_mask"] if op is not None and "col_mask" in op.files else None

    # vessels / volume from phase-1 pickle (optional, VERY heavy) ----------
    # Cache-first: if a vessel mesh (.vtp) already exists we must NOT read the
    # multi-GB pickle -- geometry.vessels() will load the small cache. We only
    # pay the big load when there is no cache (or the dense volume is wanted),
    # and even then a RAM guard refuses it rather than thrashing the machine.
    neur_ves = neur_vol = None
    if load_vessels or load_volume:
        import glob
        has_vessel_cache = bool(glob.glob(
            os.path.join(run_dir, "viz_cache", "vessels_*.vtp")))
        if load_vessels and has_vessel_cache and not load_volume:
            if verbose:
                print("[viz] using cached vessel mesh (phase-1 pickle skipped)")
        else:
            p1 = meta.get("phase1_cache")
            if p1 and os.path.exists(p1):
                guard_load(p1)   # raises MemoryBudgetError if unsafe
                if verbose:
                    gb = os.path.getsize(p1) / 1e9
                    print(f"[viz] loading phase-1 volume ({gb:.1f} GB): {p1}")
                with open(p1, "rb") as f:
                    vol, _vp = pickle.load(f)
                if load_vessels and getattr(vol, "neur_ves", None) is not None:
                    neur_ves = np.asarray(vol.neur_ves)
                if load_volume and getattr(vol, "neur_vol", None) is not None:
                    neur_vol = np.asarray(vol.neur_vol)
                if grid is None:
                    grid = (neur_ves.shape if neur_ves is not None
                            else neur_vol.shape)
                del vol  # free the rest of the multi-GB structure
            elif verbose:
                print(f"[viz] phase-1 cache not found ({p1!r}); vessels disabled.")

    run = SimRun(
        run_dir=run_dir,
        metadata=meta,
        grid_shape=grid,
        vres=vres,
        dt=dt,
        nt=nt,
        n_neur=n_neur,
        soma_neurons=soma_neurons,
        soma_locs=soma_locs,
        spikes_neurons=(np.asarray(spikes) if spikes is not None else None),
        gp_vals=gp_vals,
        mov_clean=mov_clean,
        mov_noisy=mov_noisy,
        neur_ves=neur_ves,
        neur_vol=neur_vol,
        psf=psf,
        col_mask=col_mask,
    )
    if verbose:
        print(f"[viz] run={os.path.basename(run_dir)} grid={grid} vres={vres} "
              f"N={n_neur} T={nt} vessels={'yes' if neur_ves is not None else 'no'}")
    return run
