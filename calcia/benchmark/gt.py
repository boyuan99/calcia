"""Ground-truth container for a calcia simulation run.

Loads the per-neuron ground truth (soma centres, clean fluorescence traces,
spikes) plus the physical scan parameters needed to (a) map volume coordinates
to movie pixels and (b) reason about optical detectability (depth attenuation,
illumination / collection weighting).

The heavy per-cell footprint pickle is NOT loaded here — only ``traces.npz``,
``params.pkl``, ``metadata.json`` and (optionally) ``optics.npz``.

Coordinate conventions (verified against the pipeline):
  * ``soma_locs`` are ``(N,3)`` in microns; with ``vres`` they become voxels.
  * A neuron at volume ``(x_um, y_um)`` lands at movie pixel
    ``p = (x_um * vres - scan_buff) / sfrac`` along each lateral axis.  The
    *assignment* of (x,y) to movie (row,col) and any residual motion offset are
    resolved empirically by :mod:`calcia.benchmark.matching`; here we only
    expose the scale/offset so callers do not hard-code them.
  * Optical masks (``illum_mask`` = excitation, ``col_mask`` = collection) are
    ``(X, Y)`` weight images in ``[0, 1]`` sampled at the neuron's voxel.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GroundTruth:
    """Per-neuron simulation ground truth + scan geometry."""

    locs_um: np.ndarray            # (N, 3) soma centres, microns
    traces: np.ndarray            # (N, T) clean per-neuron fluorescence (expression baked in)
    spikes: Optional[np.ndarray]  # (N, T) spike counts, or None
    dt: float                     # seconds per frame
    movie_shape: tuple            # (H, W) of the scanned movie
    vres: float
    sfrac: int
    scan_buff: int
    scatter_length_um: float      # widefield depth attenuation length L (exp(-2z/L))
    focal_depth_um: float         # depth of the focal plane
    illum_mask: Optional[np.ndarray] = None   # (X, Y) excitation weight
    col_mask: Optional[np.ndarray] = None     # (X, Y) collection weight
    run_dir: Optional[str] = None

    # ---- derived ----
    @property
    def n(self) -> int:
        return self.locs_um.shape[0]

    @property
    def nt(self) -> int:
        return self.traces.shape[1]

    @property
    def z(self) -> np.ndarray:
        return self.locs_um[:, 2]

    def base_px(self) -> np.ndarray:
        """Lateral positions mapped to movie pixels, ``(N, 2)`` as (a, b) where
        (a, b) = (x_um, y_um) * vres / sfrac - scan_buff / sfrac.  The a/b -> col/row
        assignment (and residual offset) is calibrated in matching."""
        s = self.vres / self.sfrac
        off = self.scan_buff / self.sfrac
        return np.column_stack([self.locs_um[:, 0] * s - off,
                                self.locs_um[:, 1] * s - off])

    def px_per_um(self) -> float:
        return self.vres / self.sfrac

    def sample_mask(self, mask: np.ndarray) -> np.ndarray:
        """Sample an ``(X, Y)`` optical weight image at each neuron's voxel."""
        x = np.clip(np.round(self.locs_um[:, 0] * self.vres).astype(int), 0, mask.shape[0] - 1)
        y = np.clip(np.round(self.locs_um[:, 1] * self.vres).astype(int), 0, mask.shape[1] - 1)
        return mask[x, y]

    # ---- loader ----
    @classmethod
    def from_run(cls, run_dir: str, load_optics: bool = True) -> "GroundTruth":
        tr = np.load(os.path.join(run_dir, "traces.npz"), allow_pickle=True)
        locs = np.asarray(tr["soma_locs"], dtype=np.float64)
        traces = np.asarray(tr["soma_neurons"], dtype=np.float32)
        spikes = np.asarray(tr["spikes_neurons"]) if "spikes_neurons" in tr.files else None

        meta = json.load(open(os.path.join(run_dir, "metadata.json")))
        dt = float(meta.get("dt", 1.0 / meta.get("fps", 20.0)))
        movie_shape = tuple(int(v) for v in meta["movie_shape"][:2])

        params = pickle.load(open(os.path.join(run_dir, "params.pkl"), "rb"))
        sc = params["scan_params"]
        vp = params["vol_params"]
        psf = params["psf_params"]
        vres = float(getattr(vp, "vres", meta.get("vres", 1)))
        sfrac = int(getattr(sc, "sfrac", 2))
        scan_buff = int(getattr(sc, "scan_buff", 0))
        scatter = float(getattr(psf, "scatter_length_um_wf", None) or 1e9)
        focal = getattr(psf, "wf_focal_depth_um", None)
        if focal is None:
            focal = float(meta.get("config", {}).get("focal_depth_um", 0.0) or 0.0)
        focal = float(focal)

        illum = colm = None
        opt_path = os.path.join(run_dir, "optics.npz")
        if load_optics and os.path.isfile(opt_path):
            o = np.load(opt_path, allow_pickle=True)
            if "mask" in o.files:
                illum = np.asarray(o["mask"], dtype=np.float32)
            if "col_mask" in o.files:
                colm = np.asarray(o["col_mask"], dtype=np.float32)

        return cls(locs_um=locs, traces=traces, spikes=spikes, dt=dt,
                   movie_shape=movie_shape, vres=vres, sfrac=sfrac,
                   scan_buff=scan_buff, scatter_length_um=scatter,
                   focal_depth_um=focal, illum_mask=illum, col_mask=colm,
                   run_dir=run_dir)
