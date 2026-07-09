"""The ``neuron_id`` spine: trace <-> 3D geometry <-> 2D footprint.

Everything the three views share is keyed by a single 0-based neuron index
``i`` in ``0 .. n_neur-1``.  This module provides the lookups that turn a click
in any view into that index, and the 2D footprint used to highlight the
selected neuron on the movie.

Picking policy (kept deliberately simple and O(N) per click):
  * 3D  : nearest soma to the picked world point.
  * 2D  : nearest soma to the clicked movie pixel (in movie-pixel space).
  * plot: the trace line carries the index directly.
"""

from __future__ import annotations

import os

import numpy as np

from .model import SimRun


class NeuronTable:
    """Cross-view lookups + lazy 2D footprints for one :class:`SimRun`."""

    def __init__(self, run: SimRun):
        self.run = run
        self._grid_locs = run.soma_grid_locs()            # (N,3) voxel coords
        self._mov_hw = run.mov_clean.shape[1:]             # (H, W)
        gx, gy, _ = run.grid_shape
        H, W = self._mov_hw
        self._sxy = np.array([H / gx, W / gy], dtype=np.float32)  # grid->movie
        # soma centres in movie-pixel coordinates
        self._soma_movie = self._grid_locs[:, :2] * self._sxy
        self._fp_cache: dict[tuple, np.ndarray] = {}
        self._contour_cache: dict[int, list] = {}
        self._all_contours_xy = None

    # ------------------------------------------------------------- picking
    def pick_from_3d(self, world_xyz) -> int:
        """Nearest soma to a picked 3D point (grid/voxel coordinates)."""
        p = np.asarray(world_xyz, dtype=np.float32)[None, :3]
        d = np.linalg.norm(self._grid_locs - p, axis=1)
        return int(np.argmin(d))

    def pick_from_movie(self, row: float, col: float) -> int:
        """Nearest soma to a clicked movie pixel (row, col)."""
        p = np.array([row, col], dtype=np.float32)[None, :]
        d = np.linalg.norm(self._soma_movie - p, axis=1)
        return int(np.argmin(d))

    def soma_movie_xy(self, i: int) -> tuple:
        """(row, col) of neuron ``i``'s soma in movie-pixel space."""
        r, c = self._soma_movie[i]
        return float(r), float(c)

    # ---------------------------------------------------------- footprints
    def footprint(self, i: int, part: str = "all") -> np.ndarray:
        """(H, W) float mask of neuron ``i`` projected onto the movie plane.

        Built by collapsing the neuron's voxels along z, then rasterising to
        movie resolution.  ``part`` in {"all", "soma", "dend"}.  Cached.
        """
        key = (i, part)
        if key in self._fp_cache:
            return self._fp_cache[key]
        vox = self.run.neuron_voxels(i, part=part)
        H, W = self._mov_hw
        fp = np.zeros((H, W), dtype=np.float32)
        if vox.size:
            rows = np.clip((vox[:, 0] * self._sxy[0]).astype(int), 0, H - 1)
            cols = np.clip((vox[:, 1] * self._sxy[1]).astype(int), 0, W - 1)
            np.add.at(fp, (rows, cols), 1.0)
        self._fp_cache[key] = fp
        return fp

    def soma_contours(self, i: int) -> list:
        """Outline polylines of neuron ``i``'s soma footprint.

        Returns a list of ``(M, 2)`` arrays in **(x=col, y=row)** movie
        coordinates -- ready to hand straight to a pyqtgraph curve.  Cached.
        """
        if i in self._contour_cache:
            return self._contour_cache[i]
        from skimage.measure import find_contours
        fp = self.footprint(i, part="soma")
        polys = []
        if fp.max() > 0:
            for c in find_contours(fp, 0.5):
                # find_contours returns (row, col); swap to (x=col, y=row)
                polys.append(np.column_stack([c[:, 1], c[:, 0]]))
        self._contour_cache[i] = polys
        return polys

    def _outlines_cache_path(self):
        return os.path.join(self.run.run_dir, "viz_cache", "soma_outlines.npz")

    def soma_contour_buckets(self, k: int = 16):
        """Soma outlines split into ``k`` colour buckets (neuron id mod k).

        Returns a list of ``k`` ``(x, y)`` NaN-separated arrays -- one pyqtgraph
        curve per bucket then gives neighbouring neurons distinct colours so
        overlapping outlines stay legible.

        Cached in memory and **on disk** (``viz_cache/soma_outlines.npz``): the
        ``prep`` step and the viewer share this, so the (slow) contour extraction
        for many neurons happens once.
        """
        cache = getattr(self, "_bucket_cache", None)
        if cache is not None and cache[0] == k:
            return cache[1]
        path = self._outlines_cache_path()
        if os.path.exists(path):
            try:
                d = np.load(path)
                if int(d["k"]) == k:
                    buckets = [(d[f"x{i}"], d[f"y{i}"]) for i in range(k)]
                    self._bucket_cache = (k, buckets)
                    return buckets
            except Exception:
                pass

        nan = np.array([np.nan])
        xs = [[] for _ in range(k)]
        ys = [[] for _ in range(k)]
        for i in range(self.run.n_neur):
            b = i % k
            for poly in self.soma_contours(i):
                xs[b].append(poly[:, 0]); xs[b].append(nan)
                ys[b].append(poly[:, 1]); ys[b].append(nan)
        buckets = [
            (np.concatenate(x) if x else np.empty(0),
             np.concatenate(y) if y else np.empty(0))
            for x, y in zip(xs, ys)
        ]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            arrs = {"k": np.int32(k)}
            for i, (x, y) in enumerate(buckets):
                arrs[f"x{i}"] = x.astype(np.float32)
                arrs[f"y{i}"] = y.astype(np.float32)
            np.savez_compressed(path, **arrs)
        except Exception:
            pass
        self._bucket_cache = (k, buckets)
        return buckets

    def all_soma_contours_xy(self):
        """All soma outlines as one NaN-separated (x, y) pair.

        Concatenating every neuron's contour polylines with NaN breaks lets a
        single pyqtgraph curve (``connect='finite'``) draw them all at once.
        Built lazily and cached (can take a moment for many neurons).
        """
        if self._all_contours_xy is not None:
            return self._all_contours_xy
        xs, ys = [], []
        nan = np.array([np.nan])
        for i in range(self.run.n_neur):
            for poly in self.soma_contours(i):
                xs.append(poly[:, 0]); xs.append(nan)
                ys.append(poly[:, 1]); ys.append(nan)
        x = np.concatenate(xs) if xs else np.empty(0)
        y = np.concatenate(ys) if ys else np.empty(0)
        self._all_contours_xy = (x, y)
        return self._all_contours_xy
