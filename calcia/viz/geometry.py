"""Voxel -> mesh conversion (the geometry-first strategy).

Sparse structures are turned into PyVista meshes *once* and cached, so the
interactive views only ever push static geometry to the GPU:

  * vessels       : marching cubes on ``neur_ves`` (thick enough) -> smoothed,
                    decimated surface.  Cached to ``<run>/viz_cache/vessels.vtp``.
  * somas         : rendered as GPU point-sprites in the scene (no mesh needed
                    for the overview); this module provides the point cloud and
                    a per-soma marching-cubes surface for the *selected* neuron.
  * dendrites     : thin 1-voxel trees -> point cloud (default) or a best-effort
                    tube built from a 3D skeleton.  Built lazily per neuron.

All meshes live in **grid/voxel coordinates** so vessels, somas and dendrites
share one coordinate frame with ``soma_grid_locs``.

Everything here uses only ``numpy`` / ``scikit-image`` / ``pyvista`` and returns
plain ``pyvista.PolyData`` -- nothing Qt-specific -- so the same meshes feed both
the desktop and the browser (trame) backends.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pyvista as pv
from skimage import measure


# --------------------------------------------------------------------- helpers
def _mc_polydata(mask: np.ndarray, origin=(0, 0, 0), level: float = 0.5) -> pv.PolyData:
    """Marching cubes on a boolean/float sub-volume -> PolyData (voxel coords)."""
    vol = mask.astype(np.float32)
    if vol.max() <= level:
        return pv.PolyData()
    verts, faces, _normals, _vals = measure.marching_cubes(vol, level=level)
    verts = verts + np.asarray(origin, dtype=np.float32)
    faces = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(verts, faces)


def _bbox(vox: np.ndarray, pad: int, shape) -> tuple:
    """Padded bounding box (lo, hi) around integer voxel coords."""
    lo = np.maximum(vox.min(0) - pad, 0)
    hi = np.minimum(vox.max(0) + pad + 1, shape)
    return lo, hi


def _local_mask(vox: np.ndarray, pad: int, shape) -> tuple:
    """Dense boolean sub-volume containing ``vox`` + its origin (lo corner)."""
    lo, hi = _bbox(vox, pad, shape)
    dims = hi - lo
    m = np.zeros(tuple(dims), dtype=bool)
    local = vox - lo
    m[local[:, 0], local[:, 1], local[:, 2]] = True
    return m, lo


# ------------------------------------------------------------------ main class
class GeometryCache:
    """Builds and caches meshes for one run (disk cache for vessels)."""

    def __init__(self, run, cache_dir: Optional[str] = None):
        self.run = run
        self.cache_dir = cache_dir or os.path.join(run.run_dir, "viz_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._dend: dict[tuple, pv.PolyData] = {}
        self._soma: dict[int, pv.PolyData] = {}

    # ------------------------------------------------------------- vessels
    def vessels_cache_path(self, downsample: int = 1, decimate: float = 0.5):
        return os.path.join(self.cache_dir,
                            f"vessels_ds{downsample}_dec{decimate}.vtp")

    def vessels(self, downsample: int = 1, decimate: float = 0.5,
                smooth_iter: int = 20, force: bool = False) -> pv.PolyData:
        """Vessel surface mesh.

        **Cache-first**: if the ``.vtp`` exists we read that small file and never
        touch the multi-GB phase-1 pickle.  Only when the cache is missing *and*
        the vessel voxels are loaded do we build (and cache) the mesh; otherwise
        an empty mesh is returned so the caller can decide whether to pay the
        big load.
        """
        path = self.vessels_cache_path(downsample, decimate)
        if os.path.exists(path) and not force:
            return pv.read(path)
        if self.run.neur_ves is None:
            return pv.PolyData()   # would need the 7.5 GB pickle -- caller opts in

        ves = self.run.neur_ves > 0
        origin = (0.0, 0.0, 0.0)
        if downsample > 1:
            from skimage.measure import block_reduce
            ves = block_reduce(ves, (downsample,) * 3, np.max)
        mesh = _mc_polydata(ves, origin=origin)
        if downsample > 1 and mesh.n_points:
            mesh.points *= downsample
        if mesh.n_points:
            if smooth_iter:
                mesh = mesh.smooth_taubin(n_iter=smooth_iter, pass_band=0.1)
            if decimate and mesh.n_faces > 20000:
                mesh = mesh.decimate(decimate)
            mesh = mesh.compute_normals(auto_orient_normals=True)
            mesh.save(path)
        return mesh

    # --------------------------------------------------------------- somas
    def soma_points(self) -> pv.PolyData:
        """Point cloud of all soma centres (grid coords) with a 0-based 'nid'."""
        pts = self.run.soma_grid_locs().astype(np.float32)
        poly = pv.PolyData(pts)
        poly["nid"] = np.arange(self.run.n_neur, dtype=np.int32)
        return poly

    def all_soma_surfaces(self, verbose: bool = True, force: bool = False,
                          decimate: float = 0.8) -> pv.DataSet:
        """One merged mesh of every neuron's soma surface.

        Each vertex carries a ``nid`` (0-based neuron index) so the scene can
        recolour the whole mesh by per-frame dF/F cheaply.  Building marching
        cubes for all neurons is expensive, so the merged mesh is cached both
        in memory (per session) **and on disk** (``viz_cache/soma_mesh_dec*.vtu``):
        the first enable ever builds it; every launch after that just reads the
        file.

        Each soma is decimated (``decimate`` = fraction of faces removed, 0.8 by
        default) **before** merging.  Full-resolution marching cubes for tens of
        thousands of neurons would merge into tens of millions of points and can
        exhaust memory; decimating per soma keeps peak memory + the on-disk file
        bounded while the blobs still read clearly.
        """
        if getattr(self, "_all_soma", None) is not None:
            return self._all_soma
        path = os.path.join(self.cache_dir, f"soma_mesh_dec{decimate}.vtu")
        if os.path.exists(path) and not force:
            if verbose:
                print(f"[viz] loading cached soma mesh: {path}")
            self._all_soma = pv.read(path)
            return self._all_soma

        # rough peak-memory estimate. A full-res soma is ~2000 pts; after
        # decimation ~2000*(1-decimate) survive. ~100 B/pt covers coords, nid,
        # faces and the transient per-soma copies. Refuse if it won't fit.
        from ._memory import enough_ram_for
        est_bytes = self.run.n_neur * 2000 * (1.0 - decimate) * 100
        if not enough_ram_for(est_bytes):
            raise MemoryError(
                f"soma mesh for {self.run.n_neur} neurons (~{est_bytes/1e9:.1f} "
                "GB) may exhaust RAM; close other apps or skip the 'soma mesh' "
                "layer.")

        blocks = []
        n = self.run.n_neur
        for i in range(n):
            # compute WITHOUT caching the full-res mesh: caching all N would
            # hold tens of millions of points in memory (the OOM we hit).
            surf = self._soma_surface_raw(i)
            if surf.n_points:
                if decimate and surf.n_faces > 60:
                    try:
                        surf = surf.decimate(decimate)
                    except Exception:
                        pass
                surf["nid"] = np.full(surf.n_points, i, dtype=np.int32)
                blocks.append(surf)
            if verbose and n > 500 and i % 2000 == 0 and i:
                print(f"[viz] building soma meshes {i}/{n}...")
        if blocks:
            # combined surfaces already hold the marching-cubes triangles;
            # keep the merged grid as-is (point_data 'nid' is preserved for
            # per-frame recolouring).  Both the merge and the (~100s of MB)
            # save are slow and silent, so announce them: otherwise the
            # terminal looks stuck at the last "N/n" progress line.
            if verbose and n > 500:
                print(f"[viz] merging {len(blocks)} soma meshes + saving cache…")
            self._all_soma = pv.MultiBlock(blocks).combine()
            self._all_soma.save(path)   # build once, reuse across launches
            if verbose:
                print(f"[viz] soma mesh cache saved: {path}")
        else:
            self._all_soma = pv.PolyData()
        return self._all_soma

    def _soma_surface_raw(self, i: int, pad: int = 1) -> pv.PolyData:
        """Marching-cubes surface of neuron ``i``'s soma voxels (NOT cached)."""
        vox = self.run.neuron_voxels(i, part="soma")
        if vox.size == 0:
            return pv.PolyData()
        m, lo = _local_mask(vox, pad, self.run.grid_shape)
        mesh = _mc_polydata(m, origin=lo)
        if mesh.n_points:
            mesh = mesh.smooth_taubin(n_iter=10, pass_band=0.1)
        return mesh

    def soma_surface(self, i: int, pad: int = 1) -> pv.PolyData:
        """Marching-cubes surface of neuron ``i``'s soma voxels (cached).

        Used for the *selected* neuron's highlight -- full resolution.
        """
        if i not in self._soma:
            self._soma[i] = self._soma_surface_raw(i, pad)
        return self._soma[i]

    # ----------------------------------------------------------- dendrites
    def dendrite(self, i: int, mode: str = "points",
                 radius: float = 0.6) -> pv.PolyData:
        """Neuron ``i`` dendrites as a point cloud (default) or tube (cached)."""
        key = (i, mode, radius)
        if key in self._dend:
            return self._dend[key]
        vox = self.run.neuron_voxels(i, part="dend")
        if vox.size == 0:
            self._dend[key] = pv.PolyData()
            return self._dend[key]

        if mode == "points":
            out = pv.PolyData(vox.astype(np.float32))
        elif mode == "tube":
            out = self._skeleton_tube(vox, radius)
        else:
            raise ValueError(f"unknown dendrite mode {mode!r}")
        self._dend[key] = out
        return out

    def _skeleton_tube(self, vox: np.ndarray, radius: float) -> pv.PolyData:
        """Best-effort tube: 3D skeleton -> adjacency line segments -> tube."""
        from skimage.morphology import skeletonize
        m, lo = _local_mask(vox, 1, self.run.grid_shape)
        try:
            sk = skeletonize(m)            # skimage >=0.19 handles 3D
        except (TypeError, ValueError):
            sk = m
        sv = np.argwhere(sk)
        if sv.shape[0] < 2:
            return pv.PolyData(vox.astype(np.float32))
        # 26-connectivity edges between adjacent skeleton voxels
        from scipy.spatial import cKDTree
        tree = cKDTree(sv)
        pairs = tree.query_pairs(r=np.sqrt(3) + 1e-6, output_type="ndarray")
        if pairs.size == 0:
            return pv.PolyData((sv + lo).astype(np.float32))
        pts = (sv + lo).astype(np.float32)
        lines = np.hstack(
            [np.full((pairs.shape[0], 1), 2, dtype=np.int64), pairs.astype(np.int64)]
        ).ravel()
        poly = pv.PolyData(pts)
        poly.lines = lines
        try:
            return poly.tube(radius=radius, n_sides=6)
        except Exception:
            return poly
