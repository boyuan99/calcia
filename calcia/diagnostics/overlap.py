"""Voxel-level overlap diagnostics for neural volume outputs.

Measures whether soma / basal dendrites / apical dendrites / background
dendrites / axons and blood vessels claim disjoint voxels, or whether
multiple components coexist at the same location. Intended to verify
Issue M2 in ``docs/KNOWN_ISSUES.md``.

Workflow:

    >>> from calcia.diagnostics import summarize
    >>> report = summarize(vol_out)
    >>> print(report)

All masks are boolean arrays with the **imaging-region shape**
``(Nx, Ny, Nz)`` matching ``neur_num.shape``. The full-depth
``neur_ves`` (which may include the surface region) is cropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


COMPONENT_ORDER: Tuple[str, ...] = (
    "soma",
    "basal_dendrite",
    "apical_dendrite",
    "bg_dendrite",
    "axon",
    "vessel",
)


def _build_soma_mask(gp_soma, shape: Tuple[int, ...]) -> np.ndarray:
    """Build soma mask from ``gp_soma``.

    Entries may be either a single index array (pre-dendrite shape) or a
    ``(cytoplasm_idx, smoothed_body_idx)`` tuple (post-``grow_neuron_dendrites``
    shape). Both forms are treated as soma tissue.
    """
    mask = np.zeros(shape, dtype=bool)
    flat = mask.ravel()
    for entry in gp_soma:
        if entry is None:
            continue
        # Tuple form: aggregate both arrays
        if isinstance(entry, tuple):
            for arr in entry:
                arr = np.asarray(arr)
                if arr.size > 0:
                    flat[arr] = True
        else:
            arr = np.asarray(entry)
            if arr.size > 0:
                flat[arr] = True
    return mask


def _build_axon_mask(gp_bgvals, shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    flat = mask.ravel()
    for entry in gp_bgvals:
        idx_arr = entry[0] if isinstance(entry, tuple) else entry.indices
        arr = np.asarray(idx_arr)
        if arr.size > 0:
            flat[arr] = True
    return mask


def _crop_vessel_mask(
    neur_ves: Optional[np.ndarray],
    imaging_shape: Tuple[int, ...],
) -> np.ndarray:
    """Return imaging-region vessel mask.

    ``neur_ves`` from the pipeline is full-depth: ``(Nx, Ny, full_z)``
    where ``full_z = surface_z + imaging_z``. The imaging region is the
    **trailing** slice along axis 2 (``vessel_mask_full[:, :, z_offset:]``
    in ``pipeline.py``). Handles the vessels-disabled case (returns
    all-False mask).
    """
    if neur_ves is None:
        return np.zeros(imaging_shape, dtype=bool)

    if neur_ves.shape == imaging_shape:
        return neur_ves.astype(bool)

    if neur_ves.shape[:2] != imaging_shape[:2]:
        raise ValueError(
            f"neur_ves XY shape {neur_ves.shape[:2]} does not match "
            f"imaging XY shape {imaging_shape[:2]}"
        )

    full_z = neur_ves.shape[2]
    imaging_z = imaging_shape[2]
    if full_z < imaging_z:
        raise ValueError(
            f"neur_ves depth {full_z} is smaller than imaging depth {imaging_z}"
        )

    z_offset = full_z - imaging_z
    return neur_ves[:, :, z_offset:].astype(bool)


def component_masks(vol_out) -> Dict[str, np.ndarray]:
    """Extract boolean masks for each neural component plus vessels.

    Args:
        vol_out: :class:`calcia.pipeline.NeuralVolumeOutput` instance.

    Returns:
        Dict with keys in :data:`COMPONENT_ORDER`. Each value is a bool
        array with the imaging-region shape. Apical voxels are **not**
        double-counted in basal; basal excludes both soma and apical.

    Notes:
        ID scheme inside ``neur_num`` (uint16):

        * ``0``                                — empty
        * ``1 .. N_neur``                      — neuron (soma + basal)
        * ``N_neur+1 .. N_neur+N_den``         — apical dendrites
        * ``N_neur+N_den+1 .. +N_den2``        — background dendrites

        Soma is disambiguated from basal via ``gp_soma``; apical is
        read from the separate ``neur_num_ad`` volume.
    """
    neur_num = vol_out.neur_num
    shape = neur_num.shape

    vol_params = vol_out.params["vol_params"]
    n_neur = int(vol_params.N_neur)
    n_den = int(vol_params.N_den)

    soma = _build_soma_mask(vol_out.gp_soma, shape)

    neuron_ids = (neur_num >= 1) & (neur_num <= n_neur)
    basal = neuron_ids & ~soma

    apical = (vol_out.neur_num_ad > 0)
    # Make basal strictly exclusive of apical too (apical voxels shouldn't
    # also carry a basal-ID in neur_num, but guard against any residual)
    basal = basal & ~apical

    bg_dend = (neur_num > (n_neur + n_den))

    axon = _build_axon_mask(vol_out.gp_bgvals, shape)

    vessel = _crop_vessel_mask(vol_out.neur_ves, shape)

    return {
        "soma": soma,
        "basal_dendrite": basal,
        "apical_dendrite": apical,
        "bg_dendrite": bg_dend,
        "axon": axon,
        "vessel": vessel,
    }


def owner_count_histogram(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Histogram of how many components co-own each voxel.

    Vessels are **not** counted as an owner (they are separate tissue).

    Returns:
        1D int64 array of length ``n_components + 1`` (here: 6). Entry
        ``i`` = number of voxels claimed by exactly ``i`` neural components.
    """
    component_keys = [k for k in masks if k != "vessel"]
    if not component_keys:
        return np.zeros(1, dtype=np.int64)

    shape = masks[component_keys[0]].shape
    total = np.zeros(shape, dtype=np.uint8)
    for k in component_keys:
        total += masks[k].astype(np.uint8)

    hist = np.bincount(total.ravel(), minlength=len(component_keys) + 1)
    return hist.astype(np.int64)


def pairwise_overlap(
    masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Pairwise intersection voxel counts.

    Returns:
        ``(matrix, keys)`` where ``matrix[i, j]`` is the number of voxels
        in both ``masks[keys[i]]`` and ``masks[keys[j]]``. The diagonal
        ``matrix[i, i]`` is the total voxels in that mask.
    """
    keys = [k for k in COMPONENT_ORDER if k in masks]
    n = len(keys)
    mat = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        mi = masks[keys[i]]
        mat[i, i] = int(mi.sum())
        for j in range(i + 1, n):
            mj = masks[keys[j]]
            v = int(np.logical_and(mi, mj).sum())
            mat[i, j] = v
            mat[j, i] = v
    return mat, keys


def component_vs_vessel(
    masks: Dict[str, np.ndarray],
) -> Dict[str, Tuple[int, float]]:
    """Per-component vessel-intersection voxel count and fraction.

    Returns:
        Dict keyed by component name (excluding ``vessel``). Each value
        is ``(intersection_voxels, fraction_of_component)``.
        Empty dict when vessels are disabled.
    """
    vessel = masks.get("vessel")
    out: Dict[str, Tuple[int, float]] = {}
    if vessel is None or not vessel.any():
        return out

    for k, m in masks.items():
        if k == "vessel":
            continue
        n_inter = int(np.logical_and(m, vessel).sum())
        n_comp = int(m.sum())
        frac = (n_inter / n_comp) if n_comp > 0 else 0.0
        out[k] = (n_inter, frac)
    return out


@dataclass
class OverlapReport:
    """Aggregated overlap diagnostic.

    Attributes:
        total_voxels: Total imaging-region voxels.
        component_counts: Per-component occupied voxel counts.
        owner_hist: ``owner_count_histogram`` output. ``owner_hist[0]``
            = voxels claimed by no neural component.
        pair_matrix: ``pairwise_overlap`` matrix.
        pair_keys: Row/column labels for ``pair_matrix``.
        vessel_overlap: ``component_vs_vessel`` output.
    """

    total_voxels: int
    component_counts: Dict[str, int]
    owner_hist: np.ndarray
    pair_matrix: np.ndarray
    pair_keys: List[str]
    vessel_overlap: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines: List[str] = []
        lines.append("=" * 64)
        lines.append(f"OverlapReport  (total imaging voxels: {self.total_voxels:,})")
        lines.append("=" * 64)

        lines.append("\nPer-component occupied voxels:")
        for k, n in self.component_counts.items():
            frac = 100 * n / self.total_voxels if self.total_voxels else 0.0
            lines.append(f"  {k:<18s} {n:>12,d}  ({frac:5.2f}%)")

        lines.append("\nOwner histogram (how many neural components per voxel):")
        for i, n in enumerate(self.owner_hist):
            frac = 100 * n / self.total_voxels if self.total_voxels else 0.0
            marker = "  (empty)" if i == 0 else (
                "  (single owner)" if i == 1 else "  (multi-owner)"
            )
            lines.append(f"  {i} component(s): {int(n):>12,d}  ({frac:5.2f}%){marker}")

        lines.append("\nPairwise intersection (voxels; diagonal = total):")
        header = "                " + " ".join(f"{k[:10]:>10s}" for k in self.pair_keys)
        lines.append(header)
        for i, k in enumerate(self.pair_keys):
            row = f"  {k[:14]:<14s}  " + " ".join(
                f"{int(self.pair_matrix[i, j]):>10d}" for j in range(len(self.pair_keys))
            )
            lines.append(row)

        if self.vessel_overlap:
            lines.append("\nComponent vs vessel (intersection voxels, fraction of component):")
            for k, (n, frac) in self.vessel_overlap.items():
                lines.append(f"  {k:<18s} {n:>10,d}  ({100*frac:6.3f}% of {k})")
        else:
            lines.append("\nComponent vs vessel: (vessels disabled or empty)")

        lines.append("=" * 64)
        return "\n".join(lines)


def summarize(vol_out) -> OverlapReport:
    """Run all diagnostics against a ``NeuralVolumeOutput`` and aggregate.

    Args:
        vol_out: Pipeline output (:class:`calcia.pipeline.NeuralVolumeOutput`).

    Returns:
        :class:`OverlapReport` with owner histogram, pairwise matrix, and
        vessel-overlap fractions. Does not mutate ``vol_out``.
    """
    masks = component_masks(vol_out)
    shape = masks["soma"].shape
    total_voxels = int(np.prod(shape))

    component_counts = {k: int(m.sum()) for k, m in masks.items() if k != "vessel"}
    component_counts["vessel"] = int(masks["vessel"].sum())

    hist = owner_count_histogram(masks)
    mat, keys = pairwise_overlap(masks)
    vessel_stats = component_vs_vessel(masks)

    return OverlapReport(
        total_voxels=total_voxels,
        component_counts=component_counts,
        owner_hist=hist,
        pair_matrix=mat,
        pair_keys=keys,
        vessel_overlap=vessel_stats,
    )
