"""Background neuropil and axon generation (Step 7).

Part A: Background dendrites originating from volume edges.
Part B: Axon processes with branching that fill available volume.
Part C: Axon sorting into correlated background components.

Port of MATLAB ``generate_bgdendrites.m``, ``generate_axons.m``,
``sort_axons.m``.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..algorithms.random_walk import dendrite_random_walk
from ..config.params import AxonParams, BgParams, DendParams, VolumeParams
from ..volume.fluorescence import CellFluorescenceData

FLT_MAX = np.finfo(np.float32).max


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class BgDendriteResult:
    """Result of background dendrite generation (Step 7A).

    Attributes:
        neur_num: Updated 3D uint16 volume with bg dendrite IDs.
        neur_vol: Updated 3D float32 fluorescence volume.
        gp_vals: Extended list of CellFluorescenceData.
        neur_locs: Updated (N_total, 3) float32 neuron/bg positions (um).
        N_den2: Number of background dendrite processes generated.
    """
    neur_num: np.ndarray
    neur_vol: np.ndarray
    gp_vals: List[CellFluorescenceData]
    neur_locs: np.ndarray
    N_den2: int


@dataclass
class AxonResult:
    """Result of axon generation (Step 7B).

    Attributes:
        neur_vol: Updated 3D float32 fluorescence volume.
        gp_bgvals: List of (indices, fluorescence) tuples per axon.
            indices are C-order int32 linear indices.
        N_bg_actual: Actual number of axon processes generated.
    """
    neur_vol: np.ndarray
    gp_bgvals: List[Tuple[np.ndarray, np.ndarray]]
    N_bg_actual: int


@dataclass
class BgProcessData:
    """A grouped background process (from sort_axons).

    Attributes:
        indices: 1D int32 C-order linear indices.
        fluorescence: 1D float32 per-voxel fluorescence values.
    """
    indices: np.ndarray
    fluorescence: np.ndarray


# ------------------------------------------------------------------
# Helper: import dilation from dendrites module
# ------------------------------------------------------------------

def _import_dilate():
    """Lazy import to avoid circular dependency."""
    from ..volume.dendrites import _dilate_dendrite_paths
    return _dilate_dendrite_paths


# ------------------------------------------------------------------
# Part A: Background dendrites
# ------------------------------------------------------------------

def generate_bg_dendrites(
    vol_params: VolumeParams,
    bg_params: BgParams,
    dend_params: DendParams,
    neur_vol: np.ndarray,
    neur_num: np.ndarray,
    gp_vals: List[CellFluorescenceData],
    gp_nuc: List[Tuple[np.ndarray, float]],
    neur_locs: np.ndarray,
    neur_vol_flag: bool = True,
    verbose: Optional[int] = None,
) -> BgDendriteResult:
    """Generate background neuropil dendrite processes.

    Background dendrites originate from outside the volume and grow
    inward through random walks.  They are then dilated to give width
    and assigned fluorescence values.

    Port of MATLAB ``generate_bgdendrites.m``.

    Args:
        vol_params: Volume parameters.
        bg_params: Background parameters.
        dend_params: Dendrite parameters (dtParams, thicknessScale, weightScale).
        neur_vol: 3D float32 fluorescence volume (copied internally).
        neur_num: 3D uint16 neuron ID volume (copied internally).
        gp_vals: Existing fluorescence data list (will be extended).
        gp_nuc: Nucleus data from NeuralVolumeResult.
        neur_locs: (N, 3) neuron positions in um.
        neur_vol_flag: If True, rebuild and update neur_vol.
        verbose: Verbosity level.

    Returns:
        BgDendriteResult with updated volumes, extended gp_vals, and N_den2.
    """
    if verbose is None:
        verbose = vol_params.verbose

    vres = vol_params.vres
    N_neur = vol_params.N_neur
    N_den = vol_params.N_den

    # Extract and scale dendrite parameters (MATLAB lines 107-111)
    dtParams = list(dend_params.dtParams)
    thicknessScale = dend_params.thicknessScale
    dtParams_scaled = dtParams.copy()
    dtParams_scaled[1] = dtParams[1] * vres  # horiz radius in voxels
    dtParams_scaled[2] = dtParams[2] * vres  # vert radius in voxels
    thicknessScale = thicknessScale * vres * vres

    volsize = np.array(vol_params.vol_sz, dtype=np.int32) * vres

    # Copy inputs
    neur_num_out = neur_num.copy()
    neur_locs_out = neur_locs.copy()

    # Background mask: available space (MATLAB lines 101-104)
    bg_pix = (neur_num_out == 0)
    for kk in range(N_neur):
        if kk < len(gp_nuc) and len(gp_nuc[kk][0]) > 0:
            nuc_idx = gp_nuc[kk][0]
            bg_pix.ravel()[nuc_idx] = False

    # Rebuild neur_vol from gp_vals + gp_nuc if requested (MATLAB lines 120-131)
    if neur_vol_flag:
        neur_vol_out = np.zeros(neur_num.shape, dtype=np.float32)
        for kk in range(len(gp_vals)):
            g = gp_vals[kk]
            if len(g.indices) > 0:
                neur_vol_out.ravel()[g.indices] = g.fluorescence
            if kk < len(gp_nuc):
                nuc_idx, nuc_fl = gp_nuc[kk]
                if len(nuc_idx) > 0:
                    neur_vol_out.ravel()[nuc_idx] = nuc_fl
    else:
        neur_vol_out = neur_vol.astype(np.float32).copy()

    # Cost matrix (MATLAB lines 139-146)
    M = np.random.random(tuple(volsize)).astype(np.float32)
    M[~bg_pix] = FLT_MAX
    # Boundary walls
    M[0, :, :] = FLT_MAX
    M[-1, :, :] = FLT_MAX
    M[:, 0, :] = FLT_MAX
    M[:, -1, :] = FLT_MAX
    M[:, :, 0] = FLT_MAX
    M[:, :, -1] = FLT_MAX

    dendVar = dend_params.dendVar if dend_params.dendVar is not None else 0.25

    idxvol = np.zeros(tuple(volsize), dtype=np.uint16)
    numvol = np.zeros(tuple(volsize), dtype=np.float32)

    maxlength = int(bg_params.maxlength)
    distsc = bg_params.distsc
    fillweight = bg_params.fillweight
    maxel = bg_params.maxel
    minlength = int(bg_params.minlength)
    dtSize = np.array([dtParams_scaled[1], dtParams_scaled[1], dtParams_scaled[2]])
    shiftdist = 3

    target_processes = int(
        (np.prod(volsize + 2 * dtSize) / np.prod(volsize) - 1) * N_neur
    )

    if verbose >= 1:
        print(f"Generating background dendrites... target: {target_processes}")

    idx = 0  # bg dendrite counter
    num_branches = int(dtParams[0])

    for j in range(target_processes):
        dendpts = []

        # Pick root OUTSIDE volume (MATLAB lines 179-182, 0-based)
        while True:
            root = np.floor(
                np.random.random(3) * (volsize + 2 * dtSize) - dtSize
            ).astype(np.int32)
            # Root must be outside [0, volsize)
            if not (np.all(root >= 0) and np.all(root < volsize)):
                break

        neur_locs_out = np.vstack([
            neur_locs_out,
            (root / vres).astype(np.float32).reshape(1, 3),
        ])

        # Generate branches (MATLAB lines 185-231)
        for _br in range(num_branches):
            theta = np.random.random() * 2 * np.pi
            r = np.sqrt(np.random.random()) * dtParams_scaled[1]
            dends = np.floor([
                r * np.cos(theta) + root[0],
                r * np.sin(theta) + root[1],
                2 * dtParams_scaled[2] * (np.random.random() - 0.5) + root[2],
            ]).astype(np.int32)

            # Only proceed if proposed endpoint is inside volume (MATLAB line 190)
            if not (np.all(dends >= 0) and np.all(dends < volsize)):
                continue

            # Project root→endpoint line to volume boundary (MATLAB line 191)
            diff = dends - root
            shifts = np.zeros(6, dtype=np.float64)
            # Lower bounds: root[i] < 0 → shift to boundary 0
            for ax in range(3):
                if root[ax] < 0 and diff[ax] != 0:
                    shifts[ax] = (0 - root[ax]) / diff[ax]
                if root[ax] >= volsize[ax] and diff[ax] != 0:
                    shifts[ax + 3] = ((volsize[ax] - 1) - root[ax]) / diff[ax]

            max_shift_idx = np.argmax(shifts)
            max_shift = shifts[max_shift_idx]

            bgpts_arr = np.zeros((0, 3), dtype=np.int32)
            for _attempt in range(30):
                root2 = np.round(max_shift * diff + root).astype(np.int32)

                # Add random jitter (MATLAB lines 197-203)
                # The axis of entry gets 0 jitter, others get random
                entry_axis = max_shift_idx % 3
                for ax in range(3):
                    if ax != entry_axis:
                        root2[ax] += np.random.randint(1, shiftdist + 1)

                # Clamp to volume bounds (0-based)
                root2 = np.clip(root2, 0, volsize - 1)

                bgpts_arr = dendrite_random_walk(
                    M, root2, dends, distsc, maxlength,
                    fillweight, maxel, minlength,
                )

                if len(bgpts_arr) > 0:
                    # Prepend root2
                    bgpts_arr = np.vstack([
                        root2.reshape(1, 3), bgpts_arr
                    ])

                    # Compute thickness weights (MATLAB lines 216-221)
                    n_pts = len(bgpts_arr)
                    dend_sz = max(0, np.random.normal(1, dendVar)) ** 2
                    if n_pts > 2:
                        d1 = np.diff(bgpts_arr.astype(np.float32), axis=0)
                        d2 = np.diff(np.abs(d1), axis=0)
                        curvature = np.sum(np.abs(d2), axis=1) / 2.0
                        bgptsW = np.ones(n_pts, dtype=np.float32)
                        bgptsW[0] = 0.0
                        bgptsW[1:-1] = 1.0 - (1.0 - 1.0 / np.sqrt(2)) * curvature
                        bgptsW[-1] = 0.0
                        bgptsW *= dend_sz
                    else:
                        bgptsW = dend_sz * np.ones(n_pts, dtype=np.float32)

                    # Convert to linear indices (C-order)
                    bgptsI = np.ravel_multi_index(
                        (bgpts_arr[:, 0], bgpts_arr[:, 1], bgpts_arr[:, 2]),
                        tuple(volsize),
                    )
                    dendpts.extend(bgptsI.tolist())
                    numvol.ravel()[bgptsI] = bgptsW
                    break

        # Finalize this process (MATLAB lines 233-238)
        if len(dendpts) > 0:
            idx += 1
            dendpts_arr = np.array(dendpts, dtype=np.int64)
            idxvol.ravel()[dendpts_arr] = idx
            numvol.ravel()[dendpts_arr] *= thicknessScale * dtParams[3]

    if verbose >= 1:
        print(f"  Generated {idx} background processes, dilating...")

    # Dilate paths (MATLAB line 250)
    dilate_fn = _import_dilate()
    dilated_ids = dilate_fn(
        numvol.astype(np.uint16), idxvol, (neur_num_out > 0).astype(np.uint16),
        tuple(volsize),
    )

    # Merge into neur_num (MATLAB lines 251-254)
    # dilated_ids contains BOTH original neuron IDs (from neur_num.copy())
    # and new bg dendrite IDs (1..idx). We only want the new bg IDs,
    # which are at voxels that were previously empty.
    N_den2 = idx
    Ncomps = N_neur + N_den
    bg_new_mask = (dilated_ids > 0) & (neur_num_out == 0)
    neur_num_out[bg_new_mask] = (
        dilated_ids[bg_new_mask] + Ncomps
    ).astype(np.uint16)

    # Assign fluorescence (MATLAB lines 255-265)
    wtSc = dend_params.weightScale
    gp_vals_out = list(gp_vals)  # shallow copy

    for i in range(Ncomps + 1, Ncomps + N_den2 + 1):
        indices = np.flatnonzero(neur_num_out.ravel() == i).astype(np.int32)
        n_vox = len(indices)
        if n_vox > 0:
            fluorescence = np.float32(
                (wtSc[1] * np.exp(-(dtParams[1] / wtSc[0]))
                 + (1 - wtSc[1]))
                * (1 - wtSc[2] * np.random.random(n_vox))
            ).astype(np.float32)
        else:
            fluorescence = np.array([], dtype=np.float32)

        soma_mask = np.zeros(n_vox, dtype=bool)

        if neur_vol_flag and n_vox > 0:
            neur_vol_out.ravel()[indices] = fluorescence

        gp_vals_out.append(CellFluorescenceData(
            indices=indices,
            fluorescence=fluorescence,
            soma_mask=soma_mask,
        ))

    if verbose >= 1:
        print(f"done. {N_den2} background dendrite processes.")

    return BgDendriteResult(
        neur_num=neur_num_out,
        neur_vol=neur_vol_out if neur_vol_flag else neur_vol,
        gp_vals=gp_vals_out,
        neur_locs=neur_locs_out,
        N_den2=N_den2,
    )


# ------------------------------------------------------------------
# Part B: Axon generation
# ------------------------------------------------------------------

def generate_axons(
    vol_params: VolumeParams,
    axon_params: AxonParams,
    neur_vol: np.ndarray,
    neur_num: np.ndarray,
    gp_vals: List[CellFluorescenceData],
    gp_nuc: List[Tuple[np.ndarray, float]],
    neur_vol_flag: bool = True,
    verbose: Optional[int] = None,
) -> AxonResult:
    """Generate axon processes with branching.

    Axons grow from random positions in the volume via random walks,
    with stochastic branching.  The volume is padded to allow smooth
    boundary behaviour.  Fluorescence is additive.

    Port of MATLAB ``generate_axons.m``.

    Args:
        vol_params: Volume parameters (N_bg controls target count).
        axon_params: Axon generation parameters.
        neur_vol: 3D float32 fluorescence volume (copied internally).
        neur_num: 3D uint16 neuron ID volume.
        gp_vals: Fluorescence data list.
        gp_nuc: Nucleus data from NeuralVolumeResult.
        neur_vol_flag: If True, rebuild and update neur_vol.
        verbose: Verbosity level.

    Returns:
        AxonResult with updated neur_vol, gp_bgvals, N_bg_actual.
    """
    if verbose is None:
        verbose = vol_params.verbose

    vres = vol_params.vres
    volsize = np.array(vol_params.vol_sz, dtype=np.int32) * vres
    N_bg = vol_params.N_bg
    padsize = axon_params.padsize

    # Background mask (MATLAB lines 104-108)
    bg_pix = (neur_num == 0)
    for kk in range(len(gp_nuc)):
        nuc_idx = gp_nuc[kk][0]
        if len(nuc_idx) > 0:
            bg_pix.ravel()[nuc_idx] = False

    fillnum = int(round(axon_params.maxfill * axon_params.maxvoxel * np.sum(bg_pix)))

    # Rebuild neur_vol (MATLAB lines 120-132)
    if neur_vol_flag:
        neur_vol_out = np.zeros(neur_num.shape, dtype=np.float32)
        for kk in range(len(gp_vals)):
            g = gp_vals[kk]
            if len(g.indices) > 0:
                neur_vol_out.ravel()[g.indices] = g.fluorescence
            if kk < len(gp_nuc):
                nuc_idx, nuc_fl = gp_nuc[kk]
                if len(nuc_idx) > 0:
                    neur_vol_out.ravel()[nuc_idx] = nuc_fl
    else:
        neur_vol_out = neur_vol.astype(np.float32).copy()

    # Padded cost matrix (MATLAB lines 139-144)
    volpad = volsize + 2 * padsize
    M = np.random.random(tuple(volpad)).astype(np.float32)

    # Pad bg_pix with False (occupied), set occupied to FLT_MAX
    bg_pix_inv_padded = np.pad(
        ~bg_pix, pad_width=padsize, constant_values=True,
    )
    M[bg_pix_inv_padded] = FLT_MAX

    if verbose >= 1:
        print(f"Generating axons... target: {N_bg}")

    gp_bgvals: List[Tuple[np.ndarray, np.ndarray]] = []
    j = 0
    numit2 = 0
    nummax = 10000

    while fillnum > 0 and j < N_bg and numit2 < nummax:
        # Main trunk (MATLAB lines 160-179)
        bgpts = np.zeros((0, 3), dtype=np.int32)
        numit2 = 0  # Reset per outer iteration (MATLAB line 161)
        numit2_local = 0

        while len(bgpts) < axon_params.minlength and numit2 < nummax:
            numit2 += 1
            numit2_local += 1

            # Random root in low-cost region (0-based)
            root = np.ceil(
                (volpad - 2) * np.random.random(3)
            ).astype(np.int32)
            while M[root[0], root[1], root[2]] > (axon_params.fillweight * axon_params.maxvoxel):
                root = np.ceil(
                    (volpad - 2) * np.random.random(3)
                ).astype(np.int32)

            # Random endpoint
            ends = np.ceil(
                root + 2 * axon_params.maxdist * vres * (np.random.random(3) - 0.5)
            ).astype(np.int32)
            ends = np.clip(ends, 0, volpad - 1)

            bgpts = dendrite_random_walk(
                M, root, ends, axon_params.distsc,
                int(axon_params.maxlength), axon_params.fillweight,
                axon_params.maxvoxel, int(axon_params.minlength),
            )

        if len(bgpts) == 0:
            continue

        # Branching (MATLAB lines 181-207)
        nbranches = max(0, int(round(
            axon_params.numbranches + axon_params.varbranches * np.random.randn()
        )))

        for _br in range(nbranches):
            bgpts2 = np.zeros((0, 3), dtype=np.int32)
            for _attempt in range(100):
                # Random branch root from existing path
                br_idx = np.random.randint(0, len(bgpts))
                br_root = bgpts[br_idx].copy()

                # Skip boundary points
                if (br_root[0] == 0 or br_root[0] == volpad[0] - 1 or
                        br_root[1] == 0 or br_root[1] == volpad[1] - 1 or
                        br_root[2] == 0 or br_root[2] == volpad[2] - 1):
                    continue

                br_ends = np.ceil(
                    br_root + 2 * axon_params.maxdist * vres
                    * (np.random.random(3) - 0.5)
                ).astype(np.int32)
                br_ends = np.clip(br_ends, 0, volpad - 1)

                bgpts2 = dendrite_random_walk(
                    M, br_root, br_ends, axon_params.distsc,
                    int(axon_params.maxlength), axon_params.fillweight,
                    axon_params.maxvoxel, int(axon_params.minlength),
                )

                if len(bgpts2) >= axon_params.minlength:
                    break

            if len(bgpts2) > 0:
                bgpts = np.vstack([bgpts, bgpts2])

        # Strip padding, keep only in-volume voxels (MATLAB lines 209-214)
        bgpts = bgpts - padsize
        in_bounds = (
            (bgpts[:, 0] >= 0) & (bgpts[:, 0] < volsize[0]) &
            (bgpts[:, 1] >= 0) & (bgpts[:, 1] < volsize[1]) &
            (bgpts[:, 2] >= 0) & (bgpts[:, 2] < volsize[2])
        )
        bgpts = bgpts[in_bounds]

        if len(bgpts) == 0:
            continue

        # Convert to C-order linear indices (MATLAB lines 216-217 use Fortran)
        indices = np.ravel_multi_index(
            (bgpts[:, 0], bgpts[:, 1], bgpts[:, 2]),
            tuple(volsize),
        ).astype(np.int32)

        # Fluorescence (MATLAB lines 218-219)
        scale = max(0.0, 1.0 + axon_params.varfill * np.random.randn())
        fluorescence = np.float32(1.0 / axon_params.maxel) * np.ones(
            len(bgpts), dtype=np.float32
        ) * np.float32(scale)

        fillnum -= len(bgpts)

        if neur_vol_flag:
            neur_vol_out.ravel()[indices] += fluorescence

        gp_bgvals.append((indices, fluorescence))
        j += 1

    if verbose >= 1:
        print(f"done. Generated {j} axon processes.")

    return AxonResult(
        neur_vol=neur_vol_out if neur_vol_flag else neur_vol,
        gp_bgvals=gp_bgvals,
        N_bg_actual=j,
    )


# ------------------------------------------------------------------
# Part C: Axon sorting
# ------------------------------------------------------------------

def sort_axons(
    vol_params: VolumeParams,
    axon_params: AxonParams,
    gp_bgvals: List[Tuple[np.ndarray, np.ndarray]],
    cell_pos: np.ndarray,
    verbose: Optional[int] = None,
) -> List[BgProcessData]:
    """Sort axon processes into correlated background components.

    When N_proc > N_comps, the first N_comps processes are assigned to
    the nearest cell via greedy nearest-neighbour matching.  Remaining
    axons are distributed randomly.

    Port of MATLAB ``sort_axons.m``.

    Args:
        vol_params: Volume parameters.
        axon_params: Axon parameters (N_proc controls output size).
        gp_bgvals: List of (indices, fluorescence) per axon.
        cell_pos: (N_total, 3) positions in voxel coordinates.
        verbose: Verbosity level.

    Returns:
        List of BgProcessData, length = N_proc.
    """
    if verbose is None:
        verbose = vol_params.verbose

    N_proc = axon_params.N_proc
    N_comps = vol_params.N_neur + vol_params.N_den
    volsize = np.array(vol_params.vol_sz, dtype=np.int32) * vol_params.vres

    if verbose >= 1:
        print("Sorting axons...", end="")

    # Initialize output bins
    bg_proc = [BgProcessData(
        indices=np.array([], dtype=np.int32),
        fluorescence=np.array([], dtype=np.float32),
    ) for _ in range(N_proc)]

    n_axons = len(gp_bgvals)
    if n_axons == 0:
        if verbose >= 1:
            print("done.")
        return bg_proc

    if N_proc > N_comps and N_comps > 0 and n_axons > 0:
        # Compute axon centroids (MATLAB lines 77-84)
        gp_bgpos = np.zeros((n_axons, 3), dtype=np.float32)
        for kk in range(n_axons):
            idx = gp_bgvals[kk][0]
            if len(idx) > 0:
                coords = np.array(np.unravel_index(idx, tuple(volsize)))
                gp_bgpos[kk] = coords.mean(axis=1)

        # Greedy nearest-available-axon per cell, via a KDTree over the axon
        # centroids. Semantically identical to the old full (N_comps x n_axons)
        # distance matrix + per-cell argmin + column-removal, but O((N+M) log M)
        # time and O(M) memory instead of O(N*M): the matrix materialised
        # ~N_comps*n_axons floats (>60 GB at high neuropil density -> OOM). The
        # only possible divergence from the old result is on EXACT distance ties
        # (argmin picks the lowest index; the tree may pick another) — negligible
        # for float centroids and immaterial to the ownerless-neuropil grouping.
        from scipy.spatial import cKDTree

        cell_pos2 = cell_pos[:N_comps]
        tree = cKDTree(gp_bgpos)
        taken = np.zeros(n_axons, dtype=bool)
        assigned = set()
        for ii in range(min(N_comps, n_axons)):
            k = 8
            idx = None
            while True:
                kq = min(k, n_axons)
                _d, cand = tree.query(cell_pos2[ii], k=kq)
                for c in np.atleast_1d(cand):
                    c = int(c)
                    if not taken[c]:
                        idx = c
                        break
                if idx is not None or kq >= n_axons:
                    break
                k *= 4
            if idx is None:   # every axon already assigned (n_axons <= N_comps)
                break
            taken[idx] = True
            assigned.add(idx)
            bg_proc[ii] = BgProcessData(
                indices=gp_bgvals[idx][0].copy(),
                fluorescence=gp_bgvals[idx][1].copy(),
            )

        # Random assignment of remaining (MATLAB lines 100-106)
        for kk in range(n_axons):
            if kk not in assigned:
                bin_idx = N_comps + np.random.randint(0, max(1, N_proc - N_comps))
                if bin_idx >= N_proc:
                    bin_idx = N_proc - 1
                bg_proc[bin_idx] = BgProcessData(
                    indices=np.concatenate([
                        bg_proc[bin_idx].indices, gp_bgvals[kk][0],
                    ]),
                    fluorescence=np.concatenate([
                        bg_proc[bin_idx].fluorescence, gp_bgvals[kk][1],
                    ]),
                )
    else:
        # Random assignment of all axons (MATLAB lines 108-112)
        for kk in range(n_axons):
            bin_idx = np.random.randint(0, N_proc)
            bg_proc[bin_idx] = BgProcessData(
                indices=np.concatenate([
                    bg_proc[bin_idx].indices, gp_bgvals[kk][0],
                ]),
                fluorescence=np.concatenate([
                    bg_proc[bin_idx].fluorescence, gp_bgvals[kk][1],
                ]),
            )

    if verbose >= 1:
        print("done.")

    return bg_proc
