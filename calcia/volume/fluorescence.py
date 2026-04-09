"""Cell fluorescence distribution (Step 6).

Sets non-uniform fluorescence for neuron somas (via 3D Gaussian Process),
dendrites (exponential distance decay), and through-volume apical dendrites
(uniform).

Port of MATLAB ``setCellFluoresence.m``.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..algorithms.gaussian_process import sample_3d_gp
from ..config.params import DendParams, NeuronParams, VolumeParams


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellFluorescenceData:
    """Per-component fluorescence data.

    Each component is either a neuron (ID 1..N_neur) or a through-volume
    apical dendrite (ID N_neur+1..N_neur+N_den).

    Attributes:
        indices: 1D int32 array of C-order linear indices into the volume.
        fluorescence: 1D float32 array of per-voxel fluorescence values.
        soma_mask: 1D bool array indicating which voxels are soma.
    """
    indices: np.ndarray
    fluorescence: np.ndarray
    soma_mask: np.ndarray


@dataclass
class FluorescenceResult:
    """Result of cell fluorescence setup (Step 6).

    Attributes:
        gp_vals: List of CellFluorescenceData, one per component.
                 Length = N_neur + N_den.
        neur_vol: 3D float32 array with fluorescence values at all
                  occupied voxels.
    """
    gp_vals: List[CellFluorescenceData]
    neur_vol: np.ndarray


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def set_cell_fluorescence(
    vol_params: VolumeParams,
    neur_params: NeuronParams,
    dend_params: DendParams,
    neur_num: np.ndarray,
    neur_soma: np.ndarray,
    neur_num_ad: np.ndarray,
    positions: np.ndarray,
    neur_vol: np.ndarray,
    verbose: Optional[int] = None,
) -> FluorescenceResult:
    """Set non-uniform fluorescence distributions for all cells.

    For each neuron: soma gets GP-sampled heterogeneous fluorescence,
    dendrites get exponential distance decay, apical dendrites get 1.0.
    Through-volume apical dendrites get uniform 1.0.

    Port of MATLAB ``setCellFluoresence.m``.

    Args:
        vol_params: Volume parameters.
        neur_params: Neuron parameters (fluor_dist, avg_rad).
        dend_params: Dendrite parameters (weightScale).
        neur_num: 3D uint16 volume with component IDs (1..N_neur+N_den).
                  From ApicalDendriteResult.neur_num.
        neur_soma: 3D uint16 volume with soma IDs (1..N_neur).
                   From NeuralVolumeResult.neur_soma.
        neur_num_ad: 3D uint16 volume with apical dendrite IDs.
                     From ApicalDendriteResult.neur_num_ad.
        positions: (N_neur, 3) array of neuron center locations in um.
        neur_vol: 3D float32 volume (may already have nucleus fluorescence).
                  Copied internally; caller's array is not modified.
        verbose: Verbosity level (0=silent, 1=summary, 2=per-neuron).

    Returns:
        FluorescenceResult with gp_vals and updated neur_vol.
    """
    if verbose is None:
        verbose = vol_params.verbose

    N_neur = vol_params.N_neur
    N_den = vol_params.N_den
    vres = vol_params.vres
    grid_shape = tuple(neur_num.shape)
    numcomps = N_neur + N_den

    fluor_dist = neur_params.fluor_dist
    w1, w2, w3 = dend_params.weightScale

    if verbose >= 1:
        print("Setting cell fluorescence distributions...")

    # Copy neur_vol to avoid mutating caller's data
    neur_vol_out = neur_vol.astype(np.float32).copy()

    # ------------------------------------------------------------------
    # Phase 1: Collect per-component voxel indices (MATLAB lines 126-141)
    # ------------------------------------------------------------------
    neur_num_flat = neur_num.ravel()
    valid_mask = (neur_num_flat >= 1) & (neur_num_flat <= numcomps)
    valid_ids = neur_num_flat[valid_mask].astype(np.int32)
    valid_flat_indices = np.flatnonzero(valid_mask).astype(np.int32)

    # Sort by component ID, then split
    sort_order = np.argsort(valid_ids, kind='stable')
    sorted_ids = valid_ids[sort_order]
    sorted_flat = valid_flat_indices[sort_order]

    numvox = np.bincount(valid_ids, minlength=numcomps + 1)[1:]
    splits = np.cumsum(numvox[:-1])
    component_indices = np.split(sorted_flat, splits)

    neur_soma_flat = neur_soma.ravel()
    neur_num_ad_flat = neur_num_ad.ravel()

    gp_vals: List[CellFluorescenceData] = []

    # ------------------------------------------------------------------
    # Phase 2: Neuron fluorescence (MATLAB lines 154-202)
    # ------------------------------------------------------------------
    for kk in range(N_neur):
        indices = component_indices[kk]

        if len(indices) == 0:
            gp_vals.append(CellFluorescenceData(
                indices=np.array([], dtype=np.int32),
                fluorescence=np.array([], dtype=np.float32),
                soma_mask=np.array([], dtype=bool),
            ))
            continue

        # 1. Generate 3D GP for soma fluorescence pattern (MATLAB line 158)
        grid_side = int(round(neur_params.avg_rad * 6 * vres))
        gp_sample = sample_3d_gp(
            grid_sz=(grid_side, grid_side, grid_side),
            l_scale=np.array([[fluor_dist[0] * vres]], dtype=np.float32),
            p_scale=fluor_dist[1],
            mu=0.0,
        )

        # 2. Find soma and apical dendrite voxels (MATLAB lines 160-166)
        soma_flags = neur_soma_flat[indices] == (kk + 1)
        soma_local_idx = np.where(soma_flags)[0]
        soma_global = indices[soma_local_idx]

        ad_flags = neur_num_ad_flat[indices] == (kk + 1)
        ad_local_idx = np.where(ad_flags)[0]

        # 3. Sample GP at soma locations (MATLAB lines 167-177)
        if len(soma_global) > 0:
            lx, ly, lz = np.unravel_index(soma_global, grid_shape)
            soma_coords = np.column_stack([lx, ly, lz])

            center_voxel = np.floor(vres * positions[kk]).astype(int)
            gp_center = grid_side // 2
            gp_coords = soma_coords - center_voxel + gp_center

            # Clip to GP grid bounds (MATLAB lines 171-172, 0-based)
            gp_coords = np.clip(
                gp_coords, 0,
                np.array(gp_sample.shape) - 1
            )

            gp_vals_at_soma = gp_sample[
                gp_coords[:, 0], gp_coords[:, 1], gp_coords[:, 2]
            ]

            # Normalize to [0.5, 1.5] (MATLAB lines 175-177)
            mean_val = np.mean(gp_vals_at_soma)
            max_abs = np.max(np.abs(gp_vals_at_soma - mean_val))
            if max_abs > 0:
                gp_vals_at_soma = (
                    0.5 * (gp_vals_at_soma - mean_val) / max_abs + 1.0
                )
            else:
                gp_vals_at_soma = np.ones_like(gp_vals_at_soma)
            gp_vals_at_soma = np.nan_to_num(
                gp_vals_at_soma, nan=1.0
            ).astype(np.float32)
        else:
            gp_vals_at_soma = np.array([], dtype=np.float32)

        # 4. Distance decay for ALL voxels (MATLAB lines 179-184)
        rx, ry, rz = np.unravel_index(indices, grid_shape)
        neuron_center = vres * positions[kk]
        dist = np.sqrt(
            (rx - neuron_center[0]) ** 2
            + (ry - neuron_center[1]) ** 2
            + (rz - neuron_center[2]) ** 2
        ).astype(np.float32)

        fluorescence = (
            (w2 * np.exp(-dist / (vres * w1)) + (1 - w2)) * (1 - w3)
        ).astype(np.float32)

        # 5. Override soma and apical dendrite values (MATLAB lines 186-189)
        if len(soma_local_idx) > 0:
            fluorescence[soma_local_idx] = gp_vals_at_soma
        if len(ad_local_idx) > 0:
            fluorescence[ad_local_idx] = 1.0

        # 6. Build soma mask
        soma_mask = np.zeros(len(indices), dtype=bool)
        if len(soma_local_idx) > 0:
            soma_mask[soma_local_idx] = True

        # 7. Write to volume (MATLAB lines 190-192)
        neur_vol_out.ravel()[indices] = fluorescence

        gp_vals.append(CellFluorescenceData(
            indices=indices,
            fluorescence=fluorescence,
            soma_mask=soma_mask,
        ))

        if verbose >= 2:
            n_soma = int(np.sum(soma_flags))
            n_dend = len(indices) - n_soma
            print(f"    Neuron {kk+1}: {n_soma} soma, {n_dend} dendrite voxels")

    # ------------------------------------------------------------------
    # Phase 3: Through-volume apical dendrites (MATLAB lines 221-240)
    # ------------------------------------------------------------------
    for kk in range(N_neur, numcomps):
        indices = component_indices[kk]

        fluorescence = np.ones(len(indices), dtype=np.float32)
        soma_mask = np.zeros(len(indices), dtype=bool)

        if len(indices) > 0:
            neur_vol_out.ravel()[indices] = fluorescence

        gp_vals.append(CellFluorescenceData(
            indices=indices,
            fluorescence=fluorescence,
            soma_mask=soma_mask,
        ))

        if verbose >= 2:
            print(f"    Apical dendrite {kk+1}: {len(indices)} voxels")

    if verbose >= 1:
        total_components = sum(1 for g in gp_vals if len(g.indices) > 0)
        print(f"done. Active components: {total_components}")

    return FluorescenceResult(gp_vals=gp_vals, neur_vol=neur_vol_out)
