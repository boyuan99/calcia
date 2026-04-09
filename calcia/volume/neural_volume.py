"""
Neural volume voxelization module.

Voxelizes neuron soma and nucleus surface meshes into a 3D volume grid.
This is Step 3 of the NAOMi simulation pipeline.

Corresponds to MATLAB: generateNeuralVolume.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from ..config.params import VolumeParams, NeuronParams
from ..geometry.triangulation import in_triangulation_batch


@dataclass
class NeuronVoxelData:
    """Per-neuron voxelization data.

    Attributes:
        neuron_id: 1-based neuron identifier.
        soma_indices: Linear indices (int32) of soma cytoplasm voxels
                      in the volume grid. Excludes nucleus voxels.
        nucleus_indices: Linear indices (int32) of nucleus voxels.
        nucleus_fluorescence: Fluorescence value for this neuron's nucleus.
    """

    neuron_id: int
    soma_indices: np.ndarray
    nucleus_indices: np.ndarray
    nucleus_fluorescence: float


@dataclass
class NeuralVolumeResult:
    """Result of neural volume voxelization.

    Contains the 3D voxelized representations and per-neuron data
    needed by downstream pipeline steps (dendrite generation,
    activity simulation).

    Attributes:
        neur_soma: 3D uint16 array of shape (vol_sz * vres). Each voxel
                   contains the 1-based neuron ID occupying it, or 0 if empty.
        neur_vol: 3D float32 array of same shape. Contains fluorescence
                  values (nucleus fluorescence where nuclei are located).
        gp_nuc: List of (nucleus_indices, fluorescence_value) tuples,
                one per neuron. nucleus_indices are int32 linear indices.
        gp_soma: List of soma cytoplasm index arrays (int32 linear indices),
                 one per neuron.
        neuron_data: List of NeuronVoxelData objects for per-neuron access.
        grid_shape: Shape of the voxel grid as (nx, ny, nz) tuple.
        voxel_resolution: Voxels per micrometer (vres).
    """

    neur_soma: np.ndarray
    neur_vol: np.ndarray
    gp_nuc: List[Tuple[np.ndarray, float]]
    gp_soma: List[np.ndarray]
    neuron_data: List[NeuronVoxelData] = field(default_factory=list)
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    voxel_resolution: int = 2


def generate_neural_volume(
    neurons: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    positions: np.ndarray,
    vol_params: Optional[VolumeParams] = None,
    neur_params: Optional[NeuronParams] = None,
    vessel_mask: Optional[np.ndarray] = None,
    verbose: Optional[int] = None,
) -> NeuralVolumeResult:
    """
    Voxelize neuron soma and nucleus meshes into a 3D volume grid.

    Takes pre-generated neuron surface meshes (from sample_dense_neurons)
    and rasterizes them into a voxel grid, producing per-neuron soma and
    nucleus voxel maps needed by downstream pipeline steps.

    Corresponds to MATLAB: generateNeuralVolume.m

    Args:
        neurons: List of (Vcell, Vnuc, faces) tuples.
                 Vcell: (N, 3) soma vertices in micrometers (already positioned).
                 Vnuc: (N, 3) nucleus vertices in micrometers (already positioned).
                 faces: (M, 3) face connectivity (0-based).
        positions: (n_neurons, 3) array of neuron center positions in micrometers.
        vol_params: Volume parameters. Uses defaults if None.
        neur_params: Neuron parameters (needed for nuc_fluorsc). Uses defaults if None.
        vessel_mask: Optional 3D boolean/uint8 array of vessel-occupied voxels.
                     Shape must be (vol_sz[0]*vres, vol_sz[1]*vres, vol_sz[2]*vres).
        verbose: Verbosity override. If None, uses vol_params.verbose.

    Returns:
        NeuralVolumeResult containing voxelized soma/nucleus data.
    """
    if vol_params is None:
        vol_params = VolumeParams()
    if neur_params is None:
        neur_params = NeuronParams()

    vol_sz = np.array(vol_params.vol_sz)
    vres = vol_params.vres
    n_neurons = len(neurons)
    verbosity = verbose if verbose is not None else vol_params.verbose

    # Grid shape in voxels
    grid_shape = (
        int(vol_sz[0] * vres),
        int(vol_sz[1] * vres),
        int(vol_sz[2] * vres),
    )

    # Initialize output arrays
    neur_soma = np.zeros(grid_shape, dtype=np.uint16)
    neur_vol = np.zeros(grid_shape, dtype=np.float32)
    gp_nuc: List[Tuple[np.ndarray, float]] = []
    gp_soma: List[np.ndarray] = []
    neuron_data_list: List[NeuronVoxelData] = []

    # Initialize taken_pts from vessel mask
    if vessel_mask is not None:
        if vessel_mask.shape != grid_shape:
            raise ValueError(
                f"vessel_mask shape {vessel_mask.shape} does not match "
                f"expected grid shape {grid_shape}"
            )
        taken_pts = vessel_mask.astype(bool).copy()
    else:
        taken_pts = np.zeros(grid_shape, dtype=bool)

    if n_neurons == 0:
        return NeuralVolumeResult(
            neur_soma=neur_soma,
            neur_vol=neur_vol,
            gp_nuc=gp_nuc,
            gp_soma=gp_soma,
            neuron_data=neuron_data_list,
            grid_shape=grid_shape,
            voxel_resolution=vres,
        )

    if verbosity >= 1:
        print(f"Setting up volume... grid shape {grid_shape}")
        print(f"Finding interior points for {n_neurons} neurons...")

    nuc_fluorsc = neur_params.nuc_fluorsc

    for kk in range(n_neurons):
        Vcell, Vnuc, faces = neurons[kk]
        center = positions[kk]

        # Empty result helper for skip cases
        def _append_empty():
            empty_idx = np.array([], dtype=np.int32)
            gp_nuc.append((empty_idx, nuc_fluorsc))
            gp_soma.append(empty_idx)
            neuron_data_list.append(
                NeuronVoxelData(
                    neuron_id=kk + 1,
                    soma_indices=empty_idx,
                    nucleus_indices=empty_idx,
                    nucleus_fluorescence=nuc_fluorsc,
                )
            )

        # --- Compute bounding box ---
        distances_from_center = np.linalg.norm(Vcell - center, axis=1)
        max_ext = np.ceil(np.max(distances_from_center))
        m_ext_res = int(np.ceil(max_ext * vres))

        # Neuron center in voxel coordinates (0-based)
        idx_pos = np.round(vres * center).astype(int)

        # Local bounding box indices (clamped to grid bounds)
        ix_start = max(0, idx_pos[0] - m_ext_res)
        ix_end = min(grid_shape[0], idx_pos[0] + m_ext_res + 1)
        iy_start = max(0, idx_pos[1] - m_ext_res)
        iy_end = min(grid_shape[1], idx_pos[1] + m_ext_res + 1)
        iz_start = max(0, idx_pos[2] - m_ext_res)
        iz_end = min(grid_shape[2], idx_pos[2] + m_ext_res + 1)

        if ix_end <= ix_start or iy_end <= iy_start or iz_end <= iz_start:
            _append_empty()
            continue

        # --- Create local meshgrid ---
        # Voxel center at index i is at (i + 0.5) / vres micrometers
        local_x = np.arange(ix_start, ix_end, dtype=np.float32)
        local_y = np.arange(iy_start, iy_end, dtype=np.float32)
        local_z = np.arange(iz_start, iz_end, dtype=np.float32)

        gx, gy, gz = np.meshgrid(
            (local_x + 0.5) / vres,
            (local_y + 0.5) / vres,
            (local_z + 0.5) / vres,
            indexing="ij",
        )
        local_shape = gx.shape

        # --- Sphere pre-filter ---
        dx = gx - center[0]
        dy = gy - center[1]
        dz = gz - center[2]
        dist_sq = dx**2 + dy**2 + dz**2
        sphere_mask = dist_sq <= max_ext**2

        n_test = int(np.sum(sphere_mask))
        if n_test == 0:
            _append_empty()
            continue

        # Extract test points (absolute micrometer coordinates)
        test_points = np.column_stack(
            [gx[sphere_mask], gy[sphere_mask], gz[sphere_mask]]
        )

        # --- Point-in-mesh tests ---
        inside_soma = in_triangulation_batch(Vcell, faces, test_points)
        inside_nucleus = in_triangulation_batch(Vnuc, faces, test_points)

        # --- Expand results to local bounding box ---
        soma_local = np.zeros(local_shape, dtype=bool)
        soma_local[sphere_mask] = inside_soma

        nuc_local = np.zeros(local_shape, dtype=bool)
        nuc_local[sphere_mask] = inside_nucleus

        # --- Soma cytoplasm = soma AND NOT nucleus ---
        cytoplasm_local = soma_local & (~nuc_local)

        # --- Remove already-taken voxels ---
        taken_local = taken_pts[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end]
        cytoplasm_local = cytoplasm_local & (~taken_local)

        # --- Update taken_pts ---
        taken_pts[ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] |= (
            cytoplasm_local
        )

        # --- Convert soma cytoplasm to global flat indices ---
        local_cyto_indices = np.nonzero(cytoplasm_local)
        global_ix = local_cyto_indices[0] + ix_start
        global_iy = local_cyto_indices[1] + iy_start
        global_iz = local_cyto_indices[2] + iz_start
        soma_flat = np.ravel_multi_index(
            (global_ix, global_iy, global_iz), grid_shape
        ).astype(np.int32)

        neur_soma[global_ix, global_iy, global_iz] = np.uint16(kk + 1)
        gp_soma.append(soma_flat)

        # --- Convert nucleus to global flat indices ---
        local_nuc_indices = np.nonzero(nuc_local)
        global_nuc_ix = local_nuc_indices[0] + ix_start
        global_nuc_iy = local_nuc_indices[1] + iy_start
        global_nuc_iz = local_nuc_indices[2] + iz_start
        nuc_flat = np.ravel_multi_index(
            (global_nuc_ix, global_nuc_iy, global_nuc_iz), grid_shape
        ).astype(np.int32)

        neur_vol.ravel()[nuc_flat] = nuc_fluorsc
        gp_nuc.append((nuc_flat, nuc_fluorsc))

        # --- Store per-neuron data ---
        neuron_data_list.append(
            NeuronVoxelData(
                neuron_id=kk + 1,
                soma_indices=soma_flat,
                nucleus_indices=nuc_flat,
                nucleus_fluorescence=nuc_fluorsc,
            )
        )

        if verbosity >= 2:
            print(
                f"  Neuron {kk + 1}/{n_neurons}: "
                f"{len(soma_flat)} soma voxels, {len(nuc_flat)} nucleus voxels"
            )

    if verbosity >= 1:
        total_soma = int(np.sum(neur_soma > 0))
        total_nuc = int(np.sum(neur_vol > 0))
        print(
            f"Interior points complete. "
            f"Total soma voxels: {total_soma}, nucleus voxels: {total_nuc}"
        )

    return NeuralVolumeResult(
        neur_soma=neur_soma,
        neur_vol=neur_vol,
        gp_nuc=gp_nuc,
        gp_soma=gp_soma,
        neuron_data=neuron_data_list,
        grid_shape=grid_shape,
        voxel_resolution=vres,
    )
