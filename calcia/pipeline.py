"""Unified Phase 1 pipeline: simulate_neural_volume.

Runs all 7 steps of neural volume generation in sequence and returns
a single result object containing the complete simulated tissue.

Port of ``simulate_neural_volume.m``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config.params import (
    AxonParams,
    BgParams,
    DendParams,
    NeuronParams,
    VascParams,
    VolumeParams,
)
from .volume.background import (
    AxonResult,
    BgDendriteResult,
    BgProcessData,
    generate_axons,
    generate_bg_dendrites,
    sort_axons,
)
from .volume.dendrites import grow_apical_dendrites, grow_neuron_dendrites
from .volume.fluorescence import CellFluorescenceData, set_cell_fluorescence
from .volume.neural_volume import NeuralVolumeResult, generate_neural_volume
from .volume.neurons import sample_dense_neurons
from .volume.vasculature import VesselNetwork, simulate_blood_vessels


@dataclass
class NeuralVolumeOutput:
    """Complete output of the Phase 1 neural volume simulation.

    Mirrors ``vol_out`` struct from ``simulate_neural_volume.m``.

    Attributes:
        neur_vol: 3D float32 fluorescence volume (shape = vol_sz * vres).
        gp_nuc: Per-neuron nucleus data. List of (indices, fluorescence)
            tuples, one per neuron.
        gp_soma: Per-neuron soma cytoplasm indices (int32 linear, C-order).
        gp_vals: Per-component fluorescence data. Length =
            N_neur + N_den + N_den2 (neurons + apical + bg dendrites).
        neur_ves: 3D vessel mask (uint8), or None if vessels disabled.
        bg_proc: Sorted background processes. List of BgProcessData,
            or empty list if axons disabled.
        locs: (N_total, 3) float32 neuron/bg-dendrite positions in um.
        neur_num: 3D uint16 component ID volume (1..N_neur+N_den+N_den2).
        neur_num_ad: 3D uint16 apical dendrite map.
        gp_bgvals: Per-axon (indices, fluorescence) tuples, or empty
            list if axons disabled.
        params: Dict of final parameter objects keyed by name.
    """

    neur_vol: np.ndarray
    gp_nuc: List[Tuple[np.ndarray, float]]
    gp_soma: list
    gp_vals: List[CellFluorescenceData]
    neur_ves: Optional[np.ndarray]
    bg_proc: List[BgProcessData]
    locs: np.ndarray
    neur_num: np.ndarray
    neur_num_ad: np.ndarray
    gp_bgvals: List[Tuple[np.ndarray, np.ndarray]]
    params: Dict


def simulate_neural_volume(
    vol_params: Optional[VolumeParams] = None,
    neur_params: Optional[NeuronParams] = None,
    vasc_params: Optional[VascParams] = None,
    dend_params: Optional[DendParams] = None,
    bg_params: Optional[BgParams] = None,
    axon_params: Optional[AxonParams] = None,
    *,
    seed: Optional[int] = None,
    verbose: Optional[int] = None,
) -> NeuralVolumeOutput:
    """Run the full Phase 1 neural volume simulation pipeline.

    Executes all 7 steps sequentially:
      1. Blood vessel simulation
      2. Neuron sampling and shape generation
      3. Neural volume voxelization
      4. Basal dendrite growth
      5. Through-volume apical dendrite growth
      6. Cell fluorescence distribution
      7. Background dendrites and axon generation

    Port of ``simulate_neural_volume.m``.

    Args:
        vol_params: Volume parameters. Uses defaults if None.
        neur_params: Neuron parameters. Uses defaults if None.
        vasc_params: Vasculature parameters. Uses defaults if None.
        dend_params: Dendrite parameters. Uses defaults if None.
        bg_params: Background dendrite parameters. Uses defaults if None.
        axon_params: Axon parameters. Uses defaults if None.
        seed: Random seed for reproducibility. If None, no seed is set.
        verbose: Verbosity override (0=silent, 1=progress, 2=detailed).
            If None, uses vol_params.verbose.

    Returns:
        NeuralVolumeOutput containing the complete simulated tissue.

    Example:
        >>> from calcia import simulate_neural_volume
        >>> from calcia.config.params import VolumeParams
        >>> result = simulate_neural_volume(
        ...     VolumeParams(vol_sz=(60, 60, 30), N_neur=5),
        ...     seed=42,
        ... )
        >>> print(result.neur_vol.shape)  # (120, 120, 60)
    """
    # --- Default parameters ---
    if vol_params is None:
        vol_params = VolumeParams()
    if neur_params is None:
        neur_params = NeuronParams()
    if vasc_params is None:
        vasc_params = VascParams()
    if dend_params is None:
        dend_params = DendParams()
    if bg_params is None:
        bg_params = BgParams()
    if axon_params is None:
        axon_params = AxonParams()

    if verbose is not None:
        vol_params.verbose = verbose
    v = vol_params.verbose

    if seed is not None:
        np.random.seed(seed)

    grid_shape = tuple(s * vol_params.vres for s in vol_params.vol_sz)

    if v >= 1:
        print("=" * 60)
        print("simulate_neural_volume  (Phase 1)")
        print(f"  Volume: {vol_params.vol_sz} um, "
              f"vres={vol_params.vres}, "
              f"grid={grid_shape}")
        print(f"  N_neur={vol_params.N_neur}, "
              f"N_den={vol_params.N_den}, "
              f"N_bg={vol_params.N_bg}")
        print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Blood vessels
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[1/7] Simulating blood vessels...")

    vessel_mask = None
    vessel_mask_full = None   # Full-depth volume (surface + imaging), matches MATLAB neur_ves
    if vasc_params.flag:
        vessel_network = simulate_blood_vessels(
            vol_params, vasc_params, verbose=v,
        )
        vessel_mask_full = vessel_network.vessel_volume   # shape (Nx, Ny, full_z)
        # Crop to imaging region for downstream steps that expect imaging-only shape

        z_offset = int(vol_params.vol_depth * vol_params.vres)
        vessel_mask = vessel_mask_full[:, :, z_offset:]   # shape (Nx, Ny, Nz)
    else:
        if v >= 1:
            print("  Vasculature disabled.")

    # ----------------------------------------------------------------
    # Step 2: Neuron sampling
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[2/7] Sampling neurons...")

    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params,
        vessel_mask=vessel_mask, verbose=v,
    )

    # ----------------------------------------------------------------
    # Step 3: Neural volume voxelization
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[3/7] Generating neural volume...")

    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params,
        vessel_mask=vessel_mask, verbose=v,
    )

    # ----------------------------------------------------------------
    # Step 4: Basal dendrite growth
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[4/7] Growing dendrites...")

    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions, rotation_angles=angles, verbose=v,
    )
    # dend_params may be updated (e.g. dims/dimsSS adjusted)
    dend_params = dend_result.dend_params

    # ----------------------------------------------------------------
    # Step 5: Through-volume apical dendrites
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[5/7] Growing apical dendrites...")

    apical_result = grow_apical_dendrites(
        vol_params, dend_params,
        dend_result, vol_result, verbose=v,
    )
    dend_params = apical_result.dend_params

    # ----------------------------------------------------------------
    # Step 6: Cell fluorescence distribution
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[6/7] Setting cell fluorescence...")

    fluor_result = set_cell_fluorescence(
        vol_params, neur_params, dend_params,
        neur_num=apical_result.neur_num,
        neur_soma=vol_result.neur_soma,
        neur_num_ad=apical_result.neur_num_ad,
        positions=positions,
        neur_vol=vol_result.neur_vol,
        verbose=v,
    )

    # ----------------------------------------------------------------
    # Step 7: Background dendrites & axons
    # ----------------------------------------------------------------
    if v >= 1:
        print("\n[7/7] Generating background/neuropil and axons...")

    # 7A: Background dendrites
    if bg_params.flag:
        bg_result = generate_bg_dendrites(
            vol_params, bg_params, dend_params,
            neur_vol=fluor_result.neur_vol,
            neur_num=apical_result.neur_num,
            gp_vals=fluor_result.gp_vals,
            gp_nuc=vol_result.gp_nuc,
            neur_locs=positions,
            verbose=v,
        )
        final_neur_num = bg_result.neur_num
        final_neur_vol = bg_result.neur_vol
        final_gp_vals = bg_result.gp_vals
        final_locs = bg_result.neur_locs
    else:
        if v >= 1:
            print("  Background dendrites disabled.")
        bg_result = None
        final_neur_num = apical_result.neur_num
        final_neur_vol = fluor_result.neur_vol
        final_gp_vals = fluor_result.gp_vals
        final_locs = positions

    # 7B: Axons
    gp_bgvals: List[Tuple[np.ndarray, np.ndarray]] = []
    bg_proc: List[BgProcessData] = []
    if axon_params.flag:
        axon_result = generate_axons(
            vol_params, axon_params,
            neur_vol=final_neur_vol,
            neur_num=final_neur_num,
            gp_vals=final_gp_vals,
            gp_nuc=vol_result.gp_nuc,
            verbose=v,
        )
        final_neur_vol = axon_result.neur_vol
        gp_bgvals = axon_result.gp_bgvals

        # 7C: Sort axons into correlated background processes
        axon_params_N_proc = len(final_gp_vals)
        axon_params.N_proc = axon_params_N_proc
        bg_proc = sort_axons(
            vol_params, axon_params,
            gp_bgvals=gp_bgvals,
            cell_pos=final_locs * vol_params.vres,
            verbose=v,
        )
    else:
        if v >= 1:
            print("  Axon generation disabled.")

    # ----------------------------------------------------------------
    # Assemble output
    # ----------------------------------------------------------------
    if v >= 1:
        total_voxels = int(np.prod(grid_shape))
        occupied = int(np.sum(final_neur_num > 0))
        fl_vals = final_neur_vol[final_neur_vol > 0]
        print("\n" + "=" * 60)
        print("Simulation complete.")
        print(f"  Grid: {grid_shape}")
        print(f"  Occupied voxels: {occupied:,} / {total_voxels:,} "
              f"({100 * occupied / total_voxels:.1f}%)")
        if len(fl_vals) > 0:
            print(f"  Fluorescence range: [{fl_vals.min():.3f}, "
                  f"{fl_vals.max():.3f}]")
        print("=" * 60)

    return NeuralVolumeOutput(
        neur_vol=final_neur_vol,
        gp_nuc=vol_result.gp_nuc,
        gp_soma=dend_result.gp_soma,
        gp_vals=final_gp_vals,
        neur_ves=vessel_mask_full,   # Full-depth volume, matches MATLAB vol_out.neur_ves
        bg_proc=bg_proc,
        locs=final_locs,
        neur_num=final_neur_num,
        neur_num_ad=apical_result.neur_num_ad,
        gp_bgvals=gp_bgvals,
        params={
            "vol_params": vol_params,
            "neur_params": neur_params,
            "vasc_params": vasc_params,
            "dend_params": dend_params,
            "bg_params": bg_params,
            "axon_params": axon_params,
        },
    )
