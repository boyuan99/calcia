"""Export Step 7 background/axon data to JSON for Three.js visualization."""

import json
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\boyuan\Documents\GitHub\calcia")

from calcia.config.params import (
    AxonParams, BgParams, DendParams, NeuronParams, VolumeParams,
)
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import grow_neuron_dendrites, grow_apical_dendrites
from calcia.volume.fluorescence import set_cell_fluorescence
from calcia.volume.background import generate_bg_dendrites, generate_axons


def export_background(output_path: str, seed: int = 42):
    """Run Steps 1-7 and export background/axon data as JSON."""
    np.random.seed(seed)

    vol_params = VolumeParams(vol_sz=(60, 60, 30), vres=2, N_neur=5, N_bg=20)
    neur_params = NeuronParams(n_samps=150, nuc_fluorsc=0.5)
    dend_params = DendParams(
        dtParams=(8, 25, 15, 1, 3),
        atParams=(2, 5, 5, 5, 1),
        dims=(12, 12, 12),
        dimsSS=(5, 5, 5),
    )
    bg_params = BgParams()
    axon_params = AxonParams()

    # --- Steps 2-6 ---
    print("Step 2: Sampling neurons...")
    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=1,
    )

    print("Step 3: Generating neural volume...")
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=1,
    )

    np.random.seed(seed)
    print("Step 4: Growing dendrites...")
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions, rotation_angles=angles, verbose=1,
    )

    np.random.seed(seed)
    print("Step 5: Growing apical dendrites...")
    apical_result = grow_apical_dendrites(
        vol_params, dend_result.dend_params,
        dend_result, vol_result, verbose=1,
    )

    np.random.seed(seed + 1)
    print("Step 6: Setting cell fluorescence...")
    fluor_result = set_cell_fluorescence(
        vol_params, neur_params, dend_result.dend_params,
        neur_num=apical_result.neur_num,
        neur_soma=vol_result.neur_soma,
        neur_num_ad=apical_result.neur_num_ad,
        positions=positions,
        neur_vol=vol_result.neur_vol,
        verbose=1,
    )

    # --- Step 7A: Background dendrites ---
    np.random.seed(seed + 2)
    print("Step 7A: Generating background dendrites...")
    bg_result = generate_bg_dendrites(
        vol_params, bg_params, dend_result.dend_params,
        neur_vol=fluor_result.neur_vol,
        neur_num=apical_result.neur_num,
        gp_vals=fluor_result.gp_vals,
        gp_nuc=vol_result.gp_nuc,
        neur_locs=positions,
        verbose=1,
    )

    # --- Step 7B: Axons ---
    np.random.seed(seed + 3)
    print("Step 7B: Generating axons...")
    axon_result = generate_axons(
        vol_params, axon_params,
        neur_vol=bg_result.neur_vol,
        neur_num=bg_result.neur_num,
        gp_vals=bg_result.gp_vals,
        gp_nuc=vol_result.gp_nuc,
        verbose=1,
    )

    # --- Gather data for JSON ---
    grid_shape = tuple(int(x) for x in bg_result.neur_num.shape)
    Ncomps_before = vol_params.N_neur + vol_params.N_den
    MAX_VOXELS = 5000

    def subsample(indices, values, max_n):
        if len(indices) <= max_n:
            return indices, values
        choice = np.sort(np.random.choice(len(indices), max_n, replace=False))
        return indices[choice], values[choice]

    def indices_to_coords(indices):
        """Convert C-order linear indices to (N, 3) coordinates."""
        return np.array(np.unravel_index(indices, grid_shape)).T

    # 1) Neurons (Steps 2-6) — soma + dendrites combined
    neuron_list = []
    neur_soma_flat = vol_result.neur_soma.ravel()
    for kk in range(vol_params.N_neur):
        g = fluor_result.gp_vals[kk]
        if len(g.indices) == 0:
            continue
        soma_mask = g.soma_mask
        idx_s, fl_s = subsample(g.indices[soma_mask], g.fluorescence[soma_mask], 2000)
        idx_d, fl_d = subsample(g.indices[~soma_mask], g.fluorescence[~soma_mask], 3000)
        coords_s = indices_to_coords(idx_s)
        coords_d = indices_to_coords(idx_d)
        neuron_list.append({
            "id": kk + 1,
            "soma_positions": coords_s.ravel().tolist(),
            "soma_fluorescence": np.round(fl_s, 4).tolist(),
            "dend_positions": coords_d.ravel().tolist(),
            "dend_fluorescence": np.round(fl_d, 4).tolist(),
        })

    # 2) Background dendrites (Step 7A)
    bg_dend_list = []
    for kk in range(Ncomps_before, Ncomps_before + bg_result.N_den2):
        g = bg_result.gp_vals[kk]
        if len(g.indices) == 0:
            continue
        idx, fl = subsample(g.indices, g.fluorescence, MAX_VOXELS)
        coords = indices_to_coords(idx)
        bg_dend_list.append({
            "id": kk + 1,
            "total_voxels": int(len(g.indices)),
            "positions": coords.ravel().tolist(),
            "fluorescence": np.round(fl, 4).tolist(),
        })

    # 3) Axons (Step 7B)
    axon_list = []
    for kk, (ax_idx, ax_fl) in enumerate(axon_result.gp_bgvals):
        if len(ax_idx) == 0:
            continue
        idx, fl = subsample(ax_idx, ax_fl, MAX_VOXELS)
        coords = indices_to_coords(idx)
        axon_list.append({
            "id": kk + 1,
            "total_voxels": int(len(ax_idx)),
            "positions": coords.ravel().tolist(),
            "fluorescence": np.round(fl, 4).tolist(),
        })

    # Stats
    neur_num_final = bg_result.neur_num
    total_neuron_voxels = int(np.sum(
        (neur_num_final >= 1) & (neur_num_final <= Ncomps_before)
    ))
    total_bg_voxels = int(np.sum(
        (neur_num_final > Ncomps_before) &
        (neur_num_final <= Ncomps_before + bg_result.N_den2)
    ))
    # Axon voxels counted from gp_bgvals (additive, not in neur_num)
    total_axon_voxels = sum(len(a[0]) for a in axon_result.gp_bgvals)
    total_empty = int(np.sum(neur_num_final == 0))

    data = {
        "grid_shape": list(grid_shape),
        "params": {
            "vol_sz": list(vol_params.vol_sz),
            "vres": vol_params.vres,
            "N_neur": vol_params.N_neur,
            "N_den": vol_params.N_den,
            "N_den2": bg_result.N_den2,
            "N_bg": vol_params.N_bg,
            "N_bg_actual": axon_result.N_bg_actual,
        },
        "stats": {
            "total_neuron_voxels": total_neuron_voxels,
            "total_bg_dendrite_voxels": total_bg_voxels,
            "total_axon_voxels": total_axon_voxels,
            "total_empty_voxels": total_empty,
        },
        "neurons": neuron_list,
        "bg_dendrites": bg_dend_list,
        "axons": axon_list,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = len(json.dumps(data)) / 1024 / 1024
    print(f"\nExported to {output_path} ({size_mb:.1f} MB)")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Neurons: {vol_params.N_neur} ({total_neuron_voxels} voxels)")
    print(f"  Bg dendrites: {bg_result.N_den2} ({total_bg_voxels} voxels)")
    print(f"  Axons: {axon_result.N_bg_actual} ({total_axon_voxels} voxels)")
    print(f"  Empty: {total_empty} voxels")


if __name__ == "__main__":
    out = r"C:\Users\boyuan\Documents\GitHub\calcia\visualization\background_data.json"
    export_background(out)
