"""Export Step 6 fluorescence data to JSON for Three.js visualization."""

import json
import sys
import numpy as np

# Add calcia to path
sys.path.insert(0, r"C:\Users\boyuan\Documents\GitHub\calcia")

from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import grow_neuron_dendrites, grow_apical_dendrites
from calcia.volume.fluorescence import set_cell_fluorescence


def export_fluorescence(output_path: str, seed: int = 42):
    """Run Steps 1-6 and export fluorescence data as JSON."""
    np.random.seed(seed)

    vol_params = VolumeParams(vol_sz=(60, 60, 30), vres=2, N_neur=5)
    neur_params = NeuronParams(n_samps=150, nuc_fluorsc=0.5)
    dend_params = DendParams(
        dtParams=(8, 25, 15, 1, 3),
        atParams=(2, 5, 5, 5, 1),
        dims=(12, 12, 12),
        dimsSS=(5, 5, 5),
    )

    print("Step 2: Sampling neurons...")
    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=1
    )

    print("Step 3: Generating neural volume...")
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=1
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

    grid_shape = tuple(int(x) for x in apical_result.neur_num.shape)
    neur_soma_flat = vol_result.neur_soma.ravel()
    neur_num_ad_flat = apical_result.neur_num_ad.ravel()

    # Max voxels per category for subsampling
    MAX_SOMA = 3000
    MAX_DEND = 8000
    MAX_APICAL = 5000

    def subsample(indices, values, max_n):
        if len(indices) <= max_n:
            return indices, values
        choice = np.sort(np.random.choice(len(indices), max_n, replace=False))
        return indices[choice], values[choice]

    # Build neuron data
    neuron_list = []
    for kk in range(vol_params.N_neur):
        g = fluor_result.gp_vals[kk]
        if len(g.indices) == 0:
            neuron_list.append({
                "id": kk + 1,
                "position": positions[kk].tolist(),
                "soma": {"count": 0, "positions": [], "fluorescence": []},
                "dendrite": {"count": 0, "positions": [], "fluorescence": []},
                "apical": {"count": 0, "positions": [], "fluorescence": []},
            })
            continue

        # Classify voxels
        soma_mask = g.soma_mask
        ad_flags = neur_num_ad_flat[g.indices] == (kk + 1)
        dend_mask = ~soma_mask & ~ad_flags

        # Soma
        soma_idx = g.indices[soma_mask]
        soma_fl = g.fluorescence[soma_mask]
        soma_idx, soma_fl = subsample(soma_idx, soma_fl, MAX_SOMA)
        soma_coords = np.array(np.unravel_index(soma_idx, grid_shape)).T

        # Dendrites (basal)
        dend_idx = g.indices[dend_mask]
        dend_fl = g.fluorescence[dend_mask]
        dend_idx, dend_fl = subsample(dend_idx, dend_fl, MAX_DEND)
        dend_coords = np.array(np.unravel_index(dend_idx, grid_shape)).T

        # Apical dendrites (owned by this neuron)
        ap_idx = g.indices[ad_flags]
        ap_fl = g.fluorescence[ad_flags]
        ap_idx, ap_fl = subsample(ap_idx, ap_fl, MAX_APICAL)
        ap_coords = np.array(np.unravel_index(ap_idx, grid_shape)).T

        neuron_list.append({
            "id": kk + 1,
            "position": positions[kk].tolist(),
            "soma": {
                "count": int(np.sum(soma_mask)),
                "positions": soma_coords.ravel().tolist(),
                "fluorescence": np.round(soma_fl, 4).tolist(),
            },
            "dendrite": {
                "count": int(np.sum(dend_mask)),
                "positions": dend_coords.ravel().tolist(),
                "fluorescence": np.round(dend_fl, 4).tolist(),
            },
            "apical": {
                "count": int(np.sum(ad_flags)),
                "positions": ap_coords.ravel().tolist(),
                "fluorescence": np.round(ap_fl, 4).tolist(),
            },
        })

    # Through-volume apical dendrites
    through_list = []
    for kk in range(vol_params.N_neur, len(fluor_result.gp_vals)):
        g = fluor_result.gp_vals[kk]
        idx = g.indices
        fl = g.fluorescence
        idx, fl = subsample(idx, fl, MAX_APICAL)
        coords = np.array(np.unravel_index(idx, grid_shape)).T

        through_list.append({
            "id": kk + 1,
            "count": len(g.indices),
            "positions": coords.ravel().tolist(),
            "fluorescence": np.round(fl, 4).tolist(),
        })

    # Fluorescence statistics
    all_fl = fluor_result.neur_vol[fluor_result.neur_vol > 0]
    fl_stats = {
        "min": float(np.min(all_fl)) if len(all_fl) > 0 else 0,
        "max": float(np.max(all_fl)) if len(all_fl) > 0 else 0,
        "mean": float(np.mean(all_fl)) if len(all_fl) > 0 else 0,
        "std": float(np.std(all_fl)) if len(all_fl) > 0 else 0,
    }

    data = {
        "source": "python",
        "grid_shape": list(grid_shape),
        "params": {
            "vol_sz": list(vol_params.vol_sz),
            "vres": vol_params.vres,
            "N_neur": vol_params.N_neur,
            "N_den": vol_params.N_den,
        },
        "fluorescence_stats": fl_stats,
        "stats": {
            "total_soma_voxels": sum(n["soma"]["count"] for n in neuron_list),
            "total_dendrite_voxels": sum(n["dendrite"]["count"] for n in neuron_list),
            "total_apical_voxels": sum(n["apical"]["count"] for n in neuron_list),
            "total_through_volume_voxels": sum(t["count"] for t in through_list),
        },
        "neurons": neuron_list,
        "through_volume_dendrites": through_list,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = len(json.dumps(data)) / 1024 / 1024
    print(f"\nExported to {output_path} ({size_mb:.1f} MB)")
    print(f"  Neurons: {vol_params.N_neur}")
    print(f"  Through-volume dendrites: {vol_params.N_den}")
    print(f"  Soma voxels: {data['stats']['total_soma_voxels']}")
    print(f"  Dendrite voxels: {data['stats']['total_dendrite_voxels']}")
    print(f"  Apical voxels: {data['stats']['total_apical_voxels']}")
    print(f"  Through-volume voxels: {data['stats']['total_through_volume_voxels']}")
    print(f"  Fluorescence range: [{fl_stats['min']:.3f}, {fl_stats['max']:.3f}]")


if __name__ == "__main__":
    out = r"C:\Users\boyuan\Documents\GitHub\calcia\calcia\visualization\fluorescence_python.json"
    export_fluorescence(out)
