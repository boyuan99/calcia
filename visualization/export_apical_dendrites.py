"""Export Python apical dendrite (Step 5) results as JSON for Three.js visualization."""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)

from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import grow_neuron_dendrites, grow_apical_dendrites


def main():
    # Parameters (match MATLAB export)
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
    n_actual = len(neurons)
    print(f"  Placed {n_actual} neurons")

    print("Step 3: Generating neural volume...")
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=1
    )

    print("Step 4: Growing dendrites...")
    np.random.seed(42)
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions,
        rotation_angles=angles,
        verbose=2,
    )

    print("Step 5: Growing apical dendrites...")
    np.random.seed(42)
    apical_result = grow_apical_dendrites(
        vol_params, dend_result.dend_params,
        dend_result, vol_result, verbose=2
    )

    # Gather data
    neur_soma = vol_result.neur_soma
    neur_num_before = dend_result.neur_num  # Step 4 output
    neur_num_after = apical_result.neur_num  # Step 5 output
    neur_num_ad = apical_result.neur_num_ad
    gp_nuc = vol_result.gp_nuc
    grid_shape = neur_soma.shape
    N_neur = vol_params.N_neur

    total_soma = int(np.sum(neur_soma > 0))
    total_step4 = int(np.sum(neur_num_before > 0))
    total_step5 = int(np.sum(neur_num_after > 0))
    total_apical = int(np.sum(neur_num_ad > 0))
    total_through_volume = int(np.sum(neur_num_after > N_neur))

    print(f"\nResults:")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Soma voxels: {total_soma}")
    print(f"  After Step 4 (neuron dendrites): {total_step4}")
    print(f"  After Step 5 (+ apical): {total_step5}")
    print(f"  Apical dendrite (neur_num_ad) voxels: {total_apical}")
    print(f"  Through-volume dendrite (ID > N_neur) voxels: {total_through_volume}")

    # Build JSON structure
    result = {
        "source": "python",
        "grid_shape": list(grid_shape),
        "params": {
            "vol_sz": list(vol_params.vol_sz),
            "vres": vol_params.vres,
            "N_neur": N_neur,
            "N_den": vol_params.N_den,
        },
        "stats": {
            "total_soma": total_soma,
            "total_step4": total_step4,
            "total_step5": total_step5,
            "total_apical_ad": total_apical,
            "total_through_volume": total_through_volume,
        },
        "neurons": [],
        "apical_dendrites": [],
    }

    # Per-neuron data (soma + per-neuron dendrites)
    for k in range(1, n_actual + 1):
        soma_mask = neur_soma == k
        dend_mask = (neur_num_after == k) & ~soma_mask

        soma_coords = np.argwhere(soma_mask)
        dend_coords = np.argwhere(dend_mask)

        # Nucleus
        nuc_idx = gp_nuc[k - 1][0]
        nuc_coords = np.array(
            np.unravel_index(nuc_idx, grid_shape)
        ).T if len(nuc_idx) > 0 else np.zeros((0, 3), dtype=int)

        # Subsample for performance
        if len(soma_coords) > 2000:
            idx = np.round(np.linspace(0, len(soma_coords) - 1, 2000)).astype(int)
            soma_coords = soma_coords[idx]
        if len(dend_coords) > 5000:
            idx = np.round(np.linspace(0, len(dend_coords) - 1, 5000)).astype(int)
            dend_coords = dend_coords[idx]
        if len(nuc_coords) > 1000:
            idx = np.round(np.linspace(0, len(nuc_coords) - 1, 1000)).astype(int)
            nuc_coords = nuc_coords[idx]

        neuron = {
            "id": k,
            "position": positions[k - 1].tolist(),
            "soma_count": int(np.sum(soma_mask)),
            "dendrite_count": int(np.sum(dend_mask)),
            "nucleus_count": len(nuc_idx),
            "soma_positions": soma_coords.ravel().tolist(),
            "dendrite_positions": dend_coords.ravel().tolist(),
            "nucleus_positions": nuc_coords.ravel().tolist(),
        }
        result["neurons"].append(neuron)

    # Through-volume apical dendrites (IDs > N_neur)
    unique_ids = np.unique(neur_num_after[neur_num_after > N_neur])
    for dendrite_id in unique_ids:
        apical_mask = neur_num_after == dendrite_id
        apical_coords = np.argwhere(apical_mask)

        if len(apical_coords) > 8000:
            idx = np.round(np.linspace(0, len(apical_coords) - 1, 8000)).astype(int)
            apical_coords = apical_coords[idx]

        apical = {
            "id": int(dendrite_id),
            "voxel_count": int(np.sum(apical_mask)),
            "positions": apical_coords.ravel().tolist(),
        }
        result["apical_dendrites"].append(apical)

    out_dir = Path(__file__).parent
    out_path = out_dir / "python_apical_dendrites.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"\nSaved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"  Neurons: {len(result['neurons'])}")
    print(f"  Through-volume dendrites: {len(result['apical_dendrites'])}")


if __name__ == "__main__":
    main()
