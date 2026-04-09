"""Export Python dendrite growth results as JSON for Three.js visualization."""
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

np.random.seed(42)

from calcia.config.params import VolumeParams, NeuronParams, DendParams
from calcia.volume.neurons import sample_dense_neurons
from calcia.volume.neural_volume import generate_neural_volume
from calcia.volume.dendrites import grow_neuron_dendrites


def main():
    # Parameters (match MATLAB export_dendrites_json.m)
    vol_params = VolumeParams(vol_sz=(60, 60, 30), vres=2, N_neur=5)
    neur_params = NeuronParams(n_samps=150, nuc_fluorsc=0.5)
    dend_params = DendParams(
        dtParams=(8, 25, 15, 1, 3),
        atParams=(2, 5, 5, 5, 1),
        dims=(12, 12, 12),
        dimsSS=(5, 5, 5),
    )

    print("Step 1: Sampling neurons...")
    neurons, angles, positions = sample_dense_neurons(
        vol_params, neur_params, verbose=1
    )
    n_actual = len(neurons)
    print(f"  Placed {n_actual} neurons")

    print("Step 2: Generating neural volume...")
    vol_result = generate_neural_volume(
        neurons, positions, vol_params, neur_params, verbose=1
    )

    print("Step 3: Growing dendrites...")
    dend_result = grow_neuron_dendrites(
        vol_params, dend_params, vol_result,
        positions=positions,
        rotation_angles=angles,
        verbose=2,
    )

    neur_soma = vol_result.neur_soma
    neur_num = dend_result.neur_num
    gp_nuc = vol_result.gp_nuc
    grid_shape = neur_soma.shape

    total_soma = int(np.sum(neur_soma > 0))
    total_after = int(np.sum(neur_num > 0))
    total_dend = total_after - total_soma

    print(f"\nResults:")
    print(f"  Grid shape: {grid_shape}")
    print(f"  Soma voxels: {total_soma}")
    print(f"  Total voxels: {total_after}")
    print(f"  Dendrite voxels: {total_dend}")

    # Build JSON structure
    result = {
        "source": "python",
        "grid_shape": list(grid_shape),
        "params": {
            "vol_sz": list(vol_params.vol_sz),
            "vres": vol_params.vres,
            "N_neur": vol_params.N_neur,
        },
        "stats": {
            "total_soma": total_soma,
            "total_after": total_after,
            "total_dendrites": total_dend,
        },
        "neurons": [],
    }

    for k in range(1, n_actual + 1):
        soma_mask = neur_soma == k
        dend_mask = (neur_num == k) & ~soma_mask

        soma_coords = np.argwhere(soma_mask)  # (N, 3)
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

    out_dir = Path(__file__).parent
    out_path = out_dir / "python_dendrites.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"\nSaved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
