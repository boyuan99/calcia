"""Export Python Phase 1 simulation results as JSON for Three.js visualization.

Reads examples/output/output.npz and writes comparison_tools/python_phase1_viz.json.

Usage:
    conda run -n calcia --cwd "C:/Users/boyuan/Documents/GitHub/calcia" \\
        python comparison_tools/export_phase1_viz.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calcia import import_pipeline_output


def subsample(coords, max_n):
    """Subsample rows of an (N, 3) int array to at most max_n rows."""
    n = len(coords)
    if n == 0:
        return coords
    if n <= max_n:
        return coords
    idx = np.round(np.linspace(0, n - 1, max_n)).astype(int)
    return coords[idx]


def lin_to_xyz(lin_indices, grid_shape):
    """Convert C-order linear indices to (N, 3) 0-based xyz coordinate array."""
    if len(lin_indices) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.column_stack(np.unravel_index(lin_indices, grid_shape)).astype(np.int32)


def main():
    output_path = Path(__file__).parent.parent / "examples" / "output" / "output.npz"
    print(f"Loading {output_path}...")
    out = import_pipeline_output(str(output_path))

    vp = out.params["vol_params"]
    N_neur = vp.N_neur
    N_den = vp.N_den
    vres = vp.vres
    grid_shape = out.neur_num.shape   # (500, 500, 200) C-order

    print(f"Grid: {grid_shape}, N_neur={N_neur}, N_den={N_den}")

    result = {
        "source": "python",
        "seed": 42,
        "grid_shape": list(grid_shape),
        "params": {
            "N_neur": N_neur,
            "N_den": N_den,
            "vres": vres,
            "vol_sz": list(vp.vol_sz),
            "vol_depth": vp.vol_depth,
        },
        "neurons": [],
    }

    # --- Per-neuron data ---
    print(f"Extracting {N_neur} neurons...")
    neur_num = out.neur_num

    for k_idx in range(N_neur):
        k = k_idx + 1   # 1-based neuron ID

        # Soma: from gp_soma linear indices
        soma_lin = out.gp_soma[k_idx][0]
        soma_coords = subsample(lin_to_xyz(soma_lin, grid_shape), 300)

        # Dendrites: neur_num == k minus soma
        neur_mask = neur_num == k
        if len(soma_lin) > 0:
            soma_vol = np.zeros(grid_shape, dtype=bool)
            soma_vol.ravel()[soma_lin] = True
            dend_coords = subsample(np.argwhere(neur_mask & ~soma_vol).astype(np.int32), 600)
        else:
            dend_coords = subsample(np.argwhere(neur_mask).astype(np.int32), 600)

        # Nucleus
        nuc_lin = out.gp_nuc[k_idx][0]
        nuc_coords = subsample(lin_to_xyz(nuc_lin, grid_shape), 150)

        neuron = {
            "id": k,
            "position": out.locs[k_idx].tolist(),
            "soma_count": int(np.sum(neur_mask)),
            "dendrite_count": int(np.sum(neur_mask)) - len(soma_lin),
            "nucleus_count": len(nuc_lin),
            "soma_positions": soma_coords.ravel().tolist(),
            "dendrite_positions": dend_coords.ravel().tolist(),
            "nucleus_positions": nuc_coords.ravel().tolist(),
        }
        result["neurons"].append(neuron)

        if (k_idx + 1) % 100 == 0:
            print(f"  {k_idx + 1}/{N_neur} done")

    # --- Apical dendrites ---
    print("Extracting apical dendrites...")
    ad_mask = out.neur_num_ad > 0
    ap_coords = subsample(np.argwhere(ad_mask).astype(np.int32), 5000)
    result["apical_positions"] = ap_coords.ravel().tolist()
    result["apical_count"] = int(np.sum(ad_mask))
    print(f"  Apical voxels: {result['apical_count']} (showing {len(ap_coords)})")

    # --- Vessels ---
    print("Extracting vessels...")
    if out.neur_ves is not None:
        ves_coords = subsample(np.argwhere(out.neur_ves > 0).astype(np.int32), 3000)
        result["vessel_positions"] = ves_coords.ravel().tolist()
        result["vessel_count"] = int(np.sum(out.neur_ves > 0))
    else:
        result["vessel_positions"] = []
        result["vessel_count"] = 0
    print(f"  Vessel voxels: {result['vessel_count']} (showing {len(ves_coords) if out.neur_ves is not None else 0})")

    # --- Background dendrites ---
    print("Extracting background dendrites...")
    all_bg_lin = []
    for proc in out.bg_proc:
        if proc.indices is not None and len(proc.indices) > 0:
            all_bg_lin.append(proc.indices)
    if all_bg_lin:
        all_bg_lin = np.concatenate(all_bg_lin)
        bg_coords = subsample(lin_to_xyz(all_bg_lin, grid_shape), 3000)
        result["bg_positions"] = bg_coords.ravel().tolist()
        result["bg_count"] = len(all_bg_lin)
    else:
        result["bg_positions"] = []
        result["bg_count"] = 0
    print(f"  BG voxels: {result['bg_count']} (showing {len(bg_coords) if all_bg_lin is not None and len(all_bg_lin) > 0 else 0})")

    # --- Stats ---
    neur_vol_nz = out.neur_vol[out.neur_vol > 0]
    result["stats"] = {
        "n_neurons": N_neur,
        "vessel_voxels": result["vessel_count"],
        "fluor_mean": float(neur_vol_nz.mean()) if len(neur_vol_nz) > 0 else 0.0,
        "fluor_fraction": float(len(neur_vol_nz) / out.neur_vol.size),
        "mean_soma_size": float(np.mean([len(out.gp_soma[k][0]) for k in range(N_neur)])),
        "mean_nuc_size": float(np.mean([len(out.gp_nuc[k][0]) for k in range(N_neur)])),
        "n_bg_proc": len(out.bg_proc),
        "apical_count": result["apical_count"],
    }

    # --- Write JSON ---
    out_path = Path(__file__).parent / "python_phase1_viz.json"
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        json.dump(result, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    print("\nDone! Open viewer_phase1.html (serve with: python -m http.server 8080)")


if __name__ == "__main__":
    main()
