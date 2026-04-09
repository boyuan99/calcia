"""Export Phase 1 results as meshes + point clouds for Three.js visualization.

Soma and nucleus are exported as convex-hull meshes (scipy.spatial.ConvexHull),
giving proper 3D solid shapes with lighting.
Dendrites, vessels, and background are exported as subsampled point clouds.

Output: comparison_tools/python_phase1_mesh.json

Usage:
    conda run -n calcia --cwd "C:/Users/boyuan/Documents/GitHub/calcia" \
        python comparison_tools/export_phase1_mesh.py
"""
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calcia import import_pipeline_output

# 25-color palette cycling over 625 neurons
PALETTE = [
    0xff4444, 0xff8844, 0xffcc44, 0x88ff44, 0x44ff88,
    0x44ffcc, 0x44ccff, 0x4488ff, 0x8844ff, 0xcc44ff,
    0xff44cc, 0xff4488, 0xff6600, 0x00ff66, 0x0066ff,
    0xff0066, 0x66ff00, 0x6600ff, 0xffff00, 0x00ffff,
    0xff00ff, 0xff9900, 0x99ff00, 0x0099ff, 0xff0099,
]


def neuron_rgb(k):
    """Return (r, g, b) in [0, 1] for 1-based neuron ID k."""
    c = PALETTE[(k - 1) % len(PALETTE)]
    return ((c >> 16) & 0xff) / 255.0, ((c >> 8) & 0xff) / 255.0, (c & 0xff) / 255.0


def convex_hull_mesh(lin_indices, grid_shape, max_pts=3000):
    """Compute convex-hull mesh from C-order linear indices.

    Returns (vertices, faces) as numpy arrays, or (None, None) on failure.
    vertices: (V, 3) float32 — world voxel coordinates
    faces:    (F, 3) int32  — triangle indices into vertices
    """
    if len(lin_indices) < 4:
        return None, None

    coords = np.column_stack(
        np.unravel_index(lin_indices, grid_shape)
    ).astype(np.float32)

    # Subsample before hull (hull only needs surface points, not interior)
    if len(coords) > max_pts:
        idx = np.random.choice(len(coords), max_pts, replace=False)
        coords = coords[idx]

    try:
        hull = ConvexHull(coords)
    except QhullError:
        return None, None

    # hull.vertices: indices into coords that form hull vertices
    # hull.simplices: triangle faces, indices into coords
    hull_verts = coords[hull.vertices]                       # (V, 3)
    v_map = {old: new for new, old in enumerate(hull.vertices)}
    hull_faces = np.array([[v_map[v] for v in f] for f in hull.simplices], dtype=np.int32)
    return hull_verts, hull_faces


def subsample_lin(lin_indices, grid_shape, max_n):
    """Convert linear indices to (N, 3) xyz, subsample to max_n rows."""
    if len(lin_indices) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    coords = np.column_stack(
        np.unravel_index(lin_indices, grid_shape)
    ).astype(np.int32)
    if len(coords) <= max_n:
        return coords
    idx = np.round(np.linspace(0, len(coords) - 1, max_n)).astype(int)
    return coords[idx]


def subsample_coords(coords, max_n):
    if len(coords) <= max_n:
        return coords
    idx = np.round(np.linspace(0, len(coords) - 1, max_n)).astype(int)
    return coords[idx]


def main():
    output_path = Path(__file__).parent.parent / "examples" / "output" / "output.npz"
    print(f"Loading {output_path}...")
    out = import_pipeline_output(str(output_path))

    vp = out.params["vol_params"]
    N_neur = vp.N_neur
    N_den = vp.N_den
    grid_shape = out.neur_num.shape   # (Nx, Ny, Nz) C-order

    print(f"Grid: {grid_shape}, N_neur={N_neur}, N_den={N_den}")

    # Merged mesh accumulators (all neurons concatenated)
    soma_verts, soma_faces, soma_colors = [], [], []
    nuc_verts,  nuc_faces,  nuc_colors  = [], [], []
    dend_pts, dend_colors = [], []

    soma_v_offset = 0
    nuc_v_offset  = 0
    n_failed = 0

    t0 = time.time()
    print(f"Building convex-hull meshes for {N_neur} neurons...")

    for k_idx in range(N_neur):
        k = k_idx + 1
        r, g, b = neuron_rgb(k)

        # --- Soma mesh ---
        soma_lin = out.gp_soma[k_idx][0]
        sv, sf = convex_hull_mesh(soma_lin, grid_shape)
        if sv is not None:
            for v in sv:
                soma_verts += [round(float(v[0]), 1), round(float(v[1]), 1), round(float(v[2]), 1)]
                soma_colors += [round(r, 3), round(g, 3), round(b, 3)]
            for f in sf:
                soma_faces += [int(soma_v_offset + f[0]),
                                int(soma_v_offset + f[1]),
                                int(soma_v_offset + f[2])]
            soma_v_offset += len(sv)
        else:
            n_failed += 1

        # --- Nucleus mesh ---
        nuc_lin = out.gp_nuc[k_idx][0]
        nv, nf = convex_hull_mesh(nuc_lin, grid_shape, max_pts=2000)
        if nv is not None:
            dr, dg, db = r * 0.55, g * 0.55, b * 0.55   # darker shade
            for v in nv:
                nuc_verts += [round(float(v[0]), 1), round(float(v[1]), 1), round(float(v[2]), 1)]
                nuc_colors += [round(dr, 3), round(dg, 3), round(db, 3)]
            for f in nf:
                nuc_faces += [int(nuc_v_offset + f[0]),
                               int(nuc_v_offset + f[1]),
                               int(nuc_v_offset + f[2])]
            nuc_v_offset += len(nv)

        # --- Dendrite point cloud (with per-neuron color) ---
        neur_mask = out.neur_num == k
        soma_vol = np.zeros(grid_shape, dtype=bool)
        if len(soma_lin) > 0:
            soma_vol.ravel()[soma_lin] = True
        dc = subsample_coords(np.argwhere(neur_mask & ~soma_vol).astype(np.int32), 250)
        ir, ig, ib = int(r * 255), int(g * 255), int(b * 255)
        for c in dc:
            dend_pts   += [int(c[0]), int(c[1]), int(c[2])]
            dend_colors += [ir, ig, ib]

        if (k_idx + 1) % 100 == 0:
            print(f"  {k_idx + 1}/{N_neur}  ({time.time() - t0:.1f}s)")

    print(f"  Done: {soma_v_offset} soma verts, {len(soma_faces)//3} faces  "
          f"| {nuc_v_offset} nuc verts, {len(nuc_faces)//3} faces  "
          f"| {n_failed} hull failures")

    # --- Apical dendrites ---
    print("Extracting apical dendrites...")
    ap_coords = subsample_coords(np.argwhere(out.neur_num_ad > 0).astype(np.int32), 6000)
    apical_pts = ap_coords.ravel().tolist()

    # --- Vessels (marching cubes mesh) ---
    print("Extracting vessels...")
    vessel_verts, vessel_faces = [], []
    vessel_count = 0
    if out.neur_ves is not None:
        ves_mask = out.neur_ves > 0
        vessel_count = int(np.sum(ves_mask))
        print(f"  Vessel voxels: {vessel_count}")
        if vessel_count > 0:
            # Downsample 2x for speed (500×500×200 → 250×250×100)
            ves_small = ves_mask[::2, ::2, ::2].astype(np.float32)
            try:
                vv, vf, _, _ = marching_cubes(ves_small, level=0.5)
                vv = vv * 2.0   # scale back to full resolution
                vessel_verts = np.round(vv, 2).ravel().tolist()
                vessel_faces = vf.ravel().tolist()
                print(f"  Vessel mesh: {len(vv)} verts, {len(vf)} faces")
            except Exception as e:
                print(f"  Vessel marching cubes failed: {e}")
    else:
        print("  neur_ves is None — vessels disabled in this simulation")

    # --- Background ---
    print("Extracting background dendrites...")
    all_bg_lin = []
    for proc in out.bg_proc:
        if proc.indices is not None and len(proc.indices) > 0:
            all_bg_lin.append(proc.indices)
    if all_bg_lin:
        all_bg_lin = np.concatenate(all_bg_lin)
        bg_coords = subsample_lin(all_bg_lin, grid_shape, 5000)
        bg_pts = bg_coords.ravel().tolist()
        bg_count = len(all_bg_lin)
    else:
        bg_pts = []
        bg_count = 0

    # --- Stats ---
    neur_vol_nz = out.neur_vol[out.neur_vol > 0]
    stats = {
        "n_neurons": N_neur,
        "vessel_voxels": vessel_count,
        "fluor_mean": float(neur_vol_nz.mean()) if len(neur_vol_nz) > 0 else 0.0,
        "fluor_fraction": float(len(neur_vol_nz) / out.neur_vol.size),
        "mean_soma_size": float(np.mean([len(out.gp_soma[k][0]) for k in range(N_neur)])),
        "mean_nuc_size":  float(np.mean([len(out.gp_nuc[k][0])  for k in range(N_neur)])),
        "n_bg_proc": len(out.bg_proc),
        "apical_count": int(np.sum(out.neur_num_ad > 0)),
        "soma_mesh_verts": soma_v_offset,
        "soma_mesh_faces": len(soma_faces) // 3,
        "vessel_mesh_verts": len(vessel_verts) // 3,
        "vessel_mesh_faces": len(vessel_faces) // 3,
    }

    # --- Assemble result ---
    result = {
        "source": "python",
        "seed": 42,
        "grid_shape": list(grid_shape),
        "params": {"N_neur": N_neur, "N_den": N_den, "vres": vp.vres},
        "soma_verts":    soma_verts,    # flat [x,y,z, ...]
        "soma_faces":    soma_faces,    # flat [i,j,k, ...]
        "soma_colors":   soma_colors,   # flat [r,g,b, ...] per vertex, 0-1
        "nuc_verts":     nuc_verts,
        "nuc_faces":     nuc_faces,
        "nuc_colors":    nuc_colors,
        "vessel_verts":  vessel_verts,  # flat [x,y,z, ...]
        "vessel_faces":  vessel_faces,  # flat [i,j,k, ...]
        "dend_pts":      dend_pts,      # flat [x,y,z, ...]
        "dend_colors":   dend_colors,   # flat [r,g,b, ...] per point, 0-255
        "apical_pts":    apical_pts,    # flat [x,y,z, ...]
        "bg_pts":        bg_pts,        # flat [x,y,z, ...]
        "stats":         stats,
    }

    out_path = Path(__file__).parent / "python_phase1_mesh.json"
    print(f"\nWriting {out_path}...")
    with open(out_path, "w") as f:
        json.dump(result, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)  total={time.time()-t0:.1f}s")
    print("Open viewer_phase1_mesh.html (serve: python -m http.server 8080)")


if __name__ == "__main__":
    main()
