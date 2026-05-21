"""Export neural volume meshes from Phase 1 cache to GLB for Three.js viewing.

Usage:
    conda run -n calcia python examples/export_volume_mesh.py
    conda run -n calcia python examples/export_volume_mesh.py --downsample 2

Outputs GLB files into examples/output/meshes/
"""
import argparse
import os
import pickle
import time

import numpy as np
from skimage.measure import marching_cubes
import trimesh


def extract_mesh(binary_vol, voxel_spacing=(1, 1, 1), level=0.5):
    """Run marching cubes on a binary volume, return trimesh.Trimesh."""
    verts, faces, normals, _ = marching_cubes(
        binary_vol.astype(np.float32),
        level=level,
        spacing=voxel_spacing,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def color_vertices_by_id(mesh, verts, id_vol, voxel_spacing, colormap):
    """Assign per-vertex color by looking up the component ID at each vertex."""
    # Convert vertex positions back to voxel indices
    ijk = (verts / np.array(voxel_spacing)).astype(int)
    ijk = np.clip(ijk, 0, np.array(id_vol.shape) - 1)
    ids = id_vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    colors = np.zeros((len(verts), 4), dtype=np.uint8)
    unique_ids = np.unique(ids)
    for uid in unique_ids:
        if uid == 0:
            continue
        mask = ids == uid
        ci = uid % len(colormap)
        colors[mask] = colormap[ci]

    # Set alpha
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors


def make_colormap(n=64):
    """Generate a distinguishable colormap."""
    rng = np.random.RandomState(123)
    colors = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        hue = (i * 137.508) % 360  # golden angle
        # HSV to RGB (S=0.7, V=0.9)
        h = hue / 60
        c = 0.9 * 0.7
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors[i] = [int((r + m) * 255), int((g + m) * 255),
                      int((b + m) * 255), 255]
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="examples/output/phase12_cache_250.pkl")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor (default 2: 500->250 per axis)")
    parser.add_argument("--outdir", default="examples/output/meshes")
    parser.add_argument("--max-neurons", type=int, default=None,
                        help="Max neurons to include (None=all)")
    parser.add_argument("--decimate", type=float, default=0.05,
                        help="Target fraction of faces to keep (default 0.05 = 5%%)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ds = args.downsample

    # --- Load data ---
    print(f"Loading cache: {args.cache}")
    t0 = time.time()
    with open(args.cache, "rb") as f:
        vol_out, vol_params, opt_out = pickle.load(f)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    neur_num = vol_out.neur_num      # (Nx, Ny, Nz) uint16
    neur_ves = vol_out.neur_ves      # (Nx, Ny, Nz_full) uint8
    N_neur = vol_params.N_neur
    vres = vol_params.vres
    voxel_um = 1.0 / vres  # microns per voxel

    shape = neur_num.shape
    print(f"  Volume shape: {shape}, N_neur: {N_neur}, vres: {vres}")

    # --- Downsample if needed ---
    if ds > 1:
        print(f"  Downsampling {ds}x ...")
        # Use max-pooling for label volumes (preserve structure)
        from scipy.ndimage import maximum_filter
        # Subsample after smoothing
        neur_num = neur_num[::ds, ::ds, ::ds]
        ves_depth = shape[2]  # match neur_num z range
        neur_ves_crop = neur_ves[:, :, :ves_depth] if neur_ves is not None else None
        if neur_ves_crop is not None:
            neur_ves_crop = neur_ves_crop[::ds, ::ds, ::ds]
        print(f"  Downsampled shape: {neur_num.shape}")
    else:
        neur_ves_crop = neur_ves[:, :, :shape[2]] if neur_ves is not None else None

    voxel_spacing = (voxel_um * ds, voxel_um * ds, voxel_um * ds)

    decimate = args.decimate
    colormap = make_colormap(64)
    scene = trimesh.Scene()

    def simplify_mesh(mesh, keep_frac):
        """Simplify mesh using fast_simplification quadric decimation.

        Parameters
        ----------
        mesh : trimesh.Trimesh
        keep_frac : float
            Fraction of faces to *keep* (e.g. 0.05 = keep 5%).
        """
        if keep_frac >= 1.0 or len(mesh.faces) < 2000:
            return mesh
        target_reduction = 1.0 - keep_frac  # fraction to *remove*
        try:
            simplified = mesh.simplify_quadric_decimation(
                percent=target_reduction)
            return simplified
        except TypeError:
            # Older trimesh: face_count API
            target_faces = max(1000, int(len(mesh.faces) * keep_frac))
            simplified = mesh.simplify_quadric_decimation(target_faces)
            return simplified

    # --- 1. Neuron soma meshes ---
    print("\nExtracting neuron soma meshes ...")
    t0 = time.time()

    max_n = args.max_neurons or N_neur
    soma_mask = (neur_num >= 1) & (neur_num <= min(max_n, N_neur))
    if soma_mask.any():
        mesh = extract_mesh(soma_mask, voxel_spacing)
        print(f"  Raw: {mesh.faces.shape[0]} faces, simplifying to {decimate*100:.0f}% ...")
        mesh = simplify_mesh(mesh, decimate)
        color_vertices_by_id(mesh, mesh.vertices, neur_num, voxel_spacing, colormap)
        mesh.metadata["name"] = "neurons"
        scene.add_geometry(mesh, node_name="neurons")
        print(f"  Neurons: {mesh.vertices.shape[0]} vertices, "
              f"{mesh.faces.shape[0]} faces ({time.time() - t0:.1f}s)")
    else:
        print("  No neuron voxels found!")

    # --- 2. Dendrite meshes (components > N_neur in neur_num) ---
    print("\nExtracting dendrite meshes ...")
    t0 = time.time()

    dend_mask = neur_num > N_neur
    if dend_mask.any():
        mesh_d = extract_mesh(dend_mask, voxel_spacing)
        print(f"  Raw: {mesh_d.faces.shape[0]} faces, simplifying ...")
        mesh_d = simplify_mesh(mesh_d, decimate)
        dend_color = np.full((mesh_d.vertices.shape[0], 4), [80, 200, 120, 255],
                             dtype=np.uint8)
        mesh_d.visual.vertex_colors = dend_color
        mesh_d.metadata["name"] = "dendrites"
        scene.add_geometry(mesh_d, node_name="dendrites")
        print(f"  Dendrites: {mesh_d.vertices.shape[0]} vertices, "
              f"{mesh_d.faces.shape[0]} faces ({time.time() - t0:.1f}s)")
    else:
        print("  No dendrite voxels found.")

    # --- 3. Blood vessel meshes ---
    if neur_ves_crop is not None:
        print("\nExtracting vessel meshes ...")
        t0 = time.time()

        ves_mask = neur_ves_crop > 0
        if ves_mask.any():
            mesh_v = extract_mesh(ves_mask, voxel_spacing)
            print(f"  Raw: {mesh_v.faces.shape[0]} faces, simplifying ...")
            mesh_v = simplify_mesh(mesh_v, decimate * 2)  # keep more detail for vessels
            ves_color = np.full((mesh_v.vertices.shape[0], 4), [200, 50, 50, 255],
                                dtype=np.uint8)
            mesh_v.visual.vertex_colors = ves_color
            mesh_v.metadata["name"] = "vessels"
            scene.add_geometry(mesh_v, node_name="vessels")
            print(f"  Vessels: {mesh_v.vertices.shape[0]} vertices, "
                  f"{mesh_v.faces.shape[0]} faces ({time.time() - t0:.1f}s)")
        else:
            print("  No vessel voxels found.")

    # --- Export GLB ---
    glb_path = os.path.join(args.outdir, "neural_volume.glb")
    print(f"\nExporting GLB: {glb_path}")
    t0 = time.time()
    scene.export(glb_path)
    sz_mb = os.path.getsize(glb_path) / 1024 / 1024
    print(f"  Saved: {sz_mb:.1f} MB ({time.time() - t0:.1f}s)")

    # Also export individual meshes for flexibility
    for name, geom in scene.geometry.items():
        ind_path = os.path.join(args.outdir, f"{name}.glb")
        geom.export(ind_path)
        sz = os.path.getsize(ind_path) / 1024 / 1024
        print(f"  {name}.glb: {sz:.1f} MB")

    print("\nDone! Open examples/output/meshes/viewer.html to view.")


if __name__ == "__main__":
    main()
