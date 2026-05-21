"""Generate per-component mesh visualization data from Phase 1 cache.

For each neuron, soma, dendrite, etc., extract a small bounding-box subvolume
and run marching cubes locally. This produces 625 distinct meshes per layer
instead of one merged blob (which is what the original export_volume_mesh.py
produced).

Usage:
    conda run -n calcia python examples/generate_mesh_viz.py
    conda run -n calcia python examples/generate_mesh_viz.py --decimate 0.5

Outputs in examples/output/mesh_viz/:
    manifest.json
    step1_vessels.glb       - single vessel mesh
    step3_somas.glb         - 625 individual soma meshes (with vertex colors)
    step3_nuclei.glb        - 625 nucleus meshes
    step4_dendrites.glb     - per-neuron basal dendrite meshes
    step5_apical.glb        - per-parent apical dendrite meshes
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def golden_color(idx, sat=0.65, val=0.95):
    """Return RGB color (0-1) for a given component ID using golden-angle hue."""
    hue = (idx * 137.508) % 360
    h = hue / 60
    c = val * sat
    x = c * (1 - abs(h % 2 - 1))
    if h < 1: r, g, b = c, x, 0
    elif h < 2: r, g, b = x, c, 0
    elif h < 3: r, g, b = 0, c, x
    elif h < 4: r, g, b = 0, x, c
    elif h < 5: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    m = val - c
    return np.array([r + m, g + m, b + m], dtype=np.float32)


def get_soma_indices(vol_out, n):
    """Concatenate soma cytoplasm + smoothed body indices for neuron n."""
    s = vol_out.gp_soma[n]
    if isinstance(s, tuple):
        if len(s[1]) > 0:
            return np.concatenate([s[0], s[1]])
        return s[0]
    return s


def extract_local_mesh(indices, shape, voxel_um, padding=3):
    """Extract a small mesh from a set of voxel linear indices via marching cubes.

    Returns a trimesh.Trimesh in global µm coordinates, or None if too small.
    """
    if len(indices) < 8:
        return None

    coords = np.unravel_index(indices, shape)
    i_min, i_max = coords[0].min(), coords[0].max() + 1
    j_min, j_max = coords[1].min(), coords[1].max() + 1
    k_min, k_max = coords[2].min(), coords[2].max() + 1

    # Add padding
    i_min = max(0, i_min - padding)
    j_min = max(0, j_min - padding)
    k_min = max(0, k_min - padding)
    i_max = min(shape[0], i_max + padding)
    j_max = min(shape[1], j_max + padding)
    k_max = min(shape[2], k_max + padding)

    sub_shape = (i_max - i_min, j_max - j_min, k_max - k_min)
    if min(sub_shape) < 2:
        return None

    # Build local binary mask
    local = np.zeros(sub_shape, dtype=np.float32)
    li = coords[0] - i_min
    lj = coords[1] - j_min
    lk = coords[2] - k_min
    local[li, lj, lk] = 1.0

    # Marching cubes
    try:
        verts, faces, normals, _ = marching_cubes(
            local, level=0.5, spacing=(voxel_um, voxel_um, voxel_um))
    except (RuntimeError, ValueError):
        return None

    if len(verts) < 4 or len(faces) < 1:
        return None

    # Translate to global µm coordinates
    verts[:, 0] += i_min * voxel_um
    verts[:, 1] += j_min * voxel_um
    verts[:, 2] += k_min * voxel_um

    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def color_mesh(mesh, rgb01):
    """Apply uniform RGB color (0-1 floats) as vertex colors."""
    n = len(mesh.vertices)
    colors = np.zeros((n, 4), dtype=np.uint8)
    colors[:, 0] = int(rgb01[0] * 255)
    colors[:, 1] = int(rgb01[1] * 255)
    colors[:, 2] = int(rgb01[2] * 255)
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors


def simplify(mesh, keep_frac):
    """Decimate mesh, keeping `keep_frac` of original faces."""
    if keep_frac >= 1.0 or len(mesh.faces) < 200:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(percent=1.0 - keep_frac)
    except Exception:
        return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",
                        default="examples/output/phase12_cache_250.pkl")
    parser.add_argument("--outdir",
                        default="examples/output/mesh_viz")
    parser.add_argument("--soma-decimate", type=float, default=0.5,
                        help="Soma mesh face fraction to keep (default 0.5)")
    parser.add_argument("--dendrite-decimate", type=float, default=0.15,
                        help="Dendrite face fraction to keep (default 0.15)")
    parser.add_argument("--vessel-decimate", type=float, default=0.3,
                        help="Vessel face fraction to keep (default 0.3)")
    parser.add_argument("--padding", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ============================================================
    # Load
    # ============================================================
    log(f"Loading cache: {args.cache}")
    t0 = time.time()
    with open(args.cache, "rb") as f:
        loaded = pickle.load(f)
    # Accept either a 2-tuple (vol_out, vol_params) from the optics-independent
    # Phase 1 cache, or a 3-tuple (..., opt_out) from demo_full_pipeline.
    vol_out, vol_params = loaded[0], loaded[1]
    log(f"  Loaded in {time.time() - t0:.1f}s")

    N_neur = vol_params.N_neur
    vres = vol_params.vres
    voxel_um = 1.0 / vres
    neur_num = vol_out.neur_num
    neur_num_ad = vol_out.neur_num_ad
    neur_ves = vol_out.neur_ves
    shape = neur_num.shape
    Nx, Ny, Nz = shape

    log(f"  Volume: {shape}, N_neur={N_neur}, vres={vres}")

    manifest = {
        "vol_shape": list(shape),
        "vol_sz_um": list(vol_params.vol_sz),
        "vres": vres,
        "voxel_um": voxel_um,
        "N_neur": N_neur,
        "steps": {},
    }

    # ============================================================
    # Step 1: Vessels (single mesh)
    # ============================================================
    log("\n=== Step 1: Vessels ===")
    t0 = time.time()
    if neur_ves is not None:
        ves_mask = (neur_ves > 0).astype(np.float32)
        log(f"  Vessel voxels: {int(ves_mask.sum()):,}")

        try:
            verts, faces, normals, _ = marching_cubes(
                ves_mask, level=0.5,
                spacing=(voxel_um, voxel_um, voxel_um))
            mesh = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals)
            log(f"  Raw mesh: {len(mesh.faces):,} faces, decimating to {args.vessel_decimate*100:.0f}%")
            mesh = simplify(mesh, args.vessel_decimate)
            color_mesh(mesh, np.array([0.85, 0.2, 0.2]))
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="vessels")
            path = os.path.join(args.outdir, "step1_vessels.glb")
            scene.export(path)
            sz = os.path.getsize(path) / 1024 / 1024
            manifest["steps"]["1"] = {
                "name": "Vessels",
                "file": "step1_vessels.glb",
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "color": [217, 51, 51],
                "size_mb": round(sz, 2),
            }
            log(f"  Saved {sz:.1f} MB ({len(mesh.faces):,} faces) "
                f"in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"  Failed: {e}")

    # ============================================================
    # Step 2: Neuron centers (sphere markers, no marching cubes)
    # ============================================================
    log("\n=== Step 2: Neuron centers ===")
    t0 = time.time()
    locs = vol_out.locs[:N_neur]

    # Compute approximate radii
    radii = np.zeros(N_neur, dtype=np.float32)
    for n in range(N_neur):
        soma_vox = len(get_soma_indices(vol_out, n))
        nuc_vox = len(vol_out.gp_nuc[n][0])
        total_vox = soma_vox + nuc_vox
        if total_vox > 0:
            vol_um3 = total_vox * voxel_um ** 3
            radii[n] = (vol_um3 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        else:
            radii[n] = 5.9

    scene = trimesh.Scene()
    for n in range(N_neur):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(radii[n]))
        sphere.apply_translation(locs[n])
        color_mesh(sphere, golden_color(n + 1))
        scene.add_geometry(sphere, node_name=f"neuron_{n}")
    path = os.path.join(args.outdir, "step2_centers.glb")
    scene.export(path)
    sz = os.path.getsize(path) / 1024 / 1024
    manifest["steps"]["2"] = {
        "name": "Neuron centers",
        "file": "step2_centers.glb",
        "count": N_neur,
        "size_mb": round(sz, 2),
    }
    log(f"  Saved {sz:.1f} MB ({N_neur} spheres) in {time.time()-t0:.1f}s")

    # ============================================================
    # Step 3: Per-neuron soma + nucleus meshes
    # ============================================================
    log("\n=== Step 3: Soma + nucleus meshes ===")
    t0 = time.time()

    soma_scene = trimesh.Scene()
    nuc_scene = trimesh.Scene()
    n_soma_faces = 0
    n_nuc_faces = 0
    n_skipped = 0

    for n in range(N_neur):
        idx_soma_only = get_soma_indices(vol_out, n)
        idx_nuc = vol_out.gp_nuc[n][0]
        idx_all = np.concatenate([idx_soma_only, idx_nuc]) if len(idx_nuc) else idx_soma_only

        # Full soma+nucleus surface (egg-shaped cell body)
        mesh_s = extract_local_mesh(idx_all, shape, voxel_um, args.padding)
        if mesh_s is None:
            n_skipped += 1
            continue
        mesh_s = simplify(mesh_s, args.soma_decimate)
        color_mesh(mesh_s, golden_color(n + 1))
        soma_scene.add_geometry(mesh_s, node_name=f"soma_{n}")
        n_soma_faces += len(mesh_s.faces)

        # Just nucleus
        if len(idx_nuc) > 0:
            mesh_n = extract_local_mesh(idx_nuc, shape, voxel_um, args.padding)
            if mesh_n is not None:
                mesh_n = simplify(mesh_n, args.soma_decimate)
                color_mesh(mesh_n, golden_color(n + 1) * 0.3)  # darker
                nuc_scene.add_geometry(mesh_n, node_name=f"nuc_{n}")
                n_nuc_faces += len(mesh_n.faces)

        if (n + 1) % 100 == 0:
            log(f"  Processed {n+1}/{N_neur}, "
                f"{n_soma_faces:,} soma faces, {n_nuc_faces:,} nuc faces")

    soma_path = os.path.join(args.outdir, "step3_somas.glb")
    nuc_path = os.path.join(args.outdir, "step3_nuclei.glb")
    soma_scene.export(soma_path)
    nuc_scene.export(nuc_path)
    soma_sz = os.path.getsize(soma_path) / 1024 / 1024
    nuc_sz = os.path.getsize(nuc_path) / 1024 / 1024

    manifest["steps"]["3"] = {
        "name": "Soma + nucleus",
        "files": {
            "somas": "step3_somas.glb",
            "nuclei": "step3_nuclei.glb",
        },
        "soma_faces": n_soma_faces,
        "nucleus_faces": n_nuc_faces,
        "skipped": n_skipped,
        "soma_size_mb": round(soma_sz, 2),
        "nucleus_size_mb": round(nuc_sz, 2),
    }
    log(f"  Soma {soma_sz:.1f} MB ({n_soma_faces:,} faces), "
        f"Nucleus {nuc_sz:.1f} MB ({n_nuc_faces:,} faces), "
        f"skipped {n_skipped}, {time.time()-t0:.1f}s")

    # ============================================================
    # Step 4: Per-neuron basal dendrites
    # ============================================================
    log("\n=== Step 4: Basal dendrite meshes ===")
    t0 = time.time()

    # Build a per-neuron flat-index lookup: which soma/nuc voxels to exclude
    flat_size = Nx * Ny * Nz
    soma_flat_set_per_n = []
    for n in range(N_neur):
        idx_s = get_soma_indices(vol_out, n)
        idx_nuc = vol_out.gp_nuc[n][0]
        all_excl = np.concatenate([idx_s, idx_nuc]) if len(idx_nuc) else idx_s
        soma_flat_set_per_n.append(set(all_excl.tolist()))

    # Build neuron-to-voxel mapping using neur_num
    log("  Indexing neur_num by neuron ID ...")
    flat_neur = neur_num.ravel()
    # Use np.where on each ID would be slow; use argsort trick
    order = np.argsort(flat_neur, kind='stable')
    sorted_ids = flat_neur[order]
    # Find boundaries between IDs
    boundaries = np.searchsorted(sorted_ids,
                                  np.arange(N_neur + 2),
                                  side='left')
    log(f"  Done indexing in {time.time()-t0:.1f}s")

    dend_scene = trimesh.Scene()
    n_dend_faces = 0
    n_skipped = 0

    for n in range(N_neur):
        # Get voxels with neur_num == n+1
        start = boundaries[n + 1]
        end = boundaries[n + 2]
        all_vox = order[start:end]
        if len(all_vox) == 0:
            n_skipped += 1
            continue
        # Exclude soma+nucleus voxels
        excl = soma_flat_set_per_n[n]
        dend_vox = np.array([v for v in all_vox if v not in excl],
                             dtype=np.int64)
        if len(dend_vox) < 50:
            n_skipped += 1
            continue

        mesh_d = extract_local_mesh(dend_vox, shape, voxel_um, args.padding)
        if mesh_d is None:
            n_skipped += 1
            continue
        mesh_d = simplify(mesh_d, args.dendrite_decimate)
        color_mesh(mesh_d, golden_color(n + 1))
        dend_scene.add_geometry(mesh_d, node_name=f"dend_{n}")
        n_dend_faces += len(mesh_d.faces)

        if (n + 1) % 100 == 0:
            log(f"  Processed {n+1}/{N_neur}, {n_dend_faces:,} dendrite faces")

    path = os.path.join(args.outdir, "step4_dendrites.glb")
    dend_scene.export(path)
    sz = os.path.getsize(path) / 1024 / 1024
    manifest["steps"]["4"] = {
        "name": "Basal dendrites",
        "file": "step4_dendrites.glb",
        "faces": n_dend_faces,
        "skipped": n_skipped,
        "size_mb": round(sz, 2),
    }
    log(f"  Saved {sz:.1f} MB ({n_dend_faces:,} faces), "
        f"skipped {n_skipped}, {time.time()-t0:.1f}s")

    # ============================================================
    # Step 5: Per-parent apical dendrites
    # ============================================================
    log("\n=== Step 5: Apical dendrite meshes ===")
    t0 = time.time()

    flat_ad = neur_num_ad.ravel()
    order_ad = np.argsort(flat_ad, kind='stable')
    sorted_ad = flat_ad[order_ad]
    boundaries_ad = np.searchsorted(sorted_ad,
                                     np.arange(N_neur + 2),
                                     side='left')

    apical_scene = trimesh.Scene()
    n_ap_faces = 0
    n_skipped = 0

    for n in range(N_neur):
        start = boundaries_ad[n + 1]
        end = boundaries_ad[n + 2]
        ap_vox = order_ad[start:end]
        if len(ap_vox) < 30:
            n_skipped += 1
            continue

        mesh_a = extract_local_mesh(ap_vox.astype(np.int64), shape, voxel_um,
                                     args.padding)
        if mesh_a is None:
            n_skipped += 1
            continue
        mesh_a = simplify(mesh_a, args.dendrite_decimate)
        color_mesh(mesh_a, golden_color(n + 1))
        apical_scene.add_geometry(mesh_a, node_name=f"apical_{n}")
        n_ap_faces += len(mesh_a.faces)

    path = os.path.join(args.outdir, "step5_apical.glb")
    apical_scene.export(path)
    sz = os.path.getsize(path) / 1024 / 1024
    manifest["steps"]["5"] = {
        "name": "Apical dendrites",
        "file": "step5_apical.glb",
        "faces": n_ap_faces,
        "skipped": n_skipped,
        "size_mb": round(sz, 2),
    }
    log(f"  Saved {sz:.1f} MB ({n_ap_faces:,} faces), "
        f"skipped {n_skipped}, {time.time()-t0:.1f}s")

    # ============================================================
    # Manifest
    # ============================================================
    manifest_path = os.path.join(args.outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_sz = sum(os.path.getsize(os.path.join(args.outdir, f))
                   for f in os.listdir(args.outdir))
    log(f"\nTotal output: {total_sz/1024/1024:.1f} MB")
    log(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
