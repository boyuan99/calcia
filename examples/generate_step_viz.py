"""Generate step-by-step visualization data from Phase 1 cache pickle.

Extracts per-step data for interactive visualization in the browser.
Produces compact binary files + a JSON manifest for the viewer.

Usage:
    conda run -n calcia python examples/generate_step_viz.py
    conda run -n calcia python examples/generate_step_viz.py --cache PATH --outdir DIR

Output files (in --outdir):
    manifest.json          - metadata + file layout
    step1_vessels.bin      - Float32 (N, 3) vessel point positions
    step1_mip.bin          - Float32 (Nx, Ny) vessel XY max projection
    step2_neurons.bin      - Float32 (N_neur, 4) [x,y,z,radius] per neuron
    step3_soma_surface.bin - Float32 (N, 4) [x,y,z,neuron_id] surface voxels
    step3_slices.bin       - Uint16 (K, Nx, Ny) K component-ID slices
    step4_dendrites.bin    - Float32 (N, 4) [x,y,z,neuron_id] basal dendrites
    step5_apical.bin       - Float32 (N, 4) [x,y,z,parent_id] apical dendrites
    step6_mip.bin          - Float32 (Nx, Ny) fluorescence XY MIP
    step6_slices.bin       - Float32 (K, Nx, Ny) K fluorescence slices
    step6_bright.bin       - Float32 (N, 4) [x,y,z,intensity] bright voxels
    step7_bg_sample.bin    - Float32 (N, 3) background voxel sample
    step7_mip_final.bin    - Float32 (Nx, Ny) final fluorescence MIP
"""
import argparse
import json
import os
import pickle
import time

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def subsample_nonzero(mask, stride):
    """Return coordinates of nonzero voxels, taking every `stride` one."""
    coords = np.argwhere(mask)
    if stride > 1:
        coords = coords[::stride]
    return coords


def extract_surface_voxels(label_vol):
    """Extract voxels where at least one 6-neighbor has a different label.

    Returns boolean mask of surface voxels (same shape as label_vol).
    Label 0 (background) is never surface.
    """
    surf = np.zeros(label_vol.shape, dtype=bool)
    # Differ from +x neighbor
    diff_x = label_vol[:-1, :, :] != label_vol[1:, :, :]
    surf[:-1, :, :] |= diff_x
    surf[1:, :, :] |= diff_x
    # +y
    diff_y = label_vol[:, :-1, :] != label_vol[:, 1:, :]
    surf[:, :-1, :] |= diff_y
    surf[:, 1:, :] |= diff_y
    # +z
    diff_z = label_vol[:, :, :-1] != label_vol[:, :, 1:]
    surf[:, :, :-1] |= diff_z
    surf[:, :, 1:] |= diff_z
    # Mask out background voxels
    surf &= (label_vol > 0)
    return surf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",
                        default="examples/output/phase12_cache_250.pkl")
    parser.add_argument("--outdir",
                        default="examples/output/step_viz")
    parser.add_argument("--soma-surface-stride", type=int, default=2,
                        help="Subsample soma surface voxels by this factor")
    parser.add_argument("--dendrite-stride", type=int, default=2)
    parser.add_argument("--apical-stride", type=int, default=2)
    parser.add_argument("--vessel-stride", type=int, default=4)
    parser.add_argument("--bg-fraction", type=float, default=0.01,
                        help="Fraction of background voxels to sample")
    parser.add_argument("--n-slices", type=int, default=10,
                        help="Number of z-slices to export")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ============================================================
    # Load pickle
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
    vol_sz = vol_params.vol_sz
    voxel_um = 1.0 / vres  # µm per voxel

    neur_vol = vol_out.neur_vol       # (500, 500, 200) float32
    neur_num = vol_out.neur_num       # (500, 500, 200) uint16
    neur_num_ad = vol_out.neur_num_ad # (500, 500, 200) uint16
    neur_ves = vol_out.neur_ves       # (500, 500, 400) uint8, may be None
    shape = neur_num.shape
    Nx, Ny, Nz = shape
    log(f"  Volume: {shape}, N_neur={N_neur}, vres={vres}")

    # Determine apical/bg ID ranges
    max_id = int(neur_num.max())
    log(f"  neur_num max ID: {max_id}")

    # apical IDs: N_neur+1 .. N_neur+N_den (use neur_num_ad to count)
    apical_ids_in_ad = np.unique(neur_num_ad)
    apical_ids_in_ad = apical_ids_in_ad[apical_ids_in_ad > 0]
    log(f"  Apical dendrite parents: {len(apical_ids_in_ad)} neurons")

    manifest = {
        "vol_shape": list(shape),
        "vres": vres,
        "vol_sz_um": list(vol_sz),
        "voxel_um": voxel_um,
        "N_neur": N_neur,
        "max_id": max_id,
        "steps": {},
    }

    def write_bin(name, arr, dtype=np.float32):
        path = os.path.join(args.outdir, name)
        arr.astype(dtype).tofile(path)
        return os.path.getsize(path)

    # ============================================================
    # Step 1: Blood vessels
    # ============================================================
    log("\n=== Step 1: Blood vessels ===")
    t0 = time.time()

    if neur_ves is not None:
        # Full vessel volume (includes surface depth above imaging region)
        ves_mask = neur_ves > 0
        log(f"  Vessel voxels: {ves_mask.sum():,} / {ves_mask.size:,}")

        # Subsample for point cloud
        stride = args.vessel_stride
        ves_sub = ves_mask[::stride, ::stride, ::stride]
        coords = np.argwhere(ves_sub).astype(np.float32) * (stride * voxel_um)
        # coords are (N, 3) in µm
        sz = write_bin("step1_vessels.bin", coords)

        # XY MIP (max over z) — full resolution
        ves_mip = ves_mask.max(axis=2).astype(np.float32)
        sz_mip = write_bin("step1_mip.bin", ves_mip)

        manifest["steps"]["1"] = {
            "name": "Blood vessels",
            "files": {"points": "step1_vessels.bin",
                      "mip": "step1_mip.bin"},
            "point_count": int(len(coords)),
            "point_layout": "xyz",
            "mip_shape": list(ves_mip.shape),
            "z_shape": ves_mask.shape[2],  # may differ from imaging Nz!
            "z_range_um": [0, ves_mask.shape[2] * voxel_um],
            "color": [204, 50, 50],
        }
        log(f"  Wrote {sz/1024/1024:.1f} MB points + "
            f"{sz_mip/1024/1024:.1f} MB MIP ({time.time()-t0:.1f}s)")
    else:
        log("  No vessel data available!")

    # ============================================================
    # Step 2: Neuron positions
    # ============================================================
    log("\n=== Step 2: Neuron positions ===")
    t0 = time.time()

    locs = vol_out.locs[:N_neur]  # (N_neur, 3) in µm

    def get_soma_indices(n):
        """Return concatenated soma cytoplasm + smoothed body indices."""
        s = vol_out.gp_soma[n]
        # gp_soma[n] is (soma_indices, smoothed_body) tuple
        if isinstance(s, tuple):
            return np.concatenate([s[0], s[1]]) if len(s[1]) > 0 else s[0]
        return s

    # Compute radius per neuron from gp_soma voxel count
    radii = np.zeros(N_neur, dtype=np.float32)
    for n in range(N_neur):
        soma_vox = len(get_soma_indices(n)) if n < len(vol_out.gp_soma) else 0
        nuc_vox = len(vol_out.gp_nuc[n][0]) if n < len(vol_out.gp_nuc) else 0
        total_vox = soma_vox + nuc_vox
        if total_vox > 0:
            # Volume = N * voxel_volume, V = (4/3) π r³
            vox_vol_um3 = voxel_um ** 3
            vol_um3 = total_vox * vox_vol_um3
            radii[n] = (vol_um3 * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        else:
            radii[n] = 5.9  # fallback avg_rad

    # Pack as (N, 4) [x, y, z, radius]
    neur_data = np.concatenate([locs, radii[:, None]], axis=1).astype(np.float32)
    sz = write_bin("step2_neurons.bin", neur_data)

    manifest["steps"]["2"] = {
        "name": "Neuron placement",
        "files": {"neurons": "step2_neurons.bin"},
        "count": N_neur,
        "layout": "xyzr",
        "radius_range": [float(radii.min()), float(radii.max())],
        "radius_mean": float(radii.mean()),
    }
    log(f"  {N_neur} neurons, radius {radii.min():.2f}-{radii.max():.2f} µm "
        f"(mean {radii.mean():.2f}), wrote {sz/1024:.0f} KB ({time.time()-t0:.1f}s)")

    # ============================================================
    # Step 3: Soma voxelization
    # ============================================================
    log("\n=== Step 3: Soma voxelization ===")
    t0 = time.time()

    # Build a neur_soma volume from gp_soma + gp_nuc
    neur_soma = np.zeros(shape, dtype=np.uint16)
    flat_soma = neur_soma.ravel()
    for n in range(N_neur):
        # Soma cytoplasm (tuple of cytoplasm + smoothed body)
        idx_s = get_soma_indices(n)
        if idx_s is not None and len(idx_s) > 0:
            flat_soma[idx_s] = n + 1
        # Nucleus (overwrite with same neuron ID for visual unity)
        idx_n = vol_out.gp_nuc[n][0]
        if len(idx_n) > 0:
            flat_soma[idx_n] = n + 1

    log(f"  Built soma volume: {(neur_soma > 0).sum():,} voxels")

    # Extract surface voxels
    log("  Extracting surface voxels ...")
    surf_mask = extract_surface_voxels(neur_soma)
    log(f"  Surface voxels: {surf_mask.sum():,}")

    # Subsample
    stride = args.soma_surface_stride
    if stride > 1:
        sub_mask = np.zeros_like(surf_mask)
        sub_mask[::stride, ::stride, ::stride] = surf_mask[::stride, ::stride, ::stride]
        surf_mask = sub_mask
        log(f"  After stride {stride}: {surf_mask.sum():,} voxels")

    # Extract positions + neuron IDs
    coords = np.argwhere(surf_mask)
    ids = neur_soma[coords[:, 0], coords[:, 1], coords[:, 2]]
    points_xyz = coords.astype(np.float32) * voxel_um
    points_4 = np.concatenate([points_xyz, ids.astype(np.float32)[:, None]],
                               axis=1)
    sz = write_bin("step3_soma_surface.bin", points_4)

    # Also: nucleus surface separately (for distinct visualization)
    neur_nuc = np.zeros(shape, dtype=np.uint16)
    flat_nuc = neur_nuc.ravel()
    for n in range(N_neur):
        idx_n = vol_out.gp_nuc[n][0]
        if len(idx_n) > 0:
            flat_nuc[idx_n] = n + 1
    nuc_surf = extract_surface_voxels(neur_nuc)
    if stride > 1:
        sub = np.zeros_like(nuc_surf)
        sub[::stride, ::stride, ::stride] = nuc_surf[::stride, ::stride, ::stride]
        nuc_surf = sub
    nuc_coords = np.argwhere(nuc_surf)
    nuc_ids = neur_nuc[nuc_coords[:, 0], nuc_coords[:, 1], nuc_coords[:, 2]]
    nuc_points = np.concatenate(
        [nuc_coords.astype(np.float32) * voxel_um,
         nuc_ids.astype(np.float32)[:, None]], axis=1)
    sz_nuc = write_bin("step3_nucleus_surface.bin", nuc_points)

    # Export a few XY slices of neur_soma for 2D viewer
    slice_indices = np.linspace(Nz // 10, Nz - Nz // 10,
                                 args.n_slices).astype(int)
    slices = np.stack([neur_soma[:, :, z] for z in slice_indices], axis=0)
    sz_slice = write_bin("step3_slices.bin", slices, dtype=np.uint16)

    manifest["steps"]["3"] = {
        "name": "Soma voxelization",
        "files": {
            "soma": "step3_soma_surface.bin",
            "nucleus": "step3_nucleus_surface.bin",
            "slices": "step3_slices.bin",
        },
        "soma_point_count": int(len(points_4)),
        "nucleus_point_count": int(len(nuc_points)),
        "point_layout": "xyzi",
        "slice_indices": slice_indices.tolist(),
        "slice_shape": [Nx, Ny],
        "slice_count": len(slice_indices),
    }
    log(f"  Soma: {len(points_4):,} pts, Nucleus: {len(nuc_points):,} pts, "
        f"wrote {(sz+sz_nuc+sz_slice)/1024/1024:.1f} MB "
        f"({time.time()-t0:.1f}s)")

    # ============================================================
    # Step 4: Basal dendrites
    # ============================================================
    log("\n=== Step 4: Basal dendrites ===")
    t0 = time.time()

    # Basal = neur_num in [1, N_neur] AND not in neur_soma
    basal_mask = (neur_num >= 1) & (neur_num <= N_neur) & (neur_soma == 0)
    log(f"  Basal dendrite voxels: {basal_mask.sum():,}")

    stride = args.dendrite_stride
    if stride > 1:
        sub = np.zeros_like(basal_mask)
        sub[::stride, ::stride, ::stride] = basal_mask[::stride, ::stride, ::stride]
        basal_mask = sub

    coords = np.argwhere(basal_mask)
    ids = neur_num[coords[:, 0], coords[:, 1], coords[:, 2]]
    points = np.concatenate(
        [coords.astype(np.float32) * voxel_um,
         ids.astype(np.float32)[:, None]], axis=1)
    sz = write_bin("step4_dendrites.bin", points)

    manifest["steps"]["4"] = {
        "name": "Basal dendrites",
        "files": {"dendrites": "step4_dendrites.bin"},
        "point_count": int(len(points)),
        "point_layout": "xyzi",
    }
    log(f"  {len(points):,} points, wrote {sz/1024/1024:.1f} MB "
        f"({time.time()-t0:.1f}s)")

    # ============================================================
    # Step 5: Apical dendrites
    # ============================================================
    log("\n=== Step 5: Apical dendrites ===")
    t0 = time.time()

    apical_mask = neur_num_ad > 0
    log(f"  Apical dendrite voxels: {apical_mask.sum():,}")

    stride = args.apical_stride
    if stride > 1:
        sub = np.zeros_like(apical_mask)
        sub[::stride, ::stride, ::stride] = apical_mask[::stride, ::stride, ::stride]
        apical_mask = sub

    coords = np.argwhere(apical_mask)
    parents = neur_num_ad[coords[:, 0], coords[:, 1], coords[:, 2]]
    points = np.concatenate(
        [coords.astype(np.float32) * voxel_um,
         parents.astype(np.float32)[:, None]], axis=1)
    sz = write_bin("step5_apical.bin", points)

    # XZ MIP to show vertical extent
    xz_mip = apical_mask.max(axis=1).astype(np.float32)  # (Nx, Nz)
    sz_mip = write_bin("step5_xz_mip.bin", xz_mip)

    manifest["steps"]["5"] = {
        "name": "Apical dendrites",
        "files": {
            "points": "step5_apical.bin",
            "xz_mip": "step5_xz_mip.bin",
        },
        "point_count": int(len(points)),
        "point_layout": "xyzp",  # p = parent neuron id
        "mip_shape": list(xz_mip.shape),
    }
    log(f"  {len(points):,} points, wrote {(sz+sz_mip)/1024/1024:.1f} MB "
        f"({time.time()-t0:.1f}s)")

    # ============================================================
    # Step 6: Fluorescence distribution
    # ============================================================
    log("\n=== Step 6: Fluorescence distribution ===")
    t0 = time.time()

    # XY MIP
    mip = neur_vol.max(axis=2).astype(np.float32)
    sz_mip = write_bin("step6_mip.bin", mip)

    # Z slices
    slice_indices = np.linspace(Nz // 10, Nz - Nz // 10,
                                 args.n_slices).astype(int)
    slices = np.stack([neur_vol[:, :, z] for z in slice_indices], axis=0)
    sz_sl = write_bin("step6_slices.bin", slices.astype(np.float32))

    # Bright point cloud
    threshold = 0.8
    bright_mask = neur_vol > threshold
    log(f"  Voxels > {threshold}: {bright_mask.sum():,}")
    # Subsample
    stride = 2
    sub = np.zeros_like(bright_mask)
    sub[::stride, ::stride, ::stride] = bright_mask[::stride, ::stride, ::stride]
    bright_coords = np.argwhere(sub)
    bright_vals = neur_vol[bright_coords[:, 0], bright_coords[:, 1],
                            bright_coords[:, 2]]
    bright_pts = np.concatenate(
        [bright_coords.astype(np.float32) * voxel_um,
         bright_vals.astype(np.float32)[:, None]], axis=1)
    sz_b = write_bin("step6_bright.bin", bright_pts)

    manifest["steps"]["6"] = {
        "name": "Fluorescence distribution",
        "files": {
            "mip": "step6_mip.bin",
            "slices": "step6_slices.bin",
            "bright": "step6_bright.bin",
        },
        "mip_shape": list(mip.shape),
        "slice_shape": [Nx, Ny],
        "slice_count": len(slice_indices),
        "slice_indices": slice_indices.tolist(),
        "bright_count": int(len(bright_pts)),
        "bright_threshold": threshold,
        "point_layout": "xyzv",
        "fl_range": [float(neur_vol.min()), float(neur_vol.max())],
    }
    log(f"  MIP + {args.n_slices} slices + {len(bright_pts):,} bright pts, "
        f"wrote {(sz_mip+sz_sl+sz_b)/1024/1024:.1f} MB ({time.time()-t0:.1f}s)")

    # ============================================================
    # Step 7: Background fill
    # ============================================================
    log("\n=== Step 7: Background fill ===")
    t0 = time.time()

    # Background = neur_num > N_neur (apical + bg dendrites)
    # More specifically we want bg processes: neur_num IDs beyond apical range.
    # Simpler: all voxels not in soma nor basal
    bg_mask = (neur_num > N_neur) & (neur_num_ad == 0)
    log(f"  Background voxels: {bg_mask.sum():,}")

    # Random sample
    coords_all = np.argwhere(bg_mask)
    n_sample = max(1, int(len(coords_all) * args.bg_fraction))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(coords_all), size=n_sample, replace=False)
    sample = coords_all[sample_idx].astype(np.float32) * voxel_um
    sz = write_bin("step7_bg_sample.bin", sample)

    # Final MIP (includes everything in neur_vol)
    # Already computed in step 6 — this is same. Make separate naming:
    sz_mip = write_bin("step7_mip_final.bin", mip)

    manifest["steps"]["7"] = {
        "name": "Background fill",
        "files": {
            "sample": "step7_bg_sample.bin",
            "mip": "step7_mip_final.bin",
        },
        "sample_count": int(len(sample)),
        "sample_fraction": args.bg_fraction,
        "total_bg_voxels": int(bg_mask.sum()),
        "mip_shape": list(mip.shape),
    }
    log(f"  Sampled {len(sample):,} bg points, wrote "
        f"{(sz+sz_mip)/1024/1024:.1f} MB ({time.time()-t0:.1f}s)")

    # ============================================================
    # Write manifest
    # ============================================================
    manifest_path = os.path.join(args.outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"\nManifest written: {manifest_path}")

    # Total size report
    total_sz = sum(os.path.getsize(os.path.join(args.outdir, f))
                   for f in os.listdir(args.outdir))
    log(f"Total output size: {total_sz/1024/1024:.1f} MB")
    log(f"Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
