"""Verify that the end-to-end two-colour series produced GENUINELY DIFFERENT
volumes AND videos — not the same geometry re-scanned.

For each volume (a ``deepthinves_s{seed}_..._flat_stub`` dir carrying a
``volume_stats.npz`` sidecar, plus the legacy ``deepthinves_500_flat_stub``) it
measures:

  GEOMETRY (volume-level, seed-driven — SHOULD differ):
    * soma count            (N_neur / n_soma)
    * soma depth profile    (z-histogram)             -> pairwise correlation
    * soma lateral layout   (coarse XY density map)   -> pairwise correlation
    * vessel layout         (XY vessel projection)    -> pairwise correlation
    * vessel depth profile  (vessel voxels per z)     -> pairwise correlation

  VIDEO (from each volume's GCaMP + tdt run dirs — SHOULD differ):
    * GCaMP mean image      -> pairwise correlation
    * GCaMP std-over-time   -> pairwise correlation (activity footprint)
    * population activity    (mean soma trace)        -> pairwise correlation
    * dF/F p99, total spikes, brightness

Low OFF-DIAGONAL correlation == the volumes/videos are as different as possible.
Any off-diagonal > SIMILAR_FLAG (0.5) is flagged as suspiciously similar (likely
an accidentally reused volume). Emits a printed report, ``diversity_report.json``
and a ``diversity_report.png`` figure under examples/output/.

Run:
    conda run -n calcia python examples/verify_volume_diversity.py
    conda run -n calcia python examples/verify_volume_diversity.py --stubs A B C
"""
import argparse
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "output")
SIMILAR_FLAG = 0.5          # off-diagonal corr above this => suspiciously similar
GRID = 64                   # coarse grid for lateral-map correlations


# ----------------------------------------------------------------------------- helpers
def _corr(a, b):
    """Pearson correlation of two flattened, mean-subtracted arrays."""
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def _corr_matrix(vecs):
    n = len(vecs)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = _corr(vecs[i], vecs[j])
    return M


def _downsample(img, g=GRID):
    """Area-average an image down to g x g (scale-invariant map for correlation)."""
    from scipy.ndimage import zoom
    img = np.asarray(img, np.float64)
    zf = (g / img.shape[0], g / img.shape[1])
    return zoom(img, zf, order=1)


def _xy_density(soma_locs, vol_um, g=GRID):
    x = soma_locs[:, 0]; y = soma_locs[:, 1]
    H, _, _ = np.histogram2d(x, y, bins=g, range=[[0, vol_um], [0, vol_um]])
    return H


def _depth_hist(soma_locs, depth_um, nb=30):
    z = soma_locs[:, 2]
    h, _ = np.histogram(z, bins=nb, range=[0, depth_um])
    return h.astype(np.float64)


def _off_diag(M):
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)] if n > 1 else np.array([0.0])


# ----------------------------------------------------------------------------- load
def volume_stats(stub_dir):
    """Return the volume_stats dict, computing+caching the sidecar from the Phase-1
    pickle if the stub predates the sidecar (e.g. legacy seed-42 stub)."""
    sc = os.path.join(stub_dir, "volume_stats.npz")
    if os.path.exists(sc):
        d = np.load(sc, allow_pickle=True)
        return {k: d[k] for k in d.files}
    # fall back: build from the pickle (heavy, one-off) then cache
    meta = json.load(open(os.path.join(stub_dir, "metadata.json")))
    import pickle
    with open(meta["phase1_cache"], "rb") as f:
        vol_out, vp = pickle.load(f)
    ves = np.asarray(vol_out.neur_ves); ves_bin = ves > 0
    soma_ids = np.array([i for i, c in enumerate(vol_out.gp_vals)
                         if np.any(np.asarray(c.soma_mask))], dtype=np.int64)
    soma_locs = np.asarray(vol_out.locs)[soma_ids].astype(np.float32)
    vol_sz = list(meta["vol_sz"])
    d = dict(seed=meta["seed"], N_neur=int(vp.N_neur), n_soma=len(soma_ids),
             soma_locs=soma_locs, vol_sz=np.array(vol_sz),
             vessel_voxels=int(ves_bin.sum()),
             vessel_frac=float(ves_bin.sum() / ves.size),
             vessel_proj=ves_bin.any(2).astype(np.uint8),
             vessel_depth=ves_bin.reshape(-1, ves.shape[2]).sum(0).astype(np.int64))
    np.savez_compressed(sc, **d)
    return d


def find_run(stub_name, kind):
    """Latest run dir whose metadata.matched_run == stub_name and kind matches."""
    pat = ("gcamp_realistic_*" if kind == "gcamp" else "striatum_tdt_static_*")
    hits = []
    for d in glob.glob(os.path.join(OUT, pat)):
        mj = os.path.join(d, "metadata.json")
        if not os.path.exists(mj):
            continue
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        if m.get("matched_run") == stub_name and m.get("nt", 0) >= 200:
            hits.append((os.path.getmtime(d), d, m))
    if not hits:
        return None, None
    _, d, m = max(hits)
    return d, m


def video_stats(run_dir):
    mv = np.load(os.path.join(run_dir, "movies.npz"))
    mov = mv["mov_noisy"]                       # (T,H,W)
    mean_img = mov.mean(0); std_img = mov.std(0)
    out = dict(mean_img=mean_img, std_img=std_img,
               median=float(np.median(mean_img)),
               pop_trace=None, total_spikes=None)
    tp = os.path.join(run_dir, "traces.npz")
    if os.path.exists(tp):
        tr = np.load(tp, allow_pickle=True)
        if "soma" in tr:
            out["pop_trace"] = np.asarray(tr["soma"], np.float64).mean(0)
        if "spikes" in tr:
            out["total_spikes"] = int(np.asarray(tr["spikes"]).sum())
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stubs", nargs="*", default=None,
                    help="Explicit stub dir names. Default: auto-discover deep "
                         "thin-vessel flat stubs under examples/output/.")
    args = ap.parse_args()

    # Organised tree: two_color_series/volumes/seedNNNN/{stub,gcamp,tdt}
    vol_dirs = sorted(glob.glob(os.path.join(OUT, "two_color_series", "volumes", "seed*")))
    if len(vol_dirs) < 2:
        print(f"need >=2 volumes to compare diversity; found {len(vol_dirs)}")
        return

    print(f"=== volume diversity over {len(vol_dirs)} volumes ===")
    V = []
    for vd in vol_dirs:
        sd = os.path.join(vd, "stub")
        if not (os.path.exists(os.path.join(sd, "volume_stats.npz")) or
                os.path.exists(os.path.join(sd, "metadata.json"))):
            continue
        s = os.path.basename(vd)
        vs = volume_stats(sd)
        vol_um = float(np.asarray(vs["vol_sz"])[0]); depth = float(np.asarray(vs["vol_sz"])[2])
        gdir, tdir = os.path.join(vd, "gcamp"), os.path.join(vd, "tdt")
        gd = gdir if os.path.exists(os.path.join(gdir, "movies.npz")) else None
        gm = json.load(open(os.path.join(gdir, "metadata.json"))) if gd else None
        td = tdir if os.path.exists(os.path.join(tdir, "movies.npz")) else None
        V.append(dict(stub=s, seed=int(vs["seed"]), vol_um=vol_um, depth=depth,
                      N_neur=int(vs["N_neur"]), n_soma=int(vs["n_soma"]),
                      vessel_frac=float(vs["vessel_frac"]),
                      soma_locs=np.asarray(vs["soma_locs"]),
                      vessel_proj=np.asarray(vs["vessel_proj"]),
                      vessel_depth=np.asarray(vs["vessel_depth"], np.float64),
                      gcamp_dir=gd, gcamp_meta=gm, tdt_dir=td))
        print(f"  {s}: seed={V[-1]['seed']} N_neur={V[-1]['N_neur']} "
              f"n_soma={V[-1]['n_soma']} vessel_frac={V[-1]['vessel_frac']:.4f} "
              f"gcamp={'Y' if gd else '-'} tdt={'Y' if td else '-'}")

    # ---- geometry correlations ----
    depth_hists = [_depth_hist(v["soma_locs"], v["depth"]) for v in V]
    xy_dens = [_downsample(_xy_density(v["soma_locs"], v["vol_um"]), GRID) for v in V]
    ves_maps = [_downsample(v["vessel_proj"].astype(float), GRID) for v in V]
    ves_depth = [v["vessel_depth"] for v in V]
    G = dict(soma_depth=_corr_matrix(depth_hists),
             soma_xy=_corr_matrix(xy_dens),
             vessel_xy=_corr_matrix(ves_maps),
             vessel_depth=_corr_matrix(ves_depth))

    # ---- video correlations (only volumes with a gcamp run) ----
    gi = [k for k, v in enumerate(V) if v["gcamp_dir"]]
    vid = None
    if len(gi) >= 2:
        vids = [video_stats(V[k]["gcamp_dir"]) for k in gi]
        mean_maps = [_downsample(x["mean_img"], GRID) for x in vids]
        std_maps = [_downsample(x["std_img"], GRID) for x in vids]
        pop = [x["pop_trace"] for x in vids]
        vid = dict(idx=gi,
                   gcamp_mean=_corr_matrix(mean_maps),
                   gcamp_std=_corr_matrix(std_maps),
                   pop_activity=_corr_matrix(pop) if all(p is not None for p in pop) else None,
                   dff=[V[k]["gcamp_meta"].get("dff_p99") for k in gi],
                   spikes=[x["total_spikes"] for x in vids],
                   median=[x["median"] for x in vids])

    # ---- report ----
    def _summ(name, M):
        od = _off_diag(M)
        flag = "  <== SIMILAR!" if np.max(np.abs(od)) > SIMILAR_FLAG else ""
        print(f"  {name:16s} off-diag |r|: mean={np.mean(np.abs(od)):.3f} "
              f"max={np.max(np.abs(od)):.3f}{flag}")
        return dict(mean_abs=float(np.mean(np.abs(od))),
                    max_abs=float(np.max(np.abs(od))), matrix=M.tolist())

    # A shared 1-D DEPTH MARGINAL (soma z-histogram, vessel voxels-per-z) is a
    # property of the REGION's laminar density profile, not the seed: two
    # independent volumes of the same region always have correlated depth
    # marginals. So these are reported but EXCLUDED from the reuse verdict — the
    # discriminative "is this the same volume?" signals are the 2-D LATERAL
    # layouts (soma XY, vessel XY), which collapse to |r|~1 only under true reuse.
    SHARED_PHYSICS = {"soma_depth", "vessel_depth"}
    print("\n--- GEOMETRIC diversity (lower |r| = more different) ---")
    report = {"stubs": [v["stub"] for v in V], "seeds": [v["seed"] for v in V],
              "N_neur": [v["N_neur"] for v in V], "n_soma": [v["n_soma"] for v in V],
              "vessel_frac": [v["vessel_frac"] for v in V], "geometry": {}, "video": {}}
    for k, M in G.items():
        tag = "  (shared region physics; expected)" if k in SHARED_PHYSICS else ""
        r = _summ(k, M)
        r["discriminative"] = k not in SHARED_PHYSICS
        report["geometry"][k] = r
        if tag:
            print(f"      ^ {k}{tag}")
    dn = [v["n_soma"] for v in V]
    print(f"  n_soma range: {min(dn)}..{max(dn)} (spread {max(dn)-min(dn)})")

    if vid:
        print("\n--- VIDEO diversity (GCaMP, lower |r| = more different) ---")
        for k in ("gcamp_mean", "gcamp_std", "pop_activity"):
            if vid[k] is not None:
                report["video"][k] = _summ(k, np.asarray(vid[k]))
        print(f"  dff_p99: {[round(x,3) if x else None for x in vid['dff']]}")
        print(f"  spikes : {vid['spikes']}")
        report["video"]["dff_p99"] = vid["dff"]
        report["video"]["total_spikes"] = vid["spikes"]

    # verdict over the DISCRIMINATIVE axes only (lateral geometry + video); the
    # shared depth marginals are excluded (see SHARED_PHYSICS note above).
    all_od = []
    for k, M in G.items():
        if k not in SHARED_PHYSICS:
            all_od += list(np.abs(_off_diag(M)))
    if vid:
        for k in ("gcamp_mean", "gcamp_std", "pop_activity"):
            if vid[k] is not None:
                all_od += list(np.abs(_off_diag(np.asarray(vid[k]))))
    worst = max(all_od) if all_od else 0.0
    depth_od = max(list(np.abs(_off_diag(G["soma_depth"]))) +
                   list(np.abs(_off_diag(G["vessel_depth"]))))
    verdict = ("DIVERSE — all discriminative (lateral+video) off-diagonal |r| < %.2f"
               % SIMILAR_FLAG) if worst < SIMILAR_FLAG \
        else "WARNING: discriminative pairs similar (max |r|=%.3f)" % worst
    report["worst_offdiag_abs"] = float(worst)
    report["depth_marginal_offdiag_abs"] = float(depth_od)
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    print(f"  (shared depth-marginal |r| up to {depth_od:.3f} = same-region physics, "
          f"expected — not a reuse signal)")

    with open(os.path.join(OUT, "two_color_series", "reports", "diversity_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote " + os.path.join(OUT, "two_color_series", "reports", "diversity_report.json"))

    _figure(V, G, vid, xy_dens, ves_maps)


def _figure(V, G, vid, xy_dens, ves_maps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skip figure: {e})")
        return
    n = len(V)
    nrow = 4
    fig, ax = plt.subplots(nrow, max(n, 4), figsize=(3 * max(n, 4), 11))
    if ax.ndim == 1:
        ax = ax[None, :]
    for j, v in enumerate(V):
        ax[0, j].imshow(xy_dens[j], cmap="magma"); ax[0, j].set_title(
            f"seed {v['seed']}\nsoma XY  n={v['n_soma']}", fontsize=9)
        ax[1, j].imshow(ves_maps[j], cmap="bone"); ax[1, j].set_title(
            f"vessel proj  {100*v['vessel_frac']:.2f}%", fontsize=9)
        if v["gcamp_dir"]:
            mi = video_stats(v["gcamp_dir"])["mean_img"]
            ax[2, j].imshow(mi, cmap="gray"); ax[2, j].set_title("GCaMP mean", fontsize=9)
        else:
            ax[2, j].axis("off")
    for a in ax[:3].ravel():
        a.set_xticks([]); a.set_yticks([])
    for j in range(n, ax.shape[1]):
        for r in range(3):
            ax[r, j].axis("off")
    # correlation heatmaps on the bottom row
    mats = [("soma XY", G["soma_xy"]), ("vessel XY", G["vessel_xy"])]
    if vid:
        mats.append(("GCaMP mean", np.asarray(vid["gcamp_mean"])))
        if vid["pop_activity"] is not None:
            mats.append(("pop activity", np.asarray(vid["pop_activity"])))
    for j in range(ax.shape[1]):
        if j < len(mats):
            name, M = mats[j]
            im = ax[3, j].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
            ax[3, j].set_title(f"{name} corr", fontsize=9)
            ax[3, j].set_xticks(range(M.shape[0])); ax[3, j].set_yticks(range(M.shape[0]))
            for p in range(M.shape[0]):
                for q in range(M.shape[0]):
                    ax[3, j].text(q, p, f"{M[p,q]:.2f}", ha="center", va="center",
                                  fontsize=7, color="k")
        else:
            ax[3, j].axis("off")
    fig.suptitle("Two-colour series — volume & video diversity "
                 "(off-diagonal near 0 = maximally different)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "two_color_series", "reports", "diversity_report.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
