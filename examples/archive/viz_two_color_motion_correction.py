"""Two-colour correspondence + motion-correction visualisation.

Companion / diagnostic script (core simulator untouched). Takes the co-registered
GCaMP (green, dynamic) and tdTomato (red, static) KEEPER pair and shows how the
two channels correspond spatially, with emphasis on the **motion-correction
problem**:

  * each channel was acquired with its OWN independent physio motion trajectory
    (motion_shared_with_gcamp = false), so a naive overlay of the raw movies is
    mis-registered and jitters frame-to-frame;
  * the ground-truth per-frame shift (`mot_hist`, in full-res voxels) lets us
    register each channel back to a common reference; after correction the two
    channels co-localise (expressing somata are green+red = yellow);
  * because the two trajectories differ, using tdT as a fiducial to correct GCaMP
    (as one would for SIMULTANEOUS two-colour) leaves a large residual — quantified.

Outputs (into examples/output/two_color_mc_viz_<ts>/):
  * motion_correction_overlay.png  — multipanel static figure
  * two_color_mc.gif               — raw-overlay vs motion-corrected-overlay movie
  * trace_correspondence.png       — per-neuron GCaMP dF/F with tdT identity
  * summary.txt                    — quantitative numbers
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import shift as nd_shift

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / "output" / "_archive"
G_DIR = ARCH / "BEST_gcamp_2scale_h28_w0.8_1000um_KEEPER"
T_DIR = ARCH / "BEST_tdt_2scale_h28_w0.8_1000um_KEEPER"
SFRAC = 2  # movie was downsampled by this AFTER motion was applied


# ---------------------------------------------------------------------------
def _pct_norm(img, lo=1.0, hi=99.5):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-9), 0, 1)


def motion_correct(mov, mot_hist, sfrac=SFRAC):
    """Register every frame back to the zero-shift reference.

    The scan cropped ``src[buf + x_shift : ..., buf + y_shift : ...]`` at FULL
    resolution, then downsampled by ``sfrac``. The applied shift is
    ``round(mot_hist)`` full-res voxels, i.e. ``round(mot_hist)/sfrac`` movie
    pixels. Undoing it: ``ndimage.shift(frame, +applied)``.
    """
    out = np.empty_like(mov)
    applied = np.round(mot_hist[:2]) / sfrac  # (2, Nt) in movie px, rows=x, cols=y
    for k in range(mov.shape[0]):
        out[k] = nd_shift(mov[k], (applied[0, k], applied[1, k]),
                          order=1, mode="nearest")
    return out, applied


def to_rgb(green, red):
    """Compose a two-colour RGB frame (R=tdT, G=GCaMP, B=0)."""
    rgb = np.zeros(green.shape + (3,), dtype=np.float32)
    rgb[..., 0] = red
    rgb[..., 1] = green
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120,
                    help="number of frames in the GIF")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--crop", type=int, default=0,
                    help="border px to trim from the correction ROI for display")
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / f"two_color_mc_viz_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] output -> {out_dir}")

    print("[+] loading movies ...")
    zg = np.load(G_DIR / "movies.npz")
    zt = np.load(T_DIR / "movies.npz")
    g = zg["mov_noisy"].astype(np.float32)   # (Nt, H, W)
    t = zt["mov_noisy"].astype(np.float32)
    # motion_gt is the DEFAULT motion artifact: `shift_applied` is the shift the
    # pixels REALLY moved by (no need to re-derive it by rounding mot_hist), and
    # `shift_residual` is the sub-voxel error no registration can undo. Falls back
    # to mot_hist for runs generated before motion_gt existed.
    from calcia.scanning import load_motion_gt
    gt_g, gt_t = load_motion_gt(zg), load_motion_gt(zt)
    mg = gt_g["shift_applied"]                # (2|3, Nt)  full-res voxels
    mt = gt_t["shift_applied"]
    if gt_g.get("legacy") or gt_t.get("legacy"):
        print("    [!] legacy run (no motion_gt): sub-voxel residual unavailable")
    else:
        print(f"    sub-voxel residual (gcamp) |max|="
              f"{np.abs(gt_g['shift_residual']).max():.3f} vox — irreducible")
    Nt = g.shape[0]
    print(f"    gcamp {g.shape}  tdt {t.shape}  Nt={Nt}")

    # --- motion correction (GT-shift based) ---
    print("[+] motion-correcting each channel with its own GT trajectory ...")
    g_mc, g_app = motion_correct(g, mg)
    t_mc, t_app = motion_correct(t, mt)

    # keep an interior ROI so edge roll-in doesn't distort metrics/overlay
    b = max(args.crop, int(np.ceil(np.abs(np.concatenate([g_app, t_app])).max())) + 1)
    sl = (slice(b, -b), slice(b, -b))

    # --- mean images (structure) ---
    g_raw_mean = g.mean(0)
    t_raw_mean = t.mean(0)
    g_mc_mean = g_mc.mean(0)
    t_mc_mean = t_mc.mean(0)

    # ============================ Figure 1: overlay ============================
    print("[+] building overlay figure ...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.15)

    def _show(ax, img, title, cmap="magma"):
        ax.imshow(_pct_norm(img[sl]), cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 0])
    _show(ax, g_raw_mean, "GCaMP raw mean\n(motion-smeared)", "Greens_r")
    ax = fig.add_subplot(gs[0, 1])
    _show(ax, t_raw_mean, "tdT raw mean\n(motion-smeared)", "Reds_r")

    ax = fig.add_subplot(gs[0, 2])
    rgb_raw = to_rgb(_pct_norm(g_raw_mean[sl]), _pct_norm(t_raw_mean[sl]))
    ax.imshow(rgb_raw)
    ax.set_title("RAW overlay  (green=GCaMP, red=tdT)\nindependent motion ->"
                 " mis-registered", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    _show(ax, g_mc_mean, "GCaMP motion-corrected mean", "Greens_r")
    ax = fig.add_subplot(gs[1, 1])
    _show(ax, t_mc_mean, "tdT motion-corrected mean", "Reds_r")

    ax = fig.add_subplot(gs[1, 2])
    rgb_mc = to_rgb(_pct_norm(g_mc_mean[sl]), _pct_norm(t_mc_mean[sl]))
    ax.imshow(rgb_mc)
    ax.set_title("CORRECTED overlay\nco-registered (tdT+ somata = yellow)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Two-colour correspondence via per-channel motion correction",
                 fontsize=14, y=0.97)
    fig.savefig(out_dir / "motion_correction_overlay.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ============================ Figure 2: motion ============================
    print("[+] building motion-trajectory figure ...")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    tt = np.arange(Nt)
    axes[0].plot(tt, mg[0], color="tab:green", label="GCaMP x")
    axes[0].plot(tt, mt[0], color="tab:red", label="tdT x")
    axes[0].set_ylabel("x shift (voxels)")
    axes[0].set_title("Independent per-channel motion (row / x)")
    axes[0].legend(loc="upper right")

    axes[1].plot(tt, mg[1], color="tab:green", label="GCaMP y")
    axes[1].plot(tt, mt[1], color="tab:red", label="tdT y")
    axes[1].set_ylabel("y shift (voxels)")
    axes[1].set_title("Independent per-channel motion (col / y)")
    axes[1].legend(loc="upper right")

    # residual if tdT were (wrongly) used as fiducial for GCaMP
    res_x = mg[0] - mt[0]
    res_y = mg[1] - mt[1]
    axes[2].plot(tt, res_x, color="k", label="residual x")
    axes[2].plot(tt, res_y, color="gray", label="residual y")
    axes[2].set_ylabel("residual (voxels)")
    axes[2].set_xlabel("frame")
    axes[2].set_title("Residual if tdT fiducial applied to GCaMP "
                      "(non-zero -> per-channel MC required)")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "motion_trajectories.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ===================== Figure 3: trace correspondence =====================
    print("[+] building trace-correspondence figure ...")
    tr = np.load(G_DIR / "traces.npz", allow_pickle=True)
    expr = np.load(T_DIR / "tdtomato_expression.npz", allow_pickle=True)
    soma_neurons = tr["soma_neurons"]          # (Nsoma, Nt) calcium
    expr_mask = expr["expr_mask"]              # (Nsoma,) which express tdT
    # rows align: verify
    aligned = np.allclose(expr["soma_locs"], tr["soma_locs"])
    print(f"    soma rows aligned across channels: {aligned}")

    # pick a few EXPRESSING, ACTIVE neurons
    def _dff(x):
        f0 = np.percentile(x, 10)
        return (x - f0) / max(f0, 1e-6)

    expr_idx = np.where(expr_mask)[0]
    activity = soma_neurons[expr_idx].std(1)
    order = expr_idx[np.argsort(activity)[::-1]]
    pick = order[:6]

    fig, ax = plt.subplots(figsize=(11, 7))
    dt = 0.05
    tsec = np.arange(Nt) * dt
    STEP = 1.5
    for i, nid in enumerate(pick):
        d = _dff(soma_neurons[nid])
        d = d / max(d.max(), 1e-6)          # normalise each trace to [0, 1]
        ax.plot(tsec, d + i * STEP, color="tab:green", lw=1.2)
        ax.axhline(i * STEP, color="tab:red", lw=3.0, alpha=0.4)
        peak = _dff(soma_neurons[nid]).max()
        ax.text(tsec[-1] * 1.005, i * STEP, f" n{nid}  tdT+  (dF/F {peak:.1f})",
                va="center", color="tab:red", fontsize=9)
    ax.set_yticks([i * STEP for i in range(len(pick))])
    ax.set_yticklabels([f"n{nid}" for nid in pick])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised GCaMP dF/F  (stacked per neuron)")
    ax.set_title("Signal correspondence: GCaMP (green, dynamic) vs tdT "
                 "(red bar = static structural identity)\n"
                 "same co-registered neuron, two channels")
    ax.set_xlim(0, tsec[-1] * 1.18)
    fig.tight_layout()
    fig.savefig(out_dir / "trace_correspondence.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # ============================ GIF: raw vs MC ============================
    nfr = min(args.frames, Nt)
    print(f"[+] rendering GIF ({nfr} frames) ...")
    # fixed display normalisation from the corrected means (stable scaling)
    def _sc(mov, ref):
        a, bb = np.percentile(ref[sl], [1, 99.5])
        return np.clip((mov - a) / max(bb - a, 1e-9), 0, 1)

    ds = 2  # display downsample to keep the GIF small
    frames = []
    for k in range(nfr):
        raw = to_rgb(_sc(g[k][sl], g_raw_mean), _sc(t[k][sl], t_raw_mean))
        mc = to_rgb(_sc(g_mc[k][sl], g_mc_mean), _sc(t_mc[k][sl], t_mc_mean))
        gap = np.ones((raw.shape[0], 6, 3), dtype=np.float32)
        panel = np.concatenate([raw, gap, mc], axis=1)
        panel = panel[::ds, ::ds]
        frames.append((panel * 255).astype(np.uint8))
    imageio.mimsave(out_dir / "two_color_mc.gif", frames, fps=args.fps, loop=0)
    # single-frame preview (viewable inline; left=RAW overlay, right=CORRECTED)
    imageio.imwrite(out_dir / "two_color_mc_frame0.png", frames[len(frames) // 2])

    # add a simple header note file describing panels
    # ============================ Quant summary ============================
    def _sharp(img):
        gy, gx = np.gradient(img[sl].astype(np.float64))
        return float(np.sqrt(gx**2 + gy**2).mean())

    lines = []
    lines.append("Two-colour motion-correction summary")
    lines.append("=" * 42)
    lines.append(f"frames: {Nt}   movie: {g.shape[1]}x{g.shape[2]} px")
    lines.append("")
    lines.append("Per-channel GT motion (full-res voxels):")
    lines.append(f"  GCaMP  x std {mg[0].std():5.2f}  y std {mg[1].std():5.2f}"
                 f"   |max| {np.abs(mg[:2]).max():5.1f}")
    lines.append(f"  tdT    x std {mt[0].std():5.2f}  y std {mt[1].std():5.2f}"
                 f"   |max| {np.abs(mt[:2]).max():5.1f}")
    lines.append("")
    lines.append("tdT-as-fiducial-for-GCaMP residual (why per-channel MC needed):")
    lines.append(f"  residual x std {res_x.std():5.2f}  y std {res_y.std():5.2f}"
                 f"   |max| {max(np.abs(res_x).max(), np.abs(res_y).max()):5.1f} vox")
    lines.append("")
    lines.append("Mean-image sharpness (|grad|, higher = crisper):")
    lines.append(f"  GCaMP raw {_sharp(g_raw_mean):.4f} -> MC {_sharp(g_mc_mean):.4f}"
                 f"  (+{100*(_sharp(g_mc_mean)/_sharp(g_raw_mean)-1):.0f}%)")
    lines.append(f"  tdT   raw {_sharp(t_raw_mean):.4f} -> MC {_sharp(t_mc_mean):.4f}"
                 f"  (+{100*(_sharp(t_mc_mean)/_sharp(t_raw_mean)-1):.0f}%)")
    lines.append("")
    lines.append(f"expressing (tdT+) somata: {int(expr_mask.sum())} / {len(expr_mask)}")
    lines.append(f"soma rows aligned across channels: {aligned}")
    txt = "\n".join(lines)
    (out_dir / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n[✓] done -> {out_dir}")


if __name__ == "__main__":
    main()
