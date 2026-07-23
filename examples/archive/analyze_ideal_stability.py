# -*- coding: utf-8 -*-
"""Apply motion_gt BACK to a movie and measure the IDEAL stability floor.

`motion_gt.shift_applied` is the exact shift the pixels really moved by, so
undoing it is the best correction physically possible — no estimator involved.
What survives is irreducible:

  * `shift_residual` — the <=0.5-voxel rounding the renderer baked in;
  * `blur_applied`   — intra-frame streak already convolved into each frame;
  * (2P) per-row shear.

So this measures the CEILING every motion-correction algorithm is chasing.

Run:  conda run -n calcia python examples/analyze_ideal_stability.py <run_dir> [...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

from calcia.scanning import load_motion_gt

OUTROOT = os.path.join(os.path.dirname(__file__), "output")


# ------------------------------------------------------------------ measurement
def measure_shifts(mov, ref, upsample=20):
    """Per-frame (dy, dx) displacement vs `ref`, sub-pixel, in movie px."""
    out = np.zeros((2, mov.shape[0]))
    for t in range(mov.shape[0]):
        sh, _, _ = phase_cross_correlation(ref, mov[t], upsample_factor=upsample,
                                           normalization=None)
        out[:, t] = sh
    return out


def sharpness(img):
    """Gradient energy — motion smears the time-average, so this rises when the
    movie is stabilised."""
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.mean(gy ** 2 + gx ** 2))


def jitter(mov):
    """Mean |frame(t) - frame(t-1)| normalised by mean intensity."""
    d = np.abs(np.diff(mov.astype(np.float32), axis=0)).mean()
    return float(d / (mov.mean() + 1e-9))


def correct(mov, shift_applied, sfrac):
    """Undo the rendered motion.

    The scanner moves the CROP WINDOW by ``+shift`` voxels
    (``_rigid_shift_and_crop``: ``img[buf+x : H-buf+x, buf+y : W-buf+y]``), so the
    image CONTENT is displaced by ``-shift``. Undoing it therefore shifts the
    frame back by ``+shift/sfrac`` movie pixels — axis 0 is x (rows), axis 1 is y.

    With sfrac>1 the integer voxel shift lands on FRACTIONAL movie pixels, so the
    resampler matters: bilinear (order=1) is a 2-tap average at 0.5 px and
    low-passes real structure, which then masquerades as a sharpness LOSS.
    CUBIC SPLINE (order=3) keeps the detail. Verified empirically on a rescanned
    run — residual after correction: uncorrected 2.94 px, spline-3 0.31 px
    (sub-voxel floor 0.21 px). ``fourier_shift`` was tried and rejected: neither
    sign reproduced the correction (3.16 / 7.40 px).
    """
    out = np.empty_like(mov)
    for t in range(mov.shape[0]):
        dx, dy = shift_applied[0, t] / sfrac, shift_applied[1, t] / sfrac
        out[t] = ndimage.shift(mov[t], (dx, dy), order=3, mode="nearest")
    return out


# ------------------------------------------------------------------ per-run
def analyse(run_dir):
    z = np.load(os.path.join(run_dir, "movies.npz"))
    mov = z["mov_noisy"].astype(np.float32)                 # (T,H,W) realistic
    # Sharpness is measured on the CLEAN movie: bilinear resampling in `correct`
    # smooths shot noise, which would masquerade as a sharpness LOSS on the noisy
    # movie. The clean pair isolates structural alignment.
    clean = z["mov_clean"].astype(np.float32) if "mov_clean" in z.files else None
    gt = load_motion_gt(z)
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    name = os.path.basename(os.path.normpath(run_dir))

    if gt.get("legacy"):
        print(f"  [!] {name}: legacy run (no motion_gt) — skipping")
        return None

    sfrac = float(gt.get("sfrac", meta.get("config", {}).get("sfrac", 1)))
    vres = float(gt.get("vres", meta.get("vres", 1)))
    sa = np.asarray(gt["shift_applied"], float)[:2]
    sr = np.asarray(gt["shift_residual"], float)[:2]

    corr = correct(mov, sa, sfrac)
    clean_c = correct(clean, sa, sfrac) if clean is not None else None

    # structural sharpness on the clean pair (no shot-noise confound), with the
    # FOV BORDER CROPPED: the raw movie has a hard dark crop edge whose gradient
    # dwarfs the tissue (~17x the interior) and the shift's edge-fill smooths it,
    # so an uncropped metric reports the correction as a sharpness LOSS. Margin =
    # the largest applied shift, so no replicated-edge band survives.
    sharp_src = (clean, clean_c) if clean is not None else (mov, corr)
    k = int(np.ceil(np.abs(sa).max() / sfrac)) + 4
    ref_raw = sharp_src[0].mean(0)[k:-k, k:-k]
    ref_cor = sharp_src[1].mean(0)[k:-k, k:-k]
    res_crop = k
    # measure on the SAME cropped region: the FOV edge biases phase correlation
    mov_k, corr_k = mov[:, k:-k, k:-k], corr[:, k:-k, k:-k]
    est_before = measure_shifts(mov_k, mov_k[0])
    est_after = measure_shifts(corr_k, corr_k[0])

    def rms(a):
        return float(np.sqrt((a ** 2).sum(0).mean()))

    px2um = sfrac / vres                                     # movie px -> um
    res = dict(
        run=name, seed=meta.get("seed"), sfrac=sfrac, vres=vres, T=int(mov.shape[0]),
        # measured residual displacement (movie px)
        est_rms_before_px=rms(est_before), est_rms_after_px=rms(est_after),
        est_max_before_px=float(np.abs(est_before).max()),
        est_max_after_px=float(np.abs(est_after).max()),
        # irreducible floor written by the renderer
        subvox_rms_vox=rms(sr), subvox_max_vox=float(np.abs(sr).max()),
        subvox_rms_px=rms(sr) / sfrac,
        # image-quality proxies
        sharp_before=sharpness(ref_raw), sharp_after=sharpness(ref_cor),
        jitter_before=jitter(mov), jitter_after=jitter(corr),
        px2um=px2um,
    )
    res["sharp_gain"] = res["sharp_after"] / (res["sharp_before"] + 1e-12)
    res["jitter_drop"] = res["jitter_before"] / (res["jitter_after"] + 1e-12)
    if "blur_applied" in gt:
        ba = np.asarray(gt["blur_applied"], float)
        res["blur_mean_px"] = float(np.linalg.norm(ba, axis=0).mean() / sfrac)
        res["blur_max_px"] = float(np.linalg.norm(ba, axis=0).max() / sfrac)
    # ref_raw/ref_cor are the CROPPED CLEAN means the sharpness metric uses, so
    # the figure shows exactly what the numbers measure.
    return res, mov, corr, est_before, est_after, sr, sfrac, ref_raw, ref_cor, res_crop


# ------------------------------------------------------------------ figure
def figure(items, out_png):
    n = len(items)
    fig, ax = plt.subplots(n, 4, figsize=(17, 3.7 * n), squeeze=False)
    for r, (res, mov, corr, eb, ea, sr, sfrac, mb, mc, k) in enumerate(items):
        vmin, vmax = np.percentile(mb, [1, 99.5])
        ax[r, 0].imshow(mb, cmap="gray", vmin=vmin, vmax=vmax)
        ax[r, 0].set_title(f"seed {res['seed']} — raw mean\n(sharpness {res['sharp_before']:.3g})",
                           fontsize=9)
        ax[r, 1].imshow(mc, cmap="gray", vmin=vmin, vmax=vmax)
        ax[r, 1].set_title(f"GT-corrected mean\n({res['sharp_after']:.3g},  "
                           f"x{res['sharp_gain']:.2f} sharper)", fontsize=9)
        sb = mov.std(0)[k:-k, k:-k]
        sc = corr.std(0)[k:-k, k:-k]
        vs = np.percentile(sb, 99.5)
        ax[r, 2].imshow(np.concatenate([sb, sc], axis=1), cmap="inferno", vmin=0, vmax=vs)
        ax[r, 2].set_title("temporal std   raw | GT-corrected", fontsize=9)
        for a in ax[r, :3]:
            a.set_xticks([]); a.set_yticks([])
        t = np.arange(eb.shape[1])
        ax[r, 3].plot(t, np.linalg.norm(eb, axis=0), lw=1.1, color="#c0392b",
                      label=f"raw  rms {res['est_rms_before_px']:.2f} px")
        ax[r, 3].plot(t, np.linalg.norm(ea, axis=0), lw=1.1, color="#2f7d5b",
                      label=f"GT-corrected  rms {res['est_rms_after_px']:.2f} px")
        ax[r, 3].axhline(res["subvox_rms_px"], ls="--", lw=1, color="#3d6b8e",
                         label=f"sub-voxel floor {res['subvox_rms_px']:.2f} px")
        ax[r, 3].set_title("measured displacement vs frame 0", fontsize=9)
        ax[r, 3].set_xlabel("frame"); ax[r, 3].set_ylabel("|shift| (movie px)")
        ax[r, 3].legend(fontsize=7)
    fig.suptitle("Ideal stability — motion_gt applied back to the movie "
                 "(the ceiling any motion-correction algorithm chases)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=115); plt.close(fig)


def main():
    runs = sys.argv[1:]
    if not runs:
        print("usage: analyze_ideal_stability.py <run_dir> [...]"); return
    items, rows = [], []
    for rd in runs:
        print(f"[+] {rd}")
        got = analyse(rd)
        if got:
            items.append(got); rows.append(got[0])
    if not items:
        print("no runs with motion_gt"); return

    out_dir = os.path.join(OUTROOT, "ideal_stability")
    os.makedirs(out_dir, exist_ok=True)
    figure(items, os.path.join(out_dir, "ideal_stability.png"))
    with open(os.path.join(out_dir, "ideal_stability.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== IDEAL STABILITY (motion_gt applied back) ===")
    for r in rows:
        print(f"  {r['run'][:34]:34s} seed={r['seed']}")
        print(f"    displacement rms : {r['est_rms_before_px']:.2f} px -> "
              f"{r['est_rms_after_px']:.2f} px   ({r['est_rms_after_px']*r['px2um']:.2f} um)")
        print(f"    sub-voxel floor  : {r['subvox_rms_px']:.3f} px "
              f"({r['subvox_rms_vox']:.3f} vox, max {r['subvox_max_vox']:.2f})")
        print(f"    mean sharpness   : x{r['sharp_gain']:.2f}   jitter /{r['jitter_drop']:.2f}")
        if "blur_mean_px" in r:
            print(f"    intra-frame blur : mean {r['blur_mean_px']:.2f} px, "
                  f"max {r['blur_max_px']:.2f} px (cannot be undone by shifting)")
    print(f"\nwrote {out_dir}\\ideal_stability.png / .json")


if __name__ == "__main__":
    main()
