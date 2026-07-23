# -*- coding: utf-8 -*-
"""Compare real motion-correction algorithms against the motion_gt ceiling.

Four streams per volume:
    RAW                  no correction
    motion_gt            perfect ground-truth shift undone  -> the CEILING
    CaImAn rigid         pw_rigid=False
    CaImAn non-rigid     pw_rigid=True

Emits metrics (residual displacement, mean-image sharpness, jitter), a static
figure and a 4-way animated GIF.

Run:  conda run -n calcia python examples/compare_mc_methods.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import tifffile
from scipy import ndimage
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

from calcia.scanning import load_motion_gt

OUT = os.path.join(os.path.dirname(__file__), "output", "ideal_stability")
MCD = os.path.join(OUT, "mc")
TSTEP = 3
FPS = 14
PANEL = 122
LABEL_H = 17
GAP = 5
METHODS = ["RAW", "mot_hist (legacy)", "motion_gt", "CaImAn rigid", "CaImAn non-rigid"]


def _font(sz=12):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def gt_correct(mov, sa, sfrac):
    out = np.empty_like(mov)
    for t in range(mov.shape[0]):
        out[t] = ndimage.shift(mov[t], (sa[0, t] / sfrac, sa[1, t] / sfrac),
                               order=3, mode="nearest")
    return out


def resid_rms(mov):
    e = np.array([phase_cross_correlation(mov[0], mov[t], upsample_factor=20,
                                          normalization=None)[0]
                  for t in range(mov.shape[0])]).T
    return float(np.sqrt((e ** 2).sum(0).mean())), e


def sharpness(img):
    """SCALE-INVARIANT sharpness: gradient energy normalised by variance.

    CaImAn rewrites the intensity scale (mean 37760 / std 11566 vs the raw
    5087 / 822), so a bare gradient energy would report a meaningless ~300x
    "improvement". Dividing by the variance makes it invariant to any affine
    intensity change and therefore comparable across methods.
    """
    img = img.astype(np.float64)
    gy, gx = np.gradient(img)
    return float(np.mean(gy ** 2 + gx ** 2) / (img.var() + 1e-12))


def jitter(mov):
    return float(np.abs(np.diff(mov, axis=0)).mean() / (mov.mean() + 1e-9))


def find_runs():
    runs = {}
    for d in glob.glob(os.path.join(os.path.dirname(__file__), "output",
                                    "gcamp_realistic_500*um_physio-motion_20260718_*")):
        try:
            seed = json.load(open(os.path.join(d, "metadata.json")))["seed"]
        except Exception:
            continue
        runs[int(seed)] = d
    return dict(sorted(runs.items()))


def analyse(seed, run_dir):
    z = np.load(os.path.join(run_dir, "movies.npz"))
    gt = load_motion_gt(z)
    if gt.get("legacy"):
        return None
    raw = z["mov_noisy"].astype(np.float32)
    sa = np.asarray(gt["shift_applied"], float)[:2]
    sfrac = float(gt["sfrac"]); vres = float(gt["vres"])
    k = int(np.ceil(np.abs(sa).max() / sfrac)) + 4

    # The pixels really moved by the ROUNDED shift, so undoing `shift_applied` is
    # exact. `mot_hist` stores the FLOAT requested trajectory — correcting with it
    # over/under-shoots by exactly `shift_residual`. This pair is the direct
    # demonstration of why motion_gt was added.
    sq = np.asarray(gt["shift_requested"], float)[:2]
    streams = {"RAW": raw,
               "mot_hist (legacy)": gt_correct(raw, sq, sfrac),
               "motion_gt": gt_correct(raw, sa, sfrac)}
    for name, suf in (("CaImAn rigid", "rigid"), ("CaImAn non-rigid", "nonrigid")):
        p = os.path.join(MCD, f"seed{seed}_{suf}.tif")
        if os.path.exists(p):
            streams[name] = tifffile.imread(p).astype(np.float32)

    res, traces, crops = {}, {}, {}
    for name, m in streams.items():
        c = m[:, k:-k, k:-k]
        r, e = resid_rms(c)
        res[name] = dict(resid_px=r, resid_um=r * sfrac / vres,
                         sharp=sharpness(c.mean(0)), jitter=jitter(c))
        traces[name] = np.linalg.norm(e, axis=0)
        crops[name] = c
    # sharpness relative to RAW
    for name in res:
        res[name]["sharp_gain"] = res[name]["sharp"] / (res["RAW"]["sharp"] + 1e-12)
    sr = np.asarray(gt["shift_residual"], float)[:2]
    floor = float(np.sqrt((sr ** 2).sum(0).mean())) / sfrac
    return dict(seed=seed, floor_px=floor, methods=res), traces, crops


def make_gif(items, path):
    seeds = [it[0]["seed"] for it in items]
    font = _font()
    cols = len(METHODS)
    cw, ch = PANEL + GAP, PANEL + LABEL_H
    W, H = cols * cw + GAP, len(items) * ch + GAP

    norm_streams = []
    for res, traces, crops in items:
        row = {}
        for name in METHODS:
            if name not in crops:
                continue
            a = crops[name][::TSTEP]
            zf = PANEL / a.shape[1]
            a = ndimage.zoom(a, (1, zf, PANEL / a.shape[2]), order=1)
            # normalise EACH stream on its OWN range: CaImAn rewrites the
            # intensity scale (~7x), so a shared range saturates its panels to
            # white. We are comparing STEADINESS, not absolute brightness.
            lo, hi = np.percentile(a, [1, 99.5])
            row[name] = (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        norm_streams.append(row)

    T = min(min(v.shape[0] for v in row.values()) for row in norm_streams)
    frames = []
    for t in range(T):
        im = Image.new("L", (W, H), 16)
        dr = ImageDraw.Draw(im)
        for r, row in enumerate(norm_streams):
            y0 = GAP + r * ch
            for cidx, name in enumerate(METHODS):
                if name not in row:
                    continue
                x0 = GAP + cidx * cw
                im.paste(Image.fromarray(row[name][t]), (x0, y0 + LABEL_H))
                dr.text((x0 + 2, y0 + 1), f"s{seeds[r]} {name}", fill=245, font=font)
                cx, cy = x0 + PANEL // 2, y0 + LABEL_H + PANEL // 2
                dr.line([(x0, cy), (x0 + PANEL, cy)], fill=105, width=1)
                dr.line([(cx, y0 + LABEL_H), (cx, y0 + LABEL_H + PANEL)], fill=105, width=1)
        frames.append(np.asarray(im))
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {W}x{H}, {T} frames)")


def make_fig(items, path):
    n = len(items)
    fig, ax = plt.subplots(n, 2, figsize=(13, 4.0 * n), squeeze=False)
    colors = {"RAW": "#c0392b", "mot_hist (legacy)": "#8e44ad", "motion_gt": "#2f7d5b",
              "CaImAn rigid": "#3d6b8e", "CaImAn non-rigid": "#b7791f"}
    for r, (res, traces, crops) in enumerate(items):
        for name in METHODS:
            if name not in traces:
                continue
            ax[r, 0].plot(traces[name], lw=1.0, color=colors[name],
                          label=f"{name}  {res['methods'][name]['resid_px']:.2f} px")
        ax[r, 0].axhline(res["floor_px"], ls="--", lw=1, color="k",
                         label=f"sub-voxel floor {res['floor_px']:.2f} px")
        ax[r, 0].set_title(f"seed {res['seed']} — residual displacement", fontsize=10)
        ax[r, 0].set_xlabel("frame"); ax[r, 0].set_ylabel("|shift| (movie px)")
        ax[r, 0].legend(fontsize=7)
        names = [m for m in METHODS if m in res["methods"]]
        vals = [res["methods"][m]["resid_px"] for m in names]
        ax[r, 1].barh(names, vals, color=[colors[m] for m in names])
        ax[r, 1].axvline(res["floor_px"], ls="--", color="k", lw=1)
        for i, v in enumerate(vals):
            ax[r, 1].text(v, i, f" {v:.2f}", va="center", fontsize=8)
        ax[r, 1].set_title("residual rms (lower = steadier);  dashed = ideal floor",
                           fontsize=10)
        ax[r, 1].set_xlabel("movie px")
    fig.suptitle("Real motion correction vs the motion_gt ceiling", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=115); plt.close(fig)
    print(f"wrote {path}")


def main():
    runs = find_runs()
    items = []
    for seed, rd in runs.items():
        got = analyse(seed, rd)
        if got:
            items.append(got); print(f"[+] seed {seed}")
    if not items:
        print("no runs with motion_gt"); return
    os.makedirs(OUT, exist_ok=True)
    make_fig(items, os.path.join(OUT, "mc_comparison.png"))
    make_gif(items[:2], os.path.join(OUT, "mc_compare.gif"))
    with open(os.path.join(OUT, "mc_comparison.json"), "w") as f:
        json.dump([it[0] for it in items], f, indent=2)

    print("\n=== residual rms (movie px), lower = steadier ===")
    for res, _, _ in items:
        print(f"  seed {res['seed']}  (ideal floor {res['floor_px']:.2f})")
        for m in METHODS:
            if m in res["methods"]:
                d = res["methods"][m]
                print(f"    {m:18s} {d['resid_px']:5.2f} px  ({d['resid_um']:5.2f} um)"
                      f"   sharp x{d['sharp_gain']:.2f}")


if __name__ == "__main__":
    main()
