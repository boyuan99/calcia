# -*- coding: utf-8 -*-
"""Side-by-side stability GIF: raw movie | motion_gt-corrected movie.

Static figures cannot show whether a movie is STEADY. This renders both streams
frame-by-frame with a FIXED reference crosshair so the eye can judge the jitter
directly against it.

Run:  conda run -n calcia python examples/make_stability_gif.py <run_dir> [...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import ndimage
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from calcia.scanning import load_motion_gt

OUT = os.path.join(os.path.dirname(__file__), "output", "ideal_stability")
TSTEP = 3          # temporal downsample
FPS = 14
LABEL_H = 18
GAP = 6


def _font(sz=13):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def correct(mov, sa, sfrac):
    """Verified inverse: content moved by -shift, so shift back by +shift/sfrac
    (cubic spline — bilinear would low-pass the structure)."""
    out = np.empty_like(mov)
    for t in range(mov.shape[0]):
        out[t] = ndimage.shift(mov[t], (sa[0, t] / sfrac, sa[1, t] / sfrac),
                               order=3, mode="nearest")
    return out


def build(run_dirs, gif_path):
    panels, labels = [], []
    for rd in run_dirs:
        z = np.load(os.path.join(rd, "movies.npz"))
        gt = load_motion_gt(z)
        if gt.get("legacy"):
            print(f"  [!] {rd}: no motion_gt — skipped"); continue
        mov = z["mov_noisy"].astype(np.float32)
        sa = np.asarray(gt["shift_applied"], float)[:2]
        sfrac = float(gt["sfrac"])
        cor = correct(mov, sa, sfrac)
        k = int(np.ceil(np.abs(sa).max() / sfrac)) + 4      # drop the edge-fill band
        mov, cor = mov[::TSTEP, k:-k, k:-k], cor[::TSTEP, k:-k, k:-k]
        lo, hi = np.percentile(mov, [1, 99.5])              # SAME scale for both
        norm = lambda a: (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        panels.append((norm(mov), norm(cor)))
        seed = json.load(open(os.path.join(rd, "metadata.json")))["seed"]
        labels.append(f"seed {seed}")
    if not panels:
        print("nothing to render"); return

    T = min(p[0].shape[0] for p in panels)
    H, W = panels[0][0].shape[1:]
    cw = W * 2 + GAP
    ch = H + LABEL_H
    canvas_w, canvas_h = cw, ch * len(panels)
    font = _font()

    frames = []
    for t in range(T):
        im = Image.new("L", (canvas_w, canvas_h), 16)
        dr = ImageDraw.Draw(im)
        for i, (raw, cor) in enumerate(panels):
            y0 = i * ch
            im.paste(Image.fromarray(raw[t]), (0, y0 + LABEL_H))
            im.paste(Image.fromarray(cor[t]), (W + GAP, y0 + LABEL_H))
            dr.text((3, y0 + 2), f"{labels[i]}  RAW", fill=245, font=font)
            dr.text((W + GAP + 3, y0 + 2), "motion_gt CORRECTED", fill=245, font=font)
            # fixed reference crosshair — the eye judges jitter against it
            for xo in (0, W + GAP):
                cx, cy = xo + W // 2, y0 + LABEL_H + H // 2
                dr.line([(xo, cy), (xo + W, cy)], fill=110, width=1)
                dr.line([(cx, y0 + LABEL_H), (cx, y0 + LABEL_H + H)], fill=110, width=1)
        frames.append(np.asarray(im))

    imageio.mimsave(gif_path, frames, fps=FPS, loop=0)
    print(f"wrote {gif_path}  ({os.path.getsize(gif_path)/1e6:.1f} MB, "
          f"{len(panels)} rows x {T} frames, {canvas_w}x{canvas_h})")


def main():
    runs = sys.argv[1:]
    if not runs:
        print("usage: make_stability_gif.py <run_dir> [...]"); return
    os.makedirs(OUT, exist_ok=True)
    build(runs, os.path.join(OUT, "stability_compare.gif"))


if __name__ == "__main__":
    main()
