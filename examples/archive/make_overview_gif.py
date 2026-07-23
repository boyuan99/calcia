"""Overview GIF of ALL simulated results — one animated grid tiling every volume's
GCaMP movie, so the whole diversity series can be eyeballed at a glance.

One panel per volume (latest GCaMP run per deep thin-vessel stub), spatially and
temporally downsampled, per-panel contrast-normalised, seed-labelled. Reads only
finished run dirs, so it never interferes with a running generation series.

Run:  conda run -n calcia python examples/make_overview_gif.py
"""
import glob, json, math, os
import numpy as np
from scipy.ndimage import zoom
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import re
OUT = os.path.join(os.path.dirname(__file__), "output")
VOLS = os.path.join(OUT, "two_color_series", "volumes")
GIF = os.path.join(OUT, "two_color_series", "reports", "overview_all_sims.gif")
SHEET = os.path.join(OUT, "two_color_series", "reports", "overview_contact_sheet.png")
PANEL = 76           # px per panel side
TSTEP = 4            # temporal downsample (every Nth frame)
FPS = 12
PAD = 3
LABEL_H = 16

def _unused_volume_id(stub):
    """(sort_key, label) volume identity from the stub name (NOT the run's
    activity seed). Seed-series volumes sort by seed; others go last."""
    m = re.search(r"deepthinves_s(\d+)_", stub)
    if m:
        return (int(m.group(1)), f"seed {m.group(1)}")
    if stub == "deepthinves_500_flat_stub":
        return (42, "seed 42")
    if "1k" in stub or "1000" in stub:
        return (10**7, "1mm vol")
    return (10**7 + 1, stub.replace("deepthinves_", "")[:12])

def _font(sz=13):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()

def latest_gcamp_per_stub():
    """One panel per VOLUME from the organised tree two_color_series/volumes/
    seedNNNN/gcamp. Returns (sort_key, label, gcamp_dir), seed-sorted."""
    items = []
    for vd in glob.glob(os.path.join(VOLS, "seed*")):
        g = os.path.join(vd, "gcamp")
        if not os.path.exists(os.path.join(g, "movies.npz")):
            continue                                    # skip incomplete volumes
        m = re.search(r"seed(\d+)", os.path.basename(vd))
        seed = int(m.group(1)) if m else -1
        items.append((seed, f"seed {seed}", g))
    return sorted(items, key=lambda x: x[0])

def load_panel(run):
    mov = np.load(os.path.join(run, "movies.npz"))["mov_noisy"]      # (T,H,W)
    mov = mov[::TSTEP]
    zf = PANEL / mov.shape[1]
    small = zoom(mov, (1, zf, PANEL / mov.shape[2]), order=1).astype(np.float32)
    lo, hi = np.percentile(small, 1), np.percentile(small, 99.5)
    u8 = np.clip((small - lo) / (hi - lo + 1e-6), 0, 1)
    return (u8 * 255).astype(np.uint8)                              # (Tds,PANEL,PANEL)

def main():
    items = latest_gcamp_per_stub()
    if not items:
        print("no gcamp runs found"); return
    print(f"tiling {len(items)} volumes: {[lb for _, lb, _ in items]}")
    panels = [load_panel(d) for _, _, d in items]
    labels = [lb for _, lb, _ in items]
    Tds = min(p.shape[0] for p in panels)
    panels = [p[:Tds] for p in panels]

    n = len(panels)
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    cw, ch = PANEL + PAD, PANEL + LABEL_H + PAD
    W, H = cols * cw + PAD, rows * ch + PAD
    font = _font()

    def _grid(imgs):
        canvas = Image.new("L", (W, H), 18)
        draw = ImageDraw.Draw(canvas)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            x0, y0 = PAD + c * cw, PAD + r * ch
            canvas.paste(Image.fromarray(im), (x0, y0 + LABEL_H))
            draw.text((x0 + 2, y0 + 1), labels[i], fill=235, font=font)
        return canvas

    # animated GIF (all volumes, over time)
    frames = [np.asarray(_grid([p[t] for p in panels])) for t in range(Tds)]
    imageio.mimsave(GIF, frames, fps=FPS, loop=0)
    print(f"wrote {GIF}  ({os.path.getsize(GIF)/1e6:.1f} MB, {n} panels x {Tds} frames, {W}x{H})")

    # static contact sheet (max-over-time per panel = activity footprint) — compact,
    # embeddable, the concise at-a-glance view.
    _grid([p.max(0) for p in panels]).save(SHEET)
    print(f"wrote {SHEET}  ({os.path.getsize(SHEET)/1e6:.1f} MB, {W}x{H})")

if __name__ == "__main__":
    main()
