"""
Turn finished simulation runs into web assets a laptop can play for free.

WHY THIS EXISTS
    Every animation this project has ever produced is a GIF, and they average
    30-80 MB for six seconds (calcia/io/render.py writes them through matplotlib
    + pillow).  A GIF is decoded on the CPU, frame by frame, with no hardware
    path -- it is the single worst way to put a movie on a web page.  The same
    footage as H.264/VP9 lands around 1-2% of the size and decodes on silicon
    that is already idle.

    So: nothing here re-simulates anything.  It re-reads `movies.npz`, applies
    the same percentile window `save_tif` uses, and pipes raw frames into ffmpeg.

WHAT IT EMITS (under --out, default web/site/assets/)
    movie/*.mp4|.webm     single-pane clips, no burned-in matplotlib chrome
    movie/*_poster.jpg    first-paint posters
    data/gt.json          soma centres in movie pixels + per-neuron peak dF/F
    data/traces.json      a readable subset of calcium traces, uint8-quantised
    optics/psf.png        the PSF that actually produced those movies
    series/*              the diversity corpus overview
    manifest.json         everything the page needs to know, in one file

THE ONE THING TO GET RIGHT
    The soma overlay.  um -> movie px is `x_um * vres / sfrac - scan_buff/sfrac`
    (calcia/benchmark/gt.py:71); dropping the scan_buff term puts every dot off
    by half the buffer.  The row/col assignment is genuinely ambiguous in this
    codebase -- viz_ladders3d uses row=x, benchmark/matching searches the swap --
    so instead of trusting either, both hypotheses are scored by correlating each
    claimed pixel against that neuron's own known trace, and the margin is
    printed. On the shipped hero run that is decisive: +0.121 vs -0.003.

REAL ANIMAL DATA IS EXCLUDED
    data/real/ holds recordings from live animals. They are not copied here and
    not referenced by the manifest.

Run:
    conda run -n calcia python web/build_site.py --list
    conda run -n calcia python web/build_site.py
    conda run -n calcia python web/build_site.py --run <run_dir> --no-series
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUTPUT_ROOT = os.path.join(REPO, "examples", "output")
DEFAULT_OUT = os.path.join(HERE, "site", "assets")

HERO_RUN = "ladders3d_420um_20260804_162103"

# Channel LUTs, as three stops (black -> hue -> highlight).
#
# A straight two-stop ramp is wrong for this data. A 1P widefield striatum field
# is washed: most pixels already sit near the top of the percentile window, so a
# linear ramp into saturated green turns the whole frame into neon and destroys
# exactly the contrast the section is about. Holding the hue at mid-level and
# only opening up to a pale highlight at the very top keeps the somata readable.
LUTS = {
    "green": ((0.01, 0.02, 0.01), (0.05, 0.31, 0.14), (0.74, 1.00, 0.70)),
    "red":   ((0.02, 0.01, 0.01), (0.40, 0.08, 0.10), (1.00, 0.78, 0.74)),
    "blue":  ((0.01, 0.02, 0.04), (0.16, 0.30, 0.66), (0.80, 0.90, 1.00)),
    "gray":  ((0.02, 0.02, 0.03), (0.48, 0.50, 0.55), (0.98, 0.99, 1.00)),
}


# ======================================================================= util
def sh(cmd, label):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout.decode("utf-8", "replace")[-3000:])
        raise RuntimeError(f"ffmpeg failed: {label}")


def load_movie(run_dir, key="mov_noisy"):
    """(T, H, W) float32, whatever axis order the bundle happened to use."""
    z = np.load(os.path.join(run_dir, "movies.npz"), allow_pickle=True)
    mov = np.asarray(z[key], dtype=np.float32)
    axes = str(z["axes"]) if "axes" in z.files else None
    if axes == "HWT" or (axes is None and mov.shape[0] == mov.shape[1]
                         and mov.shape[2] != mov.shape[0]):
        mov = np.transpose(mov, (2, 0, 1))
    return mov


def window(mov, lo=0.5, hi=99.5):
    """Same self-normalisation as calcia/io/render.py:save_tif.

    Movies are raw detector counts sitting on a large pedestal (this run: 1686
    to 12110). Without the percentile window every frame renders flat grey.
    """
    sub = mov[:: max(1, len(mov) // 24)]
    return float(np.percentile(sub, lo)), float(np.percentile(sub, hi))


def to_rgb(mov, lut, clim, gamma=1.0):
    """Percentile window -> display gamma -> three-stop LUT.

    The window is exactly save_tif's, so the data mapping stays honest; gamma is
    a display choice, applied afterwards and named as such. Without it a washed
    1P field sits so high in the window that everything renders as flat neon.
    """
    lo, hi = clim
    x = np.clip((mov - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** gamma
    stops = np.array(LUTS[lut], dtype=np.float32)          # (3, 3) at 0, 0.5, 1
    knots = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    rgb = np.stack([np.interp(x, knots, stops[:, c]) for c in range(3)], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def encode_rgb(frames, out_base, fps, scale=2, crf_h264=21, crf_vp9=34):
    """Pipe raw RGB straight into ffmpeg -- no PNG round-trip, no temp files."""
    t, h, w = frames.shape[:3]
    vf = f"scale={w * scale}:{h * scale}:flags=lanczos" if scale != 1 else "null"
    common = ["-f", "rawvideo", "-pixel_format", "rgb24",
              "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-"]
    for args_out, path in (
        (["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
          "-crf", str(crf_h264), "-preset", "slow", "-movflags", "+faststart"],
         out_base + ".mp4"),
        (["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf_vp9),
          "-row-mt", "1", "-pix_fmt", "yuv420p"], out_base + ".webm"),
    ):
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + common
            + ["-vf", vf] + args_out + [path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {path}: {err[-1500:]}")
    return out_base + ".mp4", out_base + ".webm"


def save_jpg(rgb, path, scale=2, quality=4):
    h, w = rgb.shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
         "-i", "-", "-vf", f"scale={w * scale}:{h * scale}:flags=lanczos",
         "-frames:v", "1", "-q:v", str(quality), path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"jpg encode failed: {err[-800:]}")


def mb(path):
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


# ============================================================ channel -> video
def build_channel(run_dir, movie_dir, asset_root, name, lut, fps, scale,
                  clim=None, key="mov_noisy", gamma=1.0):
    mov = load_movie(run_dir, key)
    clim = clim or window(mov)
    rgb = to_rgb(mov, lut, clim, gamma)
    base = os.path.join(movie_dir, name)
    mp4, webm = encode_rgb(rgb, base, fps, scale=scale)
    save_jpg(rgb[len(rgb) // 3], base + "_poster.jpg", scale=scale)
    print(f"  {name:22s} {mov.shape[2]}x{mov.shape[1]}x{len(mov)} -> "
          f"{mb(mp4):5.2f} MB mp4 / {mb(webm):5.2f} MB webm")
    rel = lambda p: os.path.relpath(p, asset_root).replace("\\", "/")
    return {
        "name": name,
        "mp4": rel(mp4),
        "webm": rel(webm),
        "poster": rel(base + "_poster.jpg"),
        "frames": int(len(mov)),
        "height": int(mov.shape[1]),
        "width": int(mov.shape[2]),
        "display_scale": scale,
        "fps": fps,
        "clim": [round(clim[0], 2), round(clim[1], 2)],
        "gamma": gamma,
    }, clim


# =========================================================== ground truth JSON
def overlay_scores(mov, traces, px_row, px_col, rad=1):
    """Does the movie at the claimed pixel actually follow that cell's trace?

    Temporal standard deviation cannot settle the row/col question on a dense
    field -- every pixel moves, so both hypotheses score ~1.1x the median.
    Correlating each pixel's time course against the neuron's OWN known trace
    can: a transposed mapping correlates a cell with someone else's activity and
    collapses to zero. Returns (median r, fraction above 0.5, n scored).
    """
    T, H, W = mov.shape
    r = np.round(px_row).astype(int)
    c = np.round(px_col).astype(int)
    ok = (r >= rad) & (r < H - rad) & (c >= rad) & (c < W - rad)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return 0.0, 0.0, 0
    sig = np.stack([mov[:, r[i] - rad:r[i] + rad + 1,
                        c[i] - rad:c[i] + rad + 1].mean(axis=(1, 2))
                    for i in idx])
    a = sig - sig.mean(1, keepdims=True)
    b = traces[idx] - traces[idx].mean(1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12
    rho = (a * b).sum(1) / denom
    return float(np.median(rho)), float(np.mean(rho > 0.5)), int(idx.size)


def build_ground_truth(run_dir, out_dir, scale, verify, max_traces):
    from calcia.benchmark.gt import GroundTruth

    gt = GroundTruth.from_run(run_dir, load_optics=False, load_movie=False)
    px = gt.base_px()                     # (N, 2), columns follow locs_um[:,0:2]
    mov = load_movie(run_dir, "mov_clean")
    traces = gt.traces                    # (N, T), frame-aligned with the movie

    # Hypothesis A (viz_ladders3d, validated on this run): row = x, col = y.
    # Hypothesis B (benchmark/matching, swap=False): col = x, row = y.
    a_med, a_hit, a_n = overlay_scores(mov, traces, px[:, 0], px[:, 1])
    b_med, b_hit, b_n = overlay_scores(mov, traces, px[:, 1], px[:, 0])
    axis = "row=x,col=y"
    row, col = px[:, 0], px[:, 1]
    if verify and b_med > a_med:
        axis, row, col = "row=y,col=x", px[:, 1], px[:, 0]
    print(f"  overlay axis test (median r | frac r>0.5, n={a_n}):"
          f"  row=x {a_med:+.3f} | {a_hit:.2f}   row=y {b_med:+.3f} | {b_hit:.2f}"
          f"   => {axis}")
    hit_rate = a_hit if axis.startswith("row=x") else b_hit
    median_r = a_med if axis.startswith("row=x") else b_med
    f0 = np.percentile(traces, 20, axis=1, keepdims=True)
    f0[f0 <= 0] = 1.0
    dff = (traces - f0) / f0
    peak = dff.max(axis=1)

    h, w = mov.shape[1], mov.shape[2]
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    order = np.argsort(-peak)
    order = order[inside[order]]
    keep = order[:max_traces]

    gt_json = {
        "axis": axis,
        "axis_scores": {"row=x": [round(a_med, 4), round(a_hit, 4)],
                        "row=y": [round(b_med, 4), round(b_hit, 4)]},
        "verification": {"median_pixel_trace_r": round(median_r, 4),
                         "frac_r_above_0p5": round(hit_rate, 4),
                         "n_scored": a_n},
        "movie": {"height": int(h), "width": int(w), "display_scale": scale},
        "fps": round(1.0 / gt.dt, 3),
        "n_neurons_total": int(len(px)),
        "n_in_frame": int(inside.sum()),
        "cells": [
            {"i": int(i), "r": round(float(row[i]), 2), "c": round(float(col[i]), 2),
             "peak_dff": round(float(peak[i]), 3)}
            for i in np.flatnonzero(inside)
        ],
    }
    # uint8 quantisation: these drive a 200 px sparkline, not an analysis.
    sel = dff[keep]
    hi = np.maximum(sel.max(axis=1, keepdims=True), 1e-6)
    q = np.clip(np.round(sel / hi * 255), 0, 255).astype(np.uint8)
    traces_json = {
        "fps": round(1.0 / gt.dt, 3),
        "n_frames": int(sel.shape[1]),
        "cells": [
            {"i": int(keep[k]), "r": round(float(row[keep[k]]), 2),
             "c": round(float(col[keep[k]]), 2),
             "peak_dff": round(float(hi[k, 0]), 3),
             "q": q[k].tolist()}
            for k in range(len(keep))
        ],
    }

    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, payload in (("gt.json", gt_json), ("traces.json", traces_json)):
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        print(f"  {name:22s} {mb(os.path.join(data_dir, name)) * 1000:6.1f} KB")
    return gt_json, traces_json


# ================================================================ optics panel
def build_psf(run_dir, out_dir):
    """The PSF that made those movies -- xy and xz, log-scaled."""
    path = os.path.join(run_dir, "optics.npz")
    if not os.path.exists(path):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psf = np.asarray(np.load(path, allow_pickle=True)["psf"], dtype=np.float32)
    if psf.ndim != 3:
        return None
    zc = psf.shape[2] // 2
    xy = psf[:, :, zc]
    xz = psf[:, psf.shape[1] // 2, :].T

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), facecolor="#05070c")
    for ax, img, title in ((axes[0], xy, "lateral  (x-y)"),
                           (axes[1], xz, "axial  (x-z)")):
        a = img / max(img.max(), 1e-12)
        ax.imshow(np.log10(a + 1e-4), cmap="inferno", origin="lower",
                  aspect="auto", vmin=-4, vmax=0)
        ax.set_title(title, color="#c9d4e6", fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#1d2740")
    fig.tight_layout()
    optics_dir = os.path.join(out_dir, "optics")
    os.makedirs(optics_dir, exist_ok=True)
    out = os.path.join(optics_dir, "psf.png")
    fig.savefig(out, dpi=150, facecolor="#05070c")
    plt.close(fig)
    print(f"  {'psf.png':22s} {mb(out) * 1000:6.1f} KB  psf shape {psf.shape}")
    return {"png": "optics/psf.png", "shape": [int(v) for v in psf.shape]}


# =============================================================== series assets
def build_series(out_dir, fps=12):
    """The 124-seed diversity corpus, as one small looping clip."""
    src_gif = os.path.join(OUTPUT_ROOT, "two_color_series", "reports",
                           "overview_all_sims.gif")
    sheet = os.path.join(OUTPUT_ROOT, "two_color_series", "reports",
                         "overview_contact_sheet.png")
    if not os.path.exists(src_gif):
        print("  [series] overview_all_sims.gif not found; skipping")
        return None
    series_dir = os.path.join(out_dir, "series")
    os.makedirs(series_dir, exist_ok=True)
    base = os.path.join(series_dir, "overview")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src_gif,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart",
        base + ".mp4"], "series mp4")
    sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src_gif,
        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-row-mt", "1",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", base + ".webm"], "series webm")
    out = {"mp4": "series/overview.mp4", "webm": "series/overview.webm"}
    if os.path.exists(sheet):
        sh(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", sheet,
            "-q:v", "72", os.path.join(series_dir, "contact.webp")], "contact")
        out["contact"] = "series/contact.webp"
    print(f"  {'series overview':22s} {mb(base + '.mp4'):5.2f} MB mp4 / "
          f"{mb(base + '.webm'):5.2f} MB webm  (from {mb(src_gif):.1f} MB gif)")
    return out


# ======================================================================= main
def discover_runs():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "*"))):
        if not os.path.isdir(path) or os.path.basename(path).startswith("_"):
            continue
        for cand in [path] + sorted(glob.glob(os.path.join(path, "*"))):
            if os.path.exists(os.path.join(cand, "movies.npz")):
                rows.append(cand)
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Build web-ready assets from finished calcia runs.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Asset output directory.")
    p.add_argument("--run", default=os.path.join(OUTPUT_ROOT, HERO_RUN, "gcamp"),
                   help="Hero GCaMP run dir (holds movies.npz).")
    p.add_argument("--run-crisp", dest="run_crisp",
                   default=os.path.join(OUTPUT_ROOT, HERO_RUN, "gcamp_crisp"),
                   help="Same tissue with scattering off -- the compare slider.")
    p.add_argument("--run-tdt", dest="run_tdt",
                   default=os.path.join(OUTPUT_ROOT, HERO_RUN, "tdt"),
                   help="Static tdTomato channel of the same tissue.")
    p.add_argument("--scale", type=int, default=2,
                   help="Upscale factor at encode time. These movies are small.")
    p.add_argument("--gamma", type=float, default=1.6,
                   help="Display gamma after the percentile window. >1 darkens "
                        "midtones; a washed 1P field needs it or it reads neon.")
    p.add_argument("--fps", type=int, default=0,
                   help="Playback fps. 0 = use the run's own acquisition rate.")
    p.add_argument("--max-traces", dest="max_traces", type=int, default=28,
                   help="Traces shipped to the browser (highest dF/F first).")
    p.add_argument("--tdt", dest="tdt", action="store_true", default=False,
                   help="Also encode the tdTomato channel. Off by default: no "
                        "section on the page uses it, and it is ~8 MB.")
    p.add_argument("--no-series", dest="series", action="store_false",
                   default=True, help="Skip the 124-seed diversity overview.")
    p.add_argument("--no-verify-overlay", dest="verify", action="store_false",
                   default=True, help="Trust row=x without scoring the swap.")
    p.add_argument("--list", action="store_true",
                   help="List run dirs that carry a movies.npz and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        for r in discover_runs():
            print(os.path.relpath(r, REPO))
        return

    out_dir = os.path.abspath(args.out)
    movie_dir = os.path.join(out_dir, "movie")
    os.makedirs(movie_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.run, "metadata.json")))
    fps = args.fps or int(round(1.0 / float(meta.get("dt", 0.05))))

    print("=" * 68)
    print(f"Web assets -> {out_dir}")
    print(f"  hero run: {os.path.relpath(args.run, REPO)}  ({fps} fps)")
    print("=" * 68)

    channels = {}
    t0 = time.time()
    ch, clim = build_channel(args.run, movie_dir, out_dir, "gcamp_noisy",
                             "green", fps, args.scale, gamma=args.gamma)
    channels["gcamp_noisy"] = ch
    ch, _ = build_channel(args.run, movie_dir, out_dir, "gcamp_clean", "green",
                          fps, args.scale, clim=clim, key="mov_clean",
                          gamma=args.gamma)
    channels["gcamp_clean"] = ch
    if os.path.exists(os.path.join(args.run_crisp, "movies.npz")):
        # Same tissue, same window -- the only difference on screen is the PSF.
        ch, _ = build_channel(args.run_crisp, movie_dir, out_dir,
                              "gcamp_crisp", "green", fps, args.scale, clim=clim,
                              gamma=args.gamma)
        channels["gcamp_crisp"] = ch
    # The page has no red-channel section, so this is 8.4 MB of dead weight in a
    # deploy unless something is actually going to use it.
    if args.tdt and os.path.exists(os.path.join(args.run_tdt, "movies.npz")):
        ch, _ = build_channel(args.run_tdt, movie_dir, out_dir, "tdt", "red",
                              fps, args.scale, gamma=args.gamma)
        channels["tdt"] = ch

    print("[data]")
    gt_json, traces_json = build_ground_truth(args.run, out_dir, args.scale,
                                              args.verify, args.max_traces)
    print("[optics]")
    psf = build_psf(args.run, out_dir)
    series = None
    if args.series:
        print("[series]")
        series = build_series(out_dir)

    manifest = {
        "generated_by": "web/build_site.py",
        "hero_run": os.path.relpath(args.run, REPO).replace("\\", "/"),
        "fps": fps,
        "channels": channels,
        "ground_truth": {"json": "data/gt.json",
                         "axis": gt_json["axis"],
                         "n_in_frame": gt_json["n_in_frame"],
                         "n_total": gt_json["n_neurons_total"]},
        "traces": {"json": "data/traces.json",
                   "n_cells": len(traces_json["cells"]),
                   "n_frames": traces_json["n_frames"]},
        "optics": psf,
        "series": series,
        "growth": ("growth/growth.json"
                   if os.path.exists(os.path.join(out_dir, "growth",
                                                  "growth.json")) else None),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(os.path.getsize(p) for p in
                glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)
                if os.path.isfile(p))
    print("-" * 68)
    print(f"assets total {total / 1e6:.1f} MB   ({time.time() - t0:.0f}s)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
