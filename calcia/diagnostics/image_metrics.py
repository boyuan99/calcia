"""Image-quality metrics for scanned movies + sim-vs-real comparison printouts.

These operate on a scanned movie ``mov`` shaped ``(H, W, T)`` (the convention
returned by the widefield scan / demos) and summarise the spatial + temporal
structure the way the real striatum-recording analysis does, so a sim run can be
scored against real-data target ranges.
"""
import numpy as np


def brightest_frame(mov):
    """Index of the frame with the highest mean intensity. mov: (H,W,T)."""
    return int(mov.reshape(-1, mov.shape[2]).mean(0).argmax())


def cv_bright(mov):
    """Spatial coefficient of variation of the brightest frame."""
    fr = mov[:, :, brightest_frame(mov)]
    return float(fr.std() / (fr.mean() + 1e-12))


def dF(mov, pct=10):
    """dF over a static per-pixel baseline (pct-th temporal percentile)."""
    f0 = np.percentile(mov, pct, axis=2, keepdims=True)
    return mov - f0


def summary_stats(mov):
    """mov: (H,W,T). Summary matching the real-data analysis."""
    mean_img = mov.mean(2)
    std_t = mov.std(2)
    bright = mean_img > np.percentile(mean_img, 50)
    cv_t = float(np.median(std_t[bright] / (mean_img[bright] + 1e-6)))
    p = np.percentile(mean_img, [1, 50, 90, 99, 99.9])
    return dict(median=float(p[1]), mean=float(mean_img.mean()),
                max=float(mean_img.max()),
                spatial_cv=float(mean_img.std() / mean_img.mean()),
                temporal_cv=cv_t, floor_frac=float(p[0] / (p[1] + 1e-9)),
                p999_over_med=float(p[4] / (p[1] + 1e-9)),
                pctiles=p.astype(int).tolist())


def print_comparison(channel, mov, targets=None):
    """Print sim summary stats next to the real-data target ranges for a channel.

    ``targets`` maps each stat to a ``(lo, hi)`` real-data range; when omitted it
    falls back to ``calcia.config.indicator_presets.REAL_TARGETS[channel]``.
    """
    if targets is None:
        from calcia.config.indicator_presets import REAL_TARGETS
        targets = REAL_TARGETS[channel]
    st = summary_stats(mov)
    print(f"\n  {channel.upper()} static widefield vs real striatum recordings")
    print(f"    median={st['median']:.0f}  mean={st['mean']:.0f}  "
          f"max={st['max']:.0f}  pctiles[1,50,90,99,99.9]={st['pctiles']}")
    for key in ("spatial_cv", "temporal_cv", "median", "floor_frac",
                "p999_over_med"):
        lo, hi = targets[key]
        ok = "OK" if lo <= st[key] <= hi else "  "
        print(f"    [{ok}] {key:16s}= {st[key]:9.3f}   real [{lo}, {hi}]")
