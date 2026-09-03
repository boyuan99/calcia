"""The identifiability criterion: an exact Cramer-Rao bound on every neuron.

The other detectability criteria ask "does this cell put enough light on the
camera to clear the noise?".  This one asks the harder question the others only
approximate: **after every other thing in the movie has been given the chance to
explain this cell's footprint, how much of it is left?**  A cell whose light can
be reproduced by a linear combination of its neighbours and the background is not
recoverable at any SNR, and no amount of photons fixes that.

    d'_i  =  || c_i - baseline ||_2  *  || P_perp A_i ||_{Sigma^-1}  *  gain

``A_i`` is the cell's exact footprint on the movie (rendered from the simulator's
own forward model, not estimated), ``Sigma`` the run's per-pixel noise variance,
``P_perp`` the projector off the span of every OTHER component's footprint, and
``gain`` the flux-to-ADU conversion measured from the movie itself.

THE IDENTITY THAT MAKES IT AFFORDABLE
-------------------------------------
The forward model is linear-Gaussian, ``Y = A S + E`` with
``E(p) ~ N(0, sigma^2(p))`` independent over pixels.  Whitening by
``W = Sigma^-1/2`` gives ``W Y = (W A) S + N(0, I)``, so the Fisher information
for the amplitudes ``S`` is ``M = A^T Sigma^-1 A`` and the covariance of the best
linear unbiased estimate is ``M^-1``.  Partitioning ``M`` around component ``i``,
the ``(i,i)`` entry of the inverse is the reciprocal of the Schur complement,

    [M^-1]_ii = 1 / ( m_ii - m_i^T M_-i^-1 m_i )
              = 1 / ( atil_i^T (I - P_-i) atil_i )

and ``P_-i = Atil_-i (Atil_-i^T Atil_-i)^-1 Atil_-i^T`` is exactly the orthogonal
projector onto the span of every other whitened footprint.  Therefore

    || P_perp A_i ||^2_{Sigma^-1}  =  1 / [ (A^T Sigma^-1 A)^-1 ]_ii            (*)

exactly, for every ``i`` simultaneously, out of ONE Cholesky of ONE matrix.  No
per-cell design matrix, no neighbour cap.  That matters: on the reference run the
median soma is overlapped by 6311 components, and a per-cell projection that caps
the competitor list has not converged even at a cap of 4000 (surviving footprint
fraction 0.302 / 0.243 / 0.203 / 0.156 at caps 250 / 600 / 1200 / 4000).  Capping
is optimistic as a theorem, not a hunch: for subspaces ``S`` in ``S'``,
``||P_{S'perp} a|| <= ||P_{Sperp} a||``, so dropping competitors can only inflate
what looks like it survives.

WHAT A NEAR-ZERO PIVOT MEANS
----------------------------
``M`` is rank-deficient exactly when some combination of components produces the
identically-zero image -- two cells with the same footprint, or a background
process that is the sum of its neighbours.  That is a physical statement about
the sample, not a numerical accident: on such a direction the movie carries zero
information and the cells involved can be recovered as a sum and never
separately.  So the rank and the condition number are reported (in
``Detectability.identifiability``), not regularised away.  A ridge is still
needed to invert at all; it is added to the UNIT-DIAGONAL Gram so it is a pure
number -- a ridge of ``r`` asserts prior knowledge worth a fraction ``r`` of one
isolated, noise-limited look at the cell, and floors the surviving-footprint
fraction at ``sqrt(r)`` (1e-4 by default).

MEASURED on two_color_series/seed0001/gcamp (14776 components, 43030 of 48400
pixels valid in >= 95% of frames)
-------------------------------------------------------------------------------
``M`` has full rank 14776 at pivot tol 1e-12 and condition number 5.4e10, so
nothing on this run is exactly unidentifiable -- but 10.7 of float64's 16 digits
go into the conditioning, which is why the Gram is accumulated in float64 and
float32 is not an option.  Only 0.46% of the median soma's footprint survives
competition from everything else in the movie (p10 0.28%, p90 3.0%).  Against a
capped per-cell projection the exact bound is smaller for 2687 of 2687 somata,
median ratio 0.079 -- the direction the monotonicity argument requires.
Cross-checked against a real whitened least-squares solve on a synthetic movie
built from the same footprints: two completely different routes (8000
FISTA/eigensolve iterations against one Cholesky) land 5% apart.

Cost: the all-component render is ~40 min for 14776 components and is therefore
CACHED as ``footprints_all.npz`` in the run directory; the Gram is 6.5 GB and
109 s.  If the render is missing this criterion fails and says what to run.  It
never falls back to another criterion.

OPEN ISSUES -- read these before quoting a number
-------------------------------------------------
1. **The cut is inherited and was calibrated at a different scale.**  The binary
   line is the same Bonferroni d' the ``functional`` criterion uses (alpha over
   the number of resolution elements in the frame), which lands at d' = 3.40 on
   the reference run.  An oracle solve on the same footprints put the
   ``r >= 0.5`` recovery crossing at **d' = 2.34**.  The cut is therefore
   conservative by about 45% in d' and the count it produces is not calibrated.

2. **The bound is for LINEAR unmixing.**  A measured non-negative solve on the
   same design was **2.72x more precise** (median recovery correlation
   0.353 -> 0.747).  Real solvers are non-negative, so this bound systematically
   understates what they achieve.  It is a floor on identifiability, not a
   prediction of any particular algorithm's recall.

3. **The calibration against designed ladders was done on a motion-free run
   only.**  Motion is removed here before the noise map is estimated, but the
   intra-frame streak cannot be undone by translation and the criterion has not
   been re-calibrated against a designed ladder on a moving run.

An additional structural caveat: ``d'`` scales with ``gain``, which is measured by
regressing the movie's temporal modulation on ``A C``.  On the reference run that
fit has R^2 = 0.247 (motion streak dominates the residual), so the gain is a
whole-field average and not a per-pixel calibration.
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from .detectability import (CATEGORIES, Detectability, DetectabilityConfig,
                            _pixel_var_and_scale, _snr_from_alpha)
from .gt import GroundTruth

#: Name of the cached all-component render, written next to ``movies.npz``.
FOOTPRINT_FILE = "footprints_all.npz"

#: Convolution-window ladder, voxels. A component's window is the first entry
#: that holds its bounding box plus a whole PSF, so no voxel is ever dropped.
CROPS = (256, 384, 512, 640)

#: Columns whose whitened norm is below this fraction of the median emit no
#: light; normalising them would promote float32 rounding to a unit-norm
#: competitor, so they are made orthogonal instead (see :func:`_normalise`).
DROP_TOL = 1e-6

#: Pivot level below which a direction carries no information at all.
RANK_TOL = 1e-12

#: Movie rows densified at a time when accumulating the Gram.
ROW_BLOCK = 24


# ---------------------------------------------------------------------------
# 1. Motion removal
# ---------------------------------------------------------------------------
# Every ground-truth position calcia gives you -- base_px, the footprints -- lives
# in the ZERO-MOTION frame. The movie does not: frame t is that scene displaced by
# the motion of its own exposure. Nothing that compares a footprint to the movie is
# honest until the two live in the same frame.


def _read_motion(npz, sfrac: float, nt: int):
    """(T, 2) per-frame (drow, dcol) correction, in movie pixels, + its source.

    Signs, confirmed by phase correlation and not assumed::

        image displacement (row, col) px = -shift_vox[:2, t] / sfrac
        row comes from voxel axis 0 (x), col from axis 1 (y)

    so undoing the motion means shifting frame t by ``+shift_vox[:2, t]/sfrac``.
    ``mgt_shift_applied`` is the shift the simulator ACTUALLY applied (integer
    voxels); ``mot_hist`` is the request before rounding and costs 0.19 -> 0.30 px
    of median residual.  A run with no motion record needs no special case: its
    correction is all zeros and every stage below is a no-op.
    """
    for key in ("mgt_shift_applied", "mot_hist"):
        if key in npz.files:
            vox = np.asarray(npz[key], np.float64)[:2, :nt]
            return vox.T / sfrac, key
    return np.zeros((nt, 2)), "none"


def remove_motion(gt: GroundTruth, order: int = 3, fill: float = 0.0):
    """Undo the run's stored ground-truth motion on the NOISY movie.

    Returns ``(mc, valid_frac, info)``: the corrected movie ``(T, H, W)`` float32
    with ``fill`` written where a frame shifted in from outside the sensor, the
    per-pixel fraction of frames that were real, and a small dict of diagnostics.

    The shift is sub-pixel (cubic spline).  It has to be: ``sfrac`` is typically 2,
    so an integer-voxel simulator shift is a HALF-integer movie shift and nearest-
    pixel rounding would leave 0.5 px of residual on half the frames.  A spline
    reaches two samples past its support, so those two rows/columns are dropped
    from the valid mask on any axis that was actually interpolated.

    What cannot be undone is the intra-frame streak: within one exposure the
    sample kept moving, and a frame smeared over 8 px does not come back by
    translating it.
    """
    from scipy.ndimage import shift as nd_shift

    if not gt.run_dir:
        raise ValueError("motion removal needs the run on disk; load the ground "
                         "truth with GroundTruth.from_run(run_dir)")
    path = os.path.join(gt.run_dir, "movies.npz")
    if not os.path.isfile(path):
        raise ValueError(f"{path} is missing: the identifiability criterion needs "
                         "the run's noisy movie")
    npz = np.load(path, allow_pickle=True)
    if "mov_noisy" not in npz.files:
        raise ValueError(f"{path} has no 'mov_noisy'")
    noisy = np.asarray(npz["mov_noisy"], np.float32)
    T, H, W = noisy.shape
    sfrac = float(npz["mgt_sfrac"]) if "mgt_sfrac" in npz.files else float(gt.sfrac)
    corr, src = _read_motion(npz, sfrac, T)

    mc = np.empty((T, H, W), np.float32)
    vcount = np.zeros((H, W), np.int32)
    for t in range(T):
        sr, sc = float(corr[t, 0]), float(corr[t, 1])
        frame = nd_shift(noisy[t], (sr, sc), order=order, mode="constant",
                         cval=0.0, prefilter=True)
        # output row r reads input row r - sr, so keep only rows whose source
        # exists; drop the spline's invented margin on an interpolated axis only
        m_r = 2 if order > 1 and abs(sr - round(sr)) > 1e-6 else 0
        m_c = 2 if order > 1 and abs(sc - round(sc)) > 1e-6 else 0
        r0 = max(int(np.ceil(sr + m_r)), 0)
        r1 = min(int(np.floor(sr + H - 1 - m_r)), H - 1)
        c0 = max(int(np.ceil(sc + m_c)), 0)
        c1 = min(int(np.floor(sc + W - 1 - m_c)), W - 1)
        ok = np.zeros((H, W), bool)
        if r1 >= r0 and c1 >= c0:
            ok[r0:r1 + 1, c0:c1 + 1] = True
            vcount[r0:r1 + 1, c0:c1 + 1] += 1
        mc[t] = np.where(ok, frame, np.float32(fill))
    vfrac = (vcount / float(T)).astype(np.float32)
    mag = np.hypot(corr[:, 0], corr[:, 1])
    info = dict(source=src, fill=float(fill),
                displacement_px_median=float(np.median(mag)),
                displacement_px_max=float(mag.max()),
                valid_frac_mean=float(vfrac.mean()))
    return mc, vfrac, info


# ---------------------------------------------------------------------------
# 2. Exact footprints, from the simulator's own forward model
# ---------------------------------------------------------------------------
#     A_i(movie px) = downsample_sfrac( sum_z conv2d( V_i[:,:,z] * illum * collect,
#                                                     psf[:,:,z] ) )
#
# V_i is the cell's own 3-D fluorescence volume out of cell_footprints.pkl, i.e.
# the very thing the scan integrated, so this is the truth and not an estimate
# (validated at R^2 = 1.000000 against the clean movie for a single cell).
#
# Do NOT substitute a per-cell OLS regression of the movie on that cell's trace:
# on a dense run its cosine against this is ~0.49, because ensemble-mates share
# the trace and their light rides along.
#
# WHAT IS IN THE MOVIE. Reading calcia/scanning/widefield.py's frame loop, each
# frame is built from THREE voxel groups driven by three different rows of
# traces.npz -- gp_vals soma voxels (trace 'soma'), gp_vals neurite voxels (trace
# 'dend'), and bg_proc (trace 'bg', ADDED not assigned). Rendering only the
# labelled somata competes a neuron against a quarter of the movie: on the
# reference run bg_proc carries 65% of the somata's raw fluorescence and its trace
# is 6.1x hotter, and adding it lifts the reconstruction of the clean movie from
# R^2 0.728 to 0.988. Pairing neurite voxels with 'dend' instead of 'soma' is
# worth +0.00007 R^2, so they are merged into the cell's own entry.


def _fft_backend():
    """scipy's pocketfft on every core, or numpy's FFT.

    1.02 ms vs 3.14 ms per 640x640 transform here, and this does ~1e6 of them.
    The two agree to 6e-8 relative on a rendered footprint (float32 eps is
    1.2e-7), so it is a speed knob and not a modelling choice.  The float64 cast
    matters: scipy preserves float32 and would drop the accumulation to
    complex64.
    """
    try:
        import scipy.fft as sf
        return (lambda a: sf.rfft2(np.asarray(a, np.float64), workers=-1),
                lambda a, s: sf.irfft2(a, s=s, workers=-1))
    except Exception:
        from numpy.fft import irfft2, rfft2
        return rfft2, (lambda a, s: irfft2(a, s=s))


def _tight(m: np.ndarray, frac: float):
    """Smallest ``[i, j)`` of the non-negative marginal ``m`` holding ``frac`` of it."""
    m = np.maximum(m, 0.0)
    need = frac * m.sum()
    if need <= 0:
        return 0, len(m)
    i = best = 0
    best_j, run = len(m), 0.0
    for j in range(len(m)):
        run += m[j]
        while run - m[i] >= need:
            run -= m[i]
            i += 1
        if run >= need and (j + 1 - i) < (best_j - best):
            best, best_j = i, j + 1
    return best, best_j


def _volume_shape(gt: GroundTruth, pk: dict):
    """(D0, D1, D2) of the voxel grid the pickle's raveled indices address."""
    vol = pk.get("neur_vol_shape")
    if vol is not None:
        return tuple(int(v) for v in vol)
    meta = json.load(open(os.path.join(gt.run_dir, "metadata.json"), encoding="utf-8"))
    vres = float(meta.get("vres", 1))
    return tuple(int(round(v * vres)) for v in meta["vol_sz"])


def render_footprints(gt: GroundTruth, out_path: str | None = None,
                      eps: float = 0.01, verbose: bool = True) -> str:
    """Render EVERY component's exact movie footprint and cache it as an npz.

    Writes ``footprints_all.npz`` next to ``movies.npz`` (or to ``out_path``) with
    a ragged, per-component rectangle:

    ==============  ==========================================================
    ``foot_flat``   float32 pixels; ``A_i = foot_flat[off[i]:off[i+1]]``
                    ``.reshape(box[i,2], box[i,3])``
    ``foot_off``    (N+1,) offsets into ``foot_flat``
    ``box``         (N,4) r, c, nrow, ncol of that patch in MOVIE px, in-frame
    ``kind``        0 soma, 1 background dendrite (gp_vals tail), 2 bg_proc
    ``trace_key``   which traces.npz array drives it: 'soma' or 'bg'
    ``trace_row``   which ROW of it
    ``tot``         flux emitted into the whole convolution window
    ``tot_frame``   ... of which this much lands inside the movie frame
    ``tot_box``     ... of which this much is inside the stored box
    ``trunc``       1 - tot_box/tot_frame, the STORAGE loss (<= ``eps``)
    ``trunc_fov``   1 - tot_frame/tot, light that misses the FOV (physical)
    ==============  ==========================================================

    A per-component box rather than one fixed square patch, because background
    processes are not compact: a 61x61 patch centred on one keeps 64.7% of its
    flux at the median and 0.3% at p10, i.e. it would throw most of the
    background away.  With a tight box the storage truncation is <= ``eps`` for
    every entry.

    This is the expensive step -- about 40 minutes for 14776 components -- which
    is why it is a cache and not something :func:`characterize_identifiability`
    will do behind your back.
    """
    if not gt.run_dir:
        raise ValueError("rendering needs the run on disk; load the ground truth "
                         "with GroundTruth.from_run(run_dir)")
    out_path = out_path or os.path.join(gt.run_dir, FOOTPRINT_FILE)
    H, W = gt.movie_shape
    SF, BUFF = int(gt.sfrac), int(gt.scan_buff)

    with open(os.path.join(gt.run_dir, "cell_footprints.pkl"), "rb") as f:
        pk = pickle.load(f)
    gp = pk["gp_vals"]
    bgp = pk.get("bg_proc") or []
    opt = np.load(os.path.join(gt.run_dir, "optics.npz"), allow_pickle=True)
    psf = np.asarray(opt["psf"], np.float32)
    mask = np.asarray(opt["mask"], np.float32) if "mask" in opt.files else None
    colm = np.asarray(opt["col_mask"], np.float32) if "col_mask" in opt.files else None
    PS = psf.shape[0]
    D0, D1, D2 = _volume_shape(gt, pk)
    if verbose:
        print(f"  volume {D0}x{D1}x{D2} vox, psf {psf.shape}, sfrac {SF}, "
              f"buff {BUFF}, movie {H}x{W}, {len(gp)} gp_vals + {len(bgp)} bg_proc")

    _rfft2, _irfft2 = _fft_backend()
    cache = {}

    def psf_fft(crop, z):
        k = cache.get((crop, z))
        if k is None:
            p = np.zeros((crop, crop), np.float32)
            p[:PS, :PS] = psf[:, :, z]
            k = cache[(crop, z)] = _rfft2(p)
        return k

    def voxels(e):
        """(a, b, z, w): raveled (X,Y,Z) indices decoded, fluorescence already
        weighted by illumination x collection."""
        idx = np.asarray(e.indices, np.int64)
        fl = np.asarray(e.fluorescence, np.float32)
        a = idx // (D1 * D2)
        b = (idx // D2) % D1
        z = (idx % D2).astype(np.int32)
        w = fl
        if mask is not None and colm is not None:
            ga = np.clip(a, 0, mask.shape[0] - 1)
            gb = np.clip(b, 0, mask.shape[1] - 1)
            w = fl * mask[ga, gb] * colm[ga, gb]
        return a, b, z, w

    def render(a, b, z, w, a0, b0, crop):
        """-> (mv, r0, c0, lost): the crop window pushed through the PSF, summed
        over z and decimated to MOVIE px; (r0, c0) is the movie pixel of mv[0,0]."""
        la, lb = a - a0, b - b0
        keep = (la >= 0) & (la < crop) & (lb >= 0) & (lb < crop)
        la, lb, zz, ww = la[keep], lb[keep], z[keep], w[keep]
        acc = np.zeros((crop, crop // 2 + 1), np.complex128)
        for zk in np.unique(zz):
            m = zz == zk
            plane = np.zeros((crop, crop), np.float32)
            np.add.at(plane, (la[m], lb[m]), ww[m])
            acc += _rfft2(plane) * psf_fft(crop, int(zk))
        img = _irfft2(acc, s=(crop, crop)).astype(np.float32)
        img = np.roll(img, (-PS // 2, -PS // 2), axis=(0, 1))
        mv = img.reshape(crop // SF, SF, crop // SF, SF).sum((1, 3))
        s = float(np.abs(w).sum())
        lost = float(1.0 - np.abs(ww).sum() / s) if s > 0 else 0.0
        return mv, (a0 - BUFF) // SF, (b0 - BUFF) // SF, lost

    # --- plan every component: kind, convolution window, window origin ---------
    items = []                                    # (kind, row, crop, a0, b0)
    for kind, lst in ((0, gp), (2, bgp)):
        for j in range(len(lst)):
            idx = np.asarray(lst[j].indices, np.int64)
            if not len(idx):
                continue
            a = idx // (D1 * D2)
            b = (idx // D2) % D1
            span = max(int(a.max() - a.min()), int(b.max() - b.min())) + 1
            crop = next((c for c in CROPS if c >= span + PS), CROPS[-1])
            a0 = (int(a.min()) + int(a.max())) // 2 - crop // 2
            b0 = (int(b.min()) + int(b.max())) // 2 - crop // 2
            a0 -= (a0 - BUFF) % SF                # keep the decimation grid aligned
            b0 -= (b0 - BUFF) % SF
            items.append((kind if (kind != 0 or j < gt.n) else 1, j, crop, a0, b0))

    N = len(items)
    frac = 1.0 - eps / 2.0                        # per axis; the box holds >= 1 - eps
    nk = np.bincount([it[0] for it in items], minlength=3)
    if verbose:
        print(f"  {N} non-empty components: {nk[0]} somata, {nk[1]} background "
              f"dendrites (gp_vals tail), {nk[2]} background processes (bg_proc)")

    flat = [None] * N
    box = np.zeros((N, 4), np.int64)
    tot = np.zeros(N, np.float64)
    tot_frame = np.zeros(N, np.float64)
    tot_box = np.zeros(N, np.float64)
    lost = np.zeros(N, np.float64)
    crop_used = np.zeros(N, np.int32)
    kind = np.array([it[0] for it in items], np.uint8)
    trace_row = np.array([it[1] for it in items], np.int64)
    trace_key = np.array(["bg" if k == 2 else "soma" for k in kind], "<U4")

    t0 = time.time()
    order = sorted(range(N), key=lambda i: items[i][2])   # by crop: one psf cache
    cur = None
    for done, i in enumerate(order):
        k, j, crop, a0, b0 = items[i]
        if crop != cur:                           # ladder step: drop the old psf
            cache.clear()
            cur = crop
        a, b, z, w = voxels(bgp[j] if k == 2 else gp[j])
        mv, r0, c0, lo = render(a, b, z, w, a0, b0, crop)
        lost[i], crop_used[i], tot[i] = lo, crop, float(mv.sum())
        ra, rb = max(r0, 0), min(r0 + mv.shape[0], H)
        ca, cb = max(c0, 0), min(c0 + mv.shape[1], W)
        if ra >= rb or ca >= cb:
            flat[i] = np.zeros(0, np.float32)
            continue
        sub = mv[ra - r0:rb - r0, ca - c0:cb - c0]
        tot_frame[i] = float(sub.sum())
        y0, y1 = _tight(sub.sum(1), frac)
        x0, x1 = _tight(sub.sum(0), frac)
        pat = np.ascontiguousarray(sub[y0:y1, x0:x1])
        box[i] = (ra + y0, ca + x0, pat.shape[0], pat.shape[1])
        tot_box[i] = float(pat.sum())
        flat[i] = pat.ravel()
        if verbose and (done + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    {done+1}/{N}  crop {crop}  {el:.0f}s elapsed, "
                  f"~{el/(done+1)*(N-done-1):.0f}s left", flush=True)

    off = np.zeros(N + 1, np.int64)
    off[1:] = np.cumsum([len(a) for a in flat])
    foot_flat = np.concatenate(flat) if N else np.zeros(0, np.float32)
    del flat
    ctr = np.column_stack([box[:, 0] + box[:, 2] // 2, box[:, 1] + box[:, 3] // 2])
    # A component sitting entirely under a zero of illum x collect emits NOTHING.
    # Its truncation is 0/0, not 100%: say 0.
    trunc = np.where(tot_frame > 0, 1.0 - tot_box / np.maximum(tot_frame, 1e-30), 0.0)
    trunc_fov = np.where(tot > 0, 1.0 - tot_frame / np.maximum(tot, 1e-30), 0.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path, foot_flat=foot_flat, foot_off=off, box=box, ctr=ctr, ctr_ext=ctr,
        kind=kind, trace_key=trace_key, trace_row=trace_row,
        tot=tot.astype(np.float32), tot_frame=tot_frame.astype(np.float32),
        tot_box=tot_box.astype(np.float32), trunc=trunc.astype(np.float32),
        trunc_fov=trunc_fov.astype(np.float32), crop=crop_used,
        movie=np.array([H, W], np.int64), n_soma=np.int64(gt.n),
        eps=np.float32(eps))
    if verbose:
        print(f"  wrote {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB, "
              f"{foot_flat.nbytes/1e6:.0f} MB of pixels) in "
              f"{(time.time()-t0)/60:.1f} min")
        print(f"  worst per-entry storage truncation {100*trunc.max():.2f}% "
              f"(cap {100*eps:.1f}%); worst flux dropped by the convolution "
              f"window {100*lost.max():.3f}% (must be 0)")
    return out_path


# ---------------------------------------------------------------------------
# 3. The global Gram and diag(M^-1)
# ---------------------------------------------------------------------------


def _symmetrise(M: np.ndarray, blk: int = 2048) -> None:
    """Mirror an upper triangle into the lower one, in blocks.

    The one-liner ``M[triu[::-1]] = M[triu]`` builds two 109-million-element index
    arrays for a 14776^2 matrix -- 2.6 GB of temporaries to copy 1.7 GB of data.
    """
    n = M.shape[0]
    for s in range(0, n, blk):
        e = min(s + blk, n)
        d = M[s:e, s:e]
        iu = np.triu_indices(e - s, 1)
        d[(iu[1], iu[0])] = d[iu]
        if e < n:
            M[e:, s:e] = M[s:e, e:].T


def _norm1(M: np.ndarray, blk: int = 2048) -> float:
    """1-norm (max absolute column sum) without a full-size abs() temporary."""
    return max(float(np.abs(M[:, s:s + blk]).sum(0).max())
               for s in range(0, M.shape[1], blk))


class _Design:
    """The ragged all-component render, addressed by movie-row block.

    ``A`` is never materialised whole: (43030 valid px x 14776) in float64 is
    5.1 GB and would be touched exactly once.  Each block of movie rows is
    densified into a slab, used, and thrown away.
    """

    def __init__(self, F, H: int, W: int, w: np.ndarray):
        self.flat, self.off, self.box = F["foot_flat"], F["foot_off"], F["box"]
        self.H, self.W, self.w = H, W, w
        self.n = len(self.box)

    def _slab(self, r0: int, r1: int, out=None) -> np.ndarray:
        """Raw (unweighted) dense block ``(rows*W, n)``, column i holding
        component i's footprint on movie rows ``[r0, r1)``."""
        S = np.zeros(((r1 - r0) * self.W, self.n)) if out is None else out
        S[...] = 0.0
        for i in range(self.n):
            a, b = self.off[i], self.off[i + 1]
            if b <= a:
                continue
            r, c, nr, nc = self.box[i]
            lo, hi = max(r, r0), min(r + nr, r1)
            if lo >= hi:
                continue
            patch = self.flat[a:b].reshape(nr, nc)[lo - r:hi - r]
            S[(np.arange(lo, hi) - r0)[:, None] * self.W +
              np.arange(c, c + nc)[None, :], i] = patch
        return S

    def gram(self, traces: np.ndarray, blk: int = ROW_BLOCK, verbose: bool = False):
        """``A^T Sigma^-1 A``, and the unweighted prediction ``A C``, in one pass.

        float64, deliberately.  float32 would save 0.87 GB and cost the answer:
        summing 48400 products in float32 leaves ~sqrt(48400)*eps32 = 2.5e-5
        relative error in every Gram entry, and ``diag(M^-1)`` amplifies a
        perturbation of ``M`` by the condition number, measured at ~1e10 here.
        float64's sqrt(48400)*eps64 = 4.6e-14 leaves ~4 correct digits even at
        that conditioning.  (The renders are stored float32, so ~1e-7 is the floor
        on what any arithmetic can recover; the point is not to add to it.)

        ``A C`` is accumulated here rather than in a second pass so the slabs are
        built once: it is the movie prediction the flux-to-ADU gain is regressed
        on.
        """
        from scipy.linalg.blas import dsyrk

        M = np.zeros((self.n, self.n), np.float64, order="F")
        pred = np.zeros((self.H * self.W, traces.shape[1]))
        buf = np.empty((blk * self.W, self.n))
        t0 = time.time()
        for r0 in range(0, self.H, blk):
            r1 = min(r0 + blk, self.H)
            S = buf[:(r1 - r0) * self.W] if r1 - r0 == blk else None
            S = self._slab(r0, r1, out=S)
            pred[r0 * self.W:r1 * self.W] = S @ traces
            S *= self.w[r0 * self.W:r1 * self.W, None]
            # trans=0 on S.T (F-contiguous, so no copy) computes S^T S: half the
            # flops of a gemm, straight into M's upper triangle
            dsyrk(1.0, S.T, 1.0, M, trans=0, lower=0, overwrite_c=1)
            if verbose:
                print(f"    rows {r1}/{self.H}  {time.time()-t0:.0f}s", flush=True)
        _symmetrise(M)                            # syrk only filled one side
        return M, pred


def _normalise(M: np.ndarray, drop_tol: float = DROP_TOL):
    """``M`` -> unit-diagonal correlation form, with dark columns made orthogonal.

    Two reasons this is not cosmetic.  (a) Column norms span 1e5 here (a
    background process against a deep soma), so a ridge on the raw Gram would be
    enormous for one and invisible for the other; on the unit-diagonal form a
    ridge is a pure number meaning the same thing for every component.  (b) The
    answer then comes out directly as a fraction: ``1/[Ms^-1]_ii`` is the share of
    component i's footprint that survives competition, since
    ``[M^-1]_ii = [Ms^-1]_ii / ||A_i||^2``.

    A component that emits no light has an all-zero column, which makes ``M``
    exactly singular.  Its row and column are zeroed and its diagonal set to 1: an
    isolated unit direction orthogonal to everything, which changes no other
    component's projection and leaves its own surviving flux at 0.
    """
    cn = np.sqrt(np.maximum(np.diag(M).copy(), 0.0))
    dark = cn < drop_tol * np.median(cn[cn > 0])
    cn_safe = np.where(dark, 1.0, cn)
    M /= cn_safe[:, None]
    M /= cn_safe[None, :]
    M[dark, :] = 0.0
    M[:, dark] = 0.0
    M[dark, dark] = 1.0
    np.fill_diagonal(M, 1.0)                      # exact, not 1 +/- 1e-16
    return cn, dark


def _inv_diag(Ms: np.ndarray, ridge: float, anorm: float, work=None):
    """``diag((Ms + ridge I)^-1)`` and the condition number, from ONE Cholesky.

    ``dpotrf`` factors it and ``dpotri`` inverts straight from that factor:
    ``n^3/3 + 2n^3/3`` flops once, against ``7n^3/3`` for a factorise plus n
    triangular solves, and it never forms n right-hand sides.  ``dpocon`` has to
    read the factor, so it runs before ``dpotri`` overwrites it.
    """
    from scipy.linalg import lapack

    if work is None:
        C = np.array(Ms, order="F", copy=True)
    else:
        np.copyto(work, Ms)
        C = work
    C.flat[::C.shape[0] + 1] += ridge
    C, info = lapack.dpotrf(C, lower=0, overwrite_a=1, clean=0)
    if info != 0:
        raise ValueError(f"Cholesky failed at pivot {info}: the ridge {ridge:g} is "
                         "too small to make the Fisher matrix numerically "
                         "positive definite")
    rcond = float(lapack.dpocon(C, anorm + ridge)[0])
    inv, info = lapack.dpotri(C, lower=0, overwrite_c=1)
    if info != 0:
        raise ValueError(f"dpotri failed, info={info}")
    return np.diag(inv).copy(), 1.0 / max(rcond, 1e-300)


# ---------------------------------------------------------------------------
# 4. The criterion
# ---------------------------------------------------------------------------


def _dc_variance(gt: GroundTruth, mc: np.ndarray, fill_below: float | None):
    """Per-pixel temporal noise variance, guarded against the edge fill.

    Shifting each frame back leaves pixels that were outside the sensor for part
    of the recording; those samples are padding, not photons.  A plain 10th
    percentile lets the padding win at any pixel that spent >10% of the run
    outside: the DC collapses to the fill value, the shot-noise term
    ``var = qe*dc + dark + read^2`` collapses with it, and d' at the frame edge
    inflates by up to 20x.  Measured on one run: 9.8% of pixels affected, and 117
    of 122 newly "detectable" cells were within 20 px of an edge.
    """
    m = np.asarray(mc, np.float32)
    if fill_below is not None:
        m = np.where(np.isfinite(m) & (m > fill_below), m, np.nan)
        with np.errstate(invalid="ignore"):
            dc = np.nanpercentile(m, 10, axis=0)
        # a pixel with no real sample at all has no usable DC; fall back to the
        # field median rather than to zero, and let the valid mask drop it
        dc = np.where(np.isfinite(dc), dc, np.nanmedian(dc))
    else:
        dc = np.percentile(m, 10, axis=0)
    var, scale = _pixel_var_and_scale(dc, gt)
    return np.maximum(var, 1e-12), scale


def _footprint_path(gt: GroundTruth, cfg: DetectabilityConfig) -> str:
    if cfg.crb_footprints:
        return cfg.crb_footprints
    if not gt.run_dir:
        raise ValueError(
            "criterion='identifiability' needs the run on disk (it reads the "
            "noisy movie, the cell footprints and the optics); load the ground "
            "truth with GroundTruth.from_run(run_dir), or point "
            "DetectabilityConfig.crb_footprints at an existing "
            f"{FOOTPRINT_FILE}.")
    return os.path.join(gt.run_dir, FOOTPRINT_FILE)


def characterize_identifiability(gt: GroundTruth,
                                 cfg: DetectabilityConfig) -> Detectability:
    """Per-neuron ``d'`` from the exact global Cramer-Rao bound.

    Fails loudly if the cached render is missing, saying exactly what to run.  It
    never falls back to another criterion: a missing 40-minute render is a
    prerequisite, not a reason to answer a different question.
    """
    if gt.noise_params is None:
        raise ValueError(
            "criterion='identifiability' needs the run's noise model; load the "
            "run with GroundTruth.from_run(run_dir) so cam_params/noise_params "
            "are available.")
    path = _footprint_path(gt, cfg)
    if not os.path.isfile(path):
        raise ValueError(
            f"criterion='identifiability' needs the all-component footprint "
            f"render and {path} is missing. Build it once (about 40 min for "
            f"14776 components; it is then reused) with\n"
            f"    from calcia.benchmark.gt import GroundTruth\n"
            f"    from calcia.benchmark.identifiability import render_footprints\n"
            f"    render_footprints(GroundTruth.from_run(r'{gt.run_dir}', "
            f"load_movie=False))\n"
            f"or point DetectabilityConfig.crb_footprints at an existing "
            f"{FOOTPRINT_FILE}.")

    H, W = gt.movie_shape
    verbose = bool(cfg.crb_verbose)

    # --- the movie, put back into the zero-motion frame the footprints live in --
    mc, vfrac, minfo = remove_motion(gt)
    T = mc.shape[0]
    if mc.shape[1:] != (H, W):
        raise ValueError(f"the run's movie is {mc.shape[1:]}, gt.movie_shape is {(H, W)}")
    var, nscale = _dc_variance(gt, mc, cfg.crb_fill_below)
    ok = vfrac.ravel() >= cfg.crb_min_valid
    w = np.sqrt(np.where(ok, 1.0 / var.ravel(), 0.0))       # Sigma^-1/2, 0 if invalid
    if verbose:
        print(f"  movie {T}x{H}x{W}; {int(ok.sum())} of {H*W} pixels valid in >= "
              f"{100*cfg.crb_min_valid:.0f}% of frames; motion from "
              f"'{minfo['source']}' (median {minfo['displacement_px_median']:.2f} px)")

    # --- the design ---------------------------------------------------------
    F = np.load(path)
    kind, trow = F["kind"], F["trace_row"]
    tkey = np.asarray(F["trace_key"]).astype(str)
    box = F["box"]
    N = len(box)
    design = _Design(F, H, W, w)

    tz = np.load(os.path.join(gt.run_dir, "traces.npz"), allow_pickle=True)
    C = np.zeros((N, T))
    for key in ("soma", "bg"):
        sel = tkey == key
        if sel.any():
            if key not in tz.files:
                raise ValueError(f"traces.npz has no '{key}' array, which "
                                 f"{int(sel.sum())} rendered components are driven by")
            C[sel] = tz[key][trow[sel], :T]

    # which rendered component is which labelled soma
    comp_of = -np.ones(gt.n, np.int64)
    sel = (kind == 0) & (tkey == "soma")
    comp_of[trow[sel]] = np.where(sel)[0]
    n_missing = int((comp_of < 0).sum())          # somata that emit no voxels at all

    # --- the Gram, the gain, the bound --------------------------------------
    t0 = time.time()
    M, pred = design.gram(C, blk=ROW_BLOCK, verbose=verbose)
    if verbose:
        print(f"  Gram {M.shape} in {time.time()-t0:.0f}s")

    # flux -> ADU, from the movie itself: regress its temporal modulation on A C,
    # in the same whitened metric everything else is stated in. The renders are in
    # flux and the movie is in ADU; assuming one ADU per unit flux is a modelling
    # choice, so it is measured instead.
    ymat = mc.reshape(T, -1).astype(np.float64)
    w2 = w ** 2
    ydm = ymat - ymat.mean(0)
    pdm = (pred - pred.mean(1, keepdims=True)).T
    num = float((ydm * pdm * w2).sum())
    den = float((pdm * pdm * w2).sum())
    gain = num / max(den, 1e-30)
    gain_r2 = num ** 2 / max((ydm * ydm * w2).sum() * den, 1e-30)
    del ymat, ydm, pdm, pred, mc

    cn, dark = _normalise(M, DROP_TOL)            # M is now the unit-diagonal form
    anorm = _norm1(M)

    from scipy.linalg import lapack
    work = np.empty_like(M, order="F")
    np.copyto(work, M)
    fac, _piv, rank, _info = lapack.dpstrf(work, tol=RANK_TOL, lower=0)
    rank = int(rank)
    del fac

    dinv, cond = _inv_diag(M, cfg.crb_ridge, anorm, work=work)
    del work, M
    surv = 1.0 / np.sqrt(np.maximum(dinv, 1e-300))   # fraction of footprint left
    resid_all = cn * surv
    resid_all[dark] = 0.0

    # --- per-soma d' ---------------------------------------------------------
    tr = gt.traces
    amp = tr.max(1)
    infected = amp > cfg.infected_eps
    mol = tr.mean(1).astype(np.float64)
    z = gt.z
    atten = np.exp(-2.0 * z / gt.scatter_length_um)
    illum = gt.sample_mask(gt.illum_mask) if gt.illum_mask is not None else np.ones(gt.n)
    colw = gt.sample_mask(gt.col_mask) if gt.col_mask is not None else np.ones(gt.n)

    # || c_i - baseline ||_2 over the whole recording: a silent cell contributes 0
    act = np.sqrt(((tr - np.percentile(tr, 10, axis=1)[:, None]) ** 2).sum(1))
    have = comp_of >= 0
    ci = np.where(have, comp_of, 0)
    full = np.where(have, cn[ci], 0.0)            # ||A_i||_{Sigma^-1}, no competitors
    resid = np.where(have, resid_all[ci], 0.0)    # ||P_perp A_i||_{Sigma^-1}
    frac = np.where(have, surv[ci], 0.0)
    detect = act * resid * gain
    alone = act * full * gain                     # the d' it would have alone

    # in-FOV from the render's own box centre: that box IS where this component
    # puts light on the camera, clipped to the frame, so it is a more direct test
    # than re-deriving the scan geometry.
    cy = np.where(have, box[ci, 0] + box[ci, 2] / 2.0, -1.0)
    cx = np.where(have, box[ci, 1] + box[ci, 3] / 2.0, -1.0)
    in_fov = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    pool = infected & in_fov

    # --- the binary line -----------------------------------------------------
    # Bonferroni over the independent resolution elements in the frame. See OPEN
    # ISSUE 1 in the module docstring: this is inherited from the functional
    # criterion and an oracle solve puts the real r>=0.5 crossing 45% lower.
    n_res = (H * W) / max(np.pi * (cfg.crb_patch_px / 6.0) ** 2, 1.0)
    cut = _snr_from_alpha(cfg.false_alarm, n_res)

    # No separate silence gate here, unlike the other criteria: the activity norm
    # ||c_i - baseline|| is a FACTOR of d', so a cell that never fires already has
    # d' = 0 and falls below the cut on its own. Adding a spike-count gate on top
    # would drop 5 cells on the reference run that have real trace modulation but
    # no thresholded spike -- i.e. it would overrule the bound with a proxy.
    category = np.array(["uninfected"] * gt.n, dtype=object)
    category[infected & ~in_fov] = "out_of_fov"
    invisible = pool & (detect < cut)
    active = pool & ~invisible
    category[invisible] = "invisible"
    category[active & (detect < 2 * cut)] = "hard"
    category[active & (detect >= 2 * cut) & (detect < 4 * cut)] = "detectable"
    category[active & (detect >= 4 * cut)] = "easy"
    category = category.astype("<U12")
    detectable = pool & (detect >= cut)

    score = np.zeros(gt.n)
    if pool.sum() > 0:
        order_ = np.argsort(np.argsort(detect[pool]))
        score[pool] = order_ / max(len(order_) - 1, 1)

    diag = dict(
        cut=float(cut), gain=float(gain), gain_r2=float(gain_r2),
        noise_scale=float(nscale), ridge=float(cfg.crb_ridge),
        rank=rank, n_components=int(N), n_dark=int(dark.sum()),
        condition=float(cond), n_valid_px=int(ok.sum()), n_px=int(H * W),
        n_somata_without_component=n_missing,
        ridge_floor=float(np.sqrt(cfg.crb_ridge)),
        footprints=os.path.abspath(path), motion=minfo,
        surv=frac, d_alone=alone,
    )
    if pool.any():
        diag.update(surv_median=float(np.median(frac[pool])),
                    surv_p10=float(np.percentile(frac[pool], 10)),
                    surv_p90=float(np.percentile(frac[pool], 90)))
    if verbose:
        print(f"  flux -> ADU gain {gain:.4f} (whitened, time-demeaned; that fit's "
              f"R^2 = {gain_r2:.4f})")
        print(f"  rank {rank} of {N} at pivot tol {RANK_TOL:g}; ridged condition "
              f"number {cond:.3e}; {int(dark.sum())} components emit no light")
        print(f"  surviving footprint over {int(pool.sum())} pool somata: median "
              f"{np.median(frac[pool]):.4f} (ridge floor {np.sqrt(cfg.crb_ridge):.2g})")
        print(f"  cut d' >= {cut:.3f} over {n_res:.0f} resolution elements -> "
              f"{int(detectable.sum())} detectable")

    counts = {c: int((category == c).sum()) for c in CATEGORIES}
    return Detectability(
        infected=infected, in_fov=in_fov, depth_um=z,
        depth_from_focus_um=np.abs(z - gt.focal_depth_um),
        mol_brightness=mol, depth_atten=atten, illum_weight=illum,
        collection_weight=colw, optical_brightness=alone, score=score,
        category=category, detectable=detectable, cfg=cfg, counts=counts,
        detect_snr=detect, identifiability=diag,
    )
