"""Per-neuron detectability characterisation of a simulation ground truth.

Answers: *which* neurons in the data can realistically be segmented, which are
hard, and why — from the physics that governs how many photons each cell
delivers to the detector:

    optical_brightness = molecular_brightness      (expression x fluorophore)
                       x depth_attenuation         (exp(-2 z / L), 1p scatter)
                       x illumination_weight        (excitation gradient)
                       x collection_weight          (detection gradient)

Cells that were never infected by the AAV (expression modulation == 0) are
flagged separately — they carry no signal by construction and must be excluded
from any recall denominator.

Every neuron gets a continuous ``score`` in ``[0, 1]`` (its optical-brightness
rank within the infected, in-FOV population) and a discrete ``category``:

    ``uninfected`` < ``invisible`` < ``hard`` < ``detectable`` < ``easy``

Two criteria (``DetectabilityConfig.criterion``) decide the category:

  * ``"percentile"`` (default) — the *relative* rank above. Self-referential:
    "detectable" just means "brighter than the pool median", so it ranks cells
    but never says whether any of them actually clears the noise floor.
  * ``"absolute_snr"`` — a *physical* standard. Each cell's footprint is read
    straight out of the run's clean movie (the real forward render, so it carries
    the true depth-dependent PSF spread) and its single-frame matched-filter SNR
    is measured against the run's own noise model. Thresholds (``snr_*``) are
    absolute, so counts are comparable across runs and calibratable to real data.
    Needs a run loaded via :meth:`GroundTruth.from_run` (mov_clean + noise params).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .gt import GroundTruth

CATEGORIES = ["uninfected", "out_of_fov", "invisible", "hard", "detectable", "easy"]


@dataclass
class DetectabilityConfig:
    infected_eps: float = 1e-3        # trace max below this => not infected

    # Which standard decides "detectable":
    #   "percentile"   -> relative rank of the optical-brightness proxy (default,
    #                     needs only traces + optics masks; self-referential).
    #   "absolute_snr" -> physical peak SNR of each cell's rendered footprint
    #                     against the run's real noise floor (needs mov_clean +
    #                     noise params; thresholds are absolute, cross-run comparable).
    #                     Uses the ground-truth trace as a matched filter, so it
    #                     measures the information-theoretic UPPER BOUND.
    #   "functional"   -> oracle-free detection SNR of what a blind functional
    #                     segmenter could recover: the cell's own transient
    #                     brightening, discounted by how much of its footprint it
    #                     spatially owns (vs brighter overlapping neighbours),
    #                     over the real noise floor. Continuous, emergent, no fit.
    criterion: str = "percentile"

    # --- percentile-criterion band edges (optical-brightness pct in the pool) ---
    # < p_invisible -> invisible ; [p_invisible, p_hard) -> hard ;
    # [p_hard, p_easy) -> detectable ; >= p_easy -> easy.
    p_invisible: float = 25.0
    p_hard: float = 50.0
    p_easy: float = 90.0
    # "detectable" denominator = infected & in-FOV & optical >= this percentile.
    detectable_percentile: float = 50.0

    # --- absolute_snr-criterion thresholds (on the single-frame peak SNR) ---
    # < snr_invisible -> invisible ; [snr_invisible, snr_hard) -> hard ;
    # [snr_hard, snr_easy) -> detectable ; >= snr_easy -> easy.
    snr_invisible: float = 1.0
    snr_hard: float = 3.0
    snr_easy: float = 8.0
    # radius of the per-cell matched-filter window (microns) used to read the
    # rendered footprint out of the clean movie.
    foot_radius_um: float = 12.0

    # --- functional-criterion knobs ---
    # Fraction of the PSF lateral energy kept when building the movie footprint
    # kernel (sets the footprint window size).
    kernel_energy: float = 0.95
    # Per-FOV false-alarm rate for the OPTIONAL detection-theory binary cut. This
    # is a detector property (fixed across runs); the number of neurons above the
    # resulting SNR line is emergent, never fitted to a target count.
    false_alarm: float = 0.05


@dataclass
class Detectability:
    """Per-neuron detectability arrays + category labels (all length N)."""

    infected: np.ndarray          # bool, AAV expressed
    in_fov: np.ndarray            # bool, soma centre inside the movie
    depth_um: np.ndarray          # z
    depth_from_focus_um: np.ndarray
    mol_brightness: np.ndarray    # expression x fluorophore (mean trace)
    depth_atten: np.ndarray       # exp(-2z/L)
    illum_weight: np.ndarray
    collection_weight: np.ndarray
    optical_brightness: np.ndarray  # photons-to-detector proxy
    score: np.ndarray             # [0,1] rank among infected in-FOV
    category: np.ndarray          # str per neuron (CATEGORIES)
    detectable: np.ndarray        # bool, the fair recall denominator
    cfg: DetectabilityConfig
    counts: Dict[str, int] = field(default_factory=dict)
    # populated only by the absolute_snr criterion (None under percentile):
    snr_peak: np.ndarray = None       # single-frame matched-filter SNR (decision var)
    snr_temporal: np.ndarray = None   # whole-trace spatio-temporal matched-filter SNR
    # populated only by the functional criterion (None otherwise):
    detect_snr: np.ndarray = None     # continuous oracle-free detection SNR (the star output)

    def mask(self, *cats: str) -> np.ndarray:
        """Boolean mask selecting neurons in any of the given categories."""
        m = np.zeros(len(self.category), bool)
        for c in cats:
            m |= self.category == c
        return m

    def summary(self) -> str:
        lines = [f"Detectability of {len(self.category)} neurons "
                 f"({int(self.in_fov.sum())} in-FOV, {int(self.infected.sum())} infected):"]
        for c in CATEGORIES:
            n = int((self.category == c).sum())
            lines.append(f"  {c:12s} {n:6d}  ({100*n/len(self.category):5.1f}%)")
        lines.append(f"  -> detectable pool (fair recall denom): {int(self.detectable.sum())}")
        snr = self.detect_snr if self.detect_snr is not None else self.snr_peak
        if snr is not None:
            pool = self.infected & self.in_fov
            if pool.any():
                q = np.percentile(snr[pool], [50, 90, 99])
                lines.append(f"  detection SNR (infected in-FOV): "
                             f"median={q[0]:.2f} p90={q[1]:.2f} p99={q[2]:.2f} max={snr.max():.2f}")
        return "\n".join(lines)


def characterize(gt: GroundTruth, cfg: DetectabilityConfig | None = None) -> Detectability:
    """Characterise per-neuron detectability under the configured criterion.

    ``cfg.criterion == "percentile"`` (default) ranks the optical-brightness
    proxy within the pool; ``"absolute_snr"`` computes each cell's physical peak
    SNR against the run's real noise floor (requires a run loaded with its clean
    movie and noise params, i.e. ``GroundTruth.from_run``). Both return the same
    ``Detectability`` shape, so every downstream benchmark module is unaffected.
    """
    cfg = cfg or DetectabilityConfig()
    if cfg.criterion == "absolute_snr":
        return _characterize_absolute(gt, cfg)
    if cfg.criterion == "functional":
        return _characterize_functional(gt, cfg)
    if cfg.criterion != "percentile":
        raise ValueError(f"unknown detectability criterion {cfg.criterion!r} "
                         "(expected 'percentile', 'absolute_snr' or 'functional')")
    return _characterize_percentile(gt, cfg)


def _characterize_percentile(gt: GroundTruth, cfg: DetectabilityConfig) -> Detectability:
    tr = gt.traces
    amp = tr.max(1)
    infected = amp > cfg.infected_eps
    mol = tr.mean(1).astype(np.float64)

    z = gt.z
    atten = np.exp(-2.0 * z / gt.scatter_length_um)
    illum = gt.sample_mask(gt.illum_mask) if gt.illum_mask is not None else np.ones(gt.n)
    colw = gt.sample_mask(gt.col_mask) if gt.col_mask is not None else np.ones(gt.n)
    optical = mol * atten * illum * colw

    base = gt.base_px()
    H, W = gt.movie_shape
    # base px are (a, b); either assignment lands in [0, H/W); FOV test is symmetric.
    in_fov = ((base[:, 0] >= 0) & (base[:, 0] < W) & (base[:, 1] >= 0) & (base[:, 1] < H)) | \
             ((base[:, 0] >= 0) & (base[:, 0] < H) & (base[:, 1] >= 0) & (base[:, 1] < W))

    pool = infected & in_fov
    score = np.zeros(gt.n)
    category = np.array(["uninfected"] * gt.n, dtype=object)
    category[infected & ~in_fov] = "out_of_fov"
    if pool.sum() > 0:
        ref = optical[pool]
        order = np.argsort(np.argsort(ref))
        score[pool] = order / max(len(ref) - 1, 1)
        e_inv = np.percentile(ref, cfg.p_invisible)
        e_hard = np.percentile(ref, cfg.p_hard)
        e_easy = np.percentile(ref, cfg.p_easy)
        ov = optical
        category[pool & (ov < e_inv)] = "invisible"
        category[pool & (ov >= e_inv) & (ov < e_hard)] = "hard"
        category[pool & (ov >= e_hard) & (ov < e_easy)] = "detectable"
        category[pool & (ov >= e_easy)] = "easy"
    category = category.astype("<U12")

    thr_det = np.percentile(optical[pool], cfg.detectable_percentile) if pool.sum() else np.inf
    detectable = pool & (optical >= thr_det)

    counts = {c: int((category == c).sum()) for c in CATEGORIES}
    return Detectability(
        infected=infected, in_fov=in_fov, depth_um=z,
        depth_from_focus_um=np.abs(z - gt.focal_depth_um),
        mol_brightness=mol, depth_atten=atten, illum_weight=illum,
        collection_weight=colw, optical_brightness=optical, score=score,
        category=category, detectable=detectable, cfg=cfg, counts=counts,
    )


# ---------------------------------------------------------------------------
# Absolute-SNR criterion
# ---------------------------------------------------------------------------
# Instead of ranking cells against each other, ask a physical question of each:
# does the light it actually put on the movie rise above the run's real noise
# floor?  The forward render already lives in ``mov_clean`` (produced by the
# simulation's own optics, so it carries the true depth-dependent PSF spread), so
# we recover each cell's footprint straight from it — no separate PSF model — and
# read the SNR out against the noise variance of the run's own noise model.


def _pixel_var_and_scale(dc: np.ndarray, gt: GroundTruth):
    """Per-pixel *temporal* measurement variance and the clean->measurement
    signal scale, from the run's noise model evaluated at the DC (baseline)
    clean level ``dc`` (H, W).

    Fixed-pattern terms (camera PRNU, DC offsets) are excluded on purpose: they
    do not fluctuate frame-to-frame, so they add no noise to transient detection.
    """
    p = gt.noise_params
    if gt.noise_kind == "camera":
        qe = float(getattr(p, "qe", 1.0))
        dark = float(getattr(p, "dark_rate", 0.0)) * float(getattr(p, "t_exp", 0.0))
        read = float(getattr(p, "read_noise", 0.0))
        var = qe * np.maximum(dc, 0.0) + dark + read ** 2   # electrons^2
        return var, qe
    # PMT (poisson_gauss): count ~ Poisson(clean+dark), lognormal gain (mean mu,
    # var sigma per count), + N(mu0, sigma0). Compound variance over the Poisson:
    #   Var = (clean+dark)*(sigma + mu^2) + sigma0^2 ;  scale (mean/clean) = mu.
    mu = float(getattr(p, "mu", 1.0))
    sigma = float(getattr(p, "sigma", 0.0))
    sig0 = float(getattr(p, "sigma0", 0.0))
    dark = float(getattr(p, "darkcount", 0.0))
    var = (np.maximum(dc, 0.0) + dark) * (sigma + mu ** 2) + sig0 ** 2
    return var, mu


def _transient_counts(gt: GroundTruth) -> np.ndarray:
    """Number of real transients per neuron. Uses spikes when available;
    otherwise flags any cell with a supra-baseline excursion as transient-bearing
    (so the transient gate only removes provably-silent cells)."""
    if gt.spikes is not None:
        sp = np.asarray(gt.spikes)
        if sp.shape[0] == gt.n:
            return (sp > 0).sum(1).astype(int)
    dF = gt.traces.max(1) - np.percentile(gt.traces, 10, axis=1)
    return (dF > 0).astype(int)


def _calibrate_to_clean(gt: GroundTruth, score_img: np.ndarray,
                        coarse=range(-8, 9, 1), force_swap=None) -> np.ndarray:
    """Solve the GT->movie mapping (axis swap + small residual offset).

    Anchored on the KNOWN scan geometry ``base_px`` (= x*vres/sfrac - scan_buff/sfrac),
    so only the axis-row/col assignment and a small residual (sub-pixel scan phase /
    motion) are searched. This is robust on dense washed fields where cells form no
    brightness peaks — a brightness search from the un-anchored ``x*s`` there fails
    to recover the scan_buff offset and lands ~scan_buff/sfrac px off. ``score_img``
    should be the temporal-std (activity) image, where firing cells stand out even
    when the mean image is a washed cloud. Returns (N,2) as (col, row)."""
    s = gt.vres / gt.sfrac
    off = gt.scan_buff / gt.sfrac
    a = gt.locs_um[:, 0] * s - off
    b = gt.locs_um[:, 1] * s - off
    H, W = score_img.shape

    def _score(swap, dc, dr):
        col = np.round((b if swap else a) + dc).astype(int)
        row = np.round((a if swap else b) + dr).astype(int)
        m = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        return float(score_img[row[m], col[m]].sum()) if m.any() else -np.inf

    swaps = (bool(force_swap),) if force_swap is not None else (False, True)
    best = None
    for swap in swaps:
        for dc in coarse:
            for dr in coarse:
                sc = _score(swap, dc, dr)
                if best is None or sc > best[0]:
                    best = (sc, swap, float(dc), float(dr))
    _, swap, dc, dr = best
    for ddc in np.arange(dc - 3, dc + 3.01, 1.0):
        for ddr in np.arange(dr - 3, dr + 3.01, 1.0):
            sc = _score(swap, ddc, ddr)
            if sc > best[0]:
                best = (sc, swap, float(ddc), float(ddr))
    _, swap, dc, dr = best
    col = (b if swap else a) + dc
    row = (a if swap else b) + dr
    return np.column_stack([col, row])


def _characterize_absolute(gt: GroundTruth, cfg: DetectabilityConfig) -> Detectability:
    if gt.movie_clean is None or gt.noise_params is None:
        raise ValueError(
            "criterion='absolute_snr' needs the clean movie and noise params; "
            "load the run with GroundTruth.from_run(run_dir) so mov_clean and "
            "cam_params/noise_params are available.")

    mov = gt.movie_clean                       # (T, H, W), noise-model input units
    T, H, W = mov.shape
    tr = gt.traces
    amp = tr.max(1)
    infected = amp > cfg.infected_eps
    mol = tr.mean(1).astype(np.float64)
    z = gt.z
    atten = np.exp(-2.0 * z / gt.scatter_length_um)
    illum = gt.sample_mask(gt.illum_mask) if gt.illum_mask is not None else np.ones(gt.n)
    colw = gt.sample_mask(gt.col_mask) if gt.col_mask is not None else np.ones(gt.n)

    act_img = mov.std(0)                       # firing cells stand out even if mean is washed
    px = _calibrate_to_clean(gt, act_img - float(np.median(act_img)))
    col, row = px[:, 0], px[:, 1]
    in_fov = (col >= 0) & (col < W) & (row >= 0) & (row < H)

    dc = np.percentile(mov, 10, axis=0)        # (H, W) baseline clean level
    var_img, scale = _pixel_var_and_scale(dc, gt)
    var_img = np.maximum(var_img, 1e-12)

    s = gt.vres / gt.sfrac
    r = max(1, int(round(cfg.foot_radius_um * s)))

    snr_peak = np.zeros(gt.n)
    snr_temporal = np.zeros(gt.n)
    peak_signal = np.zeros(gt.n)               # integrated peak-frame signal (brightness proxy)
    n_trans = _transient_counts(gt)

    pool = infected & in_fov
    for i in np.where(pool)[0]:
        ci, ri = int(round(col[i])), int(round(row[i]))
        r0, r1 = max(ri - r, 0), min(ri + r + 1, H)
        c0, c1 = max(ci - r, 0), min(ci + r + 1, W)
        trc = tr[i]
        base = np.percentile(trc, 10)
        dtr = trc - base
        dF = float(trc.max() - base)
        if dF <= 0:
            continue
        dtr0 = dtr - dtr.mean()
        denom = float(dtr0 @ dtr0)
        if denom <= 0:
            continue
        sub = mov[:, r0:r1, c0:c1]             # (T, h, w)
        # footprint = regression of the clean movie on this cell's trace; because
        # the movie is the real forward render, beta recovers exactly the spatial
        # spread the simulation gave this cell (incl. depth-dependent PSF width).
        beta = (sub * dtr0[:, None, None]).sum(0) / denom
        np.clip(beta, 0.0, None, out=beta)
        v = var_img[r0:r1, c0:c1]
        spatial = float(np.sqrt(np.sum((scale * beta) ** 2 / v)))   # SNR per unit trace deviation
        snr_peak[i] = spatial * dF
        snr_temporal[i] = spatial * float(np.sqrt((dtr * dtr).sum()))
        peak_signal[i] = float((scale * beta).sum() * dF)

    optical = peak_signal   # physical brightness proxy consumed by confusability
    decide = snr_peak       # single-frame peak SNR is the detection decision variable

    category = np.array(["uninfected"] * gt.n, dtype=object)
    category[infected & ~in_fov] = "out_of_fov"
    invisible = pool & ((decide < cfg.snr_invisible) | (n_trans == 0))
    active = pool & ~invisible
    category[invisible] = "invisible"
    category[active & (decide < cfg.snr_hard)] = "hard"
    category[active & (decide >= cfg.snr_hard) & (decide < cfg.snr_easy)] = "detectable"
    category[active & (decide >= cfg.snr_easy)] = "easy"
    category = category.astype("<U12")

    score = np.zeros(gt.n)
    if pool.sum() > 0:
        ref = decide[pool]
        order = np.argsort(np.argsort(ref))
        score[pool] = order / max(len(ref) - 1, 1)

    detectable = pool & (decide >= cfg.snr_hard) & (n_trans >= 1)
    counts = {c: int((category == c).sum()) for c in CATEGORIES}
    return Detectability(
        infected=infected, in_fov=in_fov, depth_um=z,
        depth_from_focus_um=np.abs(z - gt.focal_depth_um),
        mol_brightness=mol, depth_atten=atten, illum_weight=illum,
        collection_weight=colw, optical_brightness=optical, score=score,
        category=category, detectable=detectable, cfg=cfg, counts=counts,
        snr_peak=snr_peak, snr_temporal=snr_temporal,
    )


# ---------------------------------------------------------------------------
# Functional criterion (oracle-free, continuous, emergent)
# ---------------------------------------------------------------------------
# The physical question for a BLIND functional segmenter: can this cell's own
# light-up be told apart from noise AND from its neighbours, without knowing any
# ground-truth trace?  That is a single detection SNR:
#
#     detect_snr_i = || identifiable transient signal of cell i || / noise
#
# where the numerator is the cell's transient brightening (activity -> a silent
# cell contributes ~0) spread over its PSF footprint and DISCOUNTED, per pixel,
# by the fraction of that footprint the cell actually owns versus brighter
# overlapping neighbours (spatial dominance -> a dominated cell's signal is not
# separable).  No ground-truth trace is used to separate cells (that would be the
# absolute_snr upper bound); separation here is purely spatial.  The result is a
# continuous per-neuron quantity: no threshold, no fit to any target count.


def _block_reduce_sum(a: np.ndarray, f: int) -> np.ndarray:
    """Sum-pool a 2-D array by an integer factor ``f`` (downsample vres->movie)."""
    if f <= 1:
        return a
    h, w = a.shape
    a = a[:h - h % f, :w - w % f]
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).sum(axis=(1, 3))


def _energy_radius(k: np.ndarray, frac: float) -> int:
    """Smallest (Chebyshev) radius around the peak holding ``frac`` of the mass."""
    ky, kx = np.unravel_index(int(np.argmax(k)), k.shape)
    yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    rr = np.maximum(np.abs(yy - ky), np.abs(xx - kx)).ravel()
    order = np.argsort(rr)
    csum = np.cumsum(k.ravel()[order])
    idx = int(np.searchsorted(csum, frac * csum[-1]))
    return int(max(1, rr[order][min(idx, len(order) - 1)]))


def _movie_psf_kernel(gt: GroundTruth, cfg: DetectabilityConfig):
    """Normalised 2-D footprint kernel a cell casts on the movie, in movie pixels.

    For widefield the movie footprint is the depth-integrated lateral PSF, so we
    sum the run's real PSF over its axial axis and sum-pool to movie resolution.
    Falls back to a soma-scale Gaussian if no PSF was stored."""
    if gt.psf is not None and gt.psf.ndim == 3:
        k = np.asarray(gt.psf, np.float64).sum(axis=2)     # (X, Y) at vres
        k = _block_reduce_sum(k, int(gt.sfrac))
    else:
        fwhm = max(2.0, 12.0 * gt.vres / gt.sfrac)
        rad = int(np.ceil(2 * fwhm))
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        k = np.exp(-4 * np.log(2) * (yy ** 2 + xx ** 2) / fwhm ** 2)
    k = np.maximum(k, 0.0)
    k /= k.sum()
    r = _energy_radius(k, cfg.kernel_energy)
    cy, cx = np.unravel_index(int(np.argmax(k)), k.shape)
    k = k[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
    k = k / k.sum()
    return k, r


def _render_static_field(B, col, row, kernel, H, W):
    """Total static footprint field = sum of each cell's brightness-scaled kernel,
    via one FFT convolution of the point sources with the kernel."""
    from scipy.signal import fftconvolve
    pts = np.zeros((H, W), np.float64)
    ci = np.round(col).astype(int)
    ri = np.round(row).astype(int)
    m = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    np.add.at(pts, (ri[m], ci[m]), B[m])
    return fftconvolve(pts, kernel, mode="same")


def _snr_from_alpha(alpha: float, n_candidates: float) -> float:
    """Detection-theory SNR line: the d' at which a matched filter clears a
    per-FOV false-alarm rate ``alpha`` (Bonferroni over candidate locations).
    A property of noise + FOV, independent of the neuron population."""
    from scipy.stats import norm
    a = min(max(alpha, 1e-12), 0.5) / max(n_candidates, 1.0)
    return float(norm.isf(max(a, 1e-15)))


def _characterize_functional(gt: GroundTruth, cfg: DetectabilityConfig) -> Detectability:
    if gt.movie_clean is None or gt.noise_params is None:
        raise ValueError(
            "criterion='functional' needs the clean movie and noise params; "
            "load the run with GroundTruth.from_run(run_dir).")

    mov = gt.movie_clean
    T, H, W = mov.shape
    tr = gt.traces
    amp = tr.max(1)
    infected = amp > cfg.infected_eps
    mol = tr.mean(1).astype(np.float64)
    base = np.percentile(tr, 10, axis=1)
    dF = np.maximum(amp - base, 0.0)               # per-cell transient amplitude (activity)

    z = gt.z
    atten = np.exp(-2.0 * z / gt.scatter_length_um)
    illum = gt.sample_mask(gt.illum_mask) if gt.illum_mask is not None else np.ones(gt.n)
    colw = gt.sample_mask(gt.col_mask) if gt.col_mask is not None else np.ones(gt.n)
    opt = atten * illum * colw
    B = mol * opt                                  # static brightness (arb units)
    A = dF * opt                                   # transient amplitude (arb units)

    mean_img = mov.mean(0)
    act_img = mov.std(0)                        # firing cells stand out even if mean is washed
    kernel, r = _movie_psf_kernel(gt, cfg)
    ky, kx = kernel.shape[0] // 2, kernel.shape[1] // 2

    # The calcia scan writes mov[row, col] = mov[X, Y] (scanning/widefield.py), so
    # the GT->movie axis map is KNOWN — row=X, col=Y, i.e. swap=True — not something
    # to rediscover. Use it directly. Safety net (not a search): the brightness
    # field rendered from the GT MUST positively correlate with the real mean image;
    # if it does not (a different pipeline / a transposed fixture) fall back to the
    # other swap. This is the generation<->analysis self-consistency, not a guess.
    sc = act_img - float(np.median(act_img))
    px = _calibrate_to_clean(gt, sc, force_swap=True)
    F = _render_static_field(B, px[:, 0], px[:, 1], kernel, H, W)
    if float(np.corrcoef(F.ravel(), mean_img.ravel())[0, 1]) <= 0:
        px = _calibrate_to_clean(gt, sc, force_swap=False)
        F = _render_static_field(B, px[:, 0], px[:, 1], kernel, H, W)
    col, row = px[:, 0], px[:, 1]
    in_fov = (col >= 0) & (col < W) & (row >= 0) & (row < H)

    # single-scalar gain (arb -> movie photon units)
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(F.size), F.ravel()]),
                               mean_img.ravel(), rcond=None)
    G = float(max(coef[1], 0.0))

    dc = np.percentile(mov, 10, axis=0)
    var_img, nscale = _pixel_var_and_scale(dc, gt)
    var_img = np.maximum(var_img, 1e-12)

    n_trans = _transient_counts(gt)
    detect = np.zeros(gt.n)
    pool = infected & in_fov
    for i in np.where(pool)[0]:
        ci, ri = int(round(col[i])), int(round(row[i]))
        r0, r1 = max(ri - ky, 0), min(ri + ky + 1, H)
        c0, c1 = max(ci - kx, 0), min(ci + kx + 1, W)
        ksub = kernel[r0 - (ri - ky):r0 - (ri - ky) + (r1 - r0),
                      c0 - (ci - kx):c0 - (ci - kx) + (c1 - c0)]
        Bi_foot = B[i] * ksub                      # cell i's static footprint (arb)
        Ftot = F[r0:r1, c0:c1]                      # total static field (arb)
        dom = np.clip(Bi_foot / (Ftot + 1e-12), 0.0, 1.0)   # spatial ownership, no oracle
        c_i = G * A[i] * ksub * dom                # identifiable transient signal (movie units)
        v = var_img[r0:r1, c0:c1]
        detect[i] = float(np.sqrt(np.sum((nscale * c_i) ** 2 / v)))

    score = np.zeros(gt.n)
    if pool.sum() > 0:
        order = np.argsort(np.argsort(detect[pool]))
        score[pool] = order / max(len(order) - 1, 1)

    # OPTIONAL binary line from detection theory (NOT fitted to a count). The
    # number of neurons above it is emergent — it moves with AAV rate / imaging.
    n_cand = max(float(in_fov.sum()), (H * W) / max(np.pi * (r + 0.5) ** 2, 1.0))
    snr_cut = _snr_from_alpha(cfg.false_alarm, n_cand)

    category = np.array(["uninfected"] * gt.n, dtype=object)
    category[infected & ~in_fov] = "out_of_fov"
    invisible = pool & ((detect < snr_cut) | (n_trans == 0))
    active = pool & ~invisible
    category[invisible] = "invisible"
    category[active & (detect < 2 * snr_cut)] = "hard"
    category[active & (detect >= 2 * snr_cut) & (detect < 4 * snr_cut)] = "detectable"
    category[active & (detect >= 4 * snr_cut)] = "easy"
    category = category.astype("<U12")
    detectable = pool & (detect >= snr_cut) & (n_trans >= 1)

    counts = {c: int((category == c).sum()) for c in CATEGORIES}
    return Detectability(
        infected=infected, in_fov=in_fov, depth_um=z,
        depth_from_focus_um=np.abs(z - gt.focal_depth_um),
        mol_brightness=mol, depth_atten=atten, illum_weight=illum,
        collection_weight=colw, optical_brightness=B, score=score,
        category=category, detectable=detectable, cfg=cfg, counts=counts,
        detect_snr=detect,
    )
