"""Self-contained tests for the segmentation-benchmark toolkit.

Builds a small synthetic ground truth + synthetic algorithm results in memory
(no dependency on the large git-ignored benchmark data) and checks that each
stage does what it claims: detectability binning, confusability grouping,
spatial+temporal matching, merge detection, and downstream contamination.
"""

import numpy as np
import pytest

from calcia.benchmark.gt import GroundTruth
from calcia.benchmark.detectability import characterize, DetectabilityConfig
from calcia.benchmark.confusability import analyze, ConfusabilityConfig
from calcia.benchmark.loaders import AlgoResult
from calcia.benchmark import matching, metrics, downstream
from calcia.config.params import CameraNoiseParams


def _event_trace(rng, nt=200, n_events=4, amp=1.0):
    t = np.zeros(nt, np.float32)
    for _ in range(n_events):
        c = rng.integers(10, nt - 10)
        t[c:c + 8] += amp * np.exp(-np.arange(8) / 3.0)
    return t + 0.01 * rng.standard_normal(nt).astype(np.float32)


@pytest.fixture
def gt():
    """40 neurons on a 20-um grid in a 200x200 px movie (vres=1, sfrac=1)."""
    rng = np.random.default_rng(0)
    N, nt = 40, 200
    xs, ys = np.meshgrid(np.arange(4) * 40 + 20, np.arange(10) * 18 + 20)
    locs = np.column_stack([xs.ravel(), ys.ravel(),
                            rng.uniform(0, 60, N)]).astype(float)
    traces = np.zeros((N, nt), np.float32)
    for i in range(N):
        if i % 5 == 0:            # 20% uninfected -> flat
            continue
        amp = 1.0 + (i % 4)       # brightness spread
        traces[i] = _event_trace(rng, nt, amp=amp) * amp
    return GroundTruth(locs_um=locs, traces=traces, spikes=None, dt=0.05,
                       movie_shape=(220, 220), vres=1.0, sfrac=1, scan_buff=0,
                       scatter_length_um=70.0, focal_depth_um=30.0)


def test_detectability_categories(gt):
    det = characterize(gt)
    # 20% uninfected by construction
    assert det.counts["uninfected"] == 8
    assert det.infected.sum() == 32
    # every neuron gets exactly one category, scores in [0,1]
    assert set(np.unique(det.category)).issubset({
        "uninfected", "out_of_fov", "invisible", "hard", "detectable", "easy"})
    assert det.score.min() >= 0 and det.score.max() <= 1
    # detectable pool is a subset of infected & in-FOV
    assert np.all(det.infected[det.detectable] & det.in_fov[det.detectable])
    # score is the optical-brightness rank -> perfectly monotonic (Spearman==1)
    pool = det.infected & det.in_fov
    assert np.array_equal(np.argsort(det.optical_brightness[pool]),
                          np.argsort(det.score[pool]))


def _render_clean_movie(locs_px, traces, H, W, floor=200.0, sigma=1.6, swap_axes=False):
    """Forward-render a clean (photon) movie: constant floor + a unit-peak
    Gaussian blob per cell, modulated in time by that cell's trace. ``locs_px``
    are (N,2) as (x, y); with ``swap_axes`` the blob lands at (row=x, col=y) to
    exercise the calibrator's axis-swap branch."""
    N, T = traces.shape
    yy, xx = np.mgrid[0:H, 0:W]
    foot = np.zeros((N, H, W), np.float32)
    for i in range(N):
        cx, cy = locs_px[i]
        if swap_axes:
            cx, cy = cy, cx
        foot[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    mov = np.full((T, H, W), floor, np.float32)
    mov += np.tensordot(traces.T.astype(np.float32), foot, axes=(1, 0))  # (T,H,W)
    return mov


@pytest.fixture
def gt_movie():
    """16 well-separated cells on a 140x140 movie with a wide brightness spread,
    plus 4 uninfected (flat) cells, rendered into a clean photon movie so the
    absolute-SNR criterion has real footprints + a real noise floor to test."""
    rng = np.random.default_rng(7)
    H = W = 140
    N, nt = 20, 200
    xs, ys = np.meshgrid(np.arange(4) * 30 + 25, np.arange(4) * 30 + 25)
    grid = np.column_stack([xs.ravel(), ys.ravel()])              # 16 bright cells
    extra = np.array([[10, 10], [130, 10], [10, 130], [130, 130]])  # 4 uninfected
    locs2d = np.vstack([grid, extra]).astype(float)
    locs = np.column_stack([locs2d, rng.uniform(10, 60, N)])
    amps = np.zeros(N)
    amps[:16] = np.geomspace(0.15, 40.0, 16)                      # dim -> very bright
    traces = np.zeros((N, nt), np.float32)
    for i in range(16):
        traces[i] = _event_trace(rng, nt, amp=1.0) * amps[i] + 0.02 * amps[i]
    mov = _render_clean_movie(locs2d, traces, H, W)
    gt = GroundTruth(locs_um=locs, traces=traces, spikes=None, dt=0.05,
                     movie_shape=(H, W), vres=1.0, sfrac=1, scan_buff=0,
                     scatter_length_um=70.0, focal_depth_um=30.0,
                     movie_clean=mov, noise_kind="camera",
                     noise_params=CameraNoiseParams(qe=1.0, gain_e_per_adu=1.0))
    return gt, amps


def test_absolute_snr_bins_are_physical(gt_movie):
    gt, amps = gt_movie
    det = characterize(gt, DetectabilityConfig(criterion="absolute_snr"))
    # interface parity: same 6 categories, snr arrays populated, detectable subset
    assert set(np.unique(det.category)).issubset(set(
        ["uninfected", "out_of_fov", "invisible", "hard", "detectable", "easy"]))
    assert det.snr_peak is not None and det.snr_temporal is not None
    assert len(det.snr_peak) == gt.n
    assert np.all(det.infected[det.detectable] & det.in_fov[det.detectable])
    # the 4 flat cells carry no signal -> uninfected, never in the denominator
    assert det.counts["uninfected"] == 4
    assert not det.detectable[16:].any()
    # peak SNR rises with the planted amplitude (Spearman ~1; a rare adjacent
    # swap between two near-tied dim cells is within regression noise). The
    # brighter half must be strictly ordered.
    rank_amp = np.argsort(np.argsort(amps[:16]))
    rank_snr = np.argsort(np.argsort(det.snr_peak[:16]))
    assert np.corrcoef(rank_amp, rank_snr)[0, 1] > 0.98
    top = np.argsort(amps[:16])[8:]
    assert np.all(np.diff(det.snr_peak[top]) > 0)
    # brightest clears the noise floor (detectable/easy); dimmest does not
    assert det.category[np.argmax(amps[:16])] in ("detectable", "easy")
    assert det.category[np.argmin(amps[:16])] in ("invisible", "hard")
    # calibration actually located the cells -> a non-empty detectable pool
    assert det.detectable.sum() > 0


def test_absolute_snr_handles_axis_swap(gt_movie):
    """Render the same cells transposed (row<->col); the self-calibrator must
    still land footprints on the blobs, yielding a non-empty detectable pool."""
    gt, amps = gt_movie
    mov_sw = _render_clean_movie(gt.locs_um[:, :2], gt.traces,
                                 gt.movie_shape[0], gt.movie_shape[1], swap_axes=True)
    gt.movie_clean = mov_sw
    det = characterize(gt, DetectabilityConfig(criterion="absolute_snr"))
    assert det.detectable.sum() > 0
    # brightest cell still tops the SNR ranking after the swap is solved
    assert det.category[np.argmax(amps[:16])] in ("detectable", "easy")


@pytest.fixture
def gt_functional():
    """6 cells probing each physical lever of the functional detection SNR:
    0 = bright active isolated (shallow), 1 = bright but SILENT (flat expression),
    2 = dim active isolated, 3 = bright active / 4 = dim active OVERLAPPING (3
    dominates 4), 5 = bright active but DEEP."""
    rng = np.random.default_rng(11)
    H = W = 160
    nt = 200
    locs2d = np.array([[30, 30], [30, 120], [120, 30],
                       [120, 120], [124, 120], [80, 80.]])
    z = np.array([10, 10, 10, 10, 10, 150.])
    amps = [20, 20, 3, 25, 4, 20]
    active = [True, False, True, True, True, True]
    N = 6
    traces = np.zeros((N, nt), np.float32)
    for i in range(N):
        if active[i]:
            traces[i] = _event_trace(rng, nt, amp=1.0) * amps[i] + 0.02 * amps[i]
        else:                       # expressed but never fires -> flat trace
            traces[i] = np.full(nt, float(amps[i]), np.float32)
    mov = _render_clean_movie(locs2d, traces, H, W, floor=200.0, sigma=3.0)
    locs = np.column_stack([locs2d, z])
    return GroundTruth(locs_um=locs, traces=traces, spikes=None, dt=0.05,
                       movie_shape=(H, W), vres=1.0, sfrac=1, scan_buff=0,
                       scatter_length_um=70.0, focal_depth_um=10.0,
                       movie_clean=mov, noise_kind="camera",
                       noise_params=CameraNoiseParams(qe=1.0, gain_e_per_adu=1.0))


def test_functional_detection_snr_is_physical(gt_functional):
    gt = gt_functional
    det = characterize(gt, DetectabilityConfig(criterion="functional"))
    assert det.detect_snr is not None and len(det.detect_snr) == gt.n
    snr = det.detect_snr
    # activity: a bright but SILENT cell is ~undetectable despite its brightness
    assert snr[1] < 0.1 * snr[0]
    # brightness/noise: bright active >> dim active (isolated)
    assert snr[0] > snr[2]
    # spatial dominance: the dim cell buried under a bright neighbour loses out
    assert snr[4] < snr[3]
    # depth: the deep cell is strongly suppressed vs the shallow bright one
    assert snr[5] < snr[0]
    # continuous score in [0,1]; detectable subset of infected & in-FOV
    assert det.score.min() >= 0 and det.score.max() <= 1
    assert np.all(det.infected[det.detectable] & det.in_fov[det.detectable])


def test_functional_detectable_falls_with_worse_imaging(gt_functional):
    """Emergence: the detectable count is an OUTPUT of the physics — worsening the
    imaging (more read noise) must lower it, with no threshold retuning."""
    import dataclasses
    gt = gt_functional
    d0 = characterize(gt, DetectabilityConfig(criterion="functional")).detectable.sum()
    noisy = dataclasses.replace(
        gt, noise_params=CameraNoiseParams(qe=1.0, gain_e_per_adu=1.0, read_noise=300.0))
    d1 = characterize(noisy, DetectabilityConfig(criterion="functional")).detectable.sum()
    assert d1 <= d0


def test_functional_requires_movie():
    rng = np.random.default_rng(4)
    locs = np.column_stack([np.arange(3) * 20 + 10, np.full(3, 10), np.full(3, 30.)]).astype(float)
    traces = np.abs(rng.standard_normal((3, 50))).astype(np.float32) + 1
    gt = GroundTruth(locs, traces, None, 0.05, (60, 60), 1, 1, 0, 70, 30)
    with pytest.raises(ValueError, match="functional"):
        characterize(gt, DetectabilityConfig(criterion="functional"))


def test_absolute_snr_requires_movie():
    """Without a clean movie the criterion fails loudly rather than silently
    falling back to the relative standard."""
    rng = np.random.default_rng(2)
    locs = np.column_stack([np.arange(3) * 20 + 10, np.full(3, 10), np.full(3, 30.)]).astype(float)
    traces = np.abs(rng.standard_normal((3, 50))).astype(np.float32) + 1
    gt = GroundTruth(locs, traces, None, 0.05, (60, 60), 1, 1, 0, 70, 30)
    with pytest.raises(ValueError, match="absolute_snr"):
        characterize(gt, DetectabilityConfig(criterion="absolute_snr"))


def test_confusability_pairs_close_cells():
    rng = np.random.default_rng(1)
    # two bright cells 3 um apart -> must be a confusable pair
    locs = np.array([[50, 50, 30.], [50, 53, 30.], [150, 150, 30.]])
    traces = np.abs(rng.standard_normal((3, 200))).astype(np.float32) + 1
    gt = GroundTruth(locs, traces, None, 0.05, (200, 200), 1, 1, 0, 70, 30)
    det = characterize(gt, DetectabilityConfig(detectable_percentile=0))
    conf = analyze(gt, det, ConfusabilityConfig(radius_um=10, brightness_percentile=0))
    assert len(conf.pairs) == 1
    assert set(conf.pairs[0]) == {0, 1}
    assert conf.n_neighbors[0] == 1 and conf.n_neighbors[2] == 0


def _result_from_gt(gt, idxs, name="algo", jitter=0.0, traces=None, masks=None):
    px = gt.base_px()[idxs]
    cen = px + jitter
    tr = traces if traces is not None else gt.traces[idxs].copy()
    return AlgoResult(name, cen, masks, tr, gt.movie_shape[0], gt.movie_shape[1])


def test_matching_recovers_planted_detections(gt):
    det = characterize(gt); conf = analyze(gt, det)
    idxs = np.where(det.detectable)[0]
    res = _result_from_gt(gt, idxs, jitter=0.5)
    cal = matching.calibrate(gt, res)
    mt = matching.match(gt, res, cal)
    # planted at true positions -> all detectable recovered, traces identical
    m = metrics.compute(gt, res, det, conf, mt)
    assert m.recall_spatial > 0.95
    assert m.median_corr > 0.99
    assert m.recall_corr07 > 0.95
    assert m.precision > 0.95


def test_merge_detection():
    """Purpose-built GT: 3 isolated anchors + a 3-cell cluster. A single mask
    over the cluster must register as an under-segmentation merge."""
    rng = np.random.default_rng(3)
    anchors = [[20, 20, 30.], [180, 20, 30.], [20, 180, 30.]]
    cluster = [[100, 100, 30.], [104, 100, 30.], [100, 104, 30.]]
    locs = np.array(anchors + cluster)
    traces = (np.abs(rng.standard_normal((6, 200))).astype(np.float32) + 1) * 3
    gt = GroundTruth(locs, traces, None, 0.05, (200, 200), 1, 1, 0, 70, 30)
    det = characterize(gt, DetectabilityConfig(detectable_percentile=0))
    conf = analyze(gt, det, ConfusabilityConfig(radius_um=10, brightness_percentile=0))
    W = gt.movie_shape[1]

    def disk(px, r=2):
        cx, cy = int(round(px[0])), int(round(px[1]))
        rr, cc = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
        return (rr * W + cc).ravel()

    base = gt.base_px()
    cen = [base[0], base[1], base[2], base[3:6].mean(0)]     # 3 anchors + cluster centroid
    masks = [disk(base[0]), disk(base[1]), disk(base[2]),
             np.unique(np.concatenate([disk(base[i], 3) for i in (3, 4, 5)]))]
    res = AlgoResult("merger", np.array(cen), masks, traces[[0, 1, 2, 3]].copy(), 200, W)
    cal = matching.calibrate(gt, res, radius=4)
    mt = matching.match(gt, res, cal)
    m = metrics.compute(gt, res, det, conf, mt)
    # 4 masks; exactly the cluster mask (1) merges >=2 bright cells
    assert m.merge_rate == pytest.approx(0.25)
    assert m.frac_bright_not_separated > 0


def test_downstream_detects_contamination(gt):
    det = characterize(gt); conf = analyze(gt, det)
    idxs = np.where(det.detectable)[0]
    # contaminate every recovered trace with a shared component -> spurious corr
    shared = _event_trace(np.random.default_rng(9), gt.nt, amp=2.0)
    contaminated = gt.traces[idxs].copy() + 1.5 * shared
    res = _result_from_gt(gt, idxs, traces=contaminated)
    cal = matching.calibrate(gt, res); mt = matching.match(gt, res, cal)
    di = downstream.assess(gt, res, det, conf, mt)
    # shared contamination inflates off-diagonal correlations vs GT
    assert di.corr_matrix_rmse > 0.1
    assert di.n_cells == len(idxs)


def test_clean_downstream_has_low_distortion(gt):
    det = characterize(gt); conf = analyze(gt, det)
    idxs = np.where(det.detectable)[0]
    res = _result_from_gt(gt, idxs)          # exact GT traces
    cal = matching.calibrate(gt, res); mt = matching.match(gt, res, cal)
    di = downstream.assess(gt, res, det, conf, mt)
    assert di.corr_matrix_rmse < 1e-4        # identical traces -> no distortion
    assert di.corr_offdiag_r > 0.999
