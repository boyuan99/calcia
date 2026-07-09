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
