# calcia.benchmark — spatio-temporal segmentation evaluation toolkit

A system for evaluating cell-segmentation / demixing algorithms (DeepWonder,
CNMF-E, MIN1PIPE, SUNS2, DeepCaImX, …) against a **calcia simulation ground
truth**, where every neuron's true position, depth, expression, and clean
calcium trace are known.

It answers four questions the raw algorithm outputs cannot:

1. **Which neurons *can* be found?** — a per-neuron *detectability* model from
   the physics that sets photons-at-detector.
2. **Which are hard / impossible?** — the same model, binned.
3. **Which get merged together?** — a *confusability* graph of merge-prone cells.
4. **What does unclean signal cost downstream?** — connectivity distortion and
   spurious coupling from contaminated traces.

## Why a GT-anchored toolkit

On real recordings you cannot compute recall (you don't know the true cells) and
a "detection" that merges two neurons still looks like a success. Against the
simulation we know the truth, so we can (a) exclude cells that are *physically*
undetectable before scoring recall, and (b) separate *detection* (position) from
*functional recovery* (a clean single-cell trace).

## The four modules

### 1. `detectability` — who is findable
Each neuron's photons-to-detector proxy:

```
optical_brightness = mol_brightness      (expression x fluorophore; from the trace)
                   x exp(-2 z / L)        (1p depth scatter, L = scatter_length_wf)
                   x illum_mask(x,y)      (excitation gradient, from optics.npz)
                   x col_mask(x,y)        (collection gradient, from optics.npz)
```

`characterize(gt)` returns a `score` in `[0,1]` and a category per neuron:

`uninfected` (AAV never expressed) · `out_of_fov` · `invisible` · `hard` ·
`detectable` · `easy`

The **detectable pool** is the *fair* recall denominator — uninfected/invisible
cells never had a chance.

Which standard draws that line is `DetectabilityConfig.criterion`:

| criterion | asks | needs |
|---|---|---|
| `identifiability` (**default**) | after every *other* component in the movie has had the chance to explain this cell's footprint, how much is left? Exact Cramér–Rao bound, one global Cholesky, no neighbour cap. | a run on disk + a cached all-component render (~40 min once, see `identifiability.py`) |
| `percentile` | is it brighter than the pool median? (relative, self-referential) | traces + optics masks |
| `absolute_snr` | does its rendered footprint clear the real noise floor, using the GT trace as a matched filter? (upper bound) | `mov_clean` + noise params |
| `functional` | could a *blind* segmenter tell its transient from noise and from its neighbours? | `mov_clean` + noise params |

The default is expensive and fails loudly if its render is missing — it never
silently falls back to another criterion. Build the cache once with
`identifiability.render_footprints(gt)`. Read that module's docstring before
quoting a count: the binary cut is inherited from `functional` and was calibrated
at a different scale, and the bound is for *linear* unmixing.

### 2. `confusability` — who gets merged
Builds a graph over bright neighbours within `radius_um`; connected components
are *confusable groups* (cells at risk of being fused into one ROI). Per-neuron
`score` (merge risk) and `contamination` (brightness-weighted neighbour leakage)
feed the downstream analysis.

### 3. `loaders` + `matching` + `metrics` — how well an algorithm did
`loaders` normalise each algorithm to an `AlgoResult(centroids, masks, traces)`.
`matching` calibrates the GT→movie-pixel mapping (axis swap + motion offset,
solved by maximising centroid matches) and does greedy one-to-one assignment.
`metrics.compute` reports, on the detectable pool:

- **precision** — fraction of detections on a real cell;
- **recall** three ways — *spatial* (position only), and *temporally gated*
  (`corr>0.5`, `corr>0.7`: position **and** a matching trace);
- **trace-valid fraction** — strict/spatial recall (how many detections are
  functionally real, not merged/contaminated);
- **merge_rate** — fraction of masks covering ≥2 bright cells (under-segmentation);
- **recall_by_category** — recall for `hard` vs `detectable` vs `easy` cells.

### 4. `downstream` — what unclean signal costs
On matched detectable cells, compares recovered vs true traces:

- **connectivity fidelity** — correlation of the recovered vs GT pairwise
  correlation matrix (`corr_offdiag_r`), off-diagonal RMSE, spurious-edge rate;
- **confusable-pair inflation** — how much merge-prone pairs' correlation is
  spuriously inflated (the concrete harm of contamination);
- **event detection** — F1 of transients recovered vs true;
- **SNR ratio** — scale-invariant signal-quality change.

## Usage

```python
from calcia.benchmark import report

# 1) characterise the ground truth
gt, det, conf = report.characterize_run("examples/output/<run>")
print(det.summary()); print(conf.summary())

# 2) evaluate a tree of algorithm results -> figures + md + json
results = report.discover("data/<benchmark>/results")
report.run_benchmark("examples/output/<run>", results, out_dir="…/seg_eval")
```

CLI:

```
conda run -n calcia python -m calcia.benchmark \
    --gt examples/output/<run> --results data/<benchmark>/results --out <out>
```

Or the annotated demo: `examples/demo_seg_benchmark.py`.

## Outputs
- `fig_detectability.png` — category map, brightness-vs-depth, confusable groups.
- `fig_comparison.png` — recall-by-category, recall-vs-fidelity, downstream.
- `BENCHMARK_REPORT.md` + `summary.json` — the numbers.

## Adding an algorithm
Write a loader returning `AlgoResult(name, centroids(col,row), masks(flat idx),
traces(n,T), H, W)` and add it to `report.discover`. Continuous footprints are
thresholded to a mask; downsampled ones (e.g. MIN1PIPE at 0.5×) are upsampled to
the movie frame. Results without persisted masks/traces load as `count_only`.

## Extending the physics
`detectability` reads the real `optics.npz` masks and `params.pkl`
(`scatter_length_um_wf`, `sfrac`, `scan_buff`, focal depth), so it adapts to any
run automatically. Tune category edges via `DetectabilityConfig` and the merge
scale via `ConfusabilityConfig(radius_um=…)`.

## Tests
`tests/test_benchmark.py` — self-contained (synthetic GT + results), covers
detectability binning, confusability grouping, matching, merge detection, and
downstream contamination.
```
conda run -n calcia python -m pytest tests/test_benchmark.py -v
```
