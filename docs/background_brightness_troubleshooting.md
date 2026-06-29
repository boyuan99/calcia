# Background / Overall-Brightness Too High — Troubleshooting Guide

> Self-contained reference for diagnosing "the movie is too bright / whole frames
> are uniformly white (全亮)" in calcia simulations (both two-photon and
> single-photon/widefield). Written so a fresh session can use it as a test plan.

## 1. Symptom

- Some frames look **uniformly bright ("全亮")** — cells barely distinguishable
  from a glowing background — instead of sharp cells on a dark background.
- The movie "flashes" periodically (whole FOV brightens together).
- Overall mean brightness is high; baseline never returns to dark.

In a real 2-photon / GCaMP movie you expect: dark background, sparse cells that
light up individually. A uniform glow means **diffuse background dominates**.

## 2. Root causes (ranked) and the parameters that control them

The final frame brightness is a **product** of four independent contributions:
`(neuropil spatial extent) × (neuropil amplitude+baseline) × (out-of-focus integration / sectioning) × (activity synchrony)`.
Any one being too large produces the wash-out.

| Factor | Where | Parameter | Default | Effect |
|---|---|---|---|---|
| **Optical sectioning (1p)** | Phase 2 | `PsfParams.obj_na` | 0.8 | **strongest lever in 1p (~600× range)**; higher NA = sharper sectioning = far less out-of-focus background |
| **Depth attenuation (1p)** | Phase 2 | `PsfParams.scatter_length_um_wf` | 70.0 | shorter = more attenuation of deep/out-of-focus signal (~2.4×) |
| **Activity synchrony** | Phase 3 | `SpikeParams.smod_flag` | `"hawkes"` | `hawkes` = correlated bursts → whole-frame flashes. `poisson` desyncs BUT floods activity (mean ↑ ~8×) — NOT a good fix |
| **Neuropil amplitude** | Phase 3 | `SpikeParams.bg_scale` | 1.0 | scales the background/neuropil traces (knob added this session). 1.0=NAOMi default; 0.0=off. ~6.7× at 0↔1 |
| **Neuropil source presence** | Phase 1 | `SpikeParams.axonflag` / `AxonParams.flag` | True | turning axons/neuropil off removes the diffuse source (~4.3×) |
| **Neuropil spatial density** | Phase 1 | `AxonParams` / `BgParams` density (`maxfill`, etc.) | — | more/denser axon processes = larger diffuse floor (not cleanly swept here; `bg_proc` count comes from axons, not `N_proc`) |
| **Signal scale** | Phase 4 | `WidefieldParams.pavg`, fluorophore conc/QE | — | scales *everything* (not just background) |

Parameter source files:
- `calcia/config/params.py` — `SpikeParams.bg_scale`, `smod_flag`, `axonflag`; `PsfParams.obj_na`, `scatter_length_um_wf`, `hemo_abs_wf`; `AxonParams`, `BgParams`, `WidefieldParams`.
- `bg_scale` is applied in `calcia/traces/traces.py` (`generate_time_traces`): `S_bg = S_bg * bg_scale` (final background traces only; soma/dend unaffected).
- The striatum demo exposes `--bg-scale` (default 0.1): `examples/demo_widefield_striatum_v1.py`. The calibrated defaults live in `examples/_striatum_common.py`.

## 3. How calcia's scan already handles background (important)

- **F0/ΔF baseline separation is already present** in both `scan_volume` (2p) and
  `scan_widefield` (1p): each trace's temporal min is put into a static `f0vol`,
  only the time-varying part is added per frame. So the *baseline* doesn't pulse.
- **1p `scan_widefield` sums over ALL z-planes** (`single_scan`,
  `calcia/scanning/widefield.py` ~L397-399). There is **NO explicit out-of-focus
  down-weight** (unlike NAOMi1p's `×0.1`). So out-of-focus background is attenuated
  ONLY by the PSF axial profile + depth attenuation — i.e. by the *physics* (`obj_na`,
  `scatter_length_um_wf`), not by an ad-hoc factor.

## 4. Detection / troubleshooting methods

**Do not judge by eye on the saved TIFF** — `save_tiff` re-normalizes each stack to
its own 0.5/99.5 percentiles, which stretches a dimmed movie back to full range and
*hides* the change. Measure on the raw `mov_raw` arrays instead.

Key metrics (compute on `mov_raw`, the clean movie):
1. **Spatial CV of the brightest frame** = `frame.std()/frame.mean()`.
   - Low (~0.3) = **uniform glow (washed out)**; high (>1) = structured cells.
   - ⚠️ Scale-dependent: only meaningful when the FOV has enough neurons/neuropil.
     On tiny volumes (e.g. 80×80×50, ~32 neurons) CV is ~4 everywhere (few blobs)
     and does NOT discriminate. Use a **medium volume (≥100×100×50)** to see wash-out.
2. **Background-to-soma ratio**: `bg_traces.sum() / soma_traces.sum()`. >~2× flags
   neuropil dominance. (In the 2P medium case this was 2.3×.)
3. **Frame-mean vs class-total correlation**: correlate per-frame mean brightness
   with `soma.sum(0)`, `dend.sum(0)`, `bg.sum(0)`. Whichever class has r≈1 drives
   brightness. (bg had r=0.996 in the 2P case.)
4. **Out-of-focus fraction (1p only)**: call `scan_widefield(..., separate_focus=True)`
   and compute `mov_oof.mean() / mov_raw.mean()`. ~96% means the image is
   out-of-focus-dominated → sectioning/depth-attenuation is the lever.
5. **F0/ΔF ratio**: per-pixel temporal min (F0) vs activity range (ΔF). High F0/ΔF =
   static background washes the movie.
6. **Component decomposition**: scan with only soma / only dend / only bg active;
   measure each one's mean contribution + spatial CV. The class that is both large
   *and* spatially uniform is the culprit.

Procedure: decompose by class → for each, get (mean contribution, spatial CV) →
correlate frame-mean with class totals → (1p) measure oof fraction → check F0/ΔF.
The class that is large + uniform + high-F0 is the cause.

## 5. Empirical findings (this session)

**2P, medium volume (100×100×50), the original "全亮" case:**
- Cause = diffuse neuropil/axon background that (a) covers the whole FOV and
  (b) bursts synchronously (Hawkes); per-component bg amplitude was *comparable* to
  soma (soma 11.8 / dend 15.2 / bg 16.9 dynamic range) — the dominance was from
  spatial spread (bg/soma total 2.3×) + synchrony, NOT per-source over-weighting.
- `bg_scale=0.0` fixed it: peak-frame spatial CV **0.28 (uniform) → 1.04 (structured)**.
  `0.25`/`0.1` were NOT enough (synchronized neuropil still fills burst frames + TIFF
  normalization hides the dimming).

**1p / widefield factor sweep (small 80×80×50; brightness verified, wash-out NOT
reproduced at this scale because too few neurons). `total_mean` relative to baseline:**

| Config | total_mean | ×baseline | oof_frac |
|---|---|---|---|
| baseline (NA0.8, hawkes, bg1.0) | 8.96 | 1× | 96.5% |
| bg_scale=0.0 | 1.34 | 0.15× | 94.7% |
| bg_scale=0.25 | 3.52 | 0.39× | 97.2% |
| smod=poisson | 79.0 | **8.8× (worse)** | 96.5% |
| **NA=1.2 (sharp)** | **0.16** | **0.018×** | 98.9% |
| **NA=0.4 (weak)** | **95.3** | **10.6×** | 83.8% |
| psf_z=50 | 8.96 | 1× (no effect*) | 96.5% |
| scatter=20 | 3.65 | 0.41× | 98.5% |
| axons OFF | 2.06 | 0.23× | 93.9% |

\* `PsfParams.psf_sz` z has no effect in the widefield path — it's replaced by the
volume depth.

**Conclusions:**
- 1p `oof_frac ≈ 96%` is **physically correct**: single-photon widefield has a thin
  in-focus depth-of-field, so out-of-focus background genuinely dominates. NOT a bug.
- **NA (sectioning) is the dominant, physical lever** for 1p background brightness.
  `scatter_length` is a secondary physical lever.
- `bg_scale` / `axonflag` are **non-physical/cosmetic** knobs (delete or scale the
  neuropil). NAOMi1p's `×0.1` out-of-focus down-weight is likewise cosmetic.
- `poisson` is NOT a fix — it removes synchrony but floods steady activity (brighter).

**1p / widefield, MEDIUM striatum volume (250×250×100, region='striatum', burst
spikes ~0.57 Hz, GCaMP6f, nt=900–1000) — supersedes the small-volume table for the
STRUCTURE metric (brightest-frame spatial CV), which is what "looks washed/over-exposed"
actually measures:**

| Config | mean | brightest-frame CV | note |
|---|---|---|---|
| baseline (NA0.8, bg1.0) | 2280 | 0.65 | washed |
| NA 0.4→1.0 sweep | 1829→1410 | 0.595→0.613 | **CV flat — NA does NOT de-wash** |
| scatter 70→20 | 1564→481 | 0.61→0.69 | dims, mild structure |
| oof down-weight ×0.1 | 268 | 0.635 | **barely helps** |
| dend ×1% (NAOMi1p dend rescale, ad-hoc test) | 1721 | 0.637 | **near-zero effect** |
| **bg_scale=0.1 (neuropil/axon ×10%)** | **738** | **0.954** | **the lever — cells visible** |
| bg_scale=0.0 | 623 | 1.34 | (off; unrealistic) |

**Corrected conclusions (measuring STRUCTURE/CV, not just brightness):**
- The earlier "NA is the dominant lever" applies to *brightness* (`total_mean`), NOT to
  *structure*. On a realistic striatum volume **NA / scatter / oof-downweight / dendrite
  ×1% all leave brightest-frame CV ≈ 0.6 (still washed)**. They were the wrong suspects.
- **The dominant wash source is the dense axon/neuropil background**: it is volumetrically
  huge, fills the FOV, and the wide 1P PSF smears it into a uniform haze. `bg_scale=0.1`
  takes CV 0.65 → 0.95 (cells emerge), mean 2280 → 738, bg/soma 2.72 → 0.30.
- `bg_scale=0.1` is best read as a **neuropil-amplitude calibration**, not "deleting
  background" — it scales the axon traces to ~10%, the level where soma again carries the
  image. **Applied as the default** in `examples/demo_widefield_striatum_v1.py` (`--bg-scale`,
  default 0.1) and `examples/striatum_demix_dataset.py` (`BG_SCALE=0.1`).
- The NAOMi1p dendrite ×1% rescale was tested (via a temporary `ScanParams.dend_amp_scale`
  knob) and proved near-inert for striatum (dends contribute little vs axons), so it was
  NOT kept in the API. Re-port from `scan_volume_1p.m:321-326` if a cortex case needs it.

## 6. How NAOMi1p (reference MATLAB) solves it

`NAOMi1p/ScanningCode/scan_volume_1p.m` uses several layered mechanisms (kept the
background, controlled the wash-out):
1. Physically-calibrated signal scale (`wdmSignalscale.m`: power × conc × QE × …).
2. **Per-neuron activity normalization to uniform firing** (cap each neuron's std at
   the population median, zero the weakest) — prevents synchronized/bright outliers.
3. **Dendrites scaled to ~1% of soma + apical dendrites disabled.**
4. **F0/ΔF separation** (subtract each component's temporal min).
5. **Out-of-focus top/bottom contributions down-weighted to `×0.1`** (the direct
   anti-wash-out lever).
6. Vessel hemodynamic modulation (`vessel_dilation`) → vessels appear as dark shadows.
7. Two outputs: `mov_w_bg` (realistic) and `mov_wo_bg` (clean soma-only ground truth).

## 7. Suggested test plan for the new session

Use a **medium volume (≥100×100×50)** so the wash-out is reproducible. Build Phase 1
once, reuse across Phase 2/3 variants. For each variant measure: brightest-frame
**spatial CV** (target: 0.3→>1), `total_mean`, and (1p) `oof_frac` via
`separate_focus=True`. Also render the brightest frame per variant **without**
per-movie renormalization (or with a shared scale) to see uniform-vs-structured.

Recommended variants:
- **2P**: `bg_scale` ∈ {1.0, 0.25, 0.0}; `smod_flag` ∈ {hawkes, poisson}.
- **1p/widefield**: `obj_na` ∈ {0.4, 0.8, 1.2} (primary), `scatter_length_um_wf`
  ∈ {70, 20}, `bg_scale` ∈ {1.0, 0.0}.

Decision: for 1p, prefer the **physical** levers (`obj_na`, `scatter_length_um_wf`)
if a realistic widefield look is wanted; use `bg_scale`/out-of-focus down-weight only
for a deliberately cleaner-than-physical image.

## Related code / docs
- `calcia/scanning/widefield.py` (`scan_widefield`, `separate_focus`), `calcia/scanning/scanning.py` (2p).
- `calcia/traces/traces.py` (`generate_time_traces`, `bg_scale` application).
- `calcia/config/params.py` (`SpikeParams`, `PsfParams`, `AxonParams`, `WidefieldParams`).
- `docs/widefield_vs_twophoton.md`, `docs/KNOWN_ISSUES.md`.
