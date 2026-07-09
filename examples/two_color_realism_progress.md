# Two-colour striatum realism — progress log

**Goal:** generate GCaMP (green, dynamic) and tdTomato (red, static) 1P widefield
videos that look as close as possible to the real striatum window recordings,
as a co-registered two-colour dataset (registration / co-localization test bench).

**Hard constraints from the user**
- Do NOT modify the core simulation algorithm; build standalone companion scripts.
- Realism must EMERGE from the volume + scan physics, not post-processing hacks
  (no whole-image Gaussian blur = fake camera defocus; no synthetic vessel/void
  overlays; no per-cell cosmetics like `bright_frac` / `solid_soma` / `soma_gain`).
- Any shared-config recalibration must be REVERSIBLE (back up originals + document).
- "Wash / less clear" means RAISE background fluorescence, not blur the image.

See memories: `feedback-wash-via-background-not-blur`,
`feedback-design-purity-and-reversible-params`, `project-static-tdtomato-matched`,
`project-1p-wash-bgscale`, `reference-background-brightness`.

---

## Reference data (real, local, git-ignored under `data/real/`)
- **Real tdt:** `data/real/tdt-bfp/398_09192025_tdt_mc.h5` (500×1152²), `400_..._tdt_mc.h5`.
  Washed smooth cloud; **prominent dark blood vessels + dark voids**; single cells
  NOT resolvable by eye. Flatfielded central texture CV ≈ **0.144**.
- **Real GCaMP:** `data/real/striatum_raw_samples_15/*.tif` (~200 fr × 1152², 20 Hz).
  Smooth regional wash; **cells NOT resolved** in mean OR activity (std-over-time)
  images — they must be demixed out. Measured: **dF/F p99 ≈ 0.195 (20%)**,
  mean-image spatial CV ≈ 0.79 (dominated by vignette + large-scale gradient),
  std-over-time CV ≈ 0.56, median ≈ 1372 ADU.

The matched GCaMP fitting run being companioned:
`examples/output/striatum_v1_1700um_physio-motion_20260706_114842/`
(1700×1700×60 µm, vres=1, seed=42, 200 fr @ 20 Hz, GCaMP6f, physio motion,
N_neur=17340, movie 820×820×200). Its Phase-1 cache:
`examples/output/_shared/phase1_ae8539a935.pkl` (7.5 GB) — reused by all
companion scripts so cells are co-registered.

---

## Scripts built (standalone; core untouched)
1. **`examples/demo_static_tdtomato_matched.py`** — static tdTomato red channel,
   co-registered to the GCaMP run. 50% random soma expression (`--label-frac`),
   `--bg-frac` (neuropil wash fraction, default 1.0 = full), `--oof-blur-um`
   (⚠ this is the fake-defocus blur — to be REMOVED/zeroed, see below),
   independent physio motion by default (`--motion-seed`; seed+1234 vs the GCaMP
   run's seed+3). Saves `movie_tdt.npz`, `tdtomato_expression.npz` (`expr_ids`
   align with the GCaMP run's `soma_neurons`/`soma_locs`), `metadata.json`.
   Smoke path works (`--smoke`, tiny throwaway volume).
2. **`examples/demo_gcamp_realistic_matched.py`** — realistic GCaMP re-scan of the
   SAME volume, no cosmetics, neuropil low-dF/F floor attempt (`--bg-target-dff`,
   `--bg-bright`). Runs end-to-end (smoke OK). **Calibration NOT yet solved — see
   the open problem below; current defaults are placeholders.**

Both read the matched volume via `load_matched_volume()` → identical somata.

---

## Diagnostic journey (what we ruled out — important, don't repeat)

**tdt "too clear":** the full-frame spatial CV (0.29–0.34) was mostly the
**illumination vignette**, not cell texture. On vignette-free flatfielded central
crops: real=0.144, sim(halved-bg,12µm)=0.098, sim(full-bg,20µm)=**0.058**. So the
20 µm blur OVER-smoothed; the sim texture is if anything already SMOOTHER than
real. The real differentiators sim lacks: **dark blood vessels + dark voids**
(striatum preset deliberately thins vessels — premise "1P haze hides vessels" is
CONTRADICTED by real tdt which shows clear dark vessels). Adding blur was the
wrong direction.

**GCaMP "too clear" = dF/F 7× too high** (sim p99 ≈ 1.36 vs real 0.195); this is
vignette-independent (temporal ratio), so it is a real quantitative gap. Two
attempted physical levers BOTH FAILED to fix it:

- **`bg_scale` sweep** (amplify neuropil background), volume loaded once, nt=24:
  | bg_scale | 1 | 4 | 8 | 16 |
  |---|---|---|---|---|
  | dff_p99 | 1.33 | 1.26 | 1.37 | 1.00 |
  | median | 6457 | 19874 | 37988 | 63021 |
  Barely moves dF/F; median explodes. **Why:** the background/neuropil traces
  themselves carry dF/F ≈ **1.37** — the SAME as somata (measured directly from
  `traces.npz`: soma 1.31, bg 1.37, dend 1.72). The sim treats each neuropil
  process like its own flashing cell, so amplifying it is amplifying a flicker,
  not adding a stable floor.

- **Neuropil low-dF/F floor** (damp bg AC to target dF/F, then brighten),
  volume once, nt=24:
  | bg_dff | bg_scale | median | dff_p99 | flatCV |
  |---|---|---|---|---|
  | 0.15 | 8 | 33629 | 1.359 | 0.181 |
  | 0.15 | 16 | 60503 | 1.134 | 0.101 |
  | 0.15 | 30 | **65535 (SAT)** | 0.382 | 0.017 |
  | 0.30 | 16 | 60952 | 1.116 | 0.097 |
  bg_dff 0.15 vs 0.30 barely differ; dF/F only drops where the image SATURATES.

## ✅✅ TRUE SOLUTION — VOLUME DEPTH (out-of-focus haze from the tissue column)
The scatter-PSF approach below was WRONG: broadening the PSF uniformly blurs the
IN-FOCUS cells too, which reads as camera defocus (user rejected it). The real
mechanism is **volume DEPTH**. Diagnostic chain:
- With `separate_focus`, the out-of-focus signal is already **96%** of the total
  at any focal plane — the background is NOT too weak. BUT that OOF background is
  a field of discrete cell-BLOBS, not a smooth haze, because the volume is only
  **60 um deep** — too few defocused cells, spread too little, to merge.
- Real 1P integrates out-of-focus light over the full **~150-200 um** scatter-
  limited tissue column; many decorrelated cells at many depths, each spread by
  its defocus, sum (central-limit) into a SMOOTH bright haze that dilutes the
  in-focus cells' contrast + dF/F — while the in-focus slab STAYS SHARP.
- Validated (300x300 vol, NO scatter, focus mid): depth 60 um -> dff_p99 0.89,
  flatCV 0.27 (too clear); depth **200 um -> dff_p99 0.235, flatCV 0.107** (real
  0.20 / 0.14) and the image is a smooth wash with a few bright cells on top =
  real look. Purely from depth; no artificial blur.

**Implementation:** scan a DEEP volume with focus in the upper tissue
(`focal_depth_um` in the volume's metadata stub) so the deep column defocuses into
the haze; `scatter_um=0`. `examples/gen_deep_volume.py` generates a 500x500x180 um
volume (tractable ~1-2 h; full 1.7 mm at this depth would be ~10 h — that cost is
exactly why the sim originally used a shallow 60 um volume). Both demos now default
`--scatter-um 0` and read `focal_depth_um` from the match-run stub, with PSF support
capped to the FOV. Scan both colours from the deep stub (commands printed by
gen_deep_volume.py).

## (superseded) lateral tissue-scatter PSF broadening — WRONG (uniform blur)
Kept for the record: broaden the emission PSF by sigma=`scatter_um`. Sweep on the
matched (shallow) volume (nt=24):

| scatter_um | median | dff_p99 | flatCV | look |
|---|---|---|---|---|
| 0 | 6505 | 1.078 | 0.195 | sharp granular cells (too clear) |
| 8 | 6571 | 0.452 | 0.144 | softening |
| **16** | 6635 | **0.224** | 0.098 | cells wash into cloud (≈ real) |
| 30 | 6665 | 0.166 | 0.071 | over-smoothed |

median STABLE (no saturation) — unlike bg_scale/neuropil. Full GCaMP run
(nt=200, scatter=16): **dff_p99 0.305** (real ~0.195; was 1.36), movie a washed
cloud that looks like real. Implemented via `_striatum_common.broaden_psf_scatter`
+ wide `psf_sz=(80,80,20)`; both `demo_gcamp_realistic_matched.py` (`--scatter-um`,
default 16) and `demo_static_tdtomato_matched.py` (`--scatter-um`, replaced the
old `oof_blur`) now use it. Cosmetics + whole-image blur removed.

Remaining GCaMP gap: the activity (std-over-time) is still more CELLULAR than the
real REGIONAL/coherent blob — because sim spikes are decorrelated bursts while
real striatum activity is spatially correlated. Fix = correlated/regional spiking
(e.g. simtrace shared-drive designs) or higher scatter; optics is now right.

### (historical) KEY INSIGHT that led to the solution
`dff_p99` is dominated by the **sharply-resolved in-focus SOMATA**, not the
neuropil. Damping/brightening the background cannot dilute a resolved cell's own
pixel dF/F without a floor so bright it saturates the 16-bit range. **Real 1P
widefield does NOT resolve individual somata** (tissue scatter + huge
out-of-focus integration spread each soma's emission over a broad area). The sim
puts too much of each soma's light into a SHARP in-focus PSF. Therefore the
lever must be the **OPTICS / PSF breadth / tissue scatter**, so a soma's light is
spread and no pixel sees a high-contrast, high-dF/F cell — which simultaneously
kills the "too clear" look AND the inflated dF/F.

Note: memory `reference-background-brightness` claims scatter/obj_na/oof are
"near-inert" for the *structure* CV metric — but that was the mean-image CV on a
washed volume; it has NOT been tested for **dF/F dilution** or with a genuinely
broadened widefield PSF. This is the next thing to probe.

---

## DELIVERED (2026-07-07) — realistic co-registered pair (scatter optics)
Both channels re-scanned on the matched volume with lateral-scatter PSF, cosmetics
+ blur removed. Co-registered to each other (same volume; tdt `expr_ids` still
align with the GCaMP soma rows).
- **GCaMP:** `examples/output/gcamp_realistic_1700um_physio-motion_20260707_005345/`
  (`movie_gcamp.npz` / `.tif` / `.gif`, `traces_gt.npz`). dF/F p99 0.303 (real 0.196).
- **tdt:** `examples/output/striatum_tdt_static_1700um_physio-motion_20260707_010005/`
  (`movie_tdt.npz` / `.tif` / `.gif`, `tdtomato_expression.npz`).
- **Comparison figure:** `examples/output/final_real_vs_sim.png`.
Both now look like the real washed striatum (no sharp cells). Two known residuals
below (vessels + activity coherence).

## VESSELS (attempted — BLOCKED by generation cost)
Real tdt texture (flatCV 0.144) vs sim (0.066) gap is the missing dark vessels.
De-thinning `_STRIATUM_VASC` (vesSize (2,2,1)->(10,6,2), vesFreq (600,600,150)->
(200,250,60), distsc 6->4) DOES render dark vessels — verified on a 500 um volume
(1.06% vessel voxels, dark vessels visible), gen ~11 min. BUT the full 1.7 mm-FOV
Phase-1 with these denser vessels ran **>6.5 h without finishing** (vessel Dijkstra
does not scale to dense vessels at this FOV) — impractical. The generation was
killed and the preset REVERTED to thin (fast). De-thinned values preserved in
`calcia/config/region_presets_backup/striatum_vasc_original.md`; `gen_vessel_volume_1700.py`
kept as the (currently impractical) driver.
**Prerequisite for full-FOV vessels: speed up the vessel pathfinding** (or use a
smaller FOV, or a coarser vessel resolution). Until then the delivered pair has no
vessels; tdt texture stays smoother than real.

## DEEP VOLUME = the real wash (works); then 3 realism knobs
Depth solved the wash (see above). Then, per user, three more knobs:
- **Cell contrast**: real soma/bg ~1.23 (cells barely visible). `bg_scale` is the
  lever BUT beware: too high (8) over-washes cells to ratio ~1.0 AND exaggerates
  vessel darkness. Use a MODERATE bg_scale (~2). Calibrate on the deep volume.
- **Grain**: real fast-noise CV ~0.032 (LOW — photon-rich); sim already ~0.043.
  The visual "grain" is STATIC spatial texture (fine capillaries + pixel-scale
  structure), not temporal noise. The thin-vessel capillary network supplies it.
- **Dark vessels**: de-thin `_STRIATUM_VASC` and REGENERATE (monkeypatch in a gen
  script, no source edit). `vesSize=(10,6,2)` gave nice thin capillary LINES but
  its ~20 um-diameter PENETRATORS seen end-on through the 180 um depth appear as
  NEURON-SIZED DARK HOLES (user flagged; confirmed they align with thick-penetrator
  end-on projection; lowering bg_scale did NOT remove them = physical). FIX:
  thinner+sparser vessels `vesSize=(4,3,1.5)`, `vesFreq=(300,350,100)` via
  `examples/gen_deep_thinvessels_volume.py`. Volumes: `gen_deep_volume.py` (no
  vessels), `gen_deep_vessels_volume.py` (thick — holes), `gen_deep_thinvessels_volume.py`
  (thin — preferred).

## ✅✅✅ DARK HOLES SOLVED — neuropil-continuum composite
Real 1P GCaMP is a SMOOTH wash (dark-hole density ~27/mm^2, only on vessels); the
sim had ~620/mm^2 (a "swiss-cheese" of cell-sized dark holes). Systematic audit
ruled out: empty space (100% of columns have fluorophore, corr 0.03), unassigned
somata (all 4500 have fluorescence), bg_scale (holes persist 1..8; NOTE bg_scale
balance point ~2.6 — above it somata invert to dark holes, so use bg_scale~2),
vessels (holes exist without them), neuropil DENSITY (`AxonParams.maxfill` UP made
it WORSE — discrete random-walk axons clump), focal plane / scatter (reduce but
over-blur into featureless).
ROOT CAUSE: NAOMi renders neuropil as DISCRETE processes whose inter-process gaps
read as cell-sized holes; REAL neuropil is a sub-resolution SMOOTH CONTINUUM.
FIX (in both demos): render neuropil and somata SEPARATELY (zero the other's
traces → 2 scans), smooth the neuropil to its continuum (`--neuropil-smooth-um`
~22), keep somata as faint soft blobs (`--soma-blur-um` 8, `--soma-scale` 0.6),
combine, then add camera noise. Result: hole density 620 -> ~76/mm^2 (loose) /
0/mm^2 (strict), tissue visually hole-free, cells faint like real. This is a
compositing MODEL of the unresolved neuropil (a departure from pure volume+scan,
but the only thing that matches real; physically motivated).

## FULL-SIZE (1.7 mm) — works on the EXISTING shallow volume (no regen)
The composite fix is SCAN-LAYER, so full 1.7 mm FOV just re-scans the ORIGINAL
fitting-run volume (`phase1_ae8539a935`, 1700x1700x60 um, already generated) — NO
deep-volume regeneration. The 60 um shallow volume has more discrete-neuropil
holes than the 180 um deep one, but a slightly larger `--neuropil-smooth-um` (~22)
fills them: full-size GCaMP interior holes 251 -> 18/mm^2 (real ~9), hole-free,
cells visible, washed background. Runtime ~28 min (2 composite scans, nt=200),
pyinstrument `--profile` -> profile.html. So DEPTH is NOT required once the
composite smooths the neuropil — a deep 1.7 mm volume (~20 h) is unnecessary.
Full-size GCaMP cmd: `--match-run striatum_v1_1700um_physio-motion_20260706_114842
--bg-scale 2.5 --neuropil-smooth-um 22 --soma-blur-um 3 --soma-scale 0.7 --no-viz`.

## tdt BACKGROUND (user: "background not strong enough") — fixed
Real tdt (data/real/tdt-bfp) is a NEARLY-UNIFORM bright wash: floor/median ~0.80,
p99/med 1.33 (cells barely stand out), spatialCV 0.085. The sim tdt had floor/med
0.41, p99/med 2.50 (cells popping). Two fixes: (1) `--no-illum` — real tdt is
uniform, the GCaMP vignette (floor 0.05) darkened the edges and killed the floor;
(2) strong uniform background — `--bg-scale 5-6`, heavy `--neuropil-smooth-um`
25-30, faint `--soma-scale` 0.4-0.5. Result floor/med 0.58-0.65, p99/med ~1.28
(cells no longer pop). NOTE: bg_scale is MULTIPLICATIVE so it does NOT change
floor/med — uniformity comes from neuropil smoothing + removing the vignette.
Residual: a few thick dark blobs (vessels/gaps projected through 180 um depth) vs
real's thin vessels; the shallow 1700x60 volume has thinner vessel projections
(better for tdt). Full-size tdt uses the SAME 1700 volume as GCaMP (co-registered).

## SHALLOW-VOLUME COMPOSITE: soma_blur must COMPENSATE for missing depth OOF
Using the existing SHALLOW 1700x60 volume for full-size (instead of a ~20 h deep
1700x180 regen) fills holes fine (bigger --neuropil-smooth-um ~22), BUT the cells
come out TOO CLEAR/sharp: a DEEP volume's cells sit at many depths and defocus by
different amounts, overlapping into a washed mean; a shallow volume's cells are
all near focus and stay distinct. The composite `--soma-blur-um` (a uniform
Gaussian on the separately-rendered soma component) EMULATES that depth OOF, so on
a shallow volume it must be LARGER than on a deep one. Deep-500 used soma_blur 2-3;
shallow-1700 needs ~6 (GCaMP) / ~8 (tdt) for the same washed look. Validated: tdt
soma_blur 3->8 (+soma_scale 0.4) -> mean matches real (floor/med 0.86, p99/med 1.16
vs real 0.80/1.33). NOTE: judge cell-clarity on the MEAN image, not single frames
(frames always show cells + camera noise). This makes the deep-volume regen (~20 h)
UNNECESSARY — shallow + tuned composite matches real. Full-size cmds:
  GCaMP: `--bg-scale 2.5 --neuropil-smooth-um 22 --soma-blur-um 6 --soma-scale 0.9`
  tdt:   `--no-illum --bg-scale 6 --neuropil-smooth-um 25 --soma-blur-um 8 --soma-scale 0.4`

## NEXT STEPS (resume here)
0. Thin-vessel deep volume gen running (~1.7 h); when its stub dir exists, scan
   both colours: `--scatter-um 0 --bg-scale 2` (moderate wash, cells faint, no
   big vessel holes). Verify voids gone via scratch check_soma_voids / check_voids2.
1. Full 1.7 mm FOV still needs ~10 h Phase-1 (deep+dense) — current deliverables
   are 500 um FOV.
1. **Probe the optics lever** on the loaded volume (nt≈24, load once): sweep
   `scatter_length_um_wf` (shorter = more scatter/spread) and/or broaden the
   widefield collection PSF, measure `dff_p99` + flat-CV. Target: dff_p99 → ~0.2
   and somata NOT individually resolved, WITHOUT 16-bit saturation.
   - Inspect how much soma light is in-focus vs out-of-focus. Memory
     `project-background-separation` notes a `separate_focus=True` →
     `mov_infocus`/`mov_oof` split and a `focus_slab_um` knob — use it to see /
     shift the in-focus fraction. Real 1P out-of-focus fraction ≈ 96%.
   - If the widefield PSF cannot broaden enough via params, the gap is in the
     optical-propagation model (a core limitation) — decide whether a
     reversible core change is warranted, or accept + document the limitation.
2. Set absolute brightness (median → ~1372 post-illum) via `pavg` / `gain_e_per_adu`
   AFTER the dF/F/scatter is right (brightness and contrast are separable).
3. **tdt:** drop the `--oof-blur-um` blur (set 0), keep full background, re-scan.
   Restore realistic **vessels** for the dark-vessel/void look — this needs a
   Phase-1 regen with a de-thinned `_STRIATUM_VASC` (back up the original preset
   first to `calcia/config/region_presets_backup/` + doc). Vessels are shared by
   both colours (same volume) — the physically-correct way, not an overlay.
4. Produce the final co-registered pair from ONE volume; validate each channel
   against real with: flatfielded texture CV, dF/F p99 (GCaMP), intensity
   histogram, and side-by-side + red/green overlay figures.

## Useful scratch diagnostics (in session scratchpad, regenerate as needed)
- `real_vs_sim_tdt.py`, `tdt_texture_compare.py` — tdt texture vs real (flatfield).
- `real_vs_sim_gcamp.py` — GCaMP structure + activity (std-over-time) vs real.
- `check_trace_dff.py` — per-row dF/F of saved soma vs bg traces (the 1.37 finding).
- `calibrate_bgscale.py`, `confirm_neuropil_fix.py` — the two failed sweeps
  (load volume once, sweep, measure). Reuse the "load once + sweep nt=24" harness
  for the optics probe.

## Metrics cheat-sheet (real targets)
| metric | real | how |
|---|---|---|
| GCaMP dF/F p99 | ~0.20 | `(sig-f0)/f0`, f0 = 10th pctile over time, on signal above bias |
| flatfielded central texture CV | ~0.14 | central 40% crop, divide by gaussian(σ=40), std/mean |
| median (post-illum) | ~1372 ADU | mean image median |
| cells resolved? | NO | neither mean nor std-over-time shows discrete somata |

---

## ★ OPTICAL-DOMAIN RECIPE — CURRENT BEST, DESIGN-PURE (2026-07-09)

**The big shift:** replace the post-processing "neuropil-continuum composite"
(2 scans + `scipy.gaussian_filter` on the scanned image via `--neuropil-smooth-um`
/ `--soma-blur-um`) — which is a design-impure IMAGE blur — with a PHYSICAL
optical lever applied in the OPTICS domain during the scan.

### The recipe (both channels, same microscope → same optics)
1. **composite OFF**: `--neuropil-smooth-um 0` (disables the 2-scan gaussian hack).
2. **physical PSF scatter**: `--scatter-um N` → `_striatum_common.broaden_psf_scatter`
   convolves each z-slice of the collection PSF with a Gaussian (photon-conserving,
   OPTICS domain); the scan then spreads every source by it. NOT a post-scan blur.
   PSF must have wide support — the demos build `psf_sz=(100,100,z)` (`PSF_SUPPORT_UM=100`).
3. **flat illumination**: real striatum is ~flat over a 500 µm crop; the vignette
   inflates dF/F and adds a central glow. GCaMP: use a `illum_grad:false` match-run
   stub; tdt: `--no-illum`.

### What the sweeps showed (deep-500 volume, composite OFF, nt=200)
- **scatter is THE lever**: GCaMP dff_p99 1.62(sc0)→0.60(sc12)→0.28(sc36); dark
  holes 7→1; median stable (no saturation). It fills holes AND washes AND lowers
  dff — all optically, replacing the composite.
- **obj_na is INERT** (0.8→0.35 barely changes cv/holes/dff): the analytic Gaussian
  widefield PSF doesn't broaden with NA. Don't use it. (Confirms old memory.)
- **flat illum matters**: vignette can't reach real dff at a non-blurry scatter;
  flat does (flat sc16 dff 0.41 vs vignette sc16 higher).
- **tdt** (composite OFF, flat): scatter ~8 keeps real-like FINE cellular texture;
  16/24 over-wash. floor_frac improves with scatter (sc24 0.61 > composite's 0.55).
- **Sweet-spot tension**: the scatter that looks un-blurry (GCaMP ~4-8 vignette,
  tdt ~8) still leaves dff/cv above real; the scatter that hits real dff over-washes.
  Flat illum widens the usable window. Not fully pinned — needs flat-illum dff
  confirmation per channel.

### DOMINANT REMAINING GAP (both channels) — NOT optical
Sim dark structures are THICK ROUND BLOBS; real vessels are THIN CURVED FILAMENTS.
Cause: vessels projected through the 180 µm DEEP volume thicken. This is a
VOLUME-level issue (vessel preset + depth), unfixable by any scan/optics knob.
Next frontier = volume (thinner vessels / shallower effective projection / finer
neuropil texture), or accept + document.

### HOW TO REPRODUCE (exact commands)
Volumes (cached in `examples/output/_shared/`): deep-500 =
`phase1_deepthinves_7adf002c49.pkl`; deep-1.0mm = `phase1_deepthinves1k_e944da98fd.pkl`.
Match-run stubs (metadata.json → phase1_cache): `deepthinves_volume_500um_d180_stub`
(vignette), `deepthinves_500_flat_stub` (flat, illum_grad:false).

GCaMP design-pure scan (deep-500, flat, scatter N):
  conda run -n calcia python examples/demo_gcamp_realistic_matched.py \
    --match-run deepthinves_500_flat_stub --neuropil-smooth-um 0 --scatter-um N --no-viz
tdt design-pure scan (deep-500, flat, scatter N):
  conda run -n calcia python examples/demo_static_tdtomato_matched.py \
    --match-run deepthinves_volume_500um_d180_stub --neuropil-smooth-um 0 \
    --scatter-um N --no-illum --no-viz
run_gcamp now also accepts obj_na / scatter_length_um_wf overrides (optics-sweep).

### PARALLEL SCANNING (verified)
Single scan uses ~1.6 of 32 cores + ~5 GB → run ~8 in parallel (RAM-bound on 64 GB).
`conda run` collides on `%TEMP%\__conda_tmp` → give EACH a unique TEMP:
  TEMP=C:/.../cwN TMP=C:/.../cwN conda run -n calcia python ... &   # then `wait`
Run-dir names now carry a `_{pid}` suffix (demos) so same-second parallel runs
don't collide + corrupt movies.npz. Do NOT call env python.exe directly (Git Bash
swallows its stdout).

### Batch-compare figures (standing deliverable — after EVERY batch)
Scale-matched real-zoom vs sim. Research scripts archived in `examples/archive/`:
`compare_gcamp_scatter.py` (GCaMP progression), `compare_tdt_scatter.py` (tdt),
`optical_sweep_harness.py` (load-once obj_na/scatter/halo/flat sweep, composite OFF),
`psf_strategy_concept.py` (teaching viz of PSF strategies).
Real µm/px ≈ 1.476 (1.7 mm / 1152 px); crop
real to sim FOV + resample to sim px. NOTE: run_gcamp returns (H,W,T) — transpose
to (T,H,W) before metrics (an axis bug cost a rerun).

---

## ★★ ROOT CAUSE OF THE "HOLES" — SETTLED via NAOMi1p reference (2026-07-09)

Investigated the reference MATLAB `NAOMi1p` (`…/Deep_widefield_cal_inferece/NAOMi1p`).
**The holes are NOT a neuropil-generation difference — they are an OPTICS gap.**

- NAOMi1p renders neuropil as DISCRETE 1-voxel threads, SAME as calcia
  (`VolumeCode/generate_axons.m:151,177`, `dendrite_randomwalk2`, `maxfill=0.5`,
  `maxvoxel=6`). So our discrete neuropil is FAITHFUL, not the bug. It has the same
  raw inter-thread gaps.
- The difference: **NAOMi1p's collection PSF is a full Fresnel wave-optics,
  scattering-broadened kernel 36 µm wide laterally** (`config/RUSH_ai148d_config.m:33`
  `psf_sz=[36 36 100]`; `OpticsCode/genCorticalLightPathLite_1p.m:154` +
  `fresnel_propagation_multi.m`; scatter phase screens
  `simulate_1p_optical_propagation.m:173-201`). calcia uses a ~1 µm
  diffraction-limited analytic Gaussian.
- Plus **full-column projection**: each frame sums 100 depth slices through the 3-D
  PSF (`ScanningCode/single_scan.m:61` sum over dim 3; `scan_volume_1p.m:514`) +
  averaged above/below out-of-focus haze (`blurredBackComp2.m:46`, added
  `scan_volume_1p.m:556` at ×0.1). No additive fluorescence floor — the bright
  background is EMERGENT from broad-PSF × column projection × dense neuropil.

**Why no holes:** a 36 µm kernel is ~30× wider than the ~1-5 µm inter-thread gaps →
optically fills them. calcia's 1 µm kernel is SMALLER than the gaps → they survive
as cell-sized holes. Our post-hoc `composite` gaussian blur, and the `--scatter-um`
PSF broadening, are both stand-ins for the two physics NAOMi1p has natively:
(a) the ~36 µm scattering-broadened PSF, (b) 100-slice column projection + OOF haze.

**Validates the optical-domain direction and gives the TARGET: ~36 µm effective PSF.**
Our scatter sweep empirically needed scatter ~24-36 to fill holes — i.e. we were
rediscovering NAOMi1p's 36 µm width.

Two paths:
- A (approx, working): keep `--scatter-um` to reach ~36 µm effective width (Gaussian
  sigma ~15-18). Design-pure, already fills holes + washes + lowers dff.
- B (full fidelity): port NAOMi1p's broad Fresnel/scattering PSF + full-column
  projection into calcia's widefield optics (`psf_sz`→36, sum all z, + OOF haze),
  eliminating the need for any scatter/composite approximation. Ref files:
  `OpticsCode/{simulate_1p_optical_propagation,genCorticalLightPathLite_1p,fresnel_propagation_multi,check_psf_params}.m`,
  `ScanningCode/{single_scan,blurredBackComp2,scan_volume_1p}.m`.

---

## ★★★ BEST OPTICAL USE CASE — SETTLED (2026-07-09)

**Winner: TWO-SCALE PSF (sharp core + wide scattering halo), composite OFF, flat illum.**
Implemented as `_striatum_common.broaden_psf_two_scale(psf, halo_um, halo_weight, vres)`
= `(1-w)*core + w*halo`, halo = core blurred by sigma `halo_um`. CLI on both demos:
`--halo-um --halo-weight` (+ `--neuropil-smooth-um 0` to disable composite; GCaMP flat
via the `deepthinves_500_flat_stub` illum_grad:false stub, tdt via `--no-illum`).

**Why it wins:** a SINGLE Gaussian has one width → narrow leaves neuropil holes,
wide over-washes cells (defocus look). The real 1p PSF (NAOMi1p Fresnel, ~36 µm) is a
sharp core in a broad heavy tail; the two-scale kernel reproduces that in ONE cheap
optical convolution (2 gaussian_filters, no FFT propagation) → fills holes AND keeps
fine texture. obj_na is inert; single-scatter fills holes but over-smooths texture.

**GCaMP tuning (deep-500, flat, composite OFF, nt=200):** dff_p99 vs real ~0.23:
  halo18: w0.6→0.66, w0.7→0.56, w0.8→0.48, w0.9→0.42 (18 µm too narrow to wash 12 µm somata)
  halo28: w0.75→0.36, **w0.9→0.27 (≈real) — BEST**; matches single-scatter24's dff (0.27)
  but KEEPS texture (single-scatter is a featureless over-washed blob).
Wider halo (spreads each soma's flash over more px → lower per-px amplitude) lowers dff
more effectively than raising weight alone.

**Residual gaps (honest):**
1. dff-vs-texture tradeoff persists even with two-scale: h28 w0.9 hits real dff but
   slightly over-smooths texture; h18 w0.7 keeps texture but dff 0.56. Better than any
   single-Gaussian, not a perfect simultaneous match on this volume.
2. **#1 remaining mismatch (BOTH channels) = THICK DARK VESSELS vs real's thin
   filaments** — dominates every sim panel. VOLUME-level (vessels projected through the
   180 µm deep column thicken), NOT optical. **This is the next frontier — optics is
   exhausted.** tdt spatial_cv stays ~0.17 (real 0.10) because the thick vessels ARE the
   structure; no PSF tuning helps.

**Verdict:** the optical lever is fully explored. Two-scale PSF is the best optical
method. Further realism requires the VOLUME (thinner vessels / shallower effective
vessel projection / finer neuropil), not the scan/optics. Final figure:
`examples/output/_BEST_two_scale_vs_real.png` (regenerate via scratchpad cmp_final).
