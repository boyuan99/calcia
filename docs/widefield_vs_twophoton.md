# Widefield (Single-Photon) vs Two-Photon Imaging: Differences in the Simulation Pipeline

This document identifies every location where the NAOMi/calcia simulation pipeline
embeds two-photon–specific physics, and describes what must change (or be added as
an alternative path) to support widefield single-photon fluorescence imaging.

---

## Overview: Which Phases Are Affected?

| Phase | Description | Two-Photon Specific? | Reason |
|-------|-------------|---------------------|--------|
| Phase 1 | Volume generation | **No** | Neural morphology is optics-independent |
| Phase 2 | Optical propagation | **Yes** | PSF model, wavelength, masks |
| Phase 3 | Time traces | **No** | Calcium dynamics are optics-independent |
| Phase 4 | Scanning / imaging | **Yes** | Signal model, image formation, noise model |

---

## Phase 2: Optical Propagation (`calcia/optics/`)

### 2.1 PSF Generation — Intensity Squaring

**Current code:** `calcia/optics/psf.py`, lines 119-124

```python
intensity = np.exp(-2.0 * np.pi * nidx * (xg2**2 + yg**2) / denom) / (
    1.0 + (zg2 / zr) ** 2
)
# Two-photon: square the intensity
psf = (intensity ** 2).astype(np.float32)
```

**Physics:**
- Two-photon fluorescence requires **simultaneous absorption of two photons**,
  so the excitation rate is proportional to **I²** (intensity squared).
- This squaring is what gives two-photon its inherent optical sectioning:
  the squared Gaussian falls off much faster in z than the unsquared version.

**Widefield difference:**
- Single-photon excitation is **linear** in intensity: fluorescence ∝ I.
- The PSF should NOT be squared: `psf = intensity` (not `intensity ** 2`).
- Furthermore, in widefield mode, the excitation is **uniform** across the
  field of view (Köhler illumination), so the "excitation PSF" is effectively
  constant. The relevant PSF is the **detection/emission PSF** — the image of
  a point source formed by the objective onto the camera.

**Lateral and axial resolution comparison (Gaussian approximation):**

| Parameter | Two-Photon (λ=920nm, NA=0.8) | Widefield (λ=520nm emission, NA=0.8) |
|-----------|------------------------------|--------------------------------------|
| σ_xy | 0.21 × 0.92 / 0.8 ≈ 0.24 µm | 0.21 × 0.52 / 0.8 ≈ 0.14 µm |
| σ_z | much tighter (due to I²) | 0.88 × 0.52 / (n - √(n²-NA²)) ≈ broader |
| Sectioning | intrinsic (I² falloff) | **none** (all z contributes) |

Note: although the shorter emission wavelength gives widefield a smaller lateral
PSF, the lack of optical sectioning means z-resolution is effectively infinite
— every plane contributes to the final image.

### 2.2 Wavelength

**Current code:** `calcia/config/params.py`
- `PsfParams.lambda_um = 0.92` (line 285) — 920 nm, near-infrared
- `TpmParams.lambda_um = 0.92` (line 352)

**Physics:**
- Two-photon uses ~2× the single-photon excitation wavelength (GFP: 488 nm × 2 ≈ 920 nm)
  so that two photons together provide the energy of one UV/visible photon.
- Longer wavelength also means deeper tissue penetration (less scattering).

**Widefield difference:**
- Excitation wavelength: ~488 nm (for GFP).
- Emission wavelength: ~520 nm (for GFP). The detection PSF is computed at
  the **emission** wavelength.
- For PSF computation, use λ_emission (not λ_excitation), because in widefield
  there is no excitation PSF structure — only the detection PSF matters.

### 2.3 Illumination and Collection Masks

**Current code:** `calcia/optics/mask.py`

- `compute_illumination_mask()` (lines 136-212): models Beer-Lambert scattering
  attenuation of the excitation beam as it propagates into tissue.
- `compute_collection_mask()` (lines 51-129): models hemoglobin absorption on
  the collection (emission) path.

**Physics:**
- These masks account for depth-dependent signal attenuation due to tissue
  scattering and absorption.
- The scattering model uses parameters tuned for near-infrared (920 nm).

**Widefield difference:**
- Scattering is **stronger** at shorter wavelengths (488 nm excitation).
  The scatter parameters (`scatter_sz`, `scatter_wt`) would need different values.
- However, widefield imaging typically operates at shallower depths (< 100 µm)
  where scattering is less of a concern.
- For a first implementation, these masks could be kept similar or simplified
  (uniform illumination assumption).
- Collection mask logic is similar (emission path attenuation), but wavelength-
  dependent absorption coefficients differ.

### 2.4 PSF Tails (Out-of-Focus Energy)

**Current code:** `calcia/optics/psf.py`, lines 156-209 (`compute_psf_tails()`)

**Physics:**
- In two-photon, the PSF² falls off rapidly in z, so out-of-focus contributions
  are small and handled via pre-computed "tail" weights.

**Widefield difference:**
- In widefield, the PSF does NOT fall off in z (energy is conserved across
  z-planes — it just spreads laterally). Out-of-focus planes contribute
  significant diffuse background.
- The tail concept is replaced by **full volumetric contribution**: every z-plane
  must be convolved with the PSF at its defocus distance and summed.
- This is the single biggest computational difference.

---

## Phase 4: Scanning / Imaging (`calcia/scanning/`)

### 4.1 Image Formation Model

**Current code:** `calcia/scanning/scanning.py` + `calcia/scanning/convolution.py`

The current pipeline simulates **point scanning**:
1. For each time point, the laser focus is at a specific (x, y) position.
2. The signal at that pixel = PSF-weighted sum of fluorophores near the focus.
3. The image is built pixel-by-pixel as the laser rasters across the FOV.
4. Each pixel in a frame may be acquired at a slightly different time → motion
   artifacts appear as line-by-line distortions.

**Widefield difference — parallel acquisition:**
- The entire FOV is illuminated simultaneously; the camera captures a full 2D
  image in one exposure.
- Image formation is a **2D convolution** (not point-by-point scanning):

```
I(x, y) = Σ_z [ f(x,y,z) ⊛_2D h(x,y, z - z_focus) ]
```

where f is the fluorophore distribution, h is the PSF at defocus (z - z_focus),
and the sum runs over ALL z-planes (no optical sectioning).

- In the Fourier domain (efficient implementation):

```
I(x,y) = Σ_z IFFT2[ FFT2(f[:,:,z]) · FFT2(h[:,:,z-z0]) ]
```

- **No raster scanning** → no line-by-line temporal offset → motion artifacts
  are uniform shifts (or blur) per frame, not scan-line distortions.

### 4.2 Signal Scaling — Power Dependence

**Current code:** `calcia/optics/signal.py`, lines 65-68

```python
ftavg = (
    phi * eta * conc * delta * gp * 8.0 * nidx * pavg ** 2
    / (2.0 * f * tau * math.pi * lambda_m)
)
```

Called from `scanning.py` line 113: `tpm_signal_scale(tpm_params)`.

**Physics (two-photon):**
- Signal ∝ P² (quadratic power dependence)
- Uses two-photon absorption cross-section δ (in Goeppert-Mayer units, 10⁻⁵⁸ m⁴·s/photon)
- Requires pulsed laser parameters: repetition rate f, pulse width τ, coherence factor gp
- Formula derived from time-averaged two-photon excitation rate

**Widefield difference (single-photon):**
- Signal ∝ P (linear power dependence)
- Uses single-photon absorption cross-section σ_abs (in cm²), related to molar
  extinction coefficient ε:

```
σ_abs = 1000 · ln(10) · ε / N_A
```

- Signal formula:

```
F = φ · QE_det · σ_abs · C · I_exc · Ω_collection · T_optics
```

where:
  - φ = quantum yield (same as two-photon, ~0.6 for eGFP)
  - QE_det = camera quantum efficiency (~0.7-0.95 for sCMOS)
  - σ_abs = absorption cross-section (~3.8 × 10⁻¹⁶ cm² for GFP at 488 nm)
  - C = fluorophore concentration
  - I_exc = excitation intensity (photons/cm²/s) — **linear**, not squared
  - Ω_collection = fractional solid angle of collection
  - T_optics = optical path transmission

- **No pulsed laser parameters** needed (CW or quasi-CW illumination).

**Typical fluorophore parameters (single-photon):**

| Fluorophore | λ_exc (nm) | λ_em (nm) | ε (M⁻¹cm⁻¹) | QY |
|-------------|-----------|-----------|-------------|-----|
| EGFP | 488 | 509 | 56,000 | 0.60 |
| GCaMP6s | 488 | 515 | ~30,000 | ~0.6 |
| GCaMP6f | 488 | 515 | ~28,000 | ~0.6 |
| tdTomato | 554 | 581 | 138,000 | 0.69 |
| mCherry | 587 | 610 | 72,000 | 0.22 |

### 4.3 Noise Model — PMT vs Camera

**Current code:** `calcia/scanning/noise.py`, lines 17-70 (`poisson_gauss_noise()`)

```
PMT noise chain:
  photon count ~ Poisson(signal + darkcount)
  → PMT gain multiplication ~ LogNormal(mu, sigma)    [per-photon gain]
  → electronic readout ~ + Normal(mu0, sigma0)         [baseline noise]
```

**Physics (two-photon / PMT):**
- Two-photon uses PMT (photomultiplier tube) detectors because only a single
  spatial point is illuminated at a time — no spatial resolution needed in the
  detector (just total photon count).
- PMT gain introduces multiplicative noise (lognormal model of dynode cascade).
- Typical parameters: mu=100 (gain), sigma=2300 (gain variance), darkcount=0.05.

**Widefield difference — camera (sCMOS/CCD):**

```
Camera noise chain:
  photon count ~ Poisson(QE · signal)                  [shot noise]
  + dark current ~ Poisson(dark_rate · t_exp)           [thermal electrons]
  × pixel gain map (sCMOS: ~N(1, 0.01²) per pixel)     [fixed-pattern noise]
  + read noise ~ Normal(0, σ_read) per pixel            [readout electronics]
  → ADC: clip and digitize to integer ADU               [analog-to-digital]
```

**Typical camera parameter values:**

| Parameter | sCMOS | CCD | EMCCD |
|-----------|-------|-----|-------|
| Read noise (e⁻ rms) | 1-2 | 3-10 | <1 (with EM gain) |
| Dark current (e⁻/px/s) | 0.1-0.5 | 0.001-0.01 | 0.001-0.01 |
| QE (peak) | 70-95% | 60-75% | 90-95% |
| Full well (e⁻) | 30,000-50,000 | 20,000-100,000 | 80,000 |
| Pixel size (µm) | 6.5 | 6-13 | 13-16 |
| Gain (e⁻/ADU) | 0.5-2 | 1-4 | varies |
| Bit depth | 16 | 16 | 14-16 |
| Frame rate | 30-100 fps | 1-10 fps | 30-60 fps |

**Key noise difference:** The PMT lognormal gain noise is replaced by a much
simpler Poisson + read noise model. Camera noise is generally lower and more
well-characterized than PMT noise.

### 4.4 Pixel Bleed

**Current code:** `calcia/scanning/noise.py`, lines 73-122 (`pixel_bleed()`)

**Physics:**
- In raster scanning, charge from one pixel can bleed into the next due to
  electronics bandwidth limitations.
- The bleed is directional (along the scan direction).

**Widefield difference:**
- Camera pixels are physically independent — no directional bleed.
- sCMOS may have very minor crosstalk between adjacent pixels, but it is
  typically negligible and isotropic (not directional).
- This function can be skipped for widefield.

### 4.5 Motion Artifacts

**Current code:** `calcia/scanning/motion.py` + `scanning.py`

**Physics (two-photon):**
- Each line of the raster scan is acquired at a different time.
- If the tissue moves during the frame, different lines see different
  tissue positions → **line-by-line warping** artifacts.

**Widefield difference:**
- The entire frame is captured simultaneously (or within the exposure time).
- Motion during exposure causes **uniform blur** or **uniform shift**, not
  line-dependent warping.
- Motion between frames causes frame-to-frame displacement (rigid shift).
- Implementation: apply a single (dx, dy) shift per frame, optionally with
  motion blur (convolution with motion kernel during exposure).

---

## Summary: Complete List of Differences

| Aspect | Two-Photon (current) | Widefield (to add) | Phase | File |
|--------|---------------------|-------------------|-------|------|
| PSF formula | I² (squared) | I (linear) | 2 | `optics/psf.py` |
| Wavelength | 920 nm (excitation) | 520 nm (emission PSF) | 2 | `config/params.py` |
| Excitation | Point (focused laser) | Uniform (Köhler) | 2,4 | concept |
| Optical sectioning | Intrinsic (I²) | None | 2 | `optics/psf.py` |
| Out-of-focus | Tail weights (small) | Full z-sum (large) | 2,4 | `optics/psf.py`, `scanning/` |
| Signal ∝ power | P² (quadratic) | P (linear) | 4 | `optics/signal.py` |
| Cross-section | δ (GM, 10⁻⁵⁸) | σ_abs (cm²) | 4 | `optics/signal.py` |
| Laser type | Pulsed (fs, MHz) | CW | 4 | `config/params.py` |
| Detector | PMT | Camera (sCMOS/CCD) | 4 | `scanning/noise.py` |
| Noise model | Poisson+LogNormal+Gauss | Poisson+ReadNoise | 4 | `scanning/noise.py` |
| Image formation | Point-by-point scan | 2D convolution | 4 | `scanning/scanning.py` |
| Pixel bleed | Yes (scan direction) | No | 4 | `scanning/noise.py` |
| Motion artifacts | Line-by-line warping | Per-frame shift/blur | 4 | `scanning/motion.py` |
| Tissue scattering | 920 nm params | 488/520 nm params | 2 | `optics/mask.py` |

---

## References

- Born, M. & Wolf, E., "Principles of Optics," 7th ed., Cambridge University Press, 1999
- Gibson, S.F. & Lanni, F., "Experimental test of an analytical model of aberration
  in an oil-immersion objective lens," JOSA A, 9(1):154-166, 1992
- Zhang, B. et al., "Gaussian approximations of fluorescence microscope PSF models,"
  Applied Optics, 46(10):1819-1829, 2007
- Huang, F. et al., "Video-rate nanoscopy using sCMOS camera-specific single-molecule
  localization algorithms," Nature Methods, 2013
- MicroscPSF: https://github.com/MicroscPSF
- microsim: https://github.com/tlambert03/microsim
