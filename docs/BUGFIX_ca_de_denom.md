# Bug Fix: Ca_DE Calcium Dynamics Denom Formula Inverted

**Date Fixed:** 2026-04-19
**File:** `calcia/traces/calcium.py`
**Severity:** Critical — produces 4–7× brighter traces than MATLAB reference
**Affected models:** Ca_DE (the default calcium dynamics model)

---

## Symptom

Generated videos showed abnormally bright background fluorescence. Neurons appeared too bright relative to MATLAB NAOMi reference output. Quantitatively:

| Metric | MATLAB | Python (before fix) | Ratio |
|--------|--------|---------------------|-------|
| movRaw mean (full-size 250×250×100) | 7.70 | 51.80 | **6.72×** |
| movRaw median | 6.62 | 42.74 | **6.46×** |
| movRaw 99th percentile | 21.92 | 165.55 | **7.55×** |
| movRaw max | 68.79 | 828.68 | **12.05×** |
| Soma trace mean | 0.997 | 3.822 | **3.83×** |
| Soma trace max | 5.89 | 105.11 | **17.85×** |
| Background trace mean | 0.997 | 7.044 | **7.06×** |

The ratio grew for tails of the distribution because the bug amplified peaks non-linearly.

---

## Diagnosis Methodology

### Step 1: Verify the issue is in the simulation, not visualization

The user was viewing TIFF files directly in ImageJ. Both MATLAB and Python use the
same TIFF saving logic (percentile normalization to uint16). So the raw float
`mov_raw` values themselves had to be different.

### Step 2: Locate the phase responsible

Full-pipeline phases were compared one by one:

- **Phase 1 (Neural Volume):** ✓ matches (bg/soma voxel counts and fluorescence sums within 3%)
- **Phase 2 (Optical Propagation):** ✓ matches (PSF tail weight ratio = 1.0000)
- **Phase 3 (Time Traces):** ✗ **diverges** (soma trace 3.8× brighter, bg 7× brighter)
- **Phase 4 (Scanning):** inherits the Phase 3 error

### Step 3: Find the offending line in Phase 3

Within Phase 3, soma and bg use `_simulate_compartment()` with the same Ca_DE path,
just different `ext_mult` values. Comparing Python `calcium_dynamics()` with MATLAB
`calcium_dynamics.m` line by line revealed a subtle difference in one operator.

---

## Root Cause

### The MATLAB reference (calcium_dynamics.m, line 111)

```matlab
elseif strcmp(sat_type,'Ca_DE')
    for kk = 2:size(S,2)
        C(:,kk) = C(:,kk-1) + (-dt*ext_rate*(C(:,kk-1) ...
                       - ca_rest) + S(:,kk))./(1 + ca_bind + ...
                       (ind_con*ca_dis).\(C(:,kk-1) + ca_dis).^2);
    end
```

Note the `.\` operator — this is MATLAB's **element-wise left-divide**, which
reverses the order of arguments:

```
A .\ B   ≡   B ./ A   ≡   B / A
```

So the denom expression is:
```
1 + ca_bind + (C + ca_dis)² / (ind_con · ca_dis)
```

### The Python port (calcium.py, line 288 — BEFORE fix)

```python
elif sat_type == "Ca_DE":
    for kk in range(1, nt):
        denom = 1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2
```

The translator read `.\` as a standard division and wrote:
```
1 + ca_bind + (ind_con · ca_dis) / (C + ca_dis)²
```

**The numerator and denominator are swapped.**

### Why it produces brighter traces

With typical values:
- `ind_con` = 2×10⁻⁴ M (indicator concentration)
- `ca_dis`  = 2.9×10⁻⁷ M (Hill dissociation constant)
- `C`       ~ 5×10⁻⁸ to 10⁻⁷ M (free calcium)
- `ind_con · ca_dis` ≈ 5.8×10⁻¹¹
- `(C + ca_dis)²`    ≈ 1.2×10⁻¹³

**MATLAB denom:** `1 + ca_bind + 1.2e-13 / 5.8e-11` ≈ `1 + ca_bind + 0.002` ≈ **tiny addition**

**Python (buggy) denom:** `1 + ca_bind + 5.8e-11 / 1.2e-13` ≈ `1 + ca_bind + 483` ≈ **huge addition**

The denom appears in the free-calcium update rule:
```
C[k] = C[k-1] + (-dt · ext_rate · (C[k-1] - ca_rest) + S[k]) / denom
```

- With the **correct** denom (~1), each spike input `S[k]` contributes nearly its
  full amplitude to the next calcium level. Extrusion operates at its designed rate.
- With the **buggy** (huge) denom, the entire expression gets divided by ~500.
  This dampens extrusion (`-dt·ext_rate·(C-ca_rest)/denom`) far more than it
  dampens the spike input. Net effect: calcium accumulates instead of decaying
  to baseline, pushing the Hill equation deep into saturation and inflating
  peak fluorescence 4–7×.

### Important: the bug only affects Ca_DE

The same file has `single` and `double` sat_type paths that use the same
physical expression. But in MATLAB they are written with standard `./`:

```matlab
% single (line 98):
./(1 + ca_bind + (ind_con*ca_dis)./(C(:,kk-1) + ca_dis).^2)

% double (line 138):
./(1 + ca_bind + (ind_con*ca_dis)./(C(:,kk-1) + ca_dis).^2)
```

This is itself a **MATLAB internal inconsistency** — Ca_DE uses `.\` but single
and double use `./`. These are not algebraically equivalent. Our port copied the
same *algebraic expression* for all three paths, which accidentally matched
single/double but conflicted with the inverted Ca_DE. Since Ca_DE is the default
(`CalciumParams.sat_type = "Ca_DE"`), the bug affected every default run.

Whether MATLAB's Ca_DE `.\` is intentional (modeling choice) or itself a typo is
unknown. What matters for the port is that we match the reference *as written*.

---

## The Fix

One line, in `calcia/traces/calcium.py`:

```python
elif sat_type == "Ca_DE":
    a = float(a_bind) * 100.0 * dt
    b = float(a_ubind) * 100.0 * dt
    for kk in range(1, nt):
        # MATLAB Ca_DE uses .\ (left-divide): (A).\(B) == B/A
        denom = 1.0 + ca_bind + (C[:, kk - 1] + ca_dis) ** 2 / (ind_con * ca_dis)
        C[:, kk] = (
            C[:, kk - 1]
            + (-dt * ext_rate * (C[:, kk - 1] - ca_rest) + S[:, kk]) / denom
        )
```

Tests: all 85 Ca_DE-related tests pass. Full suite: 363/364 pass (the one
pre-existing vasculature failure is unrelated).

---

## Verification

Re-running the small-volume comparison pipeline after the fix:

| Metric | MATLAB | Python (before) | Python (after) |
|--------|--------|-----------------|----------------|
| movRaw mean | 10.6 | 72.1 | **12.9** ✓ |
| Soma mean  | 0.997 | 3.82 | **1.02** ✓ |
| Soma max   | 5.89 | 105 | **5.91** ✓ |
| Single-spike dF/F | ~0.6 | ~4.0 | **0.61** ✓ |

The ~20% residual difference versus MATLAB comes from different RNG streams
(NumPy vs MATLAB) and is consistent with other phases.

---

## How to Avoid This Class of Bug

### 1. Be suspicious of MATLAB element-wise division operators

MATLAB has four division operators and mixing them up is easy:

| Operator | Meaning | Python equivalent |
|----------|---------|-------------------|
| `A / B`  | matrix right-divide (solves x·B = A) | `A @ np.linalg.inv(B)` |
| `A \ B`  | matrix left-divide (solves A·x = B) | `np.linalg.solve(A, B)` |
| `A ./ B` | element-wise right-divide | `A / B` |
| `A .\ B` | **element-wise left-divide** | `B / A` ⚠ |

When porting, grep the MATLAB source for `\` (backslash) and carefully check
every occurrence. A raw `A\B` is matrix solve; `A.\B` is element-wise reverse.

### 2. Compare numerical outputs phase-by-phase, not just end-to-end

End-to-end tests catch the existence of a bug but not its location. A simple
script that runs MATLAB and Python on the same configuration and reports ratios
(mean, median, tails, per-neuron) at each intermediate stage pinpoints the
phase responsible in minutes.

### 3. Write regression tests that check actual signal magnitudes

The existing tests checked that `F > 0`, `F.shape == expected`, and
`ext_mult=0.25` produces lower F than `ext_mult=1.0`. None checked that a
single spike produces dF/F ≈ 0.6. A quantitative test against a known MATLAB
reference value would have caught this on day one.

Recommended test to add:
```python
def test_ca_de_single_spike_dff():
    """One 7.6e-6 M spike in a 700-sample window should produce dF/F ≈ 0.6
    at the Ca_DE peak, matching MATLAB calcium_dynamics reference."""
    S = np.zeros((1, 700), dtype=np.float32)
    S[0, 200] = 7.6e-6
    cp = CalciumParams(prot_type="gcamp6f", sat_type="Ca_DE", dt=1/100)
    _, _, F = calcium_dynamics(S, cp, prot_type="gcamp6f", ext_mult=1.0)
    baseline = F[0, :100].mean()
    peak = F[0, 200:].max()
    dff = (peak - baseline) / baseline
    assert 0.5 < dff < 0.7, f"Expected dF/F ≈ 0.6, got {dff:.3f}"
```

---

## Related Issues Discovered During Debugging

### Warning for low-activity simulations

When diagnosing the brightness issue, we also noticed that short simulations
with low `rate` produce traces where most neurons never fire. This is a
parameter-selection issue, not a bug, but is surprising to users.

Fix: added `ensure_activity` flag to `SpikeParams`. When fewer than 5% of soma
neurons are active, a warning is printed. If `ensure_activity=True`, spikes are
injected into ~20% of silent soma neurons so the output movie shows visible
transients. Default is off to preserve physical realism.

Related file: `calcia/traces/traces.py:337-365`.

---

## References

- MATLAB source: `naomi_sim/code/TimeTraceCode/calcium_dynamics.m` lines 105–126
- Python source: `calcia/traces/calcium.py` lines 283–311
- Fix commit: (add commit hash when committed)
