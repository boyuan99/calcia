"""
Main time-trace generation pipeline.

Port of MATLAB: TimeTraceCode/generateTimeTraces.m
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from calcia.config.params import CalciumParams, SpikeParams
from calcia.traces.calcium import calcium_dynamics, make_calcium_impulse
from calcia.traces.connectivity import gen_correlated_spike_trains
from calcia.traces.expression import expression_variation
from calcia.traces.spikes import bin_spike_trains, gen_burst_spike_times

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_L_BUFF: int = 500       # steady-state buffer (samples at 100 Hz)
_SPIKE_DT: float = 1.0 / 100.0  # internal simulation rate

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TimeTracesResult:
    """
    Output of :func:`generate_time_traces`.

    Attributes
    ----------
    soma : np.ndarray
        ``K × nt`` float32 somatic fluorescence traces.
    dend : np.ndarray or None
        ``K × nt`` float32 dendritic traces, or ``None`` when
        ``spike_params.dendflag`` is False.
    bg : np.ndarray or None
        ``K × nt`` (axon) or ``N_bg × nt`` (GP neuropil) float32
        background traces, or ``None``.
    spikes : np.ndarray or None
        ``K × nt`` float32 spike-count matrix if ``spikeflag`` is True,
        else ``None``.
    mod_vals : np.ndarray
        Length-``K`` float32 per-cell expression modulation factors.
    params : dict
        ``{'spike_params': SpikeParams, 'cal_params': CalciumParams}``.
    """

    soma: np.ndarray
    dend: Optional[np.ndarray]
    bg: Optional[np.ndarray]
    spikes: Optional[np.ndarray]
    mod_vals: np.ndarray
    params: Dict


# ---------------------------------------------------------------------------
# Helper: resample-rate selection
# ---------------------------------------------------------------------------


def _pick_resample_vals(dt: float) -> Tuple[int, int]:
    """
    Choose up/down-sampling integers for ``scipy.signal.resample_poly``.

    Port of the local ``pickResampleVals`` helper in ``generateTimeTraces.m``.

    Parameters
    ----------
    dt:
        Output frame interval in seconds.

    Returns
    -------
    r1, r2 : int
        ``resample_poly(x, r2, r1)`` will downsample from 100 Hz to ``1/dt`` Hz.
    """
    inv_dt = 1.0 / dt
    diff1 = abs(inv_dt - math.floor(inv_dt))
    diff2 = abs(100.0 * dt - math.floor(100.0 * dt))

    if diff2 < diff1:
        r1, r2 = 100, int(math.floor(inv_dt))
    else:
        r1, r2 = int(math.floor(101.0 * dt)), 1

    return max(1, round(abs(r1))), max(1, round(abs(r2)))


# ---------------------------------------------------------------------------
# Helper: convolve K×nt matrix with 1-D kernel along axis=1
# ---------------------------------------------------------------------------


def _conv_rows(mat: np.ndarray, h: np.ndarray) -> np.ndarray:
    """``fftconvolve(mat, h[np.newaxis,:], mode='same')`` along axis=1."""
    h2d = h[np.newaxis, :].astype(float)
    return fftconvolve(mat.astype(float), h2d, mode="same").astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: resample a K×nt array from 100 Hz to target dt
# ---------------------------------------------------------------------------


def _resample_array(arr: np.ndarray, r1: int, r2: int, buff2: int) -> np.ndarray:
    """
    Resample ``arr`` (K × nt_100Hz) from 100 Hz to ``r2/r1 * 100`` Hz.

    Pads edges to suppress ringing, resamples, then trims.
    """
    buff = 100
    row_min = arr.min(axis=1, keepdims=True)

    padded = np.concatenate(
        [arr[:, :1].repeat(buff, axis=1), arr, arr[:, -1:].repeat(buff, axis=1)],
        axis=1,
    )
    # resample_poly expects (n_samples, ...) — transpose, resample, transpose back
    resampled = resample_poly(padded.T.astype(float), r2, r1).T.astype(np.float32)

    # Trim the buffers
    out = resampled[:, buff2 : resampled.shape[1] - buff2]

    # Clamp below original row minimum (anti-ringing)
    out = np.maximum(out, row_min)
    return out


# ---------------------------------------------------------------------------
# Helper: simulate one compartment's fluorescence
# ---------------------------------------------------------------------------


def _simulate_compartment(
    S_times: np.ndarray,
    dyn_type: str,
    cal_params: CalciumParams,
    prot: str,
    ext_mult: float,
    ar_scale: float,
) -> np.ndarray:
    """
    Return fluorescence traces for one compartment (soma/dend/axon).

    Parameters
    ----------
    S_times:
        ``K × nt`` spike matrix (scaled by 7.6e-6 for ODE paths, or
        lognormal-modulated for AR paths).
    dyn_type:
        One of 'AR1', 'AR2', 'single', 'Ca_DE', 'double'.
    cal_params:
        Calcium dynamics parameters (``sat_type`` is overridden internally).
    prot:
        Protein name for Hill equation.
    ext_mult:
        Extrusion rate multiplier (1.0=soma, 0.5=dend, 0.25=axon for Ca_DE).
    ar_scale:
        Time-scale for AR impulse response (0.9=soma, 0.8=dend/axon).

    Returns
    -------
    np.ndarray
        ``K × nt_long`` float32 fluorescence (buffer not yet removed).
    """
    K = S_times.shape[0]

    if dyn_type == "AR1":
        h_ca = make_calcium_impulse(ar_scale, 1.0 / 100.0)
        h_ca = 0.5 * h_ca / h_ca.max()
        b_cell = np.abs(1.0 + 0.1 * np.random.randn(K, 1)).astype(np.float32)
        scale = float(0.5 + 0.5 * np.random.rand())
        out = 2.5 * _conv_rows(S_times, h_ca) * b_cell * scale + b_cell
        return out

    if dyn_type == "AR2":
        # AR2 call: dt=[1,1] → 2-element impulse evaluated at t=1
        h_ca = make_calcium_impulse(ar_scale, np.array([1, 1]))
        if h_ca.max() > 0:
            h_ca = 0.5 * h_ca / h_ca.max()
        b_cell = np.abs(1.0 + 0.1 * np.random.randn(K, 1)).astype(np.float32)
        scale = float(0.5 + 0.5 * np.random.rand())
        out = 2.5 * _conv_rows(S_times, h_ca) * b_cell * scale + b_cell
        return out

    if dyn_type == "single":
        cp = dataclasses.replace(cal_params, sat_type="single", ext_rate=800.0 * ext_mult)
        # Normalise spike amplitudes
        pos = S_times[S_times > 0]
        if pos.size > 0:
            S_in = 7.6e-6 * (S_times / pos.min())
        else:
            S_in = S_times.copy()
        _, _, F = calcium_dynamics(S_in, cp, prot_type=prot, ext_mult=1.0)
        return F

    if dyn_type == "Ca_DE":
        cp = dataclasses.replace(cal_params, sat_type="Ca_DE")
        _, _, F = calcium_dynamics(S_times, cp, prot_type=prot, ext_mult=ext_mult)
        return F

    if dyn_type == "double":
        cp = dataclasses.replace(cal_params, sat_type="double", ext_rate=800.0 * ext_mult)
        pos = S_times[S_times > 0]
        if pos.size > 0:
            S_in = 7.6e-6 * (S_times / pos.min())
        else:
            S_in = S_times.copy()
        _, _, F = calcium_dynamics(S_in, cp, prot_type=prot, ext_mult=1.0)
        return F

    raise ValueError(
        f"Unknown dyn_type={dyn_type!r}. "
        "Options: 'AR1', 'AR2', 'single', 'Ca_DE', 'double'."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_time_traces(
    spike_params: Optional[SpikeParams] = None,
    cal_params: Optional[CalciumParams] = None,
    s_times: Optional[np.ndarray] = None,
    n_locs: Optional[np.ndarray] = None,
    mod_vals: Optional[np.ndarray] = None,
    *,
    verbose: Optional[int] = None,
) -> TimeTracesResult:
    """
    Generate fluorescence time traces for a simulated neural population.

    Port of ``TimeTraceCode/generateTimeTraces.m``.

    Parameters
    ----------
    spike_params:
        Spike generation and dynamics configuration.  Defaults to
        :class:`~calcia.config.params.SpikeParams` with default values.
    cal_params:
        Calcium dynamics parameters.  Defaults to
        :class:`~calcia.config.params.CalciumParams` using
        ``spike_params.prot`` as the protein type.
    s_times:
        Pre-defined ``K × nt_internal`` spike matrix.  When provided the
        Hawkes/Poisson generation step is skipped.
    n_locs:
        ``K × D`` neuron location matrix, passed to the Hawkes process
        for spatially-aware connectivity.
    mod_vals:
        Pre-computed per-cell modulation factors (length ``K``).  When
        ``None``, expression variation is sampled automatically.
    verbose:
        Override verbosity level (falls back to ``spike_params.verbose``).

    Returns
    -------
    TimeTracesResult
    """
    # ------------------------------------------------------------------
    # Step 0: Parameter defaults
    # ------------------------------------------------------------------
    if spike_params is None:
        spike_params = SpikeParams()
    if cal_params is None:
        cal_params = CalciumParams(prot_type=spike_params.prot.lower().replace("-", ""))

    v = spike_params.verbose if verbose is None else verbose

    K = spike_params.K
    dyn_type = spike_params.dyn_type
    prot = spike_params.prot
    dendflag = spike_params.dendflag
    axonflag = spike_params.axonflag
    spikeflag = spike_params.spikeflag

    if spike_params.N_bg > 0 and axonflag:
        raise ValueError(
            "background must be either axons (axonflag=True) "
            "or GP components (N_bg > 0), not both."
        )

    # Internal simulation length at 100 Hz
    if s_times is not None:
        nt_internal = s_times.shape[1]
        n_desired = nt_internal  # user controls length via s_times
    else:
        nt_internal = math.ceil(spike_params.nt * 100.0 * spike_params.dt)
        n_desired = spike_params.nt

    # CalciumParams always runs at 100 Hz internally
    cp_internal = dataclasses.replace(cal_params, dt=_SPIKE_DT)

    if v >= 1:
        print("=" * 60)
        print("generate_time_traces  (Phase 3: time traces)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate spike trains
    # ------------------------------------------------------------------
    if v >= 1:
        print(f"\n[1/5] Generating spike trains  ({spike_params.smod_flag})")

    S_bg_hawkes: Optional[np.ndarray] = None

    if s_times is not None:
        S_times = np.asarray(s_times, dtype=np.float32)
    else:
        sp_internal = dataclasses.replace(
            spike_params,
            dt=_SPIKE_DT,
            nt=nt_internal,
        )
        if spike_params.smod_flag == "hawkes":
            cell_dict = gen_correlated_spike_trains(sp_internal, n_locs=n_locs)
            S_times = cell_dict["soma"].astype(np.float32)
            S_bg_hawkes = cell_dict["bg"].astype(np.float32)
        else:
            S_times = gen_burst_spike_times(sp_internal)
            S_bg_hawkes = None

    # ------------------------------------------------------------------
    # Step 1b: Check for low-activity condition
    # ------------------------------------------------------------------
    n_soma = spike_params.n_soma if spike_params.n_soma is not None else K
    n_soma = min(n_soma, K)

    soma_active = int((S_times[:n_soma].sum(axis=1) > 0).sum())
    soma_frac = soma_active / max(n_soma, 1)
    sim_duration = nt_internal * _SPIKE_DT
    if soma_frac < 0.05:
        msg = (
            f"Very few soma neurons are active: {soma_active}/{n_soma} "
            f"({100*soma_frac:.1f}%) over {sim_duration:.1f}s "
            f"(rate={spike_params.rate}, nt={spike_params.nt}, "
            f"dt={spike_params.dt:.4f}). "
            f"Consider increasing 'rate' or 'nt'."
        )
        if not spike_params.ensure_activity:
            msg += (" Set ensure_activity=True in SpikeParams to inject "
                     "spikes into silent neurons automatically.")
        warnings.warn(msg, stacklevel=2)

        if spike_params.ensure_activity:
            silent = np.where(S_times[:n_soma].sum(axis=1) == 0)[0]
            n_inject = max(1, len(silent) // 5)  # ~20% of silent soma neurons
            inject_neurons = np.random.choice(silent, size=n_inject, replace=False)
            for k in inject_neurons:
                t_spike = np.random.randint(nt_internal // 4, 3 * nt_internal // 4)
                S_times[k, t_spike] = 1.0
            if v >= 1:
                print(f"  [ensure_activity] Injected 1 spike into "
                      f"{n_inject}/{len(silent)} silent soma neurons")

    # ------------------------------------------------------------------
    # Step 2: Amplitude modulation / buffering
    # ------------------------------------------------------------------
    if dyn_type in ("AR1", "AR2"):
        mask = S_times == 1
        if mask.any():
            S_times[mask] = (1.0 + np.random.rand()) * np.exp(
                spike_params.mu + spike_params.sig * np.random.randn(int(mask.sum()))
            )
    elif dyn_type in ("single", "Ca_DE", "double"):
        S_times = np.concatenate(
            [np.zeros((K, _L_BUFF), dtype=np.float32), S_times], axis=1
        )
        S_times = (7.6e-6) * S_times

    # ------------------------------------------------------------------
    # Step 3: Soma fluorescence
    # ------------------------------------------------------------------
    if v >= 1:
        print(f"\n[2/5] Simulating soma fluorescence  ({dyn_type})")

    S_somas = _simulate_compartment(S_times, dyn_type, cp_internal, prot, 1.0, 0.9)

    # Remove steady-state buffer for ODE modes
    if dyn_type in ("single", "Ca_DE", "double"):
        S_somas = S_somas[:, _L_BUFF:]

    # ------------------------------------------------------------------
    # Step 4: Dendrite fluorescence
    # ------------------------------------------------------------------
    if v >= 1:
        print(f"\n[3/5] Simulating dendrite fluorescence  (dendflag={dendflag})")

    if dendflag:
        S_dend = _simulate_compartment(S_times, dyn_type, cp_internal, prot, 0.5, 0.8)
        if dyn_type in ("single", "Ca_DE", "double"):
            S_dend = S_dend[:, _L_BUFF:]
    else:
        S_dend = None

    # ------------------------------------------------------------------
    # Step 5: Axon fluorescence
    # ------------------------------------------------------------------
    if v >= 1:
        print(f"\n[4/5] Simulating axon/background fluorescence  (axonflag={axonflag})")

    if axonflag:
        S_axon = _simulate_compartment(S_times, dyn_type, cp_internal, prot, 0.25, 0.8)
        if dyn_type in ("single", "Ca_DE", "double"):
            S_axon = S_axon[:, _L_BUFF:]
    else:
        S_axon = None

    # ------------------------------------------------------------------
    # Step 6: GP background (N_bg > 0, mutually exclusive with axonflag)
    # ------------------------------------------------------------------
    S_bg: Optional[np.ndarray] = S_axon  # axon path fills S_bg

    if spike_params.N_bg > 0:
        opts_bg = dataclasses.replace(
            spike_params,
            K=spike_params.N_bg,
            rate=0.25,
            sig=0.2,
            dt=_SPIKE_DT,
            nt=nt_internal,
        )
        if spike_params.smod_flag == "hawkes" and S_bg_hawkes is not None:
            bg_times = S_bg_hawkes
        else:
            bg_times = gen_burst_spike_times(opts_bg)

        if dyn_type in ("AR1", "AR2"):
            mask_bg = bg_times == 1
            if mask_bg.any():
                bg_times[mask_bg] = (1.0 + np.random.rand()) * np.exp(
                    opts_bg.mu + 0.2 * np.random.randn(int(mask_bg.sum()))
                )
        elif dyn_type in ("single", "Ca_DE", "double"):
            bg_times = np.concatenate(
                [np.zeros((spike_params.N_bg, _L_BUFF), dtype=np.float32), bg_times],
                axis=1,
            )
            bg_times = (7.6e-6) * bg_times

        if dyn_type == "AR1":
            h_ca = make_calcium_impulse(0.8, 1.0 / 100.0)
            h_ca = 0.5 * h_ca / h_ca.max()
            nb = spike_params.N_bg
            bgscale = 0.5
            b_bg = np.abs(1.0 + 0.25 * np.random.randn(nb, bg_times.shape[1]))
            bg_conv = bgscale * _conv_rows(bg_times, h_ca) + b_bg
            S_bg = bg_conv / bg_conv.mean()
        elif dyn_type == "AR2":
            h_ca = make_calcium_impulse(0.8, np.array([1, 1]))
            if h_ca.max() > 0:
                h_ca = 0.5 * h_ca / h_ca.max()
            nb = spike_params.N_bg
            bgscale = 0.5
            b_bg = np.abs(1.0 + 0.25 * np.random.randn(nb, bg_times.shape[1]))
            bg_conv = bgscale * _conv_rows(bg_times, h_ca) + b_bg
            S_bg = bg_conv / bg_conv.mean()
        elif dyn_type in ("single", "Ca_DE", "double"):
            cp_bg = dataclasses.replace(cp_internal, sat_type="Ca_DE", ext_rate=2800.0)
            _, _, S_bg_arr = calcium_dynamics(bg_times, cp_bg, prot_type=prot)
            S_bg = S_bg_arr[:, _L_BUFF:]
        else:
            S_bg = None

    # ------------------------------------------------------------------
    # Step 7: Resample from 100 Hz to target dt
    # ------------------------------------------------------------------
    if v >= 1:
        print(f"\n[5/5] Resampling to {1/spike_params.dt:.1f} Hz")

    if abs(spike_params.dt - _SPIKE_DT) > 1e-12:
        inv_dt_int = int(round(1.0 / spike_params.dt))
        if inv_dt_int % 10 == 0:
            r1, r2 = 10, inv_dt_int // 10
        else:
            r1, r2 = _pick_resample_vals(spike_params.dt)

        buff2 = inv_dt_int

        S_somas = _resample_array(S_somas, r1, r2, buff2)
        if S_dend is not None:
            S_dend = _resample_array(S_dend, r1, r2, buff2)
        if S_bg is not None:
            S_bg = _resample_array(S_bg, r1, r2, buff2)

    # ------------------------------------------------------------------
    # Trim / pad to exactly n_desired time-steps
    # ------------------------------------------------------------------
    def _trim(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.shape[1] > n_desired:
            return arr[:, :n_desired]
        return arr

    S_somas = _trim(S_somas)
    S_dend = _trim(S_dend)

    if spike_params.N_bg > 0 and S_bg is not None:
        # GP background gets a 0.4 scale factor (MATLAB line ~497)
        if S_bg.shape[1] > n_desired:
            S_bg = 0.4 * S_bg[:, :n_desired]
    else:
        S_bg = _trim(S_bg)

    # Spike matrix for output (soma spikes, buffer portion removed)
    if spikeflag and s_times is None:
        if dyn_type in ("single", "Ca_DE", "double"):
            # S_times has the buffer prepended; strip it
            spikes_out = (S_times[:, _L_BUFF:] > 0).astype(np.float32)
        else:
            spikes_out = (S_times > 0).astype(np.float32)
        if spikes_out.shape[1] > n_desired:
            spikes_out = spikes_out[:, :n_desired]
    else:
        spikes_out = None

    # ------------------------------------------------------------------
    # Step 8: Expression modulation
    # ------------------------------------------------------------------
    if mod_vals is None:
        mod_vals = expression_variation(K, spike_params.p_off, spike_params.min_mod)

    mod_vals = np.asarray(mod_vals, dtype=np.float32).ravel()[:K]

    if S_somas is not None:
        S_somas = S_somas * mod_vals[:, np.newaxis]
    if S_dend is not None:
        S_dend = S_dend * mod_vals[:, np.newaxis]

    # Axon / bg modulation
    if S_bg is not None and axonflag:
        # Axon traces share the same neurons → use soma mod_vals
        if S_bg.shape[0] == K:
            S_bg = S_bg * mod_vals[:, np.newaxis]
    elif S_bg is None and S_dend is not None and axonflag:
        # MATLAB fallback: if bg is empty and axonflag, bg = dend
        S_bg = S_dend

    # User-facing neuropil brightness knob (1.0 = default NAOMi amplitude).
    # Applied to the final background traces only; soma/dend are unaffected.
    if S_bg is not None and spike_params.bg_scale != 1.0:
        S_bg = S_bg * np.float32(spike_params.bg_scale)

    if v >= 1:
        print("\nTime trace generation complete.")

    return TimeTracesResult(
        soma=S_somas.astype(np.float32) if S_somas is not None else np.zeros((K, n_desired), dtype=np.float32),
        dend=S_dend.astype(np.float32) if S_dend is not None else None,
        bg=S_bg.astype(np.float32) if S_bg is not None else None,
        spikes=spikes_out,
        mod_vals=mod_vals,
        params={
            "spike_params": spike_params,
            "cal_params": cal_params,
        },
    )
