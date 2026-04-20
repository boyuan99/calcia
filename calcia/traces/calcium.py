"""
Calcium dynamics simulation functions.

Ports of MATLAB functions:
  - mk_doub_exp_ker.m   (MiscCode/)
  - make_calcium_impulse.m
  - calcium_dynamics.m
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import fftconvolve, lfilter

from calcia.config.params import CalciumParams


# ---------------------------------------------------------------------------
# Double-exponential kernel
# ---------------------------------------------------------------------------

def mk_doub_exp_ker(
    t_on: float,
    t_off: float,
    A: float,
    dt: float,
    dext_type: str = "mult",
) -> np.ndarray:
    """
    Build a double-exponential calcium transient kernel.

    Port of ``MiscCode/mk_doub_exp_ker.m``.

    Default (``dext_type='mult'``) formula::

        h(t) = A * (1 - exp(-t_on * t)) * exp(-t_off * t)

    The kernel is evaluated from t=0 to the point where it falls below
    1e-3 of its peak value.

    Parameters
    ----------
    t_on:
        Rising rate constant (1/s).
    t_off:
        Falling rate constant (1/s).
    A:
        Amplitude scaling factor.
    dt:
        Sampling interval in seconds.
    dext_type:
        Kernel type.  Only ``'mult'`` is currently supported.

    Returns
    -------
    np.ndarray
        1-D float32 kernel array.
    """
    if dext_type != "mult":
        raise NotImplementedError(f"dext_type={dext_type!r} not implemented; use 'mult'")

    # Location of the peak
    loc_max = np.log((t_off + t_on) / t_off) / t_on

    def dexp(z: np.ndarray) -> np.ndarray:
        return A * (1.0 - np.exp(-t_on * z)) * np.exp(-t_off * z)

    max_val = float(dexp(np.asarray(loc_max)))

    # Stop when kernel drops below 1e-3 of its peak
    t_max = -np.log(max_val * 1e-3 / A) / t_off
    t_arr = np.arange(0.0, float(t_max) + dt, dt)
    return dexp(t_arr).astype(np.float32)


# ---------------------------------------------------------------------------
# AR impulse response
# ---------------------------------------------------------------------------

def make_calcium_impulse(
    ca_scale: float | np.ndarray,
    dt: float | np.ndarray = 1 / 30,
) -> np.ndarray:
    """
    Compute the AR impulse response for a calcium indicator.

    Port of ``TimeTraceCode/make_calcium_impulse.m``.

    MATLAB uses the ``arima`` + ``impulse`` toolbox functions.  This
    implementation uses the equivalent ``scipy.signal.lfilter`` approach:

    * ``ca_scale`` values become AR roots via ``exp(-ca_scale)``.
    * The denominator polynomial from ``numpy.poly(roots)`` is fed to
      ``lfilter([1], poly, impulse)``.

    When ``dt`` is a scalar the function returns ``int(10/dt)+1`` samples.
    When ``dt`` is a length-2 array (the AR2 call in ``generateTimeTraces``),
    the function evaluates the impulse response at the integer indices given
    by ``dt`` — matching MATLAB ``impulse(sys, t_vec)``.

    Parameters
    ----------
    ca_scale:
        Scalar or array of time-scale values.  AR roots are placed at
        ``exp(-ca_scale)``.
    dt:
        Scalar sampling interval *or* integer time indices for evaluation.

    Returns
    -------
    np.ndarray
        1-D float32 impulse response.
    """
    ca_scale = np.atleast_1d(np.asarray(ca_scale, dtype=float))
    dt_arr = np.atleast_1d(np.asarray(dt, dtype=float))

    roots = np.exp(-ca_scale)
    a_poly = np.poly(roots)  # [1, c1, c2, ...] denominator polynomial

    if dt_arr.size == 1:
        # Scalar dt: return int(10/dt)+1 samples
        n = int(10.0 / dt_arr[0]) + 1
        impulse_in = np.zeros(n, dtype=float)
        impulse_in[0] = 1.0
        h = lfilter([1.0], a_poly, impulse_in)
    else:
        # Vector dt: evaluate at specific integer time indices
        # This matches MATLAB impulse(sys, t_vec) behaviour
        n = int(dt_arr.max()) + 2
        impulse_in = np.zeros(n, dtype=float)
        impulse_in[0] = 1.0
        h_full = lfilter([1.0], a_poly, impulse_in)
        h = h_full[dt_arr.astype(int)]

    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# Hill-equation fluorescence nonlinearity
# ---------------------------------------------------------------------------

def sat_nonlin(CB: np.ndarray, prot_type: str) -> np.ndarray:
    """
    Map bound calcium concentration to fluorescence via Hill equation.

    Port of the nested ``sat_nonlin`` function in ``calcium_dynamics.m``.

    Parameters
    ----------
    CB:
        Bound calcium concentration array (any shape).
    prot_type:
        Protein name (case-insensitive, hyphens OK).

    Returns
    -------
    np.ndarray
        Fluorescence array, same shape as ``CB``.
    """
    key = prot_type.lower().replace("-", "")
    if key in ("gcamp6", "gcamp6f"):
        F0 = 1.0
        F = 25.2 * (1.0 / (1.0 + (290e-9 / CB) ** 2.7))
    elif key == "gcamp6s":
        F0 = 1.0
        F = 27.2 * (1.0 / (1.0 + (147e-9 / CB) ** 2.45))
    elif key == "gcamp3":
        F0 = 2.0
        F = 12.0 * (1.0 / (1.0 + (287e-9 / CB) ** 2.52))
    elif key in ("ogb1", "ogb1"):
        F0 = 1.0
        F = 14.0 * (1.0 / (1.0 + 250e-9 / CB))
    elif key in ("gcamp6rs09", "gcamp6rs09"):
        F0 = 1.4
        F = 25.0 * (1.0 / (1.0 + (520e-9 / CB) ** 3.2))
    elif key in ("gcamp6rs06", "gcamp6rs06"):
        F0 = 1.2
        F = 15.0 * (1.0 / (1.0 + (320e-9 / CB) ** 3.0))
    elif key == "jgcamp7f":
        F0 = 1.0
        F = 30.2 * (1.0 / (1.0 + (174e-9 / CB) ** 2.3))
    elif key == "jgcamp7s":
        F0 = 1.0
        F = 40.4 * (1.0 / (1.0 + (68e-9 / CB) ** 2.49))
    elif key == "jgcamp7b":
        F0 = 1.0
        F = 22.1 * (1.0 / (1.0 + (82e-9 / CB) ** 3.06))
    elif key == "jgcamp7c":
        F0 = 1.0
        F = 145.6 * (1.0 / (1.0 + (298e-9 / CB) ** 2.44))
    else:
        # Default to GCaMP6f
        F0 = 1.0
        F = 25.2 * (1.0 / (1.0 + (290e-9 / CB) ** 2.7))

    return F0 + F0 * F


# ---------------------------------------------------------------------------
# Calcium dynamics ODE
# ---------------------------------------------------------------------------

def calcium_dynamics(
    S: np.ndarray,
    cal_params: CalciumParams,
    prot_type: str = "gcamp6f",
    over_samp: int = 1,
    ext_mult: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate calcium concentration and fluorescence dynamics.

    Port of ``TimeTraceCode/calcium_dynamics.m``.

    Parameters
    ----------
    S:
        ``K × nt`` array of spike-induced calcium inputs (in mol/L units,
        scaled by 7.6e-6 in the caller).
    cal_params:
        Calcium simulation parameters.
    prot_type:
        Protein name for the Hill equation.
    over_samp:
        Over-sampling factor (unused in Ca_DE path).
    ext_mult:
        Multiplier on ``cal_params.ext_rate`` (used to differentiate soma
        vs dendrite vs axon extrusion rates).

    Returns
    -------
    CB : np.ndarray
        Bound calcium concentration ``K × nt``.
    C : np.ndarray
        Free calcium concentration ``K × nt``.
    F : np.ndarray
        Fluorescence ``K × nt``.
    """
    S = np.asarray(S, dtype=np.float32)
    K, nt = S.shape

    ext_rate = ext_mult * cal_params.ext_rate
    ca_bind = cal_params.ca_bind
    ca_rest = cal_params.ca_rest
    ind_con = cal_params.ind_con
    ca_dis = cal_params.ca_dis
    ca_sat = cal_params.ca_sat
    sat_type = cal_params.sat_type
    dt = cal_params.dt
    a_bind = cal_params.a_bind
    a_ubind = cal_params.a_ubind

    C = np.zeros((K, nt), dtype=np.float32)
    C[:, 0] = np.maximum(ca_rest, S[:, 0])

    if sat_type == "single":
        CB = np.zeros((K, nt), dtype=np.float32)
        a = float(a_bind)
        b = float(a_ubind)
        for kk in range(1, nt):
            denom = 1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2
            C[:, kk] = (
                C[:, kk - 1]
                + dt * b * CB[:, kk - 1]
                + (-dt * ext_rate * (C[:, kk - 1] - CB[:, kk - 1] - ca_rest) + S[:, kk])
                / denom
            )
            if 0.0 <= ca_sat < 1.0:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))
            CB[:, kk] = (
                CB[:, kk - 1]
                + dt * (-b * CB[:, kk - 1] + a * (C[:, kk - 1] - CB[:, kk - 1]) * (ind_con - CB[:, kk - 1]))
            )
        if over_samp > 1:
            C = C[:, ::over_samp]
            CB = CB[:, ::over_samp]
        # Fluorescence from bound Ca + indicator correction
        CB_fl = CB + ca_rest + (b / a) * CB / (ind_con - CB)
        F = sat_nonlin(CB_fl, prot_type)

    elif sat_type == "Ca_DE":
        # Euler step for free calcium (no explicit binding variable)
        a = float(a_bind) * 100.0 * dt   # pre-scaled decay rates at 100 Hz
        b = float(a_ubind) * 100.0 * dt
        for kk in range(1, nt):
            # MATLAB Ca_DE uses .\ (left-divide): (A).\(B) == B/A
            denom = 1.0 + ca_bind + (C[:, kk - 1] + ca_dis) ** 2 / (ind_con * ca_dis)
            C[:, kk] = (
                C[:, kk - 1]
                + (-dt * ext_rate * (C[:, kk - 1] - ca_rest) + S[:, kk]) / denom
            )
            if 0.0 <= ca_sat < 1.0:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))

        # Build double-exponential kernel and convolve each row
        h_ca = mk_doub_exp_ker(cal_params.t_on, cal_params.t_off, cal_params.ca_amp, dt)

        n_full = nt  # number of Ca_DE output samples (no over_samp in convolution)
        CB = np.zeros((K, n_full), dtype=np.float32)
        for kk in range(K):
            row = C[kk].astype(float) - ca_rest
            tmp = fftconvolve(row, h_ca.astype(float), mode="full") + ca_rest
            CB[kk] = tmp[:n_full].astype(np.float32)

        if over_samp > 1:
            C = C[:, ::over_samp]
            CB = CB[:, ::over_samp]

        CB = CB[:, : C.shape[1]]
        F = sat_nonlin(CB, prot_type)

    elif sat_type == "double":
        CB1 = np.zeros((K, nt), dtype=np.float32)
        CB2 = np.zeros((K, nt), dtype=np.float32)
        a = np.atleast_1d(a_bind).astype(float)
        b = np.atleast_1d(a_ubind).astype(float)
        if a.size == 1:
            a = np.array([a[0], a[0]])
        if b.size == 1:
            b = np.array([b[0], b[0]])
        for kk in range(1, nt):
            denom = 1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2
            C[:, kk] = (
                C[:, kk - 1]
                + dt * (b[0] * CB1[:, kk - 1] + b[1] * CB2[:, kk - 1])
                + (
                    -dt * ext_rate * (C[:, kk - 1] - CB1[:, kk - 1] - CB2[:, kk - 1] - ca_rest)
                    + S[:, kk]
                ) / denom
            )
            if 0.0 <= ca_sat < 1.0:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))
            free = C[:, kk - 1] - CB1[:, kk - 1] - CB2[:, kk - 1]
            avail = ind_con - CB1[:, kk - 1] - CB2[:, kk - 1]
            CB1[:, kk] = CB1[:, kk - 1] + dt * (-b[0] * CB1[:, kk - 1] + a[0] * free * avail)
            CB2[:, kk] = CB2[:, kk - 1] + dt * (-b[1] * CB2[:, kk - 1] + a[1] * free * avail)

        CB = (CB1 + CB2)
        if over_samp > 1:
            C = C[:, ::over_samp]
            CB = CB[:, ::over_samp]
        F = sat_nonlin(CB, prot_type)

    else:
        raise ValueError(f"Unknown sat_type={sat_type!r}.  Options: 'Ca_DE', 'single', 'double'.")

    return CB.astype(np.float32), C.astype(np.float32), F.astype(np.float32)
