"""Bridge: drive calcia's biophysical calcium dynamics with sim-trace spikes.

`sim-trace` (https://github.com/boyuan99/sim-trace) is a spike-train simulator
organised by *inter-neuron coupling structure* — a taxonomy of five categories:

    A  Independent      self-only history        (Poisson, renewal, burst)
    B  Shared drive     common latent / stimulus (shared PSTH, gain mod)
    C  Pairwise         explicit W matrix        (Hawkes / GLM / Ising over
                                                  WS / ER / BA / spatial / SBM)
    D  Higher-order     collective events        (assemblies, synfire chains)
    E  Hierarchical     state-modulated          (HMM, rSLDS, GPFA)

sim-trace's own ``spikes_to_calcium`` is a deliberately minimal double-exponential
kernel; its docstring points to *this* library (calcia) for real biophysical
calcium-binding dynamics + Hill non-linearity + indicator saturation. This module
closes that loop: it samples a spike matrix from any sim-trace ``IntensityModel``
and feeds it into :func:`calcia.generate_time_traces` via the ``s_times`` hook,
so the rich coupling structure of sim-trace flows into calcia's optics + scanning.

Note sim-trace's ``HawkesPairwise`` is itself a port of
``calcia.traces.connectivity`` — sim-trace *generalised* calcia's single spatial
Hawkes into a pluggable-topology, composable taxonomy. This bridge lets calcia
reuse that generalisation as a drop-in spike source.

Typical use::

    from calcia.traces.simtrace_bridge import (
        generate_time_traces_simtrace, hmm_gated_hawkes,
    )

    time_out = generate_time_traces_simtrace(
        spike_params=spike_params,          # calcia SpikeParams (K, nt, dt, ...)
        model_factory=hmm_gated_hawkes(),   # a sim-trace coupling design
        n_locs=vol_out.locs,
        seed=42,
    )
    # time_out plugs straight into scan_volume(...) like any calcia TimeTracesResult.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import numpy as np

from calcia.config.params import CalciumParams, SpikeParams
from calcia.traces.traces import TimeTracesResult, generate_time_traces

# calcia's internal simulation rate (must match traces._SPIKE_DT).
_SPIKE_HZ: float = 100.0
_SPIKE_DT: float = 1.0 / _SPIKE_HZ

# A sim-trace model factory builds an IntensityModel sized to ``n_neurons``.
# It receives the population size, an RNG, and optional K x D soma locations
# (so spatial topologies can use real geometry). Returns any object exposing
# the sim-trace IntensityModel interface (reset / intensity / update).
ModelFactory = Callable[[int, np.random.Generator, Optional[np.ndarray]], object]


# ---------------------------------------------------------------------------
# Core: sample a spike matrix from a sim-trace model
# ---------------------------------------------------------------------------


def simtrace_spike_matrix(
    model_factory: ModelFactory,
    n_neurons: int,
    nt_internal: int,
    *,
    dt: float = _SPIKE_DT,
    n_locs: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample an ``n_neurons x nt_internal`` binary spike matrix from sim-trace.

    Parameters
    ----------
    model_factory:
        Callable ``(n_neurons, rng, n_locs) -> IntensityModel`` producing a
        sim-trace model sized to the population. Use one of the helpers below
        (:func:`hawkes_smallworld`, :func:`hmm_gated_hawkes`, ...) or supply
        your own.
    n_neurons:
        Population size ``K`` — must equal ``spike_params.K`` downstream so the
        spike rows line up with calcia's fluorescent components.
    nt_internal:
        Number of 100 Hz samples to generate (calcia's internal length).
    dt:
        Bin width in seconds. Defaults to ``1/100`` to match calcia's internal
        simulation rate; leave it unless you know what you are doing.
    n_locs:
        Optional ``K x D`` soma locations passed to the factory for spatial
        topologies.
    seed / rng:
        Reproducibility. ``rng`` takes precedence; otherwise a generator is
        seeded from ``seed``.

    Returns
    -------
    np.ndarray
        ``n_neurons x nt_internal`` float32 spike matrix (0/1), ready to hand to
        :func:`calcia.generate_time_traces` as ``s_times``.
    """
    # Imported lazily so calcia does not hard-depend on sim-trace being installed.
    from simtrace.sampling import BernoulliBinSampler

    if rng is None:
        rng = np.random.default_rng(seed)

    model = model_factory(n_neurons, rng, n_locs)

    T = nt_internal * dt
    spikes = BernoulliBinSampler().sample(
        model, T=T, dt=dt, n_neurons=n_neurons, rng=rng
    )

    # BernoulliBinSampler uses ceil(T/dt); guard against a one-off rounding
    # so the matrix is exactly nt_internal wide.
    if spikes.shape[1] > nt_internal:
        spikes = spikes[:, :nt_internal]
    elif spikes.shape[1] < nt_internal:
        pad = np.zeros((n_neurons, nt_internal - spikes.shape[1]), dtype=spikes.dtype)
        spikes = np.concatenate([spikes, pad], axis=1)

    return spikes.astype(np.float32)


def _nt_internal(spike_params: SpikeParams) -> int:
    """Internal 100 Hz sample count matching calcia's ``generate_time_traces``."""
    return math.ceil(spike_params.nt * _SPIKE_HZ * spike_params.dt)


# ---------------------------------------------------------------------------
# High-level: sim-trace spikes -> calcia calcium dynamics -> TimeTracesResult
# ---------------------------------------------------------------------------


def generate_time_traces_simtrace(
    spike_params: SpikeParams,
    model_factory: ModelFactory,
    cal_params: Optional[CalciumParams] = None,
    n_locs: Optional[np.ndarray] = None,
    *,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: Optional[int] = None,
) -> TimeTracesResult:
    """End-to-end: sim-trace coupling design -> calcia biophysical traces.

    Drop-in replacement for :func:`calcia.generate_time_traces` where the
    spike trains come from a sim-trace ``IntensityModel`` instead of calcia's
    built-in spatial Hawkes / burst-Poisson. The returned
    :class:`~calcia.traces.traces.TimeTracesResult` plugs straight into
    ``scan_volume``.

    Parameters
    ----------
    spike_params:
        calcia :class:`SpikeParams`. ``K``, ``nt`` and ``dt`` drive the shape;
        ``dyn_type`` / ``prot`` / ``bg_scale`` etc. still control the calcium
        dynamics exactly as in the native path.
    model_factory:
        A sim-trace coupling design (see helpers below).
    cal_params:
        calcia :class:`CalciumParams`; defaults from ``spike_params.prot``.
    n_locs:
        ``K x D`` soma locations for spatial topologies (e.g. ``vol_out.locs``).
    seed / rng:
        Reproducibility for the spike sampling.

    Returns
    -------
    TimeTracesResult
    """
    K = spike_params.K
    nt_int = _nt_internal(spike_params)

    spikes = simtrace_spike_matrix(
        model_factory,
        n_neurons=K,
        nt_internal=nt_int,
        dt=_SPIKE_DT,
        n_locs=n_locs,
        seed=seed,
        rng=rng,
    )

    # Hand the sim-trace spikes to calcia. Passing s_times makes
    # generate_time_traces skip its own Hawkes/Poisson step and run the
    # biophysical calcium ODE (Hill saturation, indicator dynamics) on top.
    out = generate_time_traces(
        spike_params=spike_params,
        cal_params=cal_params,
        s_times=spikes,
        n_locs=n_locs,
        verbose=verbose,
    )

    # When s_times is supplied, calcia sizes its output to the *internal*
    # 100 Hz length and resampling to the target dt can leave a one-sample
    # tail (e.g. 201 instead of 200 frames). Enforce exactly spike_params.nt
    # frames so the bridge contract matches the native `generate_time_traces`
    # (which trims to spike_params.nt) and lines up with scan_volume.
    return _trim_result_frames(out, spike_params.nt)


def _trim_result_frames(out: TimeTracesResult, nt: int) -> TimeTracesResult:
    """Trim every time-axis array in ``out`` to exactly ``nt`` frames."""
    def _t(arr):
        if arr is None:
            return None
        return arr[:, :nt] if arr.shape[1] > nt else arr

    return dataclasses.replace(
        out,
        soma=_t(out.soma),
        dend=_t(out.dend),
        bg=_t(out.bg),
        spikes=_t(out.spikes),
    )


# ---------------------------------------------------------------------------
# Current sim-trace API: ensemble-recruitment spike source
# ---------------------------------------------------------------------------
# sim-trace was refactored from the A-E ``IntensityModel`` taxonomy (the
# ``model_factory`` helpers below, kept for reference) into a focused
# ENSEMBLE-RECRUITMENT generator: each stimulus / self-initiated event recruits a
# sub-population (an ensemble) of neurons, with a tunable overlap regime
# (high / partial / low) between the ensembles of different conditions, plus
# trial-to-trial participation noise. ``simtrace.ensemble.simulate_recruitment``
# returns a ``K x nt`` spike-count matrix — exactly what calcia's ``s_times`` hook
# wants. This is the supported bridge against the current sim-trace.


def generate_time_traces_recruitment(
    spike_params: SpikeParams,
    cal_params: Optional[CalciumParams] = None,
    n_locs: Optional[np.ndarray] = None,
    *,
    regime: str = "partial",
    seed: Optional[int] = None,
    n_conditions: int = 6,
    ensemble_frac: float = 0.06,
    iti_s: float = 1.0,
    response_window_s: float = 0.6,
    spontaneous_rate: float = 0.25,
    evoked_rate: float = 8.0,
    graded: bool = True,
    verbose: Optional[int] = None,
):
    """End-to-end: sim-trace ENSEMBLE-RECRUITMENT spikes -> calcia calcium ODE.

    Drop-in alternative to :func:`calcia.generate_time_traces` where the spike
    trains come from ``simtrace.ensemble.simulate_recruitment`` — a coupling
    design in which each trial recruits a spatially-scattered ENSEMBLE of the
    ``K`` fluorescent components, so the movie shows correlated up-events across a
    genetically-/functionally-defined sub-population instead of calcia's built-in
    independent burst-Poisson / spatial Hawkes. The ``overlap regime`` controls
    how much successive ensembles share members.

    The default recruitment parameters here are tuned for calcia's SHORT internal
    window (a ~10 s movie == 1000 samples at 100 Hz), unlike sim-trace's own
    10-minute-session defaults: short ITIs and a handful of trials so several
    distinct ensembles activate within the clip.

    Returns ``(TimeTracesResult, RecruitmentSession)`` — the session carries the
    ground-truth ``membership`` (n_conditions x K ensembles) and ``recruited``
    matrices for downstream scoring.
    """
    from simtrace.ensemble import RecruitmentParams, simulate_recruitment

    K = spike_params.K
    nt_int = _nt_internal(spike_params)
    fps = _SPIKE_HZ                       # recruitment runs at calcia's 100 Hz
    duration_s = nt_int / fps
    n_trials = max(n_conditions, int(duration_s / max(iti_s, 1e-3)))

    xy = None
    if n_locs is not None and np.asarray(n_locs).shape[0] == K:
        xy = np.asarray(n_locs)[:, :2].astype(float)
    ensemble_size = max(1, int(round(ensemble_frac * K)))

    rp = RecruitmentParams(
        K=K, n_conditions=n_conditions, ensemble_size=ensemble_size,
        n_trials=n_trials, duration_s=duration_s, frame_rate_hz=fps,
        iti_s=iti_s, response_window_s=response_window_s,
        spontaneous_rate=spontaneous_rate, evoked_rate=evoked_rate,
        graded=graded, seed=(seed if seed is not None else 0), xy=xy,
    )
    session = simulate_recruitment(regime, rp)

    spikes = np.asarray(session.spikes, dtype=np.float32)   # K x nt (counts)
    if spikes.shape[1] > nt_int:
        spikes = spikes[:, :nt_int]
    elif spikes.shape[1] < nt_int:
        spikes = np.concatenate(
            [spikes, np.zeros((K, nt_int - spikes.shape[1]), np.float32)], axis=1)

    out = generate_time_traces(
        spike_params=spike_params, cal_params=cal_params,
        s_times=spikes, n_locs=n_locs, verbose=verbose,
    )
    return _trim_result_frames(out, spike_params.nt), session


# ---------------------------------------------------------------------------
# LEGACY (A-E IntensityModel taxonomy) — for reference only. These require the
# OLD sim-trace layout (simtrace.sampling / simtrace.models / simtrace.core),
# which the current ensemble-focused sim-trace no longer ships. Use
# generate_time_traces_recruitment above with the installed sim-trace.
# ---------------------------------------------------------------------------
# Ready-made sim-trace coupling designs (factories sized to K at call time)
# ---------------------------------------------------------------------------


def hawkes_smallworld(
    *,
    rate: float = 0.25,
    k_conn: int = 10,
    beta: float = 0.3,
    selfact: float = 1.2,
    use_locs: bool = True,
) -> ModelFactory:
    """Category C: pairwise Hawkes over a Watts-Strogatz small-world network.

    The closest analogue to calcia's built-in ``smod_flag='hawkes'`` path, but
    exposed through sim-trace's pluggable topology so the network structure can
    be swapped independently of the dynamics. When ``use_locs`` and locations
    are supplied, the lattice connects spatial nearest-neighbours.
    """

    def factory(n_neurons, rng, n_locs):
        from simtrace.models.pairwise import HawkesPairwise, HawkesPairwiseParams
        from simtrace.models.pairwise.topology import watts_strogatz

        locs = n_locs if (use_locs and n_locs is not None
                          and np.asarray(n_locs).shape[0] == n_neurons) else None
        topo = watts_strogatz(
            n=n_neurons, k_conn=k_conn, beta=beta, n_locs=locs, rng=rng
        )
        return HawkesPairwise(topo, HawkesPairwiseParams(rate=rate, selfact=selfact))

    return factory


def hawkes_scale_free(
    *, rate: float = 0.25, m_attach: int = 3, selfact: float = 1.2
) -> ModelFactory:
    """Category C: pairwise Hawkes over a Barabasi-Albert scale-free network.

    Hub neurons dominate recruitment — a topology calcia's built-in path cannot
    produce. Falls back gracefully if the sim-trace BA generator signature
    differs.
    """

    def factory(n_neurons, rng, n_locs):
        from simtrace.models.pairwise import HawkesPairwise, HawkesPairwiseParams
        from simtrace.models.pairwise.topology import barabasi_albert

        topo = barabasi_albert(n=n_neurons, m=m_attach, rng=rng)
        return HawkesPairwise(topo, HawkesPairwiseParams(rate=rate, selfact=selfact))

    return factory


def hmm_gated_hawkes(
    *,
    rate: float = 0.3,
    n_states: int = 3,
    dwell_times_s=(20.0, 8.0, 12.0),
    drive_freq_hz: float = 0.2,
    baseline_hz: float = 1.0,
    drive_hz_amp: float = 4.0,
    k_conn: int = 10,
    beta: float = 0.3,
    use_locs: bool = True,
) -> ModelFactory:
    """Categories E gate (B + C): a brain-state HMM modulating a pairwise
    Hawkes network that also receives a shared oscillatory drive.

    This is sim-trace's flagship composition (README example) and showcases
    dynamics calcia's built-in generator *cannot* produce: slow global
    brain-state switches (the HMM) riding on top of coupled network activity
    (Hawkes) plus a population-wide rhythm (common PSTH). The resulting movie
    shows correlated up/down epochs across the field of view.
    """

    def factory(n_neurons, rng, n_locs):
        from simtrace.core import GatedIntensity, SumIntensity
        from simtrace.models.pairwise import HawkesPairwise, HawkesPairwiseParams
        from simtrace.models.pairwise.topology import watts_strogatz
        from simtrace.models.shared_drive import CommonPSTH, CommonPSTHParams
        from simtrace.models.shared_drive.common_psth import sinusoid
        from simtrace.models.latent import HMMGate, HMMGateParams

        locs = n_locs if (use_locs and n_locs is not None
                          and np.asarray(n_locs).shape[0] == n_neurons) else None
        topo = watts_strogatz(
            n=n_neurons, k_conn=k_conn, beta=beta, n_locs=locs, rng=rng
        )
        hawkes = HawkesPairwise(topo, HawkesPairwiseParams(rate=rate))
        drive = CommonPSTH(
            CommonPSTHParams(
                K=n_neurons, baseline_hz=baseline_hz, drive_hz_amp=drive_hz_amp
            ),
            drive=sinusoid(freq_hz=drive_freq_hz),
        )
        hmm = HMMGate(
            HMMGateParams(
                K=n_neurons, n_states=n_states, dwell_times_s=tuple(dwell_times_s)
            )
        )
        return GatedIntensity(gate=hmm, model=SumIntensity([hawkes, drive]))

    return factory


# ---------------------------------------------------------------------------
# Scalable designs (NO K x K matrix — safe for very large populations)
# ---------------------------------------------------------------------------
# The pairwise (C) and higher-order (D) designs above build a dense K x K
# coupling matrix, which is infeasible past a few thousand components
# (K = 55k -> a 24 GB matrix). Categories B (shared drive) and E (latent gate)
# have no such matrix and scale to arbitrarily large K, so use these on
# full-size volumes (e.g. a dense striatum window).


def shared_drive_osc(
    *,
    baseline_hz: float = 0.6,
    drive_hz_amp: float = 3.0,
    drive_freq_hz: float = 0.2,
    gain_std: float = 0.6,
) -> ModelFactory:
    """Category B: a shared oscillatory drive every neuron responds to.

    No inter-neuron matrix — the whole population is modulated by one common
    rhythm ``s(t)`` with heterogeneous per-neuron gains. Scales to any K.
    Produces field-wide correlated waxing/waning without pairwise coupling.
    """

    def factory(n_neurons, rng, n_locs):
        from simtrace.models.shared_drive import CommonPSTH, CommonPSTHParams
        from simtrace.models.shared_drive.common_psth import sinusoid

        return CommonPSTH(
            CommonPSTHParams(
                K=n_neurons, baseline_hz=baseline_hz,
                drive_hz_amp=drive_hz_amp, gain_std=gain_std,
            ),
            drive=sinusoid(freq_hz=drive_freq_hz),
        )

    return factory


def hmm_gated_drive(
    *,
    baseline_hz: float = 0.6,
    drive_hz_amp: float = 3.0,
    drive_freq_hz: float = 0.2,
    n_states: int = 3,
    dwell_times_s=(20.0, 8.0, 12.0),
) -> ModelFactory:
    """Categories E gate over B: a brain-state HMM modulating a shared drive.

    Slow discrete brain-state switches (the HMM) scale the amplitude of a
    population-wide oscillatory drive (common PSTH). No K x K matrix, so this
    scales to tens of thousands of components while still showcasing sim-trace
    composition the built-in generator cannot produce: correlated global
    up/down epochs riding on a shared rhythm.
    """

    def factory(n_neurons, rng, n_locs):
        from simtrace.core import GatedIntensity
        from simtrace.models.shared_drive import CommonPSTH, CommonPSTHParams
        from simtrace.models.shared_drive.common_psth import sinusoid
        from simtrace.models.latent import HMMGate, HMMGateParams

        drive = CommonPSTH(
            CommonPSTHParams(
                K=n_neurons, baseline_hz=baseline_hz, drive_hz_amp=drive_hz_amp
            ),
            drive=sinusoid(freq_hz=drive_freq_hz),
        )
        hmm = HMMGate(
            HMMGateParams(
                K=n_neurons, n_states=n_states, dwell_times_s=tuple(dwell_times_s)
            )
        )
        return GatedIntensity(gate=hmm, model=drive)

    return factory


# Registry so a demo / CLI can pick a design by name.
DESIGNS: dict[str, Callable[..., ModelFactory]] = {
    # Pairwise / composed — dense K x K matrix, use for K up to a few thousand.
    "hawkes_smallworld": hawkes_smallworld,
    "hawkes_scale_free": hawkes_scale_free,
    "hmm_gated_hawkes": hmm_gated_hawkes,
    # Scalable (no matrix) — safe for very large K (dense full-size volumes).
    "shared_drive_osc": shared_drive_osc,
    "hmm_gated_drive": hmm_gated_drive,
}

# Designs that do NOT build a K x K matrix (safe at very large K).
SCALABLE_DESIGNS = frozenset({"shared_drive_osc", "hmm_gated_drive"})
