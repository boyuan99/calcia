"""
Tests for the sim-trace bridge (calcia.traces.simtrace_bridge).

Verify that spike trains from a sim-trace IntensityModel flow into calcia's
biophysical calcium dynamics and yield a well-formed TimeTracesResult that
plugs into the scanning stage.

sim-trace is an optional dependency; these tests skip cleanly if it is not
installed in the environment.
"""

import numpy as np
import pytest

pytest.importorskip("simtrace")

from calcia.config.params import CalciumParams, SpikeParams
from calcia.traces.simtrace_bridge import (
    DESIGNS,
    generate_time_traces_simtrace,
    hawkes_smallworld,
    hmm_gated_hawkes,
    simtrace_spike_matrix,
)
from calcia.traces.traces import TimeTracesResult


def test_spike_matrix_shape_and_binary():
    """Sampler returns a K x nt_internal binary matrix at the requested size."""
    spikes = simtrace_spike_matrix(
        hawkes_smallworld(rate=0.3), n_neurons=40, nt_internal=500, seed=0
    )
    assert spikes.shape == (40, 500)
    assert spikes.dtype == np.float32
    assert set(np.unique(spikes)).issubset({0.0, 1.0})
    assert spikes.sum() > 0  # some activity


def test_spike_matrix_reproducible():
    a = simtrace_spike_matrix(hawkes_smallworld(), 30, 300, seed=7)
    b = simtrace_spike_matrix(hawkes_smallworld(), 30, 300, seed=7)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("design", list(DESIGNS.keys()))
def test_end_to_end_designs(design):
    """Each ready-made design produces valid soma fluorescence traces."""
    K = 50
    sp = SpikeParams(K=K, nt=200, dt=1 / 30, N_bg=0, axonflag=False,
                     rate=0.25, prot="GCaMP6f")
    cp = CalciumParams(prot_type="gcamp6f")
    out = generate_time_traces_simtrace(
        sp, model_factory=DESIGNS[design](), cal_params=cp, seed=1, verbose=0
    )
    assert isinstance(out, TimeTracesResult)
    assert out.soma.shape == (K, 200)
    assert np.isfinite(out.soma).all()
    assert out.soma.min() >= 0.0
    assert out.soma.max() > out.soma.min()  # non-degenerate


def test_spatial_topology_uses_locs():
    """Supplying n_locs to a spatial design runs without shape errors."""
    K = 40
    locs = np.random.default_rng(0).uniform(0, 100, size=(K, 3))
    sp = SpikeParams(K=K, nt=150, dt=1 / 30, N_bg=0, axonflag=False, rate=0.3)
    out = generate_time_traces_simtrace(
        sp, model_factory=hawkes_smallworld(use_locs=True),
        n_locs=locs, seed=2, verbose=0,
    )
    assert out.soma.shape == (K, 150)


def test_hmm_gated_matches_native_interface():
    """Bridge output is a drop-in TimeTracesResult (same fields as native)."""
    sp = SpikeParams(K=32, nt=120, dt=1 / 30, rate=0.3)
    out = generate_time_traces_simtrace(
        sp, model_factory=hmm_gated_hawkes(), seed=3, verbose=0
    )
    for field in ("soma", "dend", "bg", "spikes", "mod_vals", "params"):
        assert hasattr(out, field)
    assert out.mod_vals.shape == (32,)
