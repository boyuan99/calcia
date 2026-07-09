"""
Tests for Phase 3: time-trace generation.

Covers:
  - SpikeParams / CalciumParams dataclasses
  - mk_doub_exp_ker, make_calcium_impulse
  - gen_burst_spike_times, bin_spike_trains
  - samp_small_world_mat, gen_correlated_spike_trains
  - calcium_dynamics
  - expression_variation
  - generate_time_traces (end-to-end)
"""

from __future__ import annotations

import numpy as np
import pytest

from calcia.config.params import CalciumParams, SpikeParams
from calcia.traces.calcium import (
    calcium_dynamics,
    make_calcium_impulse,
    mk_doub_exp_ker,
    sat_nonlin,
)
from calcia.traces.connectivity import (
    gen_correlated_spike_trains,
    samp_small_world_mat,
)
from calcia.traces.expression import expression_variation
from calcia.traces.spikes import bin_spike_trains, gen_burst_spike_times
from calcia.traces.traces import (
    TimeTracesResult,
    _pick_resample_vals,
    generate_time_traces,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _spike_params_small(**kwargs) -> SpikeParams:
    """Minimal SpikeParams for fast tests (AR1, poisson, no flags overhead)."""
    defaults = dict(
        K=3,
        nt=50,
        dt=1 / 30,
        dyn_type="AR1",
        smod_flag="poisson",
        burst_mean=0,
        dendflag=True,
        axonflag=False,
        N_bg=0,
        verbose=0,
        spikeflag=True,
    )
    defaults.update(kwargs)
    return SpikeParams(**defaults)


def _cal_params_default() -> CalciumParams:
    return CalciumParams(prot_type="gcamp6f")


# ===========================================================================
# SpikeParams
# ===========================================================================


class TestSpikeParams:
    def test_defaults(self):
        p = SpikeParams()
        assert p.K == 30
        assert p.nt == 1000
        assert pytest.approx(p.dt, rel=1e-6) == 1 / 30
        assert p.dyn_type == "Ca_DE"
        assert p.smod_flag == "hawkes"
        assert p.prot == "GCaMP6f"
        assert p.N_bg == 0
        assert p.spikeflag is True
        assert p.dendflag is True
        assert p.axonflag is True

    def test_custom(self):
        p = SpikeParams(K=5, dyn_type="AR1", smod_flag="poisson")
        assert p.K == 5
        assert p.dyn_type == "AR1"
        assert p.smod_flag == "poisson"

    def test_min_mod_is_tuple(self):
        p = SpikeParams()
        assert len(p.min_mod) == 2
        assert p.min_mod[0] == pytest.approx(0.4)
        assert p.min_mod[1] == pytest.approx(2.53)

    def test_verbose_default(self):
        p = SpikeParams()
        assert p.verbose == 1


# ===========================================================================
# CalciumParams
# ===========================================================================


class TestCalciumParams:
    def test_gcamp6f_defaults(self):
        p = CalciumParams(prot_type="gcamp6f")
        assert p.ca_amp == pytest.approx(76.1251)
        assert p.t_on == pytest.approx(0.8535)
        assert p.t_off == pytest.approx(98.6173)
        assert p.ext_rate == pytest.approx(292.3)

    def test_gcamp3_defaults(self):
        p = CalciumParams(prot_type="gcamp3")
        assert p.ca_amp == pytest.approx(0.05)
        assert p.t_on == pytest.approx(1.0)
        assert p.t_off == pytest.approx(1.0)
        assert p.ext_rate == pytest.approx(265.73)

    def test_gcamp6s_defaults(self):
        p = CalciumParams(prot_type="gcamp6s")
        assert p.ca_amp == pytest.approx(54.6943)
        assert p.t_off == pytest.approx(68.5461)

    def test_override_ca_amp(self):
        p = CalciumParams(prot_type="gcamp6f", ca_amp=999.0)
        assert p.ca_amp == pytest.approx(999.0)
        # Other fields resolved from defaults
        assert p.t_on == pytest.approx(0.8535)

    def test_prot_type_case_insensitive(self):
        p1 = CalciumParams(prot_type="GCaMP6f")
        p2 = CalciumParams(prot_type="gcamp6f")
        assert p1.ca_amp == p2.ca_amp

    def test_unknown_prot_falls_back_to_gcamp6f(self):
        p = CalciumParams(prot_type="unknownprotein")
        assert p.ca_amp == pytest.approx(76.1251)

    def test_default_sat_type(self):
        p = CalciumParams()
        assert p.sat_type == "double"

    def test_default_dt(self):
        p = CalciumParams()
        assert p.dt == pytest.approx(1 / 100)


# ===========================================================================
# mk_doub_exp_ker
# ===========================================================================


class TestMkDoubExpKer:
    def test_returns_ndarray(self):
        h = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 100)
        assert isinstance(h, np.ndarray)

    def test_float32(self):
        h = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 100)
        assert h.dtype == np.float32

    def test_rise_then_fall(self):
        h = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 100)
        peak_idx = int(np.argmax(h))
        assert peak_idx > 0, "Peak should not be at t=0"
        assert h[peak_idx] > h[0], "Peak should be greater than t=0 value"

    def test_nonneg(self):
        h = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 100)
        assert float(h.min()) >= 0.0

    def test_length_scales_with_dt(self):
        h1 = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 100)
        h2 = mk_doub_exp_ker(0.8535, 98.6173, 76.1251, 1 / 50)
        # Coarser dt → fewer samples for same time span
        assert len(h2) < len(h1)

    def test_unsupported_dext_type(self):
        with pytest.raises(NotImplementedError):
            mk_doub_exp_ker(1.0, 1.0, 1.0, 0.01, dext_type="plus")


# ===========================================================================
# make_calcium_impulse
# ===========================================================================


class TestMakeCalciumImpulse:
    def test_ar1_starts_at_one(self):
        h = make_calcium_impulse(0.9, 1 / 100)
        assert h[0] == pytest.approx(1.0, abs=1e-6)

    def test_ar1_decays(self):
        h = make_calcium_impulse(0.9, 1 / 100)
        assert h[0] > h[1] > h[10]

    def test_ar1_length(self):
        dt = 1 / 100
        h = make_calcium_impulse(0.9, dt)
        expected_len = int(10.0 / dt) + 1
        assert len(h) == expected_len

    def test_ar2_two_element_output(self):
        # AR2 call: dt=[1,1] → evaluate at time indices 1 and 1
        h = make_calcium_impulse(0.9, np.array([1, 1]))
        assert h.shape == (2,)

    def test_ar2_both_values_equal(self):
        h = make_calcium_impulse(0.9, np.array([1, 1]))
        assert h[0] == pytest.approx(h[1])

    def test_float32(self):
        h = make_calcium_impulse(0.9, 1 / 100)
        assert h.dtype == np.float32

    def test_large_scale_decays_faster(self):
        h_slow = make_calcium_impulse(0.1, 1 / 100)
        h_fast = make_calcium_impulse(2.0, 1 / 100)
        # Larger ca_scale → root closer to 0 → faster decay
        assert h_fast[10] < h_slow[10]


# ===========================================================================
# sat_nonlin
# ===========================================================================


class TestSatNonlin:
    def test_gcamp6f_positive(self):
        CB = np.linspace(50e-9, 1e-6, 20)
        F = sat_nonlin(CB, "gcamp6f")
        assert float(F.min()) > 0.0

    def test_gcamp3_lower_than_gcamp6f(self):
        CB = np.array([500e-9])
        F6 = sat_nonlin(CB, "gcamp6f")
        F3 = sat_nonlin(CB, "gcamp3")
        # GCaMP6f has higher dynamic range than GCaMP3 at this concentration
        assert F6[0] > F3[0]

    def test_monotone_increasing(self):
        CB = np.linspace(50e-9, 5e-6, 50)
        F = sat_nonlin(CB, "gcamp6f")
        assert np.all(np.diff(F) >= 0)


# ===========================================================================
# gen_burst_spike_times
# ===========================================================================


class TestGenBurstSpikeTimes:
    def _params(self, **kw) -> SpikeParams:
        defaults = dict(K=5, nt=200, dt=1 / 100, alpha=1.0, burst_mean=0,
                        rate=1e-3, rate_dist="gamma", verbose=0)
        defaults.update(kw)
        return SpikeParams(**defaults)

    def test_shape(self):
        np.random.seed(42)
        p = self._params()
        S = gen_burst_spike_times(p)
        assert S.shape == (5, 200)

    def test_float32(self):
        np.random.seed(42)
        S = gen_burst_spike_times(self._params())
        assert S.dtype == np.float32

    def test_values_in_zero_one(self):
        np.random.seed(42)
        S = gen_burst_spike_times(self._params())
        assert set(np.unique(S)).issubset({0.0, 1.0})

    def test_spikes_present(self):
        np.random.seed(42)
        # High rate to ensure spikes are present
        S = gen_burst_spike_times(self._params(K=10, nt=500, rate=0.1))
        assert S.sum() > 0

    def test_uniform_rate_dist(self):
        np.random.seed(42)
        S = gen_burst_spike_times(self._params(K=3, nt=300, rate_dist="uniform"))
        assert S.shape == (3, 300)


# ===========================================================================
# bin_spike_trains
# ===========================================================================


class TestBinSpikeTrains:
    def test_empty_events(self):
        S = bin_spike_trains(np.array([]), np.array([]), 5, 0.01, 100)
        assert S.shape == (5, 100)
        np.testing.assert_array_equal(S, np.zeros((5, 100)))

    def test_single_event(self):
        evt = np.array([0.05])  # 0.05 s
        evm = np.array([2])     # neuron 2 (1-based)
        S = bin_spike_trains(evt, evm, 5, 0.01, 10)
        # bin_idx = ceil(0.05/0.01) - 1 = 5 - 1 = 4
        assert S[1, 4] == pytest.approx(1.0)
        assert S.sum() == pytest.approx(1.0)

    def test_multiple_events_same_bin(self):
        evt = np.array([0.01, 0.009])
        evm = np.array([1, 1])
        S = bin_spike_trains(evt, evm, 2, 0.01, 5)
        # Both ceil to bin 1 (0-based: 0)
        assert S[0, 0] == pytest.approx(2.0)

    def test_float32(self):
        S = bin_spike_trains(np.array([]), np.array([]), 3, 0.01, 10)
        assert S.dtype == np.float32


# ===========================================================================
# samp_small_world_mat
# ===========================================================================


class TestSampSmallWorldMat:
    def test_shape_scalar(self):
        np.random.seed(42)
        A = samp_small_world_mat(10, 4, 0.3)
        assert A.shape == (10, 10)

    def test_shape_with_bg(self):
        np.random.seed(42)
        A = samp_small_world_mat((8, 2), 4, 0.3)
        assert A.shape == (10, 10)

    def test_nonnegative(self):
        np.random.seed(42)
        A = samp_small_world_mat(10, 4, 0.3)
        assert float(A.min()) >= 0.0

    def test_self_excitation_diagonal(self):
        np.random.seed(42)
        self_ex = 4.0
        A = samp_small_world_mat(10, 4, 0.3, self_ex=self_ex)
        # All diagonal entries must be >= self_ex (added on top of lattice)
        assert np.all(A.diagonal() >= self_ex)

    def test_beta_zero_mostly_lattice(self):
        np.random.seed(42)
        # With beta=0, no rewiring, should be symmetric toeplitz + diagonal
        A = samp_small_world_mat(10, 4, 0.0, self_ex=0.0)
        assert A.shape == (10, 10)
        assert float(A.min()) >= 0.0

    def test_rand_opt_changes_weights(self):
        np.random.seed(42)
        A0 = samp_small_world_mat(10, 4, 0.3, rand_opt=0.0)
        np.random.seed(42)
        A1 = samp_small_world_mat(10, 4, 0.3, rand_opt=0.9)
        # rand_opt > 0 introduces continuous weights
        assert not np.allclose(A0, A1)


# ===========================================================================
# gen_correlated_spike_trains
# ===========================================================================


class TestGenCorrelatedSpikeTrains:
    def _params(self, **kw) -> SpikeParams:
        # K=20 > k_conn=10 to match MATLAB's expected usage (K >> k_conn)
        defaults = dict(K=20, N_bg=0, nt=50, dt=1 / 100,
                        rate=1e-3, selfact=1.2, burst_mean=10,
                        smod_flag="hawkes", verbose=0)
        defaults.update(kw)
        return SpikeParams(**defaults)

    def test_returns_dict_with_soma(self):
        np.random.seed(42)
        d = gen_correlated_spike_trains(self._params())
        assert "soma" in d
        assert "bg" in d

    def test_soma_shape(self):
        np.random.seed(42)
        d = gen_correlated_spike_trains(self._params())
        assert d["soma"].shape == (20, 50)

    def test_soma_nonneg(self):
        np.random.seed(42)
        d = gen_correlated_spike_trains(self._params())
        assert float(d["soma"].min()) >= 0.0

    def test_bg_empty_when_no_bg(self):
        np.random.seed(42)
        d = gen_correlated_spike_trains(self._params(N_bg=0))
        assert d["bg"].shape[0] == 0

    def test_bg_shape_with_bg(self):
        np.random.seed(42)
        d = gen_correlated_spike_trains(self._params(K=15, N_bg=2))
        assert d["bg"].shape == (2, 50)

    def test_discrete_only(self):
        with pytest.raises(NotImplementedError):
            gen_correlated_spike_trains(self._params(), discrete=False)


# ===========================================================================
# calcium_dynamics
# ===========================================================================


class TestCalciumDynamics:
    def _S(self, K=2, nt=200) -> np.ndarray:
        np.random.seed(42)
        S = np.zeros((K, nt), dtype=np.float32)
        S[:, 50] = 7.6e-6
        S[:, 100] = 7.6e-6
        return S

    def _cp(self, sat_type="Ca_DE") -> CalciumParams:
        return CalciumParams(prot_type="gcamp6f", sat_type=sat_type, dt=1 / 100)

    def test_ca_de_returns_three_arrays(self):
        CB, C, F = calcium_dynamics(self._S(), self._cp("Ca_DE"), "gcamp6f")
        assert CB.shape == (2, 200)
        assert C.shape == (2, 200)
        assert F.shape == (2, 200)

    def test_ca_de_float32(self):
        CB, C, F = calcium_dynamics(self._S(), self._cp("Ca_DE"), "gcamp6f")
        assert CB.dtype == np.float32
        assert F.dtype == np.float32

    def test_f_positive(self):
        CB, C, F = calcium_dynamics(self._S(), self._cp("Ca_DE"), "gcamp6f")
        assert float(F.min()) > 0.0

    def test_single_mode_shape(self):
        CB, C, F = calcium_dynamics(self._S(), self._cp("single"), "gcamp6f")
        assert F.shape == (2, 200)

    def test_double_mode_shape(self):
        CB, C, F = calcium_dynamics(self._S(), self._cp("double"), "gcamp6f")
        assert F.shape == (2, 200)

    def test_unknown_sat_type(self):
        cp = CalciumParams(prot_type="gcamp6f", sat_type="unknown")
        with pytest.raises(ValueError):
            calcium_dynamics(self._S(), cp, "gcamp6f")

    def test_ext_mult_reduces_fluorescence(self):
        # Higher extrusion rate (larger ext_mult) should produce lower peak F
        S = self._S()
        cp = self._cp("Ca_DE")
        _, _, F1 = calcium_dynamics(S, cp, "gcamp6f", ext_mult=1.0)
        _, _, F2 = calcium_dynamics(S, cp, "gcamp6f", ext_mult=4.0)
        assert F1.max() >= F2.max()


# ===========================================================================
# expression_variation
# ===========================================================================


class TestExpressionVariation:
    def test_shape(self):
        np.random.seed(42)
        x = expression_variation(10, 0.0, (0.4, 2.53))
        assert x.shape == (10,)

    def test_float32(self):
        np.random.seed(42)
        x = expression_variation(10, 0.0, (0.4, 2.53))
        assert x.dtype == np.float32

    def test_p_off_zero_all_nonzero(self):
        np.random.seed(42)
        x = expression_variation(20, 0.0, (0.4, 2.53))
        assert np.all(x > 0)

    def test_p_off_one_all_zero(self):
        # p_off=1 → all cells get zero expression
        np.random.seed(42)
        x = expression_variation(20, 1.0, (0.4, 2.53))
        np.testing.assert_array_equal(x, np.zeros(20, dtype=np.float32))

    def test_scalar_min_mod_uniform(self):
        np.random.seed(42)
        min_val = 0.3
        x = expression_variation(100, 0.0, min_val)
        assert float(x.min()) >= 0.0  # may have p_off zeros

    def test_gamma_mode(self):
        np.random.seed(42)
        x = expression_variation(50, 0.0, (0.4, 2.53))
        assert x.shape == (50,)
        assert float(x.min()) >= 0.0


# ===========================================================================
# _pick_resample_vals
# ===========================================================================


class TestPickResampleVals:
    def test_30hz(self):
        r1, r2 = _pick_resample_vals(1 / 30)
        assert r1 > 0 and r2 > 0

    def test_positive_integers(self):
        for dt in [1 / 10, 1 / 15, 1 / 30, 1 / 25]:
            r1, r2 = _pick_resample_vals(dt)
            assert isinstance(r1, int) and r1 >= 1
            assert isinstance(r2, int) and r2 >= 1

    def test_10hz_divisible(self):
        r1, r2 = _pick_resample_vals(1 / 10)
        assert r1 == 10
        assert r2 == 1


# ===========================================================================
# generate_time_traces  (end-to-end)
# ===========================================================================


class TestGenerateTimeTraces:
    def _p(self, **kw) -> SpikeParams:
        return _spike_params_small(**kw)

    def test_returns_result_type(self):
        np.random.seed(42)
        result = generate_time_traces(self._p())
        assert isinstance(result, TimeTracesResult)

    def test_soma_shape(self):
        np.random.seed(42)
        result = generate_time_traces(self._p())
        assert result.soma.shape == (3, 50)

    def test_soma_float32(self):
        np.random.seed(42)
        result = generate_time_traces(self._p())
        assert result.soma.dtype == np.float32

    def test_dend_present_when_flag(self):
        np.random.seed(42)
        result = generate_time_traces(self._p(dendflag=True))
        assert result.dend is not None
        assert result.dend.shape == (3, 50)

    def test_dend_none_when_flag_off(self):
        np.random.seed(42)
        result = generate_time_traces(self._p(dendflag=False))
        assert result.dend is None

    def test_spikes_present_when_flag(self):
        np.random.seed(42)
        result = generate_time_traces(self._p(spikeflag=True))
        assert result.spikes is not None
        assert result.spikes.shape == (3, 50)

    def test_spikes_none_when_flag_off(self):
        np.random.seed(42)
        result = generate_time_traces(self._p(spikeflag=False))
        assert result.spikes is None

    def test_mod_vals_shape(self):
        np.random.seed(42)
        result = generate_time_traces(self._p())
        assert result.mod_vals.shape == (3,)

    def test_params_stored(self):
        np.random.seed(42)
        result = generate_time_traces(self._p())
        assert "spike_params" in result.params
        assert "cal_params" in result.params

    def test_ar2_mode(self):
        np.random.seed(42)
        p = self._p(dyn_type="AR2")
        result = generate_time_traces(p)
        assert result.soma.shape == (3, 50)

    def test_ca_de_mode(self):
        np.random.seed(42)
        p = SpikeParams(K=2, nt=30, dt=1 / 30, dyn_type="Ca_DE",
                        smod_flag="poisson", burst_mean=0,
                        dendflag=False, axonflag=False, N_bg=0, verbose=0)
        result = generate_time_traces(p)
        assert result.soma.shape == (2, 30)
        assert float(result.soma.min()) > 0.0

    def test_single_mode(self):
        np.random.seed(42)
        p = SpikeParams(K=2, nt=30, dt=1 / 30, dyn_type="single",
                        smod_flag="poisson", burst_mean=0,
                        dendflag=False, axonflag=False, N_bg=0, verbose=0)
        result = generate_time_traces(p)
        assert result.soma.shape == (2, 30)

    def test_hawkes_smod(self):
        np.random.seed(42)
        p = SpikeParams(K=4, nt=30, dt=1 / 30, dyn_type="AR1",
                        smod_flag="hawkes", burst_mean=10,
                        dendflag=False, axonflag=False, N_bg=0, verbose=0)
        result = generate_time_traces(p)
        assert result.soma.shape == (4, 30)

    def test_axonflag_bg_shape(self):
        np.random.seed(42)
        p = self._p(axonflag=True, dendflag=False)
        result = generate_time_traces(p)
        # axonflag → bg has K rows
        assert result.bg is not None
        assert result.bg.shape[0] == 3

    # ------------------------------------------------------------------
    # Static (non-Ca-dependent) indicator mode: tdTomato / BFP
    # ------------------------------------------------------------------
    def test_static_shape(self):
        np.random.seed(42)
        p = SpikeParams(K=5, nt=40, dt=1 / 20, dyn_type="static",
                        prot="tdTomato", dendflag=True, axonflag=True,
                        N_bg=0, verbose=0)
        result = generate_time_traces(p)
        assert result.soma.shape == (5, 40)
        assert result.dend.shape == (5, 40)
        assert result.bg.shape == (5, 40)

    def test_static_is_constant_in_time(self):
        """A static indicator has zero activity variation: every trace is flat."""
        np.random.seed(0)
        p = SpikeParams(K=8, nt=50, dt=1 / 20, dyn_type="static",
                        prot="BFP", dendflag=True, axonflag=True,
                        N_bg=0, verbose=0)
        result = generate_time_traces(p)
        for arr in (result.soma, result.dend, result.bg):
            # max per-row temporal std is float round-off, not real variation
            assert float(np.abs(arr - arr[:, :1]).max()) < 1e-4

    def test_static_no_spikes(self):
        np.random.seed(0)
        p = SpikeParams(K=4, nt=30, dt=1 / 20, dyn_type="static",
                        prot="tdTomato", dendflag=False, axonflag=False,
                        spikeflag=True, N_bg=0, verbose=0)
        result = generate_time_traces(p)
        assert result.spikes is not None
        assert float(result.spikes.sum()) == 0.0
        assert result.dend is None
        assert result.bg is None

    def test_static_expression_heterogeneity(self):
        """Per-cell brightness comes from mod_vals; constant across time."""
        np.random.seed(1)
        p = SpikeParams(K=30, nt=20, dt=1 / 20, dyn_type="static",
                        prot="tdTomato", dendflag=False, axonflag=False,
                        p_off=0.2, N_bg=0, verbose=0)
        result = generate_time_traces(p)
        col0 = result.soma[:, 0]
        # heterogeneous cells + some switched fully off by p_off
        assert col0.std() > 0
        assert int((col0 == 0).sum()) >= 1
        # each cell's value equals its mod_val (constant baseline 1.0)
        assert np.allclose(col0, result.mod_vals, atol=1e-5)

    def test_static_no_low_activity_warning(self):
        """The static path must not emit the low-activity spike warning."""
        import warnings
        np.random.seed(2)
        p = SpikeParams(K=6, nt=40, dt=1 / 20, dyn_type="static",
                        prot="BFP", dendflag=False, axonflag=False,
                        N_bg=0, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            generate_time_traces(p)

    def test_predef_s_times(self):
        np.random.seed(42)
        # Pass a pre-built spike matrix at 100 Hz
        p = SpikeParams(K=3, nt=40, dt=1 / 100, dyn_type="AR1",
                        smod_flag="poisson", burst_mean=0,
                        dendflag=False, axonflag=False, N_bg=0, verbose=0)
        S_pre = np.zeros((3, 40), dtype=np.float32)
        S_pre[:, 10] = 1.0
        result = generate_time_traces(p, s_times=S_pre)
        assert result.soma.shape[0] == 3

    def test_expression_mod_all_zero(self):
        np.random.seed(42)
        p = self._p()
        mod = np.zeros(3, dtype=np.float32)
        result = generate_time_traces(p, mod_vals=mod)
        np.testing.assert_array_equal(result.soma, np.zeros_like(result.soma))

    def test_reproducible_with_seed(self):
        np.random.seed(0)
        r1 = generate_time_traces(self._p())
        np.random.seed(0)
        r2 = generate_time_traces(self._p())
        np.testing.assert_array_equal(r1.soma, r2.soma)

    def test_bg_n_bg(self):
        np.random.seed(42)
        p = SpikeParams(K=2, nt=30, dt=1 / 30, dyn_type="AR1",
                        smod_flag="poisson", burst_mean=0,
                        dendflag=False, axonflag=False, N_bg=2, verbose=0)
        result = generate_time_traces(p)
        assert result.bg is not None
        assert result.bg.shape[0] == 2

    def test_axon_and_bg_raises(self):
        p = SpikeParams(K=2, nt=30, dt=1 / 30, dyn_type="AR1",
                        smod_flag="poisson", burst_mean=0,
                        dendflag=False, axonflag=True, N_bg=2, verbose=0)
        with pytest.raises(ValueError, match="background"):
            generate_time_traces(p)

    def test_verbose_output(self, capsys):
        np.random.seed(42)
        p = self._p(verbose=1)
        generate_time_traces(p)
        captured = capsys.readouterr()
        assert "generate_time_traces" in captured.out
