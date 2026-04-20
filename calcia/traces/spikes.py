"""
Spike-train generation and binning.

Ports of MATLAB functions:
  - gen_burst_spike_times.m
  - binSpikeTrains.m
"""

from __future__ import annotations

import math

import numpy as np

from calcia.config.params import SpikeParams


def gen_burst_spike_times(spike_params: SpikeParams) -> np.ndarray:
    """
    Generate a binary spike matrix from independent Poisson burst processes.

    Port of ``TimeTraceCode/gen_burst_spike_times.m``.

    Each neuron fires with an exponentially distributed inter-burst interval.
    After each burst onset, ``Poisson(burst_mean)+1`` spikes are placed at
    5 + Uniform(0,2) sample offsets.

    Parameters
    ----------
    spike_params:
        Must have ``K``, ``nt``, ``rate``, ``alpha``, ``rate_dist``,
        ``burst_mean`` fields set.

    Returns
    -------
    np.ndarray
        ``K × nt`` float32 binary (0/1) spike matrix.
    """
    K = spike_params.K
    nt = spike_params.nt
    alpha = spike_params.alpha
    burst_mean = spike_params.burst_mean
    ref_time = 5  # refractory samples between burst spikes

    # Draw per-neuron firing rates
    base_rate = spike_params.rate
    if np.ndim(base_rate) == 0 or np.size(base_rate) == 1:
        if spike_params.rate_dist == "uniform":
            rates = float(base_rate) * np.ones(K)
        else:  # 'gamma'
            rate_tmp = np.random.gamma(alpha, float(base_rate), size=K)
            rates = np.clip(rate_tmp, float(base_rate) / 10.0, float(base_rate) * 10.0)
    else:
        rates = np.asarray(base_rate, dtype=float)

    # Transform rates to mean inter-spike intervals
    isi = 1.0 / rates  # samples between bursts

    S = np.zeros((K, nt), dtype=np.float32)

    for k in range(K):
        T_tot = 0.0
        while T_tot < nt:
            # Exponential inter-burst arrival time
            t_arr = -isi[k] * math.log(max(np.random.rand(), 1e-300))
            idx = int(math.ceil(T_tot + t_arr)) - 1  # 0-based
            if 0 <= idx < nt:
                S[k, idx] = 1.0
            T_tot += t_arr

            # Add intra-burst spikes
            if burst_mean > 0:
                num_in_burst = 1 + np.random.poisson(burst_mean)
                for _ in range(num_in_burst - 1):
                    t_arr2 = ref_time + 2.0 * np.random.rand()
                    idx2 = int(math.ceil(T_tot + t_arr2)) - 1
                    if 0 <= idx2 < nt:
                        S[k, idx2] = 1.0
                    T_tot += t_arr2

    return S


def bin_spike_trains(
    evt: np.ndarray,
    evm: np.ndarray,
    n_node: int,
    dt: float,
    T: int,
) -> np.ndarray:
    """
    Bin continuous-time marked spike events into a count matrix.

    Port of ``TimeTraceCode/binSpikeTrains.m``.

    Parameters
    ----------
    evt:
        1-D array of event times (seconds).
    evm:
        1-D integer array of event marks (1-based neuron IDs).
    n_node:
        Total number of neurons / marks.
    dt:
        Bin width in seconds.
    T:
        Number of time bins.

    Returns
    -------
    np.ndarray
        ``n_node × T`` float32 spike-count matrix.
    """
    evt = np.asarray(evt, dtype=float)
    evm = np.asarray(evm, dtype=int)

    S = np.zeros((n_node, T), dtype=np.float32)
    if evt.size == 0:
        return S

    # Convert to 0-based bin indices (MATLAB: ceil(evt/dt) → 1-based)
    bin_idx = np.ceil(evt / dt).astype(int) - 1
    bin_idx = np.clip(bin_idx, 0, T - 1)
    valid = (bin_idx >= 0) & (bin_idx < T) & (evm >= 1) & (evm <= n_node)

    np.add.at(S, (evm[valid] - 1, bin_idx[valid]), 1.0)
    return S
