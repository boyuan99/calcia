"""
Neural connectivity generation and correlated spike-train simulation.

Ports of MATLAB functions:
  - MiscCode/sampSmallWorldMat.m
  - TimeTraceCode/genCorrelatedSpikeTrains2.m  (discrete Hawkes path only)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.linalg import toeplitz

from calcia.config.params import SpikeParams
from calcia.traces.spikes import bin_spike_trains


def samp_small_world_mat(
    n_node: int | tuple,
    k_conn: int,
    beta: float,
    rand_opt: float = 0.0,
    self_ex: float | np.ndarray = 4.0,
    n_locs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a random small-world connectivity matrix (Watts-Strogatz model).

    Port of ``MiscCode/sampSmallWorldMat.m``.

    Parameters
    ----------
    n_node:
        Number of soma nodes.  If a 2-element tuple ``(N_soma, N_bg)``
        background nodes are appended as fully-connected rows.
    k_conn:
        Initial number of local connections per node.
    beta:
        Rewiring probability (0 = pure lattice, 1 = random).
    rand_opt:
        If > 0, connection weights are drawn from ``Uniform(0.1, 1.0)``
        scaled by ``rand_opt``.
    self_ex:
        Value added to the diagonal (self-excitation).  May be a scalar
        or an array of length ``N_soma + N_bg``.
    n_locs:
        Optional ``N_soma × D`` location matrix.  When provided, initial
        connections are to the ``k_conn`` nearest spatial neighbours.

    Returns
    -------
    np.ndarray
        Float64 ``N × N`` adjacency matrix.
    """
    if isinstance(n_node, (tuple, list, np.ndarray)) and len(n_node) > 1:
        N_soma = int(n_node[0])
        N_bg = int(n_node[1])
    else:
        N_soma = int(np.atleast_1d(n_node)[0])
        N_bg = 0

    N = N_soma  # total before background

    # ------------------------------------------------------------------
    # Build initial lattice adjacency
    # ------------------------------------------------------------------
    use_locs = (
        n_locs is not None
        and np.asarray(n_locs).shape[0] == N_soma
        and N_soma > 0
    )

    # Defensive clamp: MATLAB zeros(1, N-K/2) silently returns [] when K/2 > N
    # (negative size → empty array), so toeplitz produces an oversized matrix
    # without error.  In normal usage K >> k_conn=10; this only matters for
    # small test networks.  We clamp explicitly to avoid a ValueError.
    k_conn_eff = min(k_conn, N_soma)

    if use_locs:
        n_locs = np.asarray(n_locs)
        # Pairwise distance matrix
        dist = np.zeros((N_soma, N_soma))
        for d in range(n_locs.shape[1]):
            dist += (n_locs[:, d : d + 1] - n_locs[:, d]) ** 2
        dist = np.sqrt(dist)
        adj = np.zeros((N_soma, N_soma))
        for i in range(N_soma):
            ix = np.argsort(dist[i])
            adj[i, ix[:k_conn_eff]] = 1.0
    else:
        # Toeplitz lattice: connect k_conn_eff/2 neighbours on each side
        half = k_conn_eff // 2
        n_zeros = max(0, N_soma - half)
        row = np.concatenate([np.ones(half), np.zeros(n_zeros)])[:N_soma]
        adj = toeplitz(row).astype(float)

    # ------------------------------------------------------------------
    # Watts-Strogatz rewiring
    # ------------------------------------------------------------------
    for i in range(N_soma):
        nc_1locs = np.where(adj[i] == 1)[0]
        nc_0locs = np.where(adj[i] == 0)[0]
        if len(nc_1locs) == 0 or len(nc_0locs) == 0:
            continue
        switch_flag = np.random.rand(len(nc_1locs)) < beta
        n_switch = int(switch_flag.sum())
        if n_switch > 0 and len(nc_0locs) >= n_switch:
            new_cons = np.random.choice(len(nc_0locs), size=n_switch, replace=False)
            adj[i, nc_0locs[new_cons]] = 1.0
            adj[i, nc_1locs[switch_flag]] = 0.0

    # ------------------------------------------------------------------
    # Append background nodes
    # ------------------------------------------------------------------
    if N_bg > 0:
        adj = np.block([
            [adj,                          np.zeros((N_soma, N_bg))],
            [np.ones((N_bg, N_soma)),      np.eye(N_bg)],
        ])

    N_total = N_soma + N_bg

    # ------------------------------------------------------------------
    # Self-excitation diagonal
    # ------------------------------------------------------------------
    self_ex_arr = np.atleast_1d(np.asarray(self_ex, dtype=float))
    if self_ex_arr.size == 1:
        np.fill_diagonal(adj, adj.diagonal() + float(self_ex_arr[0]))
    else:
        adj[np.arange(N_total), np.arange(N_total)] += self_ex_arr[:N_total]

    # ------------------------------------------------------------------
    # Randomise weights
    # ------------------------------------------------------------------
    if rand_opt > 0:
        w = 1.0 - rand_opt + rand_opt * (0.1 + 0.9 * np.random.rand(N_total, N_total))
        adj = adj * w

    return adj


def gen_correlated_spike_trains(
    spike_params: SpikeParams,
    n_locs: Optional[np.ndarray] = None,
    discrete: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate correlated spike trains via a discrete Hawkes process.

    Port of ``TimeTraceCode/genCorrelatedSpikeTrains2.m`` (discrete path).

    The continuous-time marked-point-process path (``markpointproc``) is
    not implemented.

    Parameters
    ----------
    spike_params:
        Must have ``K``, ``N_bg``, ``nt``, ``dt``, ``rate``, ``selfact``,
        ``burst_mean`` set.  ``dt`` should already be set to ``1/100``
        (the internal simulation rate) by the caller.
    n_locs:
        Optional ``K × D`` soma location matrix for spatially-informed
        connectivity.
    discrete:
        Must be ``True``; the continuous path is not implemented.

    Returns
    -------
    dict
        ``{'soma': K × nt float32, 'bg': N_bg × nt float32}``
    """
    if not discrete:
        raise NotImplementedError(
            "Continuous Hawkes (markpointproc) is not implemented. Use discrete=True."
        )

    K = spike_params.K
    N_bg = spike_params.N_bg
    N_total = K + N_bg
    nt = spike_params.nt
    dt = spike_params.dt
    selfact = spike_params.selfact

    ascale = 4.0
    bscale = 2.0

    # ------------------------------------------------------------------
    # Sample network connectivity and baseline rates
    # ------------------------------------------------------------------
    MU = np.concatenate([
        np.random.gamma(1.0, spike_params.rate, size=K),
        np.random.gamma(1.0, spike_params.rate, size=N_bg),
    ])
    B = np.concatenate([
        np.random.gamma(3.0, bscale, size=K),
        np.random.gamma(3.0, bscale, size=N_bg),
    ])

    A = samp_small_world_mat(
        (K, N_bg),
        k_conn=10,
        beta=0.3,
        rand_opt=0.9,
        self_ex=0.0,  # diagonal set explicitly below
        n_locs=n_locs,
    )

    # Normalise so mean column sum ≈ ascale
    col_mean = A.sum(axis=0).mean()
    if col_mean > 0:
        A = ascale * A / col_mean

    # Set self-excitation diagonal to selfact * B
    np.fill_diagonal(A, selfact * B)

    # ------------------------------------------------------------------
    # Per-neuron time constants
    # ------------------------------------------------------------------
    extSc = np.maximum(0.3, 1.0 + 0.3 * np.random.randn(N_total))
    inbSc = extSc / 2.0

    # ------------------------------------------------------------------
    # Discrete Hawkes simulation
    # ------------------------------------------------------------------
    alpha_rect = 3.0

    def softplus(z: np.ndarray) -> np.ndarray:
        # log(1 + exp(alpha * z)), numerically stable
        return np.log1p(np.exp(np.clip(alpha_rect * z, -500, 500)))

    zt = np.zeros(N_total)
    yt = np.full(N_total, 5.0)  # MATLAB initialises at 5

    evt_times: list = []
    evt_marks: list = []

    for tt in range(nt):
        rate_t = softplus(zt - yt + 1.0) * MU * dt
        xt = np.random.rand(N_total) < (1.0 - np.exp(-rate_t))
        zt = np.exp(-extSc * dt) * zt + A @ xt
        yt = np.exp(-inbSc * dt) * yt + B * xt

        fired = np.where(xt)[0]
        if fired.size > 0:
            # Spike times uniformly spread within the time bin
            times = (tt + np.random.rand(fired.size)) * dt
            evt_times.append(times)
            evt_marks.append(fired + 1)  # 1-based marks for bin_spike_trains

    # ------------------------------------------------------------------
    # Bin events
    # ------------------------------------------------------------------
    if evt_times:
        all_times = np.concatenate(evt_times)
        all_marks = np.concatenate(evt_marks)
    else:
        all_times = np.array([], dtype=float)
        all_marks = np.array([], dtype=int)

    soma_marks = all_marks[all_marks <= K]
    soma_times = all_times[all_marks <= K]
    bg_marks = all_marks[all_marks > K] - K
    bg_times = all_times[all_marks > K]

    S_soma = bin_spike_trains(soma_times, soma_marks, K, dt, nt)
    S_bg = bin_spike_trains(bg_times, bg_marks, max(N_bg, 1), dt, nt)
    if N_bg == 0:
        S_bg = np.zeros((0, nt), dtype=np.float32)

    return {"soma": S_soma, "bg": S_bg}
