"""
Per-cell fluorescence expression modulation.

Port of MATLAB: TimeTraceCode/expression_variation.m
"""

from __future__ import annotations

import numpy as np


def expression_variation(
    n: int,
    p_off: float,
    min_mod: tuple | np.ndarray,
) -> np.ndarray:
    """
    Generate per-cell expression modulation factors.

    Port of ``TimeTraceCode/expression_variation.m``.

    Each cell is assigned a modulation value drawn from either a uniform
    distribution (when ``min_mod`` is a scalar) or a Gamma distribution
    (when ``min_mod`` is a 2-element array).  With probability ``p_off``
    the modulation is set to zero (the cell has no expression).

    Parameters
    ----------
    n:
        Number of cells.
    p_off:
        Probability that a cell has zero fluorescence expression.
    min_mod:
        Modulation parameters.  If scalar: uniform draw in
        ``[min_mod, 1]``.  If 2-element ``(shape, scale)``: Gamma draw
        with those parameters (matching MATLAB ``gamrnd(min_mod(2),
        min_mod(1), N, 1)``).

    Returns
    -------
    np.ndarray
        Length-``n`` float32 array of modulation values in ``[0, ∞)``.
    """
    N = int(n)
    mm = np.atleast_1d(np.asarray(min_mod, dtype=float))

    if mm.size == 1:
        x = mm[0] + (1.0 - mm[0]) * np.random.rand(N)
    else:
        # MATLAB: gamrnd(min_mod(2), min_mod(1), N, 1)
        # numpy.random.gamma(shape, scale) = MATLAB gamrnd(shape, scale)
        x = np.random.gamma(mm[1], mm[0], size=N)

    # Each cell set to zero expression with probability p_off
    x = x * (p_off < np.random.rand(N))
    return x.astype(np.float32)
