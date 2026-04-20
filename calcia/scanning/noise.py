"""
Noise models for scanning simulation.

Port of MATLAB: ``PoissonGaussNoiseModel.m``, ``pixel_bleed.m``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config.params import NoiseParams


def poisson_gauss_noise(
    clean: np.ndarray,
    noise_params: "NoiseParams",
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Poisson–lognormal–Gaussian noise model to a clean image.

    Port of MATLAB ``PoissonGaussNoiseModel.m``.

    Three stages:
    1. **Poisson**: draw photon counts from ``Poisson(clean + darkcount)``.
    2. **Lognormal**: scale by PMT gain (mu, sigma) → lognormal sample.
    3. **Gaussian**: add electronic readout noise ``N(mu0, sigma0)``.

    Parameters
    ----------
    clean : np.ndarray
        Non-negative fluorescence intensity (2-D image).
    noise_params : NoiseParams
        PMT / electronics noise parameters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    noisy : np.ndarray
        Noisy measurement (same shape as *clean*), rounded to integers.
    """
    mu = noise_params.mu
    sigma = noise_params.sigma
    mu0 = noise_params.mu0
    sigma0 = noise_params.sigma0
    darkcount = noise_params.darkcount

    # Stage 1: Poisson photon counts
    cnt = rng.poisson(np.maximum(clean + darkcount, 0)).astype(np.float64)

    # Stage 2: Lognormal PMT gain
    m = cnt * mu
    v = cnt * sigma
    # Lognormal parameters (avoid log of zero)
    safe = m > 0
    mu2 = np.zeros_like(m)
    sigma2 = np.zeros_like(m)
    mu2[safe] = np.log(m[safe] ** 2 / np.sqrt(v[safe] + m[safe] ** 2))
    sigma2[safe] = np.sqrt(np.log(v[safe] / m[safe] ** 2 + 1))

    noisy = np.where(safe, rng.lognormal(mu2, np.maximum(sigma2, 1e-30)), 0.0)

    # Stage 3: Gaussian electronic noise
    noisy += rng.normal(mu0, sigma0, size=noisy.shape)
    noisy = np.round(noisy).astype(np.float32)

    return noisy


def pixel_bleed(
    frame: np.ndarray,
    p: float,
    b_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate pixel-to-pixel bleed-through from readout electronics.

    Port of MATLAB ``pixel_bleed.m``.

    Each pixel bleeds charge to its right and bottom neighbours with
    probability *p* and maximum amplitude *b_max*.

    Parameters
    ----------
    frame : np.ndarray
        2-D image (float32).
    p : float
        Bleed probability (0 = no bleed).
    b_max : float
        Maximum bleed amplitude (fraction of pixel value).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    frame_out : np.ndarray
        Image with pixel bleed-through applied.
    """
    if p <= 0:
        return frame.copy()

    x_bleed = b_max * np.maximum(rng.random(frame.shape) - (1 - p), 0) / p

    # Shifted copy: each pixel receives bleed from left and above neighbour
    # MATLAB: [[0;x_bleed(1:end-1,end)], x_bleed(:,1:end-1)]
    #  → column 0 gets the last column shifted down by 1 row
    #  → columns 1..end get columns 0..end-1
    shifted_bleed = np.empty_like(x_bleed)
    shifted_bleed[:, 1:] = x_bleed[:, :-1]
    shifted_bleed[0, 0] = 0
    shifted_bleed[1:, 0] = x_bleed[:-1, -1]

    shifted_frame = np.empty_like(frame)
    shifted_frame[:, 1:] = frame[:, :-1]
    shifted_frame[0, 0] = 0
    shifted_frame[1:, 0] = frame[:-1, -1]

    frame_out = frame - x_bleed * frame + shifted_bleed * shifted_frame
    return frame_out.astype(np.float32)
