"""
Noise models for scanning simulation.

Port of MATLAB: ``PoissonGaussNoiseModel.m``, ``pixel_bleed.m``. Also
provides :func:`camera_noise`, the sCMOS/CCD counterpart used by the
widefield imaging path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config.params import CameraNoiseParams, NoiseParams


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


def camera_noise(
    clean: np.ndarray,
    cam_params: "CameraNoiseParams",
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply an sCMOS/CCD camera noise model to a clean image.

    Widefield counterpart to :func:`poisson_gauss_noise`. Replaces the PMT
    lognormal-gain chain with a Poisson + read-noise model:

    1. **Photon shot noise**: ``signal_e = Poisson(qe * clean)``.
    2. **Dark current**: ``dark_e = Poisson(dark_rate * t_exp)`` per pixel.
    3. **PRNU**: multiply ``(signal_e + dark_e)`` by a per-pixel gain
       drawn once as ``Normal(1, pixel_gain_sigma)`` (skipped when sigma<=0).
    4. **Read noise**: add ``Normal(0, read_noise)`` per pixel.
    5. **ADC**: divide by ``gain_e_per_adu``, round, clip to
       ``[0, 2**bit_depth - 1]``.

    Parameters
    ----------
    clean : np.ndarray
        Non-negative photon-rate image (2-D). Units: photons per exposure.
    cam_params : CameraNoiseParams
        Camera noise configuration.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    adu : np.ndarray
        Noisy digitized image (float32 ADU, shape matches *clean*).
    """
    qe = cam_params.qe
    dark_rate = cam_params.dark_rate
    t_exp = cam_params.t_exp
    read_noise = cam_params.read_noise
    gain = cam_params.gain_e_per_adu
    prnu_sigma = cam_params.pixel_gain_sigma
    max_adu = float(2 ** cam_params.bit_depth - 1)

    # Stage 1: photon shot noise (QE folded in)
    signal_e = rng.poisson(np.maximum(qe * clean, 0.0)).astype(np.float64)

    # Stage 2: dark current
    dark_mean = dark_rate * t_exp
    if dark_mean > 0:
        dark_e = rng.poisson(dark_mean, size=clean.shape).astype(np.float64)
    else:
        dark_e = 0.0

    total_e = signal_e + dark_e

    # Stage 3: PRNU (fixed-pattern per-pixel gain)
    if prnu_sigma > 0:
        prnu = rng.normal(1.0, prnu_sigma, size=clean.shape)
        total_e = total_e * prnu

    # Stage 4: Gaussian read noise (in electrons)
    if read_noise > 0:
        total_e = total_e + rng.normal(0.0, read_noise, size=clean.shape)

    # Stage 5: ADC
    adu = np.round(total_e / gain)
    adu = np.clip(adu, 0.0, max_adu).astype(np.float32)
    return adu
