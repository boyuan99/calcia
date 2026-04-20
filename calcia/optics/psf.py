"""
Point-spread function utilities for two-photon microscopy simulation.

Port of MATLAB files:
  - gaussian_psf_na.m
  - gaussianBeamSize.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..config.params import PsfParams, VolumeParams


@dataclass
class PsfTail:
    """PSF energy in the volume tail above or below the focal plane.

    Mirrors the psfT / psfB struct returned by genCorticalLightPathLite in
    MATLAB.

    Attributes:
        weights: 2D float32 array — lateral convolution kernel (MATLAB ``convmask``).
        mask: 2D float32 — spatial modulation mask.
        weight: Scalar float — total tail energy relative to the main PSF
            (MATLAB ``psfTS.weight``).  Used as ``pwr_ratio`` in
            ``blurredBackComp2``.  Defaults to 1.0 for backward compatibility.
        z_weights: 1D float32 — depth-dependent weights for out-of-focus slices
            (MATLAB ``psfTS.psfZ``).  Normalized to mean=1.  When ``None``,
            all z-slices contribute equally.
    """
    weights: np.ndarray
    mask: np.ndarray
    weight: float = 1.0
    z_weights: Optional[np.ndarray] = None


def gaussian_psf_na(
    na: float,
    lambda_um: float,
    sampling: Tuple[float, float, float],
    mat_size: Tuple[int, int, int],
    theta: float = 0.0,
    nidx: float = 1.33,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a two-photon Gaussian PSF volume from numerical aperture.

    Port of MATLAB ``gaussian_psf_na.m``.

    Algorithm:
      1. ``psflen = 0.626 * lambda / (nidx - sqrt(nidx^2 - na^2))``  [Abbe]
      2. ``zr = psflen / 2``  (Rayleigh range)
      3. Build coordinate grids via ``np.meshgrid(..., indexing='ij')``
         — equivalent to MATLAB ``meshgrid`` + ``permute([2 1 3])``.
      4. Apply optional tilt rotation in the x-z plane (theta in degrees).
      5. ``intensity = exp(-2*pi*nidx*(x^2+y^2) / (zr*lambda*(1+(z/zr)^2)))
                        / (1+(z/zr)^2)``
      6. ``psf = intensity^2``  (two-photon)

    Args:
        na: Excitation numerical aperture.
        lambda_um: Excitation wavelength in microns.
        sampling: (dx, dy, dz) voxel spacing in um for each axis.
        mat_size: (Nx, Ny, Nz) output array size in voxels.
        theta: Beam tilt angle in degrees (rotation in the x-z plane).
        nidx: Refractive index of the medium.

    Returns:
        Tuple ``(psf, x, y, z)`` where:
          - psf: float32 array of shape ``(Nx, Ny, Nz)`` — two-photon PSF.
          - x: 1D float64 x-coordinates in um.
          - y: 1D float64 y-coordinates in um.
          - z: 1D float64 z-coordinates in um.

    Raises:
        ValueError: If ``na`` is zero or ``na >= nidx`` (undefined Abbe formula).
    """
    if na <= 0:
        raise ValueError(f"na must be positive, got {na}")
    if na >= nidx:
        raise ValueError(f"na ({na}) must be less than nidx ({nidx})")

    # Normalize sampling to 3 elements
    dx, dy, dz = float(sampling[0]), float(sampling[1]), float(sampling[2])
    Nx, Ny, Nz = int(mat_size[0]), int(mat_size[1]), int(mat_size[2])

    # Abbe formula: PSF length (Rayleigh range * 2) from NA
    psflen = 0.626 * lambda_um / (nidx - np.sqrt(nidx ** 2 - na ** 2))
    zr = psflen / 2.0

    # Coordinate vectors centered at zero.
    # Use MATLAB convention: round(N/2) as center index, so the zero point
    # falls exactly on a grid node (guarantees peak intensity == 1.0).
    # Equivalent to MATLAB: ((0:N-1) - round(N/2)) * sampling
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    z = (np.arange(Nz) - Nz // 2) * dz

    # Build 3D grids — indexing='ij' gives shape (Nx, Ny, Nz), equivalent to
    # MATLAB meshgrid(x,y,z) followed by permute(psf,[2 1 3])
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')

    # Optional tilt: rotate x-z plane by theta degrees
    if theta != 0.0:
        theta_rad = np.deg2rad(theta)
        xg2 = np.cos(theta_rad) * xg - np.sin(theta_rad) * zg
        zg2 = np.sin(theta_rad) * xg + np.cos(theta_rad) * zg
    else:
        xg2 = xg
        zg2 = zg

    # Gaussian beam intensity envelope
    denom = zr * lambda_um * (1.0 + (zg2 / zr) ** 2)
    intensity = np.exp(-2.0 * np.pi * nidx * (xg2 ** 2 + yg ** 2) / denom) / (
        1.0 + (zg2 / zr) ** 2
    )

    # Two-photon: square the intensity
    psf = (intensity ** 2).astype(np.float32)

    return psf, x, y, z


def gaussian_beam_size(
    psf_params: "PsfParams",
    dist: float,
    apod: float = 2.0,
) -> np.ndarray:
    """Compute the lateral Gaussian beam extent at a distance from focus.

    Port of MATLAB ``gaussianBeamSize.m``.

    Formula::

        gauss_sz = ceil(tan(asin(obj_na / n)) * dist * 1.5) * apod * [1, 1, 0]

    Args:
        psf_params: PSF parameters (uses obj_na and n).
        dist: Distance from the focal point in um.
        apod: Apodization scaling factor (default 2).

    Returns:
        3-element float64 array ``[X_extent, Y_extent, 0.0]`` in um.
        The Z component is always 0 (beam size is a lateral quantity).
    """
    half_angle = np.arcsin(psf_params.obj_na / psf_params.n)
    lateral = np.ceil(np.tan(half_angle) * dist * 1.5) * apod
    return np.array([lateral, lateral, 0.0])


def compute_psf_tails(
    psf: np.ndarray,
    vol_params: "VolumeParams",
    psf_params: "PsfParams",
) -> Tuple[PsfTail, PsfTail]:
    """Compute PSF tail weights above and below the focal plane.

    Simplified port of the psfT / psfB outputs from genCorticalLightPathLite.
    Tail weights are derived by integrating PSF energy in the z-slabs
    outside the imaging volume.

    The focal plane is assumed to be the centre of the PSF along Z.

    Args:
        psf: 3D float32 PSF array of shape (Nx, Ny, Nz).
        vol_params: Volume parameters (vol_sz, vres).
        psf_params: PSF parameters (tail_length, psf_sz).

    Returns:
        Tuple ``(psf_top, psf_bot)`` — PsfTail structs for the region above
        and below the focal plane respectively.
    """
    Nx, Ny, Nz = psf.shape
    z_center = Nz // 2

    # Top half: z < center (above focal plane in tissue convention)
    top_vol = psf[:, :, :z_center]
    bot_vol = psf[:, :, z_center:]

    # Lateral weights: sum along Z, then normalize
    top_w = top_vol.sum(axis=2).astype(np.float32)
    bot_w = bot_vol.sum(axis=2).astype(np.float32)

    top_sum = float(top_w.sum()) or 1.0
    bot_sum = float(bot_w.sum()) or 1.0

    top_mask = (top_w / top_sum).astype(np.float32)
    bot_mask = (bot_w / bot_sum).astype(np.float32)

    # Interpolate masks to full volume XY size (MATLAB uses griddata).
    vres = vol_params.vres
    vol_xy = (vol_params.vol_sz[0] * vres, vol_params.vol_sz[1] * vres)
    if top_mask.shape != vol_xy:
        from scipy.ndimage import zoom
        zf = (vol_xy[0] / top_mask.shape[0], vol_xy[1] / top_mask.shape[1])
        top_mask = zoom(top_mask, zf, order=3).astype(np.float32)
        bot_mask = zoom(bot_mask, zf, order=3).astype(np.float32)

    # Scalar weight: total tail energy (MATLAB psfTS.weight = mean(psfT(:)))
    top_weight = float(top_w.mean())
    bot_weight = float(bot_w.mean())

    return (PsfTail(weights=top_w, mask=top_mask, weight=top_weight),
            PsfTail(weights=bot_w, mask=bot_mask, weight=bot_weight))
