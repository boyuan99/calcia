"""
Illumination and collection mask computation for two-photon microscopy.

Ports of the mask generation sections in MATLAB
``simulate_optical_propagation.m`` (lines 460-487 for collection mask).
The illumination mask uses a simplified Beer-Lambert scattering model in
place of the full Fresnel wave-optics path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

if TYPE_CHECKING:
    from ..config.params import PsfParams, VolumeParams


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _group_z_project(vol: np.ndarray, chunk_size: int) -> np.ndarray:
    """Sum-project a 3D volume along Z in fixed-size chunks.

    Port of MATLAB ``groupzproject(vol, chunk_size, 'sum')``.

    Args:
        vol: 3D array of shape (Nx, Ny, Nz).
        chunk_size: Number of Z-slices to sum into each output slice.

    Returns:
        3D array of shape ``(Nx, Ny, ceil(Nz / chunk_size))``.
    """
    Nx, Ny, Nz = vol.shape
    n_chunks = int(np.ceil(Nz / chunk_size))
    out = np.zeros((Nx, Ny, n_chunks), dtype=np.float32)
    for k in range(n_chunks):
        sl = vol[:, :, k * chunk_size: (k + 1) * chunk_size]
        out[:, :, k] = sl.sum(axis=2)
    return out


# ---------------------------------------------------------------------------
# Collection mask
# ---------------------------------------------------------------------------

def compute_collection_mask(
    vol_params: "VolumeParams",
    psf_params: "PsfParams",
    vessel_volume: Optional[np.ndarray],
) -> np.ndarray:
    """Compute the 2D collection mask from hemoglobin absorption.

    Direct port of MATLAB ``simulate_optical_propagation.m`` lines 460-487.

    The collection cone at depth z has radius
    ``coldist = vres * tan(asin(obj_na/n)) * (focal_depth - z_center_um)``
    where z_center_um is the depth of the chunk centre.  Each vessel chunk is
    convolved with a normalized circular aperture of that radius, accumulating
    the integrated hemoglobin column seen by emitted photons on their way out.

    Args:
        vol_params: Volume parameters (vol_sz, vres, vol_depth).
        psf_params: PSF parameters (obj_na, n, hemo_abs, prop_sz).
        vessel_volume: 3D uint8/bool array of shape (Nx, Ny, Nz_full)
            from ``NeuralVolumeOutput.neur_ves``.  If None, returns an
            all-ones mask (no absorption).

    Returns:
        col_mask: 2D float32 array of shape (Nx_vol, Ny_vol) with values
            in (0, 1].  A value of 1 means no absorption.
    """
    vres = vol_params.vres
    Nx_vol = int(vol_params.vol_sz[0] * vres)
    Ny_vol = int(vol_params.vol_sz[1] * vres)

    if vessel_volume is None:
        return np.ones((Nx_vol, Ny_vol), dtype=np.float32)

    proppx = max(1, round(psf_params.prop_sz * vres))

    # Sum-project vessel volume in Z-chunks
    vasc = vessel_volume.astype(np.float32)
    Nx_v, Ny_v, Nz_v = vasc.shape
    chunks = _group_z_project(vasc, proppx)          # (Nx_v, Ny_v, n_chunks)
    n_chunks = chunks.shape[2]

    # Focal depth in voxels from top of vessel volume
    focal_depth_um = float(vol_params.vol_depth) + float(vol_params.vol_sz[2]) / 2.0
    half_angle = np.arcsin(psf_params.obj_na / psf_params.n)

    colmask = np.zeros((Nx_v, Ny_v), dtype=np.float32)

    # Grid for the circular aperture (same XY size as vasc)
    cx, cy = Nx_v // 2, Ny_v // 2
    xv = np.arange(Nx_v) - cx
    yv = np.arange(Ny_v) - cy
    X, Y = np.meshgrid(xv, yv, indexing='ij')
    rho = np.sqrt(X ** 2 + Y ** 2)

    for i in range(n_chunks):
        # Depth of chunk centre in um (measured from top of vessel volume)
        z_center_um = (i + 0.5) * psf_params.prop_sz
        coldist = vres * np.tan(half_angle) * (focal_depth_um - z_center_um)
        if coldist <= 0:
            continue

        # Circular aperture of radius coldist (in pixels)
        aperture = (rho <= coldist).astype(np.float32)
        aperture_sum = aperture.sum()
        if aperture_sum == 0:
            continue
        aperture /= aperture_sum

        # Convolve vessel chunk with aperture; accumulate
        colmask += fftconvolve(chunks[:, :, i], aperture, mode='same').astype(np.float32)

    # Crop to the imaging FOV
    x0 = (Nx_v - Nx_vol) // 2
    y0 = (Ny_v - Ny_vol) // 2
    colmask_crop = colmask[x0: x0 + Nx_vol, y0: y0 + Ny_vol]

    # Beer-Lambert: 10^(-colmask / vres * hemo_abs)
    col_mask = np.power(10.0, -colmask_crop / vres * psf_params.hemo_abs).astype(np.float32)
    return col_mask


# ---------------------------------------------------------------------------
# Illumination mask  (simplified Beer-Lambert, replaces Fresnel wave-optics)
# ---------------------------------------------------------------------------

def compute_illumination_mask(
    vol_params: "VolumeParams",
    psf_params: "PsfParams",
    vessel_volume: Optional[np.ndarray],
) -> np.ndarray:
    """Compute the 2D illumination blockage mask.

    Simplified port of the illumination mask output from the MATLAB
    ``genCorticalLightPathLite`` / ``simulate_optical_propagation`` path.

    The full MATLAB version uses Fresnel wave-optics propagation through a
    3D scattering volume (~500 lines).  This implementation uses a physically
    motivated Beer-Lambert model:

    1. Extract the vessel volume above the focal plane (the region the beam
       travels through before reaching the FOV).
    2. For each scattering scale in ``psf_params.scatter_sz`` / ``scatter_wt``,
       blur the depth-weighted vessel projection with a Gaussian of that sigma,
       then accumulate with the corresponding weight.
    3. Exponentiate negatively: ``mask = exp(-accumulated_scatter)``.
    4. Normalize so ``mean(mask) = 1``.

    Args:
        vol_params: Volume parameters (vol_sz, vres, vol_depth).
        psf_params: PSF parameters (scatter_sz, scatter_wt, n, obj_na).
        vessel_volume: 3D uint8/bool array of shape (Nx, Ny, Nz_full).
            Nz_full covers both the tissue above and the imaging volume.
            If None, returns an all-ones mask.

    Returns:
        mask: 2D float32 array of shape (Nx_vol, Ny_vol), normalized so
            ``mean(mask) == 1``.
    """
    vres = vol_params.vres
    Nx_vol = int(vol_params.vol_sz[0] * vres)
    Ny_vol = int(vol_params.vol_sz[1] * vres)

    if vessel_volume is None:
        return np.ones((Nx_vol, Ny_vol), dtype=np.float32)

    vasc = vessel_volume.astype(np.float32)
    Nx_v, Ny_v, Nz_v = vasc.shape

    # Number of Z-slices that correspond to the tissue above the FOV
    z_above_px = int(vol_params.vol_depth * vres)
    z_above_px = min(z_above_px, Nz_v)

    if z_above_px == 0:
        # No tissue above — minimal attenuation
        return np.ones((Nx_vol, Ny_vol), dtype=np.float32)

    # Depth-weighted projection of vessels above focal plane.
    # Slices near the surface (small z index) attenuate the beam most.
    # Weight linearly: w[z] = (z_above_px - z) / z_above_px  (1 at surface, 0 at FOV)
    weights = np.linspace(1.0, 0.0, z_above_px, dtype=np.float32)
    vessel_above = vasc[:, :, :z_above_px]           # (Nx_v, Ny_v, z_above_px)
    proj = (vessel_above * weights[np.newaxis, np.newaxis, :]).sum(axis=2)  # (Nx_v, Ny_v)

    # Accumulate scatter over each scale
    accumulated = np.zeros((Nx_v, Ny_v), dtype=np.float32)
    for sigma_um, wt in zip(psf_params.scatter_sz, psf_params.scatter_wt):
        sigma_px = sigma_um * vres
        blurred = gaussian_filter(proj, sigma=sigma_px)
        accumulated += float(wt) * blurred

    mask_full = np.exp(-accumulated).astype(np.float32)

    # Crop to FOV
    x0 = (Nx_v - Nx_vol) // 2
    y0 = (Ny_v - Ny_vol) // 2
    mask = mask_full[x0: x0 + Nx_vol, y0: y0 + Ny_vol]

    # Normalize so mean == 1 (matches MATLAB: psf = psf / mean(mask))
    m = float(mask.mean())
    if m > 0:
        mask = mask / m
    return mask.astype(np.float32)
