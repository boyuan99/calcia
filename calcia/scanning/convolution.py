"""
PSF convolution utilities for scanning simulation.

Port of MATLAB: ``psf_fft.m``, ``single_scan.m``, ``blurredBackComp2.m``,
``nearest_small_prime.m``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def nearest_small_prime(n: int, max_factor: int = 7) -> int:
    """Find smallest integer >= *n* whose prime factors are all <= *max_factor*.

    Port of MATLAB ``nearest_small_prime.m``.
    """
    n = int(round(n))
    if n <= 0:
        return n
    while _max_prime_factor(n) > max_factor:
        n += 1
    return n


def _max_prime_factor(n: int) -> int:
    """Return the largest prime factor of *n* (1 if *n* <= 1)."""
    if n <= 1:
        return 1
    max_f = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            max_f = max(max_f, d)
            n //= d
        d += 1
    if n > 1:
        max_f = max(max_f, n)
    return max_f


def _nearest_small_prime_vec(sizes: Sequence[int], max_factor: int = 7) -> Tuple[int, ...]:
    """Apply :func:`nearest_small_prime` element-wise."""
    return tuple(nearest_small_prime(int(s), max_factor) for s in sizes)


# ------------------------------------------------------------------
# PSF FFT pre-computation
# ------------------------------------------------------------------

def psf_fft(
    vol_shape: Tuple[int, int, int],
    psf: np.ndarray,
    z_sub: int = 1,
) -> np.ndarray:
    """Pre-compute the 2-D FFT of the PSF for fast Fourier-domain convolution.

    Port of MATLAB ``psf_fft.m``.

    Parameters
    ----------
    vol_shape : (N1, N2, N3)
        Spatial shape of the volume that will be scanned.
    psf : np.ndarray
        3-D PSF array of shape ``(Np1, Np2, Np3)``.
    z_sub : int
        Axial pre-summing factor.  If > 1, every *z_sub* PSF slices are
        summed together before the FFT, reducing the number of per-slice
        operations during scanning.

    Returns
    -------
    freq_psf : np.ndarray
        Complex array — the 2-D FFT of the (possibly z-sub-summed) PSF,
        padded to an FFT-friendly size.
    """
    if z_sub > 1:
        psf_sub = _presub_z(psf, z_sub)
    else:
        psf_sub = psf

    conv_shape = (
        vol_shape[0] + psf_sub.shape[0] - 1,
        vol_shape[1] + psf_sub.shape[1] - 1,
    )
    fft_shape = _nearest_small_prime_vec(conv_shape)

    # 2-D FFT along axes 0 and 1 for each z-slice
    return np.fft.fft2(psf_sub, s=fft_shape, axes=(0, 1))


# ------------------------------------------------------------------
# Single-frame scan (3-D → 2-D via PSF convolution)
# ------------------------------------------------------------------

def single_scan(
    vol: np.ndarray,
    psf_shape: Tuple[int, int, int],
    freq_psf: np.ndarray,
    z_sub: int = 1,
) -> np.ndarray:
    """Convolve a 3-D volume with a PSF in the Fourier domain, producing a 2-D image.

    Port of MATLAB ``single_scan.m`` (FFT path).

    Parameters
    ----------
    vol : np.ndarray
        3-D volume of shape ``(N1, N2, N3)``.
    psf_shape : (Np1, Np2, Np3)
        Original PSF shape (before any z-sub-summing), used to determine
        the crop region.
    freq_psf : np.ndarray
        Pre-computed 2-D FFT of the PSF (from :func:`psf_fft`).
    z_sub : int
        Axial pre-summing factor (must match the value used in
        :func:`psf_fft`).

    Returns
    -------
    scan_img : np.ndarray
        2-D float32 scanned image.
    """
    if z_sub > 1:
        vol_sub = _presub_z(vol, z_sub)
    else:
        vol_sub = vol

    fft_shape = (freq_psf.shape[0], freq_psf.shape[1])

    # FFT-based convolution: FFT2(vol) * FFT2(psf), sum over z, IFFT2
    freq_vol = np.fft.fft2(vol_sub, s=fft_shape, axes=(0, 1))
    scan_full = np.fft.ifft2(
        np.sum(freq_vol * freq_psf, axis=2),
        axes=(0, 1),
    ).real

    # Crop to valid convolution region (matching MATLAB cropping)
    # MATLAB: y_ix = ceil((psf_sz(1)-1)/2) + [1, size(vol_sub,1)]
    #         y_jx = ceil((psf_sz(2)-1)/2) + [1, size(vol_sub,2)]
    if z_sub > 1:
        psf_shape_sub = (psf_shape[0], psf_shape[1],
                         int(np.ceil(psf_shape[2] / z_sub)))
    else:
        psf_shape_sub = psf_shape

    row_off = int(np.ceil((psf_shape_sub[0] - 1) / 2))
    col_off = int(np.ceil((psf_shape_sub[1] - 1) / 2))
    scan_img = scan_full[row_off:row_off + vol_sub.shape[0],
                         col_off:col_off + vol_sub.shape[1]]

    return scan_img.astype(np.float32)


# ------------------------------------------------------------------
# Out-of-focus background
# ------------------------------------------------------------------

def blurred_back_comp(
    vol: np.ndarray,
    z_indices: np.ndarray,
    freq_psf_lr: np.ndarray,
    weight: float,
    mask: Optional[np.ndarray] = None,
    z_scale: Optional[np.ndarray] = None,
    extra_vols: Optional[list] = None,
) -> np.ndarray:
    """Compute out-of-focus fluorescence contribution.

    Port of MATLAB ``blurredBackComp2.m``.

    Parameters
    ----------
    vol : np.ndarray
        3-D volume ``(N1, N2, N3)``.
    z_indices : np.ndarray
        Integer indices of the z-slices to include.
    freq_psf_lr : np.ndarray
        2-D FFT of the low-resolution PSF (already padded to FFT size).
    weight : float
        Power ratio scaling factor.
    mask : np.ndarray or None
        2-D spatial mask applied before convolution.
    z_scale : np.ndarray or None
        Depth-dependent weights for the selected z-slices.
    extra_vols : list of np.ndarray or None
        Additional 3-D volumes whose z-slices are averaged and added.

    Returns
    -------
    img_out : np.ndarray
        2-D float32 out-of-focus image.
    """
    n_idx = len(z_indices)
    if n_idx == 0:
        return np.zeros((vol.shape[0], vol.shape[1]), dtype=np.float32)

    # Average selected z-slices (optionally depth-weighted)
    if z_scale is not None and len(z_scale) >= n_idx:
        zs = z_scale[:n_idx].reshape(1, 1, -1)
        img_out = (1.0 / n_idx) * np.sum(vol[:, :, z_indices] * zs, axis=2)
    else:
        img_out = (1.0 / n_idx) * np.sum(vol[:, :, z_indices], axis=2)

    # Add contributions from extra volumes
    if extra_vols is not None:
        for ev in extra_vols:
            if z_scale is not None and len(z_scale) >= n_idx:
                v_vec = np.zeros(ev.shape[2])
                v_vec[z_indices] = z_scale[:n_idx] / n_idx
                img_out += (ev.reshape(-1, ev.shape[2]) @ v_vec).reshape(
                    ev.shape[0], ev.shape[1])
            else:
                v_vec = np.zeros(ev.shape[2])
                v_vec[z_indices] = 1.0 / n_idx
                img_out += (ev.reshape(-1, ev.shape[2]) @ v_vec).reshape(
                    ev.shape[0], ev.shape[1])

    img_out *= weight

    if mask is not None:
        img_out *= mask

    # FFT-based convolution with low-res PSF
    fft_shape = (freq_psf_lr.shape[0], freq_psf_lr.shape[1])
    freq_img = np.fft.fft2(img_out, s=fft_shape)
    conv_full = np.fft.ifft2(freq_img * freq_psf_lr).real

    # Crop back to volume size
    vol_n1, vol_n2 = vol.shape[0], vol.shape[1]
    row_off = int(np.ceil((fft_shape[0] - vol_n1 - 1) / 2))
    col_off = int(np.ceil((fft_shape[1] - vol_n2 - 1) / 2))
    img_out = conv_full[row_off:row_off + vol_n1,
                        col_off:col_off + vol_n2]

    return img_out.astype(np.float32)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _presub_z(arr: np.ndarray, z_sub: int) -> np.ndarray:
    """Pre-sum every *z_sub* slices along axis 2.

    Matches MATLAB logic in ``psf_fft.m`` / ``single_scan.m``.
    """
    n_z = arr.shape[2]
    n_slices = int(np.ceil(n_z / z_sub))
    out = np.zeros((*arr.shape[:2], n_slices), dtype=arr.dtype)
    for k in range(z_sub):
        slc = slice(k, min(z_sub * n_slices, n_z), z_sub)
        n = len(range(*slc.indices(n_z)))
        out[:, :, :n] += arr[:, :, slc]
    return out
