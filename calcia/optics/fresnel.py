"""Fresnel wave-optics propagation utilities.

Ports of MATLAB functions:
  - fresnel_propagation_multi.m
  - generateBA.m
  - generateGaussianProfile.m
  - generateZernike.m
  - applyZernike.m
  - zernike.m
  - masked_3DGP_test.m
  - groupzproject.m
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import gamma as _gamma

from ..config.params import PsfParams, VolumeParams
from .psf import gaussian_beam_size

# ---------------------------------------------------------------------------
# Fresnel propagation
# ---------------------------------------------------------------------------


def fresnel_propagation_multi(
    Uin: np.ndarray,
    wavelength: float,
    dx: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
    nidx: float,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """FFT-based angular-spectrum Fresnel propagation through phase screens.

    Port of MATLAB ``fresnel_propagation_multi.m``.

    Args:
        Uin: (N, N) complex input field.
        wavelength: Vacuum wavelength in meters.
        dx: 1-D array of lateral pixel spacings at each z-position [m].
        z: 1-D array of propagation z-positions [m], length *n*.
        phi: (N, N, n) complex phase screens.  Screen ``phi[:,:,i]`` is
            applied at position ``z[i]``.  If ``phi`` has fewer than *n*
            planes the last step propagates without a screen.
        nidx: Refractive index of medium.
        return_all: If True also return all intermediate planes.

    Returns:
        ``Uout`` (N, N) output field, or ``(Uout, UoutAll)`` where
        ``UoutAll`` is (N, N, n).
    """
    lam = wavelength / nidx  # effective wavelength in medium
    N = Uin.shape[0]
    nx, ny = np.meshgrid(
        np.arange(-N // 2, N // 2, dtype=np.float32),
        np.arange(-N // 2, N // 2, dtype=np.float32),
    )
    nx = nx.T  # match MATLAB meshgrid convention
    ny = ny.T
    k = 2 * np.pi / lam

    n = len(z)
    dx = np.asarray(dx, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    df = 1.0 / (N * dx)
    dz = z[1:] - z[:-1]
    sc = dx[1:] / dx[:-1]

    # Initial quadratic phase + first screen
    U = Uin.copy().astype(np.complex64)
    U *= np.exp(
        1j * k * ((nx * dx[0]) ** 2 + (ny * dx[0]) ** 2) * (1 - sc[0]) / (2 * dz[0])
    ).astype(np.complex64)
    U *= phi[:, :, 0].astype(np.complex64)

    if return_all:
        UoutAll = np.zeros((N, N, n), dtype=np.complex64)
        UoutAll[:, :, 0] = U

    # Pre-compute first transfer function
    tol = 1e-12
    Q = np.exp(
        -1j * np.pi * lam * dz[0] * ((nx * df[0]) ** 2 + (ny * df[0]) ** 2) / sc[0]
    ).astype(np.complex64)

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift

    for i in range(n - 1):
        # Recompute Q only when parameters change
        if i > 0 and not (
            abs(dz[i] - dz[i - 1]) < tol
            and abs(df[i] - df[i - 1]) < tol
            and abs(sc[i] - sc[i - 1]) < tol
        ):
            Q = np.exp(
                -1j * np.pi * lam * dz[i]
                * ((nx * df[i]) ** 2 + (ny * df[i]) ** 2)
                / sc[i]
            ).astype(np.complex64)

        # Propagate: FFT → multiply by Q → IFFT
        U_scaled = U / sc[i]
        U_prop = ifftshift(ifft2(ifftshift(Q * fftshift(fft2(fftshift(U_scaled))))))

        # Apply phase screen (skip last if phi has fewer planes)
        if i == n - 2 and phi.shape[2] < n:
            U = U_prop.astype(np.complex64)
        else:
            U = (phi[:, :, i + 1] * U_prop).astype(np.complex64)

        if return_all:
            UoutAll[:, :, i + 1] = U

    # Final quadratic phase correction
    Uout = U * np.exp(
        1j * k / 2 * (sc[-1] - 1) / (sc[-1] * dz[-1])
        * ((nx * dx[-1]) ** 2 + (ny * dx[-1]) ** 2)
    ).astype(np.complex64)

    if return_all:
        return Uout, UoutAll
    return Uout


# ---------------------------------------------------------------------------
# Gaussian beam / back aperture
# ---------------------------------------------------------------------------


def generate_gaussian_profile(
    X: np.ndarray,
    Y: np.ndarray,
    rad: float,
    aper: float,
    k: float,
    fl: float,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Gaussian back-aperture intensity profile with focusing phase.

    Port of MATLAB ``generateGaussianProfile.m``.
    """
    rho2 = X ** 2 + Y ** 2
    Uout = np.exp(-((X - offset[0]) ** 2 + (Y - offset[1]) ** 2) / rad ** 2)
    Uout = Uout * (rho2 < aper ** 2)
    Uout = Uout * np.exp(-1j * k / (2 * fl) * rho2)
    return Uout.astype(np.complex64)


# ---------------------------------------------------------------------------
# Zernike polynomials
# ---------------------------------------------------------------------------


def _zernike_index(i: int):
    """Noll-indexed Zernike (n, m) from sequential index *i* (1-based)."""
    n = int(math.ceil(math.sqrt(0.25 + 2 * i) - 1.5))
    # azimuthal order
    if n % 2 == 1:
        m = 2 * int(math.ceil((i - (n + 1) * n / 2) / 2)) - 1
    else:
        m = 2 * int(math.floor((i - (n + 1) * n / 2) / 2))
    return n, m


def _zernike_radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial R_n^m(r)."""
    R = np.zeros_like(r)
    for s in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** s
            * _gamma(n - s + 1)
            / (_gamma(s + 1) * _gamma((n + m) / 2 - s + 1) * _gamma((n - m) / 2 - s + 1))
        )
        R = R + coeff * r ** (n - 2 * s)
    return R


def zernike_polynomial(i: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate Noll-indexed Zernike polynomial *i* (1-based) on (x, y).

    Port of MATLAB ``zernike.m``.
    """
    theta = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    n, m = _zernike_index(i)
    if m == 0:
        return float(np.sqrt(n + 1)) * _zernike_radial(n, 0, r)
    if i % 2 == 0:
        return float(np.sqrt(2 * (n + 1))) * _zernike_radial(n, m, r) * np.cos(m * theta)
    return float(np.sqrt(2 * (n + 1))) * _zernike_radial(n, m, r) * np.sin(m * theta)


def apply_zernike(
    Uin: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: float,
    abb: np.ndarray,
) -> np.ndarray:
    """Apply Zernike aberrations to an input field.

    Port of MATLAB ``applyZernike.m``.

    Args:
        Uin: (N, N) complex field.
        X, Y: Coordinate grids (normalized by objective radius for standard Zernike).
        k: Wavenumber [rad/m].
        abb: Aberration coefficients [m].  Non-zero entries are applied.

    Returns:
        Aberrated field (N, N) complex64.
    """
    phase = np.zeros(X.shape, dtype=np.float32)
    abb = np.atleast_1d(np.asarray(abb, dtype=np.float64))
    for idx in np.nonzero(abb)[0]:
        phase += float(abb[idx]) * zernike_polynomial(idx + 1, X, Y).astype(np.float32)
    phase -= phase.mean()
    return (Uin * np.exp(1j * k * phase)).astype(np.complex64)


def generate_ba(
    vol_params: VolumeParams,
    psf_params: PsfParams,
) -> np.ndarray:
    """Generate Gaussian back-aperture field with Zernike aberrations.

    Port of MATLAB ``generateBA.m``.

    Returns:
        (N, N) complex64 input field ``Uin``.
    """
    vres = vol_params.vres
    fl = np.float32(psf_params.obj_fl / 1000)  # focal length [m]
    ss = psf_params.ss
    D2 = np.float32(1e-6 / (vres * ss))  # observation grid spacing [m]

    vasc_sz = vol_params.vasc_sz
    if vasc_sz is None:
        beam_ext = gaussian_beam_size(
            psf_params, vol_params.vol_depth + vol_params.vol_sz[2] / 2
        )
        vasc_sz = tuple(
            int(np.ceil(b + s + d))
            for b, s, d in zip(beam_ext, vol_params.vol_sz, (0, 0, vol_params.vol_depth))
        )

    vol_sz = np.array(vol_params.vol_sz, dtype=np.float32)
    vs = np.array(vasc_sz[:2], dtype=np.float32)

    N_arr = np.float32(1e-6 * (vs - vol_sz[:2]) / D2)
    N = int(N_arr[0])  # grid size (square)

    beam_at_fl = gaussian_beam_size(psf_params, fl * 1e6)
    D1 = np.float32(max(beam_at_fl[:2]) * 1e-6 / N)  # source grid spacing [m]

    nre = np.float32(psf_params.n)
    rad = np.float32(np.tan(np.arcsin(psf_params.na / nre)) * fl)
    objrad = np.float32(np.tan(np.arcsin(psf_params.obj_na / nre)) * fl)
    k = 2 * nre * np.pi / np.float32(psf_params.lambda_um * 1e-6)

    xs = (np.arange(-N // 2, N // 2, dtype=np.float32)) * D1
    ys = (np.arange(-N // 2, N // 2, dtype=np.float32)) * D1
    X, Y = np.meshgrid(xs, ys)  # MATLAB meshgrid convention

    Uout = generate_gaussian_profile(X, Y, rad, objrad, k, fl)

    # Apply Zernike aberrations
    zernike_wt = np.array(psf_params.zernike_wt, dtype=np.float64)
    abb = zernike_wt * psf_params.lambda_um * 1e-6  # weights → meters
    Uout = apply_zernike(Uout, X / objrad, Y / objrad, k, abb)

    return Uout


# ---------------------------------------------------------------------------
# Scatter volume generation
# ---------------------------------------------------------------------------


def generate_scatter_volume(
    grid_sz: Tuple[int, int, int],
    l_scale: np.ndarray,
    p_scale: float,
    mu: float = 0.0,
    l_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate a Gaussian random scattering volume.

    Port of MATLAB ``masked_3DGP_test.m``.

    Args:
        grid_sz: (Nx, Ny, Nz) output shape.
        l_scale: (K, 3) length scales for each Gaussian component.
        p_scale: Covariance scaling.
        mu: Mean value.
        l_weights: (K,) weights per component (default ones).

    Returns:
        (Nx, Ny, Nz) float32 random field.
    """
    l_scale = np.atleast_2d(np.asarray(l_scale, dtype=np.float32))
    K = l_scale.shape[0]
    if l_weights is None:
        l_weights = np.ones(K, dtype=np.float32)
    else:
        l_weights = np.asarray(l_weights, dtype=np.float32)

    Nx, Ny, Nz = int(grid_sz[0]), int(grid_sz[1]), int(grid_sz[2])
    wmx = np.pi / 2

    gx = np.linspace(-wmx, wmx, Nx, dtype=np.float32).reshape(-1, 1, 1) ** 2
    gy = np.linspace(-wmx, wmx, Ny, dtype=np.float32).reshape(1, -1, 1) ** 2
    gz = np.linspace(-wmx, wmx, Nz, dtype=np.float32).reshape(1, 1, -1) ** 2

    # Build spectral kernel
    kernel = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    for i in range(K):
        kx = np.exp(-gx * l_scale[i, 0] ** 2)
        ky = np.exp(-gy * l_scale[i, 1] ** 2)
        kz = np.exp(-gz * l_scale[i, 2] ** 2)
        ker = kx * ky * kz
        kernel += float(np.prod(l_scale[i])) * ker ** 2 * l_weights[i] ** 2
    kernel = np.sqrt(kernel)

    # Multiply by complex Gaussian noise per z-slice
    kernel_c = kernel.astype(np.complex64)
    for iz in range(Nz):
        noise = (
            np.random.randn(Nx, Ny).astype(np.float32)
            + 1j * np.random.randn(Nx, Ny).astype(np.float32)
        )
        kernel_c[:, :, iz] *= noise

    # IFFT back to spatial domain
    result = (
        2 * p_scale * np.sqrt(Nx * Ny * Nz)
        * np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kernel_c))))
    )
    return (result + mu).astype(np.float32)


# ---------------------------------------------------------------------------
# Z-projection utility
# ---------------------------------------------------------------------------


def group_z_project(
    vol: np.ndarray,
    chunk_size: int,
    mode: str = "sum",
) -> np.ndarray:
    """Project a 3-D volume along Z in fixed-size chunks.

    Port of MATLAB ``groupzproject.m``.

    Args:
        vol: (Nx, Ny, Nz) input volume.
        chunk_size: Number of z-slices per group.
        mode: Reduction type — ``'sum'``, ``'mean'``, ``'max'``, ``'min'``,
            ``'prod'``.

    Returns:
        (Nx, Ny, ceil(Nz/chunk_size)) projected volume.
    """
    Nx, Ny, Nz = vol.shape
    n_full = Nz // chunk_size
    remainder = Nz % chunk_size

    reduce_fn = {
        "sum": lambda a, ax: a.sum(axis=ax),
        "mean": lambda a, ax: a.mean(axis=ax),
        "max": lambda a, ax: a.max(axis=ax),
        "min": lambda a, ax: a.min(axis=ax),
        "prod": lambda a, ax: a.prod(axis=ax),
    }[mode]

    parts = []
    if n_full > 0:
        full = vol[:, :, : n_full * chunk_size].reshape(Nx, Ny, chunk_size, n_full)
        parts.append(reduce_fn(full, 2))  # reduce over chunk axis
    if remainder > 0:
        tail = vol[:, :, n_full * chunk_size :]
        parts.append(reduce_fn(tail, 2)[:, :, np.newaxis] if tail.ndim == 3
                      else reduce_fn(tail[:, :, np.newaxis], 2))
    if not parts:
        return np.zeros((Nx, Ny, 0), dtype=vol.dtype)
    return np.concatenate(parts, axis=2)
