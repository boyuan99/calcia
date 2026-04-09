"""
Gaussian Process sampling utilities.

Provides functions for sampling from Gaussian processes:
- Isotropic GP on sphere surface (for neural shapes)
- FFT-based 3D GP (for somatic fluorescence heterogeneity)
"""

import numpy as np
from scipy.linalg import cholesky, eigh
from typing import Optional, Tuple


def sample_gp_sphere(
    vertices: np.ndarray,
    l_scale: float = 90.0,
    p_scale: float = 1000.0,
    power: float = 1.0,
    mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample from an isotropic Gaussian process on a sphere.

    The covariance between points is based on geodesic distance:
    C(i,j) = p_scale * exp(-(geodesic_dist / l_scale)^power)

    Args:
        vertices: (N, 3) array of points on unit sphere.
        l_scale: Length scale of the GP (controls smoothness/bumpiness).
        p_scale: Variance scale of the GP.
        power: Power for the distance (default 1.0, don't change - sensitive).
        mean: Optional (N,) mean function. If None, uses zero mean.

    Returns:
        (N,) array of sampled values at each vertex.
    """
    n = len(vertices)

    # Compute geodesic distance matrix
    # geodesic = 2 * arcsin(euclidean_dist / 2)
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=2))
    geodesic_dist = 2 * np.arcsin(np.clip(euclidean_dist / 2, -1, 1))

    # Compute covariance matrix
    cov_matrix = p_scale * np.exp(-(geodesic_dist / l_scale) ** power)

    # Ensure positive semi-definite
    cov_matrix = ensure_psd(cov_matrix)

    # Sample from multivariate normal
    if mean is None:
        mean = np.zeros(n)

    sample = sample_multivariate_normal(mean, cov_matrix)

    return sample


def ensure_psd(matrix: np.ndarray, factor: float = 1.03) -> np.ndarray:
    """
    Ensure a matrix is positive semi-definite.

    Adds a small diagonal term if the minimum eigenvalue is negative.

    Args:
        matrix: Square matrix to check/fix.
        factor: Multiplier for the correction term.

    Returns:
        PSD matrix.
    """
    # Find minimum eigenvalue efficiently
    try:
        # Use only a few eigenvalues for speed
        from scipy.sparse.linalg import eigsh
        min_eig = eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
    except Exception:
        # Fallback to full eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)

    if min_eig < 0:
        # Add diagonal term to make PSD
        matrix = matrix + np.abs(min_eig) * factor * np.eye(len(matrix))

    return matrix


def sample_multivariate_normal(
    mean: np.ndarray,
    cov: np.ndarray,
    method: str = 'cholesky'
) -> np.ndarray:
    """
    Sample from a multivariate normal distribution.

    Args:
        mean: (N,) mean vector.
        cov: (N, N) covariance matrix.
        method: 'cholesky' (faster) or 'eig' (more stable).

    Returns:
        (N,) sample vector.
    """
    n = len(mean)

    if method == 'cholesky':
        try:
            # Cholesky decomposition: cov = L @ L.T
            L = cholesky(cov, lower=True)
            z = np.random.randn(n)
            sample = mean + L @ z
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue method if Cholesky fails
            return sample_multivariate_normal(mean, cov, method='eig')
    else:
        # Eigenvalue decomposition method (more stable)
        eigenvalues, eigenvectors = eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        z = np.random.randn(n)
        sample = mean + eigenvectors @ (np.sqrt(eigenvalues) * z)

    return sample


def compute_geodesic_covariance(
    vertices: np.ndarray,
    l_scale: float,
    p_scale: float,
    power: float = 1.0
) -> np.ndarray:
    """
    Compute covariance matrix based on geodesic distances.

    Args:
        vertices: (N, 3) points on unit sphere.
        l_scale: Length scale.
        p_scale: Variance scale.
        power: Distance power.

    Returns:
        (N, N) covariance matrix.
    """
    # Compute pairwise distances
    diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Convert to geodesic distance (arc length on unit sphere)
    geodesic_dist = 2 * np.arcsin(np.clip(euclidean_dist / 2, -1, 1))

    # Squared exponential (RBF) kernel
    cov = p_scale * np.exp(-(geodesic_dist / l_scale) ** power)

    return cov


def teardrop_projection(
    vertices: np.ndarray,
    p: int = 1
) -> np.ndarray:
    """
    Project sphere points onto a teardrop shape.

    Used for pyramidal neurons to give them a characteristic teardrop shape.
    The projection is:
        x' = (x/r) * sin(θ) * sin(θ/2)^p
        y' = (y/r) * sin(θ) * sin(θ/2)^p
        z' = -cos(θ)

    where θ is the polar angle and r is the xy-plane radius.

    Args:
        vertices: (N, 3) points on unit sphere.
        p: Teardrop parameter (1 for pyramidal, 2 for peanut shape).

    Returns:
        (N, 3) points on teardrop shape.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # XY-plane radius
    rr = np.sqrt(x ** 2 + y ** 2)

    # Polar angle (elevation)
    # θ = π - atan(r/|z|) - π*(z>0)
    theta = np.pi - np.arctan2(rr, np.abs(z)) - (z > 0) * np.pi

    # Compute teardrop coordinates
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.sin(theta) * (np.sin(0.5 * theta) ** p)
        x_tear = (x / rr) * scale
        y_tear = (y / rr) * scale

    z_tear = -np.cos(theta)

    # Handle NaN (occurs at poles where rr=0)
    x_tear = np.nan_to_num(x_tear, nan=0.0)
    y_tear = np.nan_to_num(y_tear, nan=0.0)

    return np.column_stack([x_tear, y_tear, z_tear])


def sample_3d_gp(
    grid_sz: Tuple[int, int, int],
    l_scale: np.ndarray,
    p_scale: float,
    mu: float = 0.0,
    bin_mask: Optional[np.ndarray] = None,
    threshold: float = 1e-10,
    l_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample from a 3D Gaussian Process using FFT-based filtering.

    Draws from X ~ N(mu, C) where C_{i,j} = p * exp(-(i-j)^2 / (2*l^2)).
    Uses frequency-domain filtering: i.i.d. complex randn multiplied by
    the kernel, then inverse FFT to return to spatial domain.

    Port of MATLAB ``masked_3DGP_v2.m``.

    Args:
        grid_sz: (gx, gy, gz) dimensions of the output sample.
        l_scale: Length scale(s). Scalar, 1D array of N scales, or
                 (N, 3) array with per-axis scales.
        p_scale: Covariance scaling (variance parameter).
        mu: Mean of the GP (scalar or array matching grid_sz).
        bin_mask: Optional binary mask to zero certain output voxels.
        threshold: Kernel sparsity threshold.
        l_weights: Per-scale weights. Defaults to ones.

    Returns:
        3D float32 array of shape grid_sz.
    """
    grid_sz = tuple(int(d) for d in grid_sz)

    # --- Normalize l_scale to shape (N, 3) ---
    l_scale = np.atleast_2d(np.asarray(l_scale, dtype=np.float32))
    if l_scale.shape[1] == 1:
        l_scale = np.tile(l_scale, (1, 3))

    n_scales = l_scale.shape[0]

    if l_weights is None:
        l_weights = np.ones(n_scales, dtype=np.float32)
    else:
        l_weights = np.asarray(l_weights, dtype=np.float32).ravel()
        if len(l_weights) == 1:
            l_weights = np.full(n_scales, l_weights[0], dtype=np.float32)

    if bin_mask is None:
        bin_mask = 1.0

    # --- Frequency grids (MATLAB lines 56-59) ---
    wmx = np.pi / 2.0
    grid_x = np.linspace(-wmx, wmx, grid_sz[0], dtype=np.float32) ** 2
    grid_y = np.linspace(-wmx, wmx, grid_sz[1], dtype=np.float32) ** 2
    grid_z = np.linspace(-wmx, wmx, grid_sz[2], dtype=np.float32) ** 2

    gp_vals = np.zeros(grid_sz, dtype=np.complex64)

    # --- Kernel loop (MATLAB lines 61-79) ---
    n_voxels = int(np.prod(grid_sz))
    for i in range(n_scales):
        ker_x = np.exp(-grid_x * l_scale[i, 0] ** 2)
        ker_y = np.exp(-grid_y * l_scale[i, 1] ** 2)
        ker_z = np.exp(-grid_z * l_scale[i, 2] ** 2)
        ker_1 = ker_x[:, None, None] * ker_y[None, :, None] * ker_z[None, None, :]

        ker_loc = ker_1.ravel() > threshold
        ker_len = int(np.sum(ker_loc))

        scale_factor = l_weights[i] * np.sqrt(np.prod(l_scale[i]))

        if ker_len < n_voxels // 2:
            # Sparse path
            tmp = (np.random.randn(ker_len).astype(np.float32)
                   + 1j * np.random.randn(ker_len).astype(np.float32))
            tmp *= scale_factor * ker_1.ravel()[ker_loc]
            gp_vals.ravel()[ker_loc] += tmp
        else:
            # Dense path
            tmp = (np.random.randn(*grid_sz).astype(np.float32)
                   + 1j * np.random.randn(*grid_sz).astype(np.float32))
            tmp *= scale_factor * ker_1
            gp_vals += tmp

    # --- Inverse FFT to spatial domain (MATLAB line 81) ---
    gp_vals = np.sqrt(float(n_voxels)) * np.real(
        np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(gp_vals)))
    )

    # --- Scale and apply mask (MATLAB lines 82-83) ---
    norm_const = 2.0 ** 4.5 / np.pi ** 1.5
    gp_vals = (p_scale * norm_const * bin_mask * gp_vals
               / np.sqrt(float(len(l_weights))) + mu)

    return gp_vals.astype(np.float32)
