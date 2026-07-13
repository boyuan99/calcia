"""Lateral tissue-scatter PSF broadening for widefield emission.

The analytic Gaussian-NA collection PSF (``gaussian_psf_na``) is diffraction-
limited and omits the lateral light diffusion that scattering tissue imposes on
1P widefield emission. These operators broaden a collection PSF in the OPTICS
domain (photon-conserving, per z-slice) so a subsequent scan spreads every
source by them — this is tissue scatter, NOT a post-hoc image blur (which would
average camera noise away and read as defocus).

Both require the PSF to carry enough lateral support to hold the broadened tail
(build it with a wide ``psf_sz``, e.g. ``(80, 80, z)`` for the single-scale
kernel or ``(100, 100, z)`` for the two-scale halo).
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def broaden_psf_scatter(psf, scatter_um, vres):
    """Broaden an emission PSF laterally by tissue scatter (physics the Gaussian-
    NA PSF omits).

    Real 1P widefield light diffuses laterally through scattering tissue on its
    way out, so each source's collected footprint is spread — single somata are
    NOT resolved. The analytic Gaussian-NA PSF is diffraction-limited (sharp) and
    lacks this. We convolve every z-slice of the collection PSF with a Gaussian
    of sigma = ``scatter_um`` (photon-conserving per slice), then the scan spreads
    every source (soma + neuropil) by it. This lives in the OPTICS domain and the
    scan adds camera noise AFTER — it is tissue scatter, NOT a post-hoc movie blur
    (which would average the noise and read as camera defocus).

    Requires the PSF to have enough lateral support to hold the tail (build it
    with a wide ``psf_sz``, e.g. (80, 80, z)). Returns a new float32 array.
    """
    if scatter_um <= 0:
        return psf.astype(np.float32)
    sig_px = scatter_um * vres
    out = psf.astype(np.float32).copy()
    for z in range(out.shape[2]):
        out[:, :, z] = gaussian_filter(out[:, :, z], sig_px)
    s0 = psf.sum(axis=(0, 1), keepdims=True)
    s1 = out.sum(axis=(0, 1), keepdims=True)
    return (out * (s0 / (s1 + 1e-12))).astype(np.float32)


def broaden_psf_two_scale(psf, halo_um, halo_weight, vres):
    """Approximate the real 1p scattering PSF as a TWO-SCALE kernel: the original
    narrow diffraction CORE plus a wide scattering HALO, in ONE optical kernel.

    A single Gaussian can only have one width — narrow keeps cells sharp but
    leaves inter-process neuropil GAPS (cell-sized holes); wide fills the gaps but
    washes the cells out (reads as defocus). The real 1p PSF (Fresnel + tissue
    scatter, ~36 um in NAOMi1p) is a SHARP CORE sitting in a BROAD HEAVY TAIL:
    ``psf' = (1-w)*core + w*halo`` where ``halo`` = core blurred by a wide Gaussian
    (sigma = ``halo_um``) and ``w`` = ``halo_weight`` = fraction of collected light
    in the scattering halo. Convolving the volume with this ONCE gives bright cell
    cores on a smooth haze — filling holes AND keeping cells — with no post-hoc
    image blur. Photon-conserving per z-slice. Lives in the OPTICS domain (scan
    adds camera noise AFTER). Needs a wide-support PSF to hold the halo tail
    (demos build psf_sz=(100,100,z))."""
    if halo_weight <= 0 or halo_um <= 0:
        return psf.astype(np.float32)
    core = psf.astype(np.float32)
    halo = np.empty_like(core)
    sig = halo_um * vres
    for z in range(core.shape[2]):
        halo[:, :, z] = gaussian_filter(core[:, :, z], sig)
    out = (1.0 - halo_weight) * core + halo_weight * halo
    s0 = core.sum(axis=(0, 1), keepdims=True)
    s1 = out.sum(axis=(0, 1), keepdims=True)
    return (out * (s0 / (s1 + 1e-12))).astype(np.float32)
