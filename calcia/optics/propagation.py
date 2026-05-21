"""
Phase 2 optical propagation simulation.

Port of MATLAB ``simulate_optical_propagation.m``.

Supports two PSF computation modes:

* ``psf_type='gaussian'`` (default) — full Fresnel wave-optics propagation
  through vessel phase screens, matching MATLAB ``genCorticalLightPathLite``.
* ``psf_type='gaussian_analytical'`` — fast analytical Gaussian PSF
  (no wave-optics, sum-normalised to 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ..config.params import PsfParams
from .mask import compute_collection_mask, compute_illumination_mask
from .psf import PsfTail, compute_psf_tails, gaussian_psf_na

if TYPE_CHECKING:
    from ..config.params import VolumeParams
    from ..pipeline import NeuralVolumeOutput


@dataclass
class OpticalPropagationResult:
    """Complete output of the Phase 2 optical propagation simulation.

    Mirrors the ``PSF_struct`` / ``opt_out`` struct from MATLAB
    ``simulate_optical_propagation.m``.

    Attributes:
        psf: 3D float32 array of shape (Nx, Ny, Nz) — two-photon PSF volume.
        mask: 2D float32 array of shape (Nx_vol, Ny_vol) — illumination
            blockage mask, normalized so ``mean(mask) == 1``.
        psf_top: PsfTail — PSF energy integrated in the volume above the
            focal plane.
        psf_bot: PsfTail — PSF energy integrated in the volume below the
            focal plane.
        col_mask: 2D float32 array of shape (Nx_vol, Ny_vol) — collection
            mask from hemoglobin absorption, values in (0, 1].
        params: Dict with keys ``'vol_params'`` and ``'psf_params'``.
    """
    psf: np.ndarray
    mask: np.ndarray
    psf_top: PsfTail
    psf_bot: PsfTail
    col_mask: np.ndarray
    params: Dict


def simulate_optical_propagation(
    vol_params: "VolumeParams",
    psf_params: Optional[PsfParams] = None,
    vol_out: Optional["NeuralVolumeOutput"] = None,
    *,
    verbose: Optional[int] = None,
) -> OpticalPropagationResult:
    """Simulate optical propagation through neural tissue (Phase 2).

    Port of MATLAB ``simulate_optical_propagation.m``.

    Args:
        vol_params: Volume parameters from Phase 1 (or fresh defaults).
        psf_params: PSF parameters.  Uses :class:`PsfParams` defaults if
            None.
        vol_out: Phase 1 output (:class:`~calcia.pipeline.NeuralVolumeOutput`).
            Blood-vessel volume ``vol_out.neur_ves`` is used to compute the
            masks and (for Fresnel mode) phase screens.
        verbose: Verbosity level (0 = silent, 1 = progress, 2 = detailed).
            If None, inherits from ``vol_params.verbose``.

    Returns:
        :class:`OpticalPropagationResult` containing psf, mask, psf_top,
        psf_bot, col_mask, and final parameter objects.
    """
    if psf_params is None:
        psf_params = PsfParams()

    v = vol_params.verbose if verbose is None else verbose

    if getattr(psf_params, "imaging_mode", "two-photon") == "widefield":
        from .widefield import simulate_optical_propagation_widefield
        return simulate_optical_propagation_widefield(
            vol_params, psf_params, vol_out, verbose=v,
        )

    if v >= 1:
        print("=" * 60)
        print("simulate_optical_propagation  (Phase 2)")
        print(f"  PSF type: {psf_params.psf_type}")
        print(f"  NA={psf_params.na}, objNA={psf_params.obj_na}, "
              f"lambda={psf_params.lambda_um} um")
        print(f"  PSF size: {psf_params.psf_sz} um")
        print("=" * 60)

    vessel_volume = vol_out.neur_ves if vol_out is not None else None

    if psf_params.psf_type == "gaussian":
        return _fresnel_path(vol_params, psf_params, vessel_volume, v)
    else:
        return _analytical_path(vol_params, psf_params, vessel_volume, v)


# ======================================================================
# Fresnel wave-optics path  (psf_type='gaussian', default)
# ======================================================================

def _fresnel_path(
    vol_params: "VolumeParams",
    psf_params: PsfParams,
    vessel_volume: Optional[np.ndarray],
    v: int,
) -> OpticalPropagationResult:
    """Full Fresnel propagation through phase screens."""
    from .fresnel import generate_ba
    from .fresnel_psf import decompose_vessel_volume, gen_cortical_light_path_lite

    # Step 1: Generate input beam
    if v >= 1:
        print("\n[1/4] Generating input beam (back aperture)...")
    Uin = generate_ba(vol_params, psf_params)
    if v >= 2:
        print(f"  Uin shape: {Uin.shape}")

    # Step 2: Decompose vessels into phase screens
    if v >= 1:
        print("\n[2/4] Building phase screens from vessel volume...")
    phzA, phzB, phzC = decompose_vessel_volume(
        vol_params, psf_params, vessel_volume,
    )
    if v >= 2:
        print(f"  phzA: {phzA.shape}, phzB: {phzB.shape}, phzC: {phzC.shape}")

    # Step 3: Fresnel propagation → PSF + mask
    if v >= 1:
        print("\n[3/4] Fresnel propagation (light path)...")
    mask, psf, psf_top, psf_bot = gen_cortical_light_path_lite(
        vol_params, psf_params, phzA, phzB, phzC, Uin, verbose=v,
    )

    # Normalize mask to mean=1.
    # MATLAB: mask = mask/mean(mask); psf = psf/mean(mask);
    # After mask normalization, mean(mask)==1, so PSF is unchanged.
    mask_mean = float(mask.mean()) if mask.mean() > 0 else 1.0
    mask = (mask / mask_mean).astype(np.float32)

    # Step 4: Collection mask
    if v >= 1:
        print("\n[4/4] Computing collection mask...")
    col_mask = compute_collection_mask(vol_params, psf_params, vessel_volume)

    if v >= 1:
        print("\n" + "=" * 60)
        print("Optical propagation complete (Fresnel).")
        print(f"  PSF shape:       {psf.shape}")
        print(f"  PSF sum:         {float(psf.sum()):.4g}")
        print(f"  Mask shape:      {mask.shape}")
        print(f"  Col mask shape:  {col_mask.shape}")
        print("=" * 60)

    return OpticalPropagationResult(
        psf=psf,
        mask=mask,
        psf_top=psf_top,
        psf_bot=psf_bot,
        col_mask=col_mask,
        params={"vol_params": vol_params, "psf_params": psf_params},
    )


# ======================================================================
# Analytical Gaussian path  (psf_type='gaussian_analytical')
# ======================================================================

def _analytical_path(
    vol_params: "VolumeParams",
    psf_params: PsfParams,
    vessel_volume: Optional[np.ndarray],
    v: int,
) -> OpticalPropagationResult:
    """Fast analytical Gaussian PSF (no wave-optics)."""
    vres = vol_params.vres

    if v >= 1:
        print("\n[1/3] Computing analytical Gaussian PSF...")

    psf_sz = psf_params.psf_sz
    sampling = (1.0 / vres, 1.0 / vres, 1.0 / vres)
    mat_size = (
        int(round(psf_sz[0] * vres)),
        int(round(psf_sz[1] * vres)),
        int(round(psf_sz[2] * vres)),
    )

    psf, _, _, _ = gaussian_psf_na(
        na=psf_params.na,
        lambda_um=psf_params.lambda_um,
        sampling=sampling,
        mat_size=mat_size,
        nidx=psf_params.n,
    )

    # Normalize to unit sum (analytical PSF has peak ~1, needs rescaling)
    psf_sum = float(psf.sum())
    if psf_sum > 0:
        psf = (psf / psf_sum).astype(np.float32)

    if v >= 2:
        print(f"  PSF shape: {psf.shape}, max={float(psf.max()):.6g}, "
              f"sum={float(psf.sum()):.4f}")

    # Illumination mask
    if v >= 1:
        print("\n[2/3] Computing illumination mask...")
    mask = compute_illumination_mask(vol_params, psf_params, vessel_volume)

    mask_mean = float(mask.mean())
    if mask_mean > 0:
        psf = (psf / mask_mean).astype(np.float32)

    # Collection mask
    if v >= 1:
        print("\n[3/3] Computing collection mask...")
    col_mask = compute_collection_mask(vol_params, psf_params, vessel_volume)

    # PSF tails
    psf_top, psf_bot = compute_psf_tails(psf, vol_params, psf_params)

    if v >= 1:
        print("\n" + "=" * 60)
        print("Optical propagation complete (analytical).")
        print(f"  PSF shape:       {psf.shape}")
        print(f"  Mask shape:      {mask.shape}")
        print(f"  Col mask shape:  {col_mask.shape}")
        print("=" * 60)

    return OpticalPropagationResult(
        psf=psf,
        mask=mask,
        psf_top=psf_top,
        psf_bot=psf_bot,
        col_mask=col_mask,
        params={"vol_params": vol_params, "psf_params": psf_params},
    )
