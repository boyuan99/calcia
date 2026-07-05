"""
Phase 2 optical propagation for widefield (single-photon) imaging.

Companion to :mod:`calcia.optics.propagation`. Differences from the
two-photon path (documented in ``docs/widefield_vs_twophoton.md``):

1. PSF uses the emission wavelength (``psf_params.lambda_em_um``), not the
   excitation wavelength.
2. Intensity is not squared (``scaling='widefield'``): the linear intensity
   profile is the emission / detection PSF.
3. PSF z-extent covers the entire volume depth, so every z-plane has a
   corresponding defocused PSF slice. The widefield scanner convolves every
   z-plane with its matching PSF slice and sums (no optical sectioning).
4. Illumination mask is uniform (Koehler illumination). The collection mask
   is reused from :mod:`calcia.optics.mask`.
5. PSF tails are unused; placeholder zero-filled ``PsfTail`` objects keep
   :class:`OpticalPropagationResult` shape stable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import numpy as np

from .mask import compute_collection_mask
from .propagation import OpticalPropagationResult
from .psf import PsfTail, gaussian_psf_na

if TYPE_CHECKING:
    from ..config.params import PsfParams, VolumeParams
    from ..pipeline import NeuralVolumeOutput


def simulate_optical_propagation_widefield(
    vol_params: "VolumeParams",
    psf_params: "PsfParams",
    vol_out: Optional["NeuralVolumeOutput"] = None,
    *,
    verbose: Optional[int] = None,
) -> OpticalPropagationResult:
    """Simulate widefield optical propagation (Phase 2, single-photon path).

    Args:
        vol_params: Volume parameters from Phase 1 (or fresh defaults).
        psf_params: PSF parameters. Must have ``imaging_mode='widefield'``
            (set by caller); ``lambda_em_um`` controls the emission PSF.
        vol_out: Phase 1 output. Its vessel volume is used by the
            collection mask; illumination mask is uniform.
        verbose: Verbosity level (0 silent, 1 progress, 2 detailed).

    Returns:
        :class:`OpticalPropagationResult` with an emission PSF whose z-axis
        spans the full volume depth, a uniform illumination mask, a
        collection mask, and zero-filled PSF tails.
    """
    v = vol_params.verbose if verbose is None else verbose

    if v >= 1:
        print("=" * 60)
        print("simulate_optical_propagation  (Phase 2, widefield)")
        print(f"  Emission lambda: {psf_params.lambda_em_um} um")
        print(f"  objNA={psf_params.obj_na}, n={psf_params.n}")
        print("=" * 60)

    vres = vol_params.vres
    vol_sz = vol_params.vol_sz
    Nx_vol = int(round(vol_sz[0] * vres))
    Ny_vol = int(round(vol_sz[1] * vres))
    Nz_vol = int(round(vol_sz[2] * vres))

    # --- Emission PSF spanning the full volume z-range ----------------
    if v >= 1:
        print("\n[1/3] Computing widefield emission PSF (full z-stack)...")

    Nx_psf = int(round(psf_params.psf_sz[0] * vres))
    Ny_psf = int(round(psf_params.psf_sz[1] * vres))
    sampling = (1.0 / vres, 1.0 / vres, 1.0 / vres)

    # Focal plane: gaussian_psf_na is sharpest at the CENTRE of its z-range.
    # Legacy (wf_focal_depth_um=None) uses a Nz_vol PSF -> in-focus at mid-
    # depth, which for a scatter-limited window prep wrongly leaves the bright
    # surface layer defocused. When wf_focal_depth_um is set, generate a
    # double-height PSF and slice it so the in-focus plane lands at the
    # requested depth (0 = surface): vol z-plane k is convolved with psf slice
    # k, whose defocus is |k - k_focus|.
    focal_um = getattr(psf_params, "wf_focal_depth_um", None)
    if focal_um is None:
        psf, _, _, _ = gaussian_psf_na(
            na=psf_params.obj_na, lambda_um=psf_params.lambda_em_um,
            sampling=sampling, mat_size=(Nx_psf, Ny_psf, Nz_vol),
            nidx=psf_params.n, scaling="widefield",
        )
    else:
        k_focus = int(round(min(max(focal_um, 0.0),
                                (Nz_vol - 1) / vres) * vres))
        psf_full, _, _, _ = gaussian_psf_na(
            na=psf_params.obj_na, lambda_um=psf_params.lambda_em_um,
            sampling=sampling, mat_size=(Nx_psf, Ny_psf, 2 * Nz_vol),
            nidx=psf_params.n, scaling="widefield",
        )
        # psf_full is in-focus at index Nz_vol; slice so vol z=k_focus is sharp.
        psf = psf_full[:, :, (Nz_vol - k_focus):(2 * Nz_vol - k_focus)]

    # Energy-conserving per-slice normalization (lateral redistribution of
    # photons by defocus conserves integral) then multiply by tissue
    # round-trip attenuation so deeper planes contribute less signal.
    slice_sums = psf.sum(axis=(0, 1), keepdims=True)
    slice_sums[slice_sums <= 0] = 1.0
    psf = psf / slice_sums

    # Depth below the imaging window, measured from the TOP of the imaging
    # volume (z=0). Widefield with an imaging window sits flush against the
    # tissue top, so VolumeParams.vol_depth (the depth of overlying tissue
    # ABOVE the volume — a 2P cranial-window concept) is NOT added here.
    # Users modelling thinned-skull or extra overlying tissue should either
    # lengthen scatter_length_um_wf or lower pavg accordingly.
    z_idx = np.arange(Nz_vol, dtype=np.float32)
    abs_depth_um = z_idx / vres
    L = float(psf_params.scatter_length_um_wf)
    attenuation = np.exp(-2.0 * abs_depth_um / L).astype(np.float32)
    psf = (psf * attenuation[np.newaxis, np.newaxis, :]).astype(np.float32)

    if v >= 2:
        print(
            f"  PSF shape: {psf.shape}, "
            f"top-slice sum={float(psf[:, :, 0].sum()):.3g}, "
            f"bottom-slice sum={float(psf[:, :, -1].sum()):.3g} "
            f"(scatter L={L:.0f} um)"
        )

    # --- Uniform illumination mask (Koehler) --------------------------
    if v >= 1:
        print("\n[2/3] Building uniform illumination mask (Koehler)...")
    mask = np.ones((Nx_vol, Ny_vol), dtype=np.float32)

    # --- Collection mask (reused from two-photon path, with the widefield
    # hemoglobin absorbance — 520 nm absorbs ~30x more strongly than 920 nm).
    # Also override vol_depth to 0 so the collection cone geometry is
    # referenced to the imaging window (at volume top), not the pial surface.
    if v >= 1:
        print("\n[3/3] Computing collection mask...")
    vessel_volume = vol_out.neur_ves if vol_out is not None else None
    psf_params_col = replace(psf_params, hemo_abs=psf_params.hemo_abs_wf)
    vol_params_col = replace(vol_params, vol_depth=0)
    col_mask = compute_collection_mask(
        vol_params_col, psf_params_col, vessel_volume,
    )

    # --- Zero-filled PSF tails (not used by widefield scanner) --------
    zero_weights = np.zeros((Nx_psf, Ny_psf), dtype=np.float32)
    zero_mask = np.zeros((Nx_vol, Ny_vol), dtype=np.float32)
    psf_top = PsfTail(weights=zero_weights, mask=zero_mask, weight=0.0)
    psf_bot = PsfTail(weights=zero_weights.copy(), mask=zero_mask.copy(), weight=0.0)

    if v >= 1:
        print("\n" + "=" * 60)
        print("Optical propagation complete (widefield).")
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
