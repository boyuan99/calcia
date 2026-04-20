"""Fresnel-propagated PSF and illumination mask computation.

Ports of MATLAB functions:
  - genCorticalLightPathLite.m  → gen_cortical_light_path_lite
  - simulate_optical_propagation.m (volume decomposition) → decompose_vessel_volume

These produce a physically-scaled PSF and spatially-varying illumination
mask that account for light propagation through blood vessels and tissue
scattering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import shift as ndshift, zoom as ndzoom

from ..config.params import PsfParams
from .fresnel import (
    fresnel_propagation_multi,
    generate_ba,
    generate_scatter_volume,
    group_z_project,
)
from .psf import PsfTail, gaussian_beam_size

if TYPE_CHECKING:
    from ..config.params import VolumeParams


# ---------------------------------------------------------------------------
# Volume decomposition into phase screens
# ---------------------------------------------------------------------------


def decompose_vessel_volume(
    vol_params: "VolumeParams",
    psf_params: PsfParams,
    vessel_volume: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split vessel/scatter data into three phase-screen regions.

    Port of the volume-setup logic in MATLAB
    ``simulate_optical_propagation.m`` lines 169-296.

    Returns:
        ``(phzA, phzB, phzC)`` — complex64 phase-screen volumes for regions
        above, through, and below the PSF focal zone.
    """
    vres = vol_params.vres
    vol_sz = np.array(vol_params.vol_sz, dtype=np.float64)
    vol_depth = vol_params.vol_depth

    vasc_sz = vol_params.vasc_sz
    if vasc_sz is None:
        beam_ext = gaussian_beam_size(
            psf_params, vol_depth + vol_sz[2] / 2
        )
        vasc_sz = tuple(
            int(np.ceil(b + s + d))
            for b, s, d in zip(beam_ext, vol_sz, (0, 0, vol_depth))
        )

    psf_sz = np.array(psf_params.psf_sz, dtype=np.float64)
    psfpx = psf_sz * vres
    proppx = psf_params.prop_sz * vres

    # Z boundaries (in voxels)
    zA = vres * (vol_depth + vol_sz[2] / 2) - psfpx[2] / 2
    zB = vres * (vol_depth + vol_sz[2] / 2) + psfpx[2] / 2

    # Region sizes
    vasc_vox = np.array(vasc_sz, dtype=np.float64) * vres
    zAsz = np.array([vasc_vox[0], vasc_vox[1], zA / proppx])

    # zBsz: full vasc_sz XY for PSF region (cropping is done during extraction
    # in gen_cortical_light_path_lite, not here)
    zBsz = np.array([vasc_vox[0], vasc_vox[1], psfpx[2]])

    tail_len = psf_params.tail_length
    zC_end = vres * (vol_depth + vol_sz[2] / 2 + tail_len) + psfpx[2] / 2
    zCsz = np.array([vasc_vox[0], vasc_vox[1], (zC_end - zB) / proppx])

    zAsz = np.ceil(zAsz).astype(int)
    zBsz = np.ceil(zBsz).astype(int)
    zCsz = np.ceil(zCsz).astype(int)

    zA_int = int(zA)
    zB_int = int(zB)

    # --- Scatter volumes ---
    scatter_sz = np.array(psf_params.scatter_sz, dtype=np.float32)
    scatter_wt = np.array(psf_params.scatter_wt, dtype=np.float32)
    n_diff = psf_params.n_diff

    if scatter_sz.ndim == 1:
        scatter_sz = scatter_sz[:, np.newaxis] * np.ones((1, 3), dtype=np.float32)
    scatter_sz_vox = scatter_sz * vres

    if n_diff > 0:
        # Adjust scatter weights by n_diff ratio if n_diff_scatter exists
        pass

    neur_ves_A = generate_scatter_volume(
        tuple(zAsz),
        scatter_sz_vox * np.array([1, 1, 1.0 / proppx]),
        proppx, 0.0, scatter_wt,
    )
    neur_ves_B = generate_scatter_volume(
        tuple(zBsz),
        scatter_sz_vox, 1.0, 0.0, scatter_wt,
    )
    neur_ves_C = generate_scatter_volume(
        tuple(zCsz),
        scatter_sz_vox * np.array([1, 1, 1.0 / proppx]),
        proppx, 0.0, scatter_wt,
    )

    # --- Merge with actual vessel volume ---
    volpx = (np.array(vol_params.vol_sz[:2]) * vres).astype(int)

    if vessel_volume is not None:
        ves = vessel_volume.astype(np.float32)
        # Pad offsets to center vessel volume within vasc domain
        x_off = int(np.floor((zAsz[0] - volpx[0]) / 2))
        y_off = int(np.floor((zAsz[1] - volpx[1]) / 2))

        # Region A: z-grouped projection of vessels above PSF
        if zA_int > 0 and zA_int <= ves.shape[2]:
            ves_A_proj = group_z_project(
                ves[:, :, :zA_int], int(proppx), "sum"
            )
            n_slices = min(ves_A_proj.shape[2], zAsz[2])
            x_end = min(x_off + volpx[0], zAsz[0])
            y_end = min(y_off + volpx[1], zAsz[1])
            vA = ves_A_proj[: x_end - x_off, : y_end - y_off, :n_slices]
            sA = neur_ves_A[x_off:x_end, y_off:y_end, :n_slices]
            neur_ves_A[x_off:x_end, y_off:y_end, :n_slices] = (
                vA + sA * (1 - vA / proppx)
            )

        # Region B: direct merge at full resolution
        if zA_int < ves.shape[2]:
            z_b_end = min(zB_int, ves.shape[2])
            ves_B = ves[:, :, zA_int:z_b_end]
            x_off_b = int(np.floor((zBsz[0] - volpx[0]) / 2))
            y_off_b = int(np.floor((zBsz[1] - volpx[1]) / 2))
            n_z_b = min(ves_B.shape[2], zBsz[2])
            x_end_b = min(x_off_b + volpx[0], zBsz[0])
            y_end_b = min(y_off_b + volpx[1], zBsz[1])
            if x_off_b >= 0 and y_off_b >= 0:
                vB = ves_B[: x_end_b - x_off_b, : y_end_b - y_off_b, :n_z_b]
                sB = neur_ves_B[x_off_b:x_end_b, y_off_b:y_end_b, :n_z_b]
                neur_ves_B[x_off_b:x_end_b, y_off_b:y_end_b, :n_z_b] = (
                    vB + sB * (1 - vB)
                )

        # Region C: z-grouped projection of vessels below PSF
        if zB_int < ves.shape[2]:
            ves_C_raw = ves[:, :, zB_int:]
            ves_C_proj = group_z_project(ves_C_raw, int(proppx), "sum")
            n_slices_c = min(ves_C_proj.shape[2], zCsz[2])
            x_end_c = min(x_off + volpx[0], zCsz[0])
            y_end_c = min(y_off + volpx[1], zCsz[1])
            vC = ves_C_proj[: x_end_c - x_off, : y_end_c - y_off, :n_slices_c]
            sC = neur_ves_C[x_off:x_end_c, y_off:y_end_c, :n_slices_c]
            neur_ves_C[x_off:x_end_c, y_off:y_end_c, :n_slices_c] = (
                vC + sC * (1 - vC / proppx)
            )

    # --- Convert to complex phase screens ---
    wvl = psf_params.lambda_um * 1e-6
    k = 2 * np.pi / wvl
    phase_scale = n_diff * k * 1e-6 / vres

    ss = psf_params.ss

    def _to_phase(vol, target_ss):
        """Convert intensity volume to complex phase screen, resized by ss."""
        phz = np.exp(1j * phase_scale * vol).astype(np.complex64)
        if target_ss != 1:
            # Resize XY by ss (keep Z)
            phz = ndzoom(phz, (target_ss, target_ss, 1), order=3)
        return phz

    phzA = _to_phase(neur_ves_A, ss)
    phzB = _to_phase(neur_ves_B, ss)
    phzC = _to_phase(neur_ves_C, ss)

    return phzA, phzB, phzC


# ---------------------------------------------------------------------------
# Main light-path computation
# ---------------------------------------------------------------------------


def gen_cortical_light_path_lite(
    vol_params: "VolumeParams",
    psf_params: PsfParams,
    phzA: np.ndarray,
    phzB: np.ndarray,
    phzC: np.ndarray,
    Uin: np.ndarray,
    *,
    verbose: int = 1,
) -> Tuple[np.ndarray, np.ndarray, PsfTail, PsfTail]:
    """Compute PSF and illumination mask via Fresnel propagation.

    Port of MATLAB ``genCorticalLightPathLite.m``.

    Args:
        vol_params: Volume parameters.
        psf_params: PSF parameters.
        phzA, phzB, phzC: Complex phase-screen volumes for the three
            propagation regions.
        Uin: (N, N) complex input field from ``generate_ba``.
        verbose: Verbosity level.

    Returns:
        ``(mask, psf_avg, psf_top, psf_bot)`` where:
          - mask: 2D float32 illumination mask (Nx_vol, Ny_vol).
          - psf_avg: 3D float32 average PSF.
          - psf_top, psf_bot: PsfTail structures.
    """
    vres = vol_params.vres
    vol_sz = np.array(vol_params.vol_sz, dtype=np.float64)
    vol_depth = vol_params.vol_depth
    psf_sz = np.array(psf_params.psf_sz, dtype=np.float64)
    ss = psf_params.ss

    vasc_sz = vol_params.vasc_sz
    if vasc_sz is None:
        beam_ext = gaussian_beam_size(
            psf_params, vol_depth + vol_sz[2] / 2
        )
        vasc_sz = tuple(
            int(np.ceil(b + s + d))
            for b, s, d in zip(beam_ext, vol_sz, (0, 0, vol_depth))
        )

    fl = np.float32(psf_params.obj_fl / 1000)
    D2 = np.float32(1e-6 / (vres * ss))
    vs = np.array(vasc_sz[:2], dtype=np.float32)
    N = int(np.float32(1e-6 * (vs[0] - vol_sz[0]) / D2))
    D1 = np.float32(max(gaussian_beam_size(psf_params, fl * 1e6)[:2]) * 1e-6 / N)

    nre = np.float32(psf_params.n)
    wvl = np.float32(psf_params.lambda_um * 1e-6)
    psf_samp = min(psf_params.sampling, 1e10)
    psfpx = (psf_sz * vres).astype(int)
    proppx = int(psf_params.prop_sz * vres)

    # Apodization window
    xs = (np.arange(-N // 2, N // 2, dtype=np.float32)) * D1
    ys = xs.copy()
    x1, y1 = np.meshgrid(xs, ys)
    x1, y1 = x1.T, y1.T  # match MATLAB meshgrid
    sg = np.exp(-(x1 / (0.47 * N * D1)) ** 16) * np.exp(-(y1 / (0.47 * N * D1)) ** 16)
    sg = sg.astype(np.complex64)

    # Free-space propagation from back aperture to tissue surface
    z_prop = np.array([0.0, fl - (vol_depth + vol_sz[2] / 2) * 1e-6])
    delta = np.array([D1, D2])
    t = np.stack([sg, sg], axis=-1)
    Uout = fresnel_propagation_multi(Uin, wvl, delta, z_prop, t, nre)
    Uout = Uout / np.sqrt(np.sum(np.abs(Uout) ** 2))

    # Sampling grid
    imax = round(vol_sz[0] / psf_samp) + 1
    jmax = round(vol_sz[1] / psf_samp) + 1

    # Z positions
    zA = vres * (vol_depth + vol_sz[2] / 2) - psfpx[2] / 2
    zB = vres * (vol_depth + vol_sz[2] / 2) + psfpx[2] / 2

    if zA % proppx:
        zApos = proppx / vres * np.concatenate(
            [[0], np.arange(phzA.shape[2]) + (zA % proppx) / proppx]
        ) * 1e-6
    else:
        zApos = proppx / vres * np.arange(phzA.shape[2] + 1) * 1e-6

    zBpos = np.arange(phzB.shape[2] + 1) * 1e-6 / vres

    tail_len = psf_params.tail_length
    zC = vres * (vol_depth + vol_sz[2] / 2 + tail_len) + psfpx[2] / 2
    if (zC - zB) % proppx:
        zCpos = proppx / vres * np.concatenate(
            [np.arange(phzC.shape[2]), [(zC - zB) / proppx]]
        ) * 1e-6
    else:
        zCpos = proppx / vres * np.arange(phzC.shape[2] + 1) * 1e-6

    # Beam-extent crop for region B
    if psf_params.prop_crop:
        N2 = min(
            int(max(gaussian_beam_size(psf_params, psfpx[2] / vres / 2, apod=3)[:2])
                * 1e-6 / (1e-6 / vres) * 2 * ss),
            N,
        )
    else:
        N2 = N
    if N % 2 != 0 and N2 % 2 == 0:
        N2 -= 1

    # Region B apodization
    xs2 = (np.arange(-N2 // 2, N2 // 2, dtype=np.float32)) * D1
    x2, y2 = np.meshgrid(xs2, xs2)
    x2, y2 = x2.T, y2.T
    sg2 = np.exp(-(x2 / (0.47 * N2 * D1)) ** 16) * np.exp(-(y2 / (0.47 * N2 * D1)) ** 16)
    sg2 = sg2.astype(np.complex64)

    step = int(psf_samp * vres * ss)
    # MATLAB line 184: psfs{i,j} = ss^2 * imresize(...psf2p...) * (vres*(1e6*wvl)^1.5)/(pi*nre)
    # psf2p already has ss^2 from line 156.  The second ss^2 in line 184
    # accounts for the imresize downsampling area factor.
    # scale_factor here does NOT include ss^2 — it's applied separately
    # in the downsample step.
    scale_factor = float((vres * (1e6 * wvl) ** 1.5) / (np.pi * nre))

    # Output accumulators
    psf_out_shape = (int(psfpx[0]), int(psfpx[1]), int(psfpx[2]))
    psfs3 = np.zeros(psf_out_shape, dtype=np.float32)
    psfmag = np.zeros((imax, jmax), dtype=np.float64)
    psfT_arr = np.zeros((imax, jmax), dtype=np.float64)
    psfB_arr = np.zeros((imax, jmax), dtype=np.float64)
    psfTM = None
    psfBM = None
    psfTMz = None  # z-profile accumulator (top)
    psfBMz = None  # z-profile accumulator (bot)

    if verbose >= 1:
        print(f"Propagating through {imax * jmax} locations:")

    count = 0
    for i in range(imax):
        for j in range(jmax):
            # Extract phase-screen sub-regions with spatial offset
            r0 = step * i
            c0 = step * j

            # Region A
            phzAi = phzA[r0: r0 + N, c0: c0 + N, :]
            phzAi = sg[:, :, np.newaxis] * phzAi

            dx_A = D2 * np.ones(len(zApos), dtype=np.float64)
            UoutA, UoutTop = fresnel_propagation_multi(
                Uout, wvl, dx_A, zApos, phzAi, nre, return_all=True
            )

            # Trim tail region
            if tail_len > 0 and zA - tail_len * vres > 0:
                trim = int(np.ceil((zA - tail_len * vres) / proppx))
                if trim < UoutTop.shape[2]:
                    UoutTop = UoutTop[:, :, trim:]

            # Region B — crop to N2
            h = (N - N2) // 2
            UoutA_crop = UoutA[h: h + N2, h: h + N2]

            phzBi = phzB[r0 + h: r0 + h + N2, c0 + h: c0 + h + N2, :]
            phzBi = sg2[:, :, np.newaxis] * phzBi

            dx_B = D2 * np.ones(phzBi.shape[2] + 1, dtype=np.float64)
            UoutB, UoutAll = fresnel_propagation_multi(
                UoutA_crop, wvl, dx_B, zBpos, phzBi, nre, return_all=True
            )

            # Region C — pad UoutB back to N
            UoutB_pad = np.zeros((N, N), dtype=np.complex64)
            UoutB_pad[h: h + N2, h: h + N2] = UoutB

            phzCi = phzC[r0: r0 + N, c0: c0 + N, :]
            phzCi = sg[:, :, np.newaxis] * phzCi

            dx_C = D2 * np.ones(len(zCpos), dtype=np.float64)
            _, UoutBot = fresnel_propagation_multi(
                UoutB_pad, wvl, dx_C, zCpos, phzCi, nre, return_all=True
            )

            # Extract PSF from center of UoutAll
            hp = int(psfpx[0] * ss // 2)
            c_b = N2 // 2
            psf2p = UoutAll[
                c_b - hp: c_b + hp,
                c_b - hp: c_b + hp,
                :-1,
            ]

            # Extract tail fields (crop to N2 region)
            psf2pTop = UoutTop[h: h + N2, h: h + N2, :]
            psf2pBot = UoutBot[h: h + N2, h: h + N2, :]

            # Two-photon: |field|^4
            psf2p = (ss ** 2 * np.abs(psf2p) ** 4).astype(np.float32)
            psf2pTop = (ss ** 2 * np.abs(psf2pTop) ** 4).astype(np.float32)
            psf2pBot = (ss ** 2 * np.abs(psf2pBot) ** 4).astype(np.float32)

            # Downsample PSF by 1/ss (with half-pixel shift to match MATLAB imtranslate)
            psf_ds = np.zeros(psf_out_shape, dtype=np.float32)
            for zz in range(psf2p.shape[2]):
                sl = ndshift(psf2p[:, :, zz], [ss / 2 - 0.5, ss / 2 - 0.5])
                psf_ds[:, :, zz] = ndzoom(sl, 1.0 / ss, order=3)
            psf_ij = (ss ** 2 * psf_ds * scale_factor).astype(np.float32)

            # Accumulate average PSF
            psfs3 += np.abs(psf_ij)
            psfmag[i, j] = float(psf_ij.sum())

            # Tail weights (trapezoidal-rule endpoint correction)
            psf2pTop[:, :, 0] *= 0.5
            if psf2pTop.shape[2] > 1:
                psf2pTop[:, :, -1] *= 0.5
            psf2pBot[:, :, 0] *= 0.5
            if psf2pBot.shape[2] > 1:
                psf2pBot[:, :, -1] *= 0.5

            # Z-profile: sum over xy → 1D vector per z-slice
            # MATLAB: psf2pZTop = squeeze(sum(sum(psf2pTop)));
            psf2pZTop = psf2pTop.sum(axis=(0, 1))
            psf2pZBot = psf2pBot.sum(axis=(0, 1))

            psfT_arr[i, j] = float(psf2pTop.sum()) * proppx / vres
            psfB_arr[i, j] = float(psf2pBot.sum()) * proppx / vres

            # Lateral tail masks (z-summed, then downsampled)
            top_lat = psf2pTop.sum(axis=2)
            bot_lat = psf2pBot.sum(axis=2)
            top_ds = ss ** 2 * ndzoom(
                ndshift(top_lat, [ss / 2 - 0.5, ss / 2 - 0.5]),
                1.0 / ss, order=3,
            )
            bot_ds = ss ** 2 * ndzoom(
                ndshift(bot_lat, [ss / 2 - 0.5, ss / 2 - 0.5]),
                1.0 / ss, order=3,
            )

            if psfTM is None:
                psfTM = top_ds.copy()
                psfBM = bot_ds.copy()
                psfTMz = psf2pZTop.copy()
                psfBMz = psf2pZBot.copy()
            else:
                psfTM += top_ds
                psfBM += bot_ds
                psfTMz += psf2pZTop
                psfBMz += psf2pZBot

            count += 1
            if verbose >= 1 and count % max(1, imax * jmax // 10) == 0:
                print(f"  {count}/{imax * jmax}")

    if verbose >= 1:
        print(f"  {count}/{imax * jmax} done.")

    psfs3 /= (imax * jmax)

    # --- Interpolate sparse mask grid to full FOV ---
    vol_nx = int(vol_sz[0] * vres)
    vol_ny = int(vol_sz[1] * vres)

    # Sparse sample coordinates
    si = np.arange(imax, dtype=np.float64) * psf_samp * vres
    sj = np.arange(jmax, dtype=np.float64) * psf_samp * vres
    sx, sy = np.meshgrid(si, sj, indexing='ij')
    sparse_pts = np.column_stack([sx.ravel(), sy.ravel()])

    # Full grid coordinates
    Xi = np.arange(vol_nx, dtype=np.float64) + 0.5  # center of each pixel
    Yj = np.arange(vol_ny, dtype=np.float64) + 0.5
    Xg, Yg = np.meshgrid(Xi, Yj, indexing='ij')
    full_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Thin-plate spline interpolation (closest to MATLAB griddata 'v4')
    def _interp_sparse(values):
        rbf = RBFInterpolator(sparse_pts, values.ravel(), kernel="thin_plate_spline")
        return rbf(full_pts).reshape(vol_nx, vol_ny).astype(np.float32)

    mask = _interp_sparse(psfmag)

    # Tail masks
    psf_top_mask = _interp_sparse(psfT_arr)
    psf_bot_mask = _interp_sparse(psfB_arr)

    # Tail convolution masks (averaged)
    psf_top_weights = (psfTM / (imax * jmax)).astype(np.float32) if psfTM is not None else np.zeros((1, 1), dtype=np.float32)
    psf_bot_weights = (psfBM / (imax * jmax)).astype(np.float32) if psfBM is not None else np.zeros((1, 1), dtype=np.float32)

    # Interpolate tail masks to full volume XY size
    if psf_top_mask.shape != (vol_nx, vol_ny):
        zf = (vol_nx / psf_top_mask.shape[0], vol_ny / psf_top_mask.shape[1])
        psf_top_mask = ndzoom(psf_top_mask, zf, order=3).astype(np.float32)
        psf_bot_mask = ndzoom(psf_bot_mask, zf, order=3).astype(np.float32)

    # Scalar tail weights: MATLAB psfTS.weight = mean(psfT(:))
    top_weight = float(psfT_arr.mean()) if psfT_arr.size > 0 else 0.0
    bot_weight = float(psfB_arr.mean()) if psfB_arr.size > 0 else 0.0

    # Z-profile weights: interpolate from coarse (prop_sz steps) to fine (voxel) resolution
    # MATLAB: psfTMz = interp1(0:prop_sz:taillength, psfTMz, 0:1/vres:taillength)
    #         psfTS.psfZ = psfTMz / mean(psfTMz)
    psf_top_z = None
    psf_bot_z = None
    if psfTMz is not None and len(psfTMz) > 1:
        prop_sz = psf_params.prop_sz
        tail_len = psf_params.tail_length
        z_coarse = np.arange(len(psfTMz)) * prop_sz
        z_fine = np.arange(0, tail_len + 1.0 / vres, 1.0 / vres)
        psf_top_z = np.interp(z_fine, z_coarse, psfTMz).astype(np.float32)
        psf_bot_z = np.interp(z_fine, z_coarse, psfBMz).astype(np.float32)
        # Normalize to mean=1
        if psf_top_z.mean() > 0:
            psf_top_z /= psf_top_z.mean()
        if psf_bot_z.mean() > 0:
            psf_bot_z /= psf_bot_z.mean()

    psf_top = PsfTail(weights=psf_top_weights, mask=psf_top_mask, weight=top_weight, z_weights=psf_top_z)
    psf_bot = PsfTail(weights=psf_bot_weights, mask=psf_bot_mask, weight=bot_weight, z_weights=psf_bot_z)

    return mask, psfs3, psf_top, psf_bot
