"""DESIGNED-LADDER demo: what depth / overlap / expression actually do to a cell
in the finished 1P widefield movie — in a volume small enough to see.

WHY THIS EXISTS
Every other demo in this folder samples neurons at random and then asks how the
population looks. That is right for producing datasets and useless for *seeing* a
mechanism: in a 4500-cell volume you can never tell whether a given blob is faint
because it is deep, because it is dim, or because a neighbour is sitting on top of
it. This demo inverts that. It places a SMALL number of neurons on three
deliberate LADDERS, one factor per lane, everything else held equal:

    lane A  DEPTH       identical cells at increasing z (the only difference)
    lane B  OVERLAP     identical pairs at decreasing lateral separation
    lane C  EXPRESSION  identical cells at increasing expression level

so the movie becomes a controlled experiment you can read off by eye. Companion
viewer: ``viz_ladders3d.py`` (3D scene + per-ladder panels + animations).

WHAT IS DESIGNED AND WHAT IS STILL PHYSICS
Designed: where the ladder somata sit, their per-cell expression, their tdTomato
labelling, and WHEN each one fires. Everything else is the unmodified calcia
pipeline — real soma/dendrite anatomy, a neuropil of randomly placed filler
neurons, the widefield PSF with its scattering halo, depth attenuation, the camera
noise model. Nothing is drawn by hand and no frame is post-processed to look a
particular way; the ladders are an experimental design, not a rendering trick.

THE ACTIVATION SCHEDULE (the part that makes it readable)
Ladder cells do not fire at random. Each rung of each lane gets its own time slot,
so at any moment only ONE cell per lane is active and its blob in the movie is
unambiguously its own — measurable without demixing anything. The two cells of an
overlap pair fire half a slot apart (A alone, then B alone), and the run ends with
a FINALE where every ladder cell fires at once: the realistic, hopelessly-mixed
case, for comparison. Filler neurons fire decorrelated Poisson bursts throughout,
so the neuropil behaves normally.

THREE CO-REGISTERED MOVIES (one volume, same cells, same schedule)
    gcamp/        GCaMP6f, scattering halo ON   <- what a real 1P rig records
    gcamp_crisp/  GCaMP6f, halo OFF             <- the same tissue, diffraction
                                                   limited: the picture your eye
                                                   wants and the optics deny
    tdt/          tdTomato, static              <- structural red channel; a
                                                   designed subset expresses, so
                                                   green<->red correspondence is
                                                   known per cell
``gcamp_crisp`` differs from ``gcamp`` in ONE parameter (halo_weight), so the
difference between those two images is exactly "tissue scatter" and nothing else.

MOTION IS OFF BY DEFAULT. With motion, a cell's ground-truth position and its
pixel in the movie differ by a per-frame shift — realistic, and it ruins a
teaching figure. ``--motion physio`` turns the real thing back on.

Run:
    # ~2 min end-to-end sanity check on a tiny volume
    conda run -n calcia python examples/demo_ladders3d.py --smoke

    # the real demo (420x420x150 um, 3 channels, then figures)
    conda run -n calcia python examples/demo_ladders3d.py

    # re-draw figures only, from an existing run
    conda run -n calcia python examples/viz_ladders3d.py examples/output/<run>
"""
import argparse
import dataclasses
import datetime as _dt
import json
import os
import subprocess
import sys
import time

import numpy as np

import _striatum_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(HERE, "output")

# Camera / photon budget shared by every channel so the three movies are
# comparable in absolute ADU (only the optics and the indicator differ).
BIAS = 470.0
DARK_RATE = 0.3
READ_NOISE = 1.6


# ======================================================================
# 1. The design — three ladders, one factor each
# ======================================================================
def build_design(vol_um, depth_um, focal_um, n_rungs=7, expr_lo=0.15,
                 expr_hi=2.7, sep_lo=8.0, sep_hi=60.0, z_lo=10.0,
                 slot=16, pre=10, finale=30, tdt_label="alternate"):
    """Lay out the ladder cells and their firing schedule.

    Returns a dict of parallel arrays, one entry per DESIGNED cell (the filler
    neurons are appended after these, so a designed cell's index here IS its
    neuron id everywhere downstream: ``soma_neurons[i]``, ``soma_locs[i]``,
    ``gp_vals[i]``).

    Geometry is expressed as fractions of ``vol_um`` so the layout survives a
    change of FOV. ``margin`` keeps every cell away from the lateral edge, where
    the widefield PSF support is truncated and the neuropil piles up.
    """
    # Lanes are inset far enough that a ~46 um analysis crop around any ladder
    # cell still fits inside the frame — the figures measure a cell against the
    # neuropil in an annulus around it, and a clipped annulus is a wrong number.
    margin = 0.17 * vol_um
    lane_y = np.array([0.24, 0.50, 0.76]) * vol_um      # A, B, C
    n_pairs = n_rungs - 1

    xyz, group, rung, pair_side, sep_um, expr, label = [], [], [], [], [], [], []
    fire = []

    def _add(x, y, z, g, r, side, sep, e, fr, lab):
        xyz.append((float(x), float(y), float(z)))
        group.append(g); rung.append(int(r)); pair_side.append(int(side))
        sep_um.append(float(sep)); expr.append(float(e))
        fire.append(int(fr)); label.append(lab)

    # --- lane A: DEPTH. Identical cells (expression 1.0), z is the only knob.
    #     The rungs span the full imaged column, and one of them is placed
    #     exactly at the focal plane so "in focus" has a reference rung.
    xs = np.linspace(margin, vol_um - margin, n_rungs)
    zs = np.linspace(z_lo, depth_um - 8.0, n_rungs)
    zs[np.argmin(np.abs(zs - focal_um))] = focal_um
    for s in range(n_rungs):
        _add(xs[s], lane_y[0], zs[s], "depth", s, -1, np.nan, 1.0,
             pre + s * slot, f"z={zs[s]:.0f}um")

    # --- lane B: OVERLAP. Identical pairs at the focal plane; only the lateral
    #     separation changes. Separation runs along y so consecutive pairs can be
    #     packed tightly along x. A fires at the slot start, B half a slot later,
    #     so each cell is seen alone before their transients overlap.
    xp = np.linspace(margin, vol_um - margin, n_pairs)
    # The widest pair must still fit inside an analysis crop centred on it, and
    # the crop is bounded by the frame edge next to the outermost pair. Clamping
    # here keeps every FOV size self-consistent instead of silently producing a
    # first rung the figures cannot measure.
    sep_hi = min(sep_hi, 1.2 * margin)
    seps = np.geomspace(sep_hi, sep_lo, n_pairs)
    for s in range(n_pairs):
        for side in (0, 1):
            y = lane_y[1] + (side - 0.5) * seps[s]
            _add(xp[s], y, focal_um, "overlap", s, side, seps[s], 1.0,
                 pre + s * slot + (0 if side == 0 else slot // 2),
                 f"d={seps[s]:.0f}um {'A' if side == 0 else 'B'}")

    # --- lane C: EXPRESSION. Identical cells at the focal plane; only the
    #     per-cell expression multiplier changes (log-spaced across ~18x).
    es = np.geomspace(expr_lo, expr_hi, n_rungs)
    for s in range(n_rungs):
        _add(xs[s], lane_y[2], focal_um, "expression", s, -1, np.nan, es[s],
             pre + s * slot, f"expr={es[s]:.2f}")

    n = len(xyz)
    nt = pre + n_rungs * slot + finale
    finale_frame = pre + n_rungs * slot + 6

    # --- tdTomato labelling: DESIGNED, not random, so every comparison the red
    #     channel is supposed to support actually exists in the data. Alternate
    #     rungs of lanes A and C are labelled (so you can see a labelled and an
    #     unlabelled cell at the same depth / brightness), and in every overlap
    #     pair exactly ONE cell is labelled — that is the two-colour question in
    #     its purest form: the green blob is a merge of two cells, which one is
    #     the red-labelled one?
    tdt = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if tdt_label == "all":
            tdt[i] = 1.0
        elif group[i] == "overlap":
            tdt[i] = 1.0 if pair_side[i] == 0 else 0.0
        else:
            tdt[i] = 1.0 if (rung[i] % 2 == 0) else 0.0
    # The expression lane keeps its brightness ladder in red too, so the red
    # channel shows the same expression axis (a dim cell is dim in both colours).
    for i in range(n):
        if group[i] == "expression" and tdt[i] > 0:
            tdt[i] = expr[i]

    return dict(
        xyz=np.asarray(xyz, dtype=np.float64),
        group=np.asarray(group), rung=np.asarray(rung, dtype=np.int32),
        pair_side=np.asarray(pair_side, dtype=np.int32),
        sep_um=np.asarray(sep_um, dtype=np.float64),
        expr=np.asarray(expr, dtype=np.float64),
        tdt_expr=tdt, fire_frame=np.asarray(fire, dtype=np.int32),
        label=np.asarray(label), nt=int(nt), slot=int(slot), pre=int(pre),
        finale_frame=int(finale_frame), n_rungs=int(n_rungs),
        n_pairs=int(n_pairs), lane_y=lane_y, focal_um=float(focal_um),
        vol_um=float(vol_um), depth_um=float(depth_um))


def build_spike_matrix(design, K, n_designed, nt, fps, n_filler_rate,
                       spikes_per_event, seed, spike_params):
    """(K, nt*100/fps) spike matrix at the 100 Hz internal rate.

    Filler + background rows get ordinary decorrelated bursts from calcia's own
    generator; the designed rows are overwritten with the schedule. Returns the
    100 Hz matrix and its per-frame binning (the spike ground truth saved with
    the run).
    """
    from calcia.traces.spikes import gen_burst_spike_times

    nt_int = int(round(nt * 100.0 / fps))
    sp = dataclasses.replace(spike_params, K=K, dt=1 / 100.0, nt=nt_int,
                             rate=n_filler_rate, verbose=0)
    rng_state = np.random.get_state()
    np.random.seed(seed + 4242)               # gen_burst_spike_times uses global
    S = np.asarray(gen_burst_spike_times(sp), dtype=np.float32)
    np.random.set_state(rng_state)
    if S.shape[1] < nt_int:                   # generator rounds; pad if short
        S = np.pad(S, ((0, 0), (0, nt_int - S.shape[1])))
    S = S[:, :nt_int]

    step = int(round(100.0 / fps))            # 100 Hz samples per movie frame
    S[:n_designed] = 0.0
    for i in range(n_designed):
        f0 = int(design["fire_frame"][i])
        S[i, f0 * step:f0 * step + spikes_per_event] = 1.0
        ff = int(design["finale_frame"])
        S[i, ff * step:ff * step + spikes_per_event] = 1.0

    binned = S.reshape(K, nt, step).sum(axis=2).astype(np.float32)
    return S, binned


# ======================================================================
# 2. Designed neuron placement (Phase-1 monkeypatch)
# ======================================================================
def make_designed_sampler(design, n_filler, keepout_um, seed, announce=True):
    """Build a drop-in replacement for ``calcia.pipeline.sample_dense_neurons``.

    Same signature and same return contract, but the positions are OURS: the
    designed ladder cells first (so their index is their neuron id), then
    ``n_filler`` randomly placed neurons that supply a normal neuropil. Fillers
    keep ``keepout_um`` of lateral clearance from every designed cell so a random
    neighbour never lands on a ladder rung and confounds it; they are otherwise
    unconstrained and overlap each other freely.
    """
    from calcia.geometry.sphere_sampling import spiral_sample_sphere
    from calcia.volume.neurons import generate_neural_body

    def sampler(vol_params, neur_params, vessel_mask=None, verbose=None):
        vol_sz = np.asarray(vol_params.vol_sz, dtype=np.float64)
        rng = np.random.default_rng(seed + 991)
        pos = [np.asarray(p, dtype=np.float32) for p in design["xyz"]]
        d_xy = design["xyz"][:, :2]
        pad = 12.0
        placed_xyz = list(design["xyz"])
        tries = 0
        while len(pos) - len(design["xyz"]) < n_filler and tries < 400 * (n_filler + 1):
            tries += 1
            c = rng.uniform([pad, pad, 6.0],
                            [vol_sz[0] - pad, vol_sz[1] - pad, vol_sz[2] - 6.0])
            if np.min(np.linalg.norm(d_xy - c[:2], axis=1)) < keepout_um:
                continue
            if placed_xyz and np.min(np.linalg.norm(
                    np.asarray(placed_xyz) - c, axis=1)) < 13.0:
                continue
            placed_xyz.append(c)
            pos.append(c.astype(np.float32))
        positions = np.asarray(pos, dtype=np.float32)

        verts, faces = spiral_sample_sphere(neur_params.n_samps,
                                            return_triangulation=True)
        neurons, angles = [], []
        for p in positions:
            Vcell, Vnuc, _, rot = generate_neural_body(
                neur_params, vertices=verts, faces=faces)
            neurons.append((Vcell + p, Vnuc + p, faces))
            angles.append(rot)
        vol_params.N_neur = len(positions)
        if announce:
            print(f"  [designed] {len(design['xyz'])} ladder cells + "
                  f"{len(positions) - len(design['xyz'])} filler "
                  f"= {len(positions)} neurons  (keepout {keepout_um:.0f} um)")
        return neurons, angles, positions

    return sampler


def normalize_designed_brightness(vol_out, n_designed):
    """Remove the per-cell RANDOM expression jitter from the ladder cells only.

    ``set_cell_fluorescence`` draws a per-cell gain ~N(1, 0.2). That is correct
    physics for a population and poison for a controlled ladder: a +/-20% random
    gain on top of a designed 18x expression sweep muddies every rung. Dividing
    each designed cell's footprint by its own mean soma value fixes its soma mean
    at exactly 1.0, so from here on the ONLY thing separating two ladder cells is
    the factor the ladder varies (depth, separation, or the designed expression
    multiplier applied later through ``mod_vals``). Filler neurons keep their
    natural jitter.
    """
    for i in range(min(n_designed, len(vol_out.gp_vals))):
        cfd = vol_out.gp_vals[i]
        sm = np.asarray(cfd.soma_mask)
        if not sm.any():
            continue
        g = float(np.mean(cfd.fluorescence[sm]))
        if g > 0:
            cfd.fluorescence = (cfd.fluorescence / g).astype(np.float32)


# ======================================================================
# 3. Channel rendering
# ======================================================================
def render_channel(vol_out, vol_params, *, kind, nt, fps, seed, mod_vals,
                   s_times, spikes_binned, halo_um, halo_weight, focal_um,
                   motion, motion_seed, bg_scale, pavg, sfrac, scan_buff,
                   separate_focus, verbose=1):
    """Render one co-registered channel of the same volume.

    ``kind`` is 'gcamp' (dynamic GCaMP6f) or 'tdt' (static tdTomato). The two
    differ only in emission wavelength, tissue scatter length, photon budget and
    the indicator dynamics — the anatomy, the cell ids and the geometry are
    shared, so the movies are pixel-co-registered by construction.
    """
    from calcia import simulate_optical_propagation, generate_time_traces
    from calcia.config import STATIC_PRESETS
    from calcia.config.params import (PsfParams, WidefieldParams, SpikeParams,
                                      ScanParams, CameraNoiseParams,
                                      MotionParams, CalciumParams)
    from calcia.scanning import scan_widefield

    is_tdt = (kind == "tdt")
    P = STATIC_PRESETS["tdt"]
    depth_um = vol_params.vol_sz[2]
    supp = min(100.0, 0.5 * vol_params.vol_sz[0])

    # --- Phase 2: optics. Wide PSF support so the out-of-focus light of the deep
    #     planes can actually spread; the two-scale halo is the tissue-scatter
    #     term that makes 1P look like 1P (halo_weight=0 -> diffraction limited).
    psf_params = PsfParams(
        imaging_mode="widefield", psf_type="gaussian_analytical",
        lambda_em_um=(P["lambda_em_um"] if is_tdt else 0.52),
        obj_na=0.8, n=1.35, psf_sz=(supp, supp, 20.0),
        wf_focal_depth_um=focal_um,
        **({"scatter_length_um_wf": P["scatter_length_um_wf"]} if is_tdt else {}))
    opt_out = simulate_optical_propagation(
        vol_params=vol_params, psf_params=psf_params, vol_out=vol_out, verbose=0)
    if halo_weight and halo_weight > 0:
        opt_out.psf = C.broaden_psf_two_scale(opt_out.psf, halo_um, halo_weight,
                                              vol_params.vres)
    if verbose:
        print(f"    optics: focal={focal_um:.0f}um  halo={halo_um:.0f}um "
              f"w={halo_weight:.2f}  psf{opt_out.psf.shape}")

    # --- Phase 3: traces. mod_vals carries the per-cell expression (designed for
    #     the ladder cells); s_times carries the designed firing schedule.
    K = len(vol_out.gp_vals)
    has_axons = len(vol_out.bg_proc) > 0
    if is_tdt:
        spike_params = SpikeParams(K=K, nt=nt, dt=1 / fps, N_bg=0,
                                   dyn_type="static", prot="tdt",
                                   dendflag=True, axonflag=has_axons,
                                   bg_scale=P["bg_scale"], verbose=0)
        time_out = generate_time_traces(spike_params=spike_params,
                                        n_locs=vol_out.locs, mod_vals=mod_vals,
                                        verbose=0)
    else:
        spike_params = SpikeParams(K=K, nt=nt, dt=1 / fps, N_bg=0,
                                   axonflag=has_axons, prot="GCaMP6f",
                                   smod_flag="burst", burst_mean=0,
                                   bg_scale=bg_scale, verbose=0)
        time_out = generate_time_traces(
            spike_params=spike_params, cal_params=CalciumParams(prot_type="gcamp6f"),
            s_times=s_times, mod_vals=mod_vals, verbose=0)
        # generate_time_traces returns spikes=None when the caller supplies
        # s_times; hand the designed schedule back so the run's traces.npz still
        # carries per-cell spike ground truth.
        time_out.spikes = spikes_binned[:, :time_out.soma.shape[1]]

    # --- Phase 4: camera scan. Motion off by default (see module docstring).
    scan_params = ScanParams(scan_buff=scan_buff, motion=(motion != "none"),
                             sfrac=sfrac, verbose=0)
    motion_params = (MotionParams(model="physio", seed=motion_seed)
                     if motion == "physio" else None)
    cam = CameraNoiseParams(
        qe=1.0, dark_rate=DARK_RATE, t_exp=1 / fps, read_noise=READ_NOISE,
        gain_e_per_adu=(P["gain"] if is_tdt else 1.0), bias=BIAS)
    wf_params = (WidefieldParams(pavg=P["pavg"], lambda_ex_um=P["lambda_ex_um"],
                                 sigma_abs=P["sigma_abs"], phi=P["phi"],
                                 qe_det=P["qe_det"])
                 if is_tdt else
                 WidefieldParams(pavg=pavg, lambda_ex_um=0.488, qe_det=0.8))

    # focus_slab_um is set explicitly: the diffraction depth-of-field here is
    # ~1.1 um, which would make "in focus" mean a single z-plane and the split
    # useless. 20 um is the slab a person would actually call the in-focus cell
    # layer, so the in-focus / out-of-focus split answers a question worth asking
    # ("how much of this frame is the layer I am imaging, and how much is fog
    # from the rest of the column?").
    scan_out = scan_widefield(vol_out=vol_out, opt_out=opt_out, time_out=time_out,
                              scan_params=scan_params, cam_params=cam,
                              wf_params=wf_params, motion_params=motion_params,
                              spike_params=spike_params, seed=seed,
                              separate_focus=separate_focus,
                              focus_slab_um=(20.0 if separate_focus else None))
    noisy, clean = scan_out.mov, scan_out.mov_raw

    params_dict = dict(vol_params=vol_params, psf_params=psf_params,
                       spike_params=spike_params, scan_params=scan_params,
                       wf_params=wf_params, cam_params=cam,
                       motion_params=motion_params)
    return dict(noisy=noisy, clean=clean, opt_out=opt_out, time_out=time_out,
                scan_out=scan_out, params_dict=params_dict,
                psf_params=psf_params)


def dff_stats(mov, bias=BIAS):
    sig = np.clip(mov - bias, 0, None)
    f0 = np.percentile(sig, 10, axis=2, keepdims=True)
    dff = (sig - f0) / (f0 + 1e-6)
    return float(np.percentile(dff, 99)), float(np.median(mov.mean(2)))


def verify_alignment(clean, design, n_designed, vres, sfrac, scan_buff,
                     search_px=6, min_snr=8.0):
    """Check each ladder cell's transient really lands where the geometry says.

    ``base_px`` maps microns to movie pixels analytically; this finds the local
    dF peak inside a small window around that prediction during the cell's own
    time slot, and reports the residual. A residual of more than ~2 px would mean
    the run and its ground truth are NOT aligned and every figure built on top
    would be quietly wrong — so it is measured, not assumed.

    Cells whose own transient does not clear ``min_snr`` times the frame-
    difference noise are reported as ``nan`` rather than as a large residual: a
    deep or dim cell that is genuinely invisible has no peak to find, and letting
    its argmax wander would turn a physics result into a fake alignment failure.
    """
    s, off = vres / sfrac, scan_buff / sfrac
    pred = np.column_stack([design["xyz"][:n_designed, 0] * s - off,
                            design["xyz"][:n_designed, 1] * s - off])
    H, W = clean.shape[:2]
    res, amp = [], []
    for i in range(n_designed):
        f = int(design["fire_frame"][i])
        a, b = int(round(pred[i, 0])), int(round(pred[i, 1]))
        if not (0 <= a < H and 0 <= b < W):
            res.append(np.nan); amp.append(np.nan); continue
        base = clean[:, :, max(0, f - 3)]
        d = clean[:, :, min(clean.shape[2] - 1, f + 3)] - base
        a0, a1 = max(0, a - search_px), min(H, a + search_px + 1)
        b0, b1 = max(0, b - search_px), min(W, b + search_px + 1)
        win = d[a0:a1, b0:b1]
        # noise scale of this difference image, measured away from the cell
        sig = 1.4826 * float(np.median(np.abs(d - np.median(d))))
        da, db = np.unravel_index(np.argmax(win), win.shape)
        amp.append(float(win.max()))
        if not np.isfinite(sig) or sig <= 0 or win.max() < min_snr * sig:
            res.append(np.nan)          # too faint to localise: not a failure
            continue
        res.append(float(np.hypot(a0 + da - pred[i, 0], b0 + db - pred[i, 1])))
    return pred, np.asarray(res), np.asarray(amp)


# ======================================================================
# 4. CLI / main
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Designed depth/overlap/expression ladders + tdTomato, "
                    "small volume, built for looking at.")
    p.add_argument("--vol-um", type=float, default=420.0, dest="vol_um",
                   help="Lateral FOV (square), um. Small on purpose: the point "
                        "is to SEE individual cells, not to fill a window.")
    p.add_argument("--depth-um", type=float, default=150.0, dest="depth_um",
                   help="Imaged tissue depth, um. Sets the span of the depth "
                        "ladder and how much out-of-focus haze there is.")
    p.add_argument("--focal-um", type=float, default=32.0, dest="focal_um",
                   help="Focal plane depth, um. The overlap and expression "
                        "lanes sit exactly here so depth is not a confound.")
    p.add_argument("--vres", type=int, default=1, help="voxels/um (grid = vol*vres)")
    p.add_argument("--sfrac", type=int, default=2,
                   help="camera binning; pixel size = sfrac/vres um")
    p.add_argument("--rungs", type=int, default=7,
                   help="rungs per ladder (depth + expression); overlap gets rungs-1 pairs")
    p.add_argument("--n-filler", type=int, default=500, dest="n_filler",
                   help="randomly placed neurons supplying a normal neuropil. "
                        "Anatomical striatum density would be ~2600 here; the "
                        "default is deliberately sparser so the ladders stay "
                        "legible. Raise it for a realistically crowded field.")
    p.add_argument("--keepout-um", type=float, default=26.0, dest="keepout_um",
                   help="lateral clearance filler neurons keep from ladder "
                        "cells (~one scatter-halo core), so a random neighbour "
                        "never lands on a rung and confounds it")
    p.add_argument("--halo-um", type=float, default=28.0, dest="halo_um",
                   help="two-scale PSF: width of the tissue-scatter halo (um)")
    p.add_argument("--halo-weight", type=float, default=0.8, dest="halo_weight",
                   help="two-scale PSF: fraction of light in the halo. 0.8 is "
                        "the production two-colour recipe (real 1P striatum); "
                        "lower it to make the ladders easier to read.")
    p.add_argument("--bg-scale", type=float, default=2.0, dest="bg_scale",
                   help="neuropil/axon wash amplitude (never exceed ~2.6: above "
                        "that the neuropil outshines the somata and cells inverta"
                        " into dark holes)")
    p.add_argument("--pavg", type=float, default=2.0, help="photon budget scale")
    p.add_argument("--rate", type=float, default=0.02,
                   help="filler-neuron firing rate (burst mode)")
    p.add_argument("--spikes-per-event", type=int, default=5,
                   dest="spikes_per_event",
                   help="spikes in one scheduled ladder-cell burst (at 100 Hz)")
    p.add_argument("--slot", type=int, default=16,
                   help="frames per ladder slot (one rung fires per slot)")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--motion", choices=["none", "physio", "randomwalk"],
                   default="none",
                   help="sample motion. 'none' (default) keeps ground truth and "
                        "movie pixels in one frame, which is what makes the "
                        "figures readable; 'physio' is the realistic model.")
    p.add_argument("--tdt-label", choices=["alternate", "all"],
                   default="alternate", dest="tdt_label",
                   help="which ladder cells express tdTomato")
    p.add_argument("--tdt-filler-frac", type=float, default=0.5,
                   dest="tdt_filler_frac",
                   help="fraction of FILLER neurons expressing tdTomato")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--no-crisp", action="store_true", dest="no_crisp",
                   help="skip the diffraction-limited comparison channel")
    p.add_argument("--no-tdt", action="store_true", dest="no_tdt",
                   help="skip the tdTomato channel")
    p.add_argument("--no-separate-focus", action="store_true",
                   dest="no_separate_focus",
                   help="skip the in-focus / out-of-focus split (halves Phase-4 cost)")
    p.add_argument("--viz-cache", action="store_true", dest="viz_cache",
                   help="also build the calcia.viz mesh cache (slow) so the run "
                        "opens in `python -m calcia.viz <run>/gcamp`")
    p.add_argument("--no-figs", action="store_true", dest="no_figs",
                   help="do not run viz_ladders3d.py at the end")
    p.add_argument("--smoke", action="store_true",
                   help="tiny 200x200x100 um / 4-rung / 60-filler run (~1 min) "
                        "to check the pipeline end to end. The ladders are "
                        "cramped at that FOV — it is a smoke test, not a demo.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.vol_um, args.depth_um, args.focal_um = 200.0, 100.0, 26.0
        args.rungs, args.n_filler, args.slot = 4, 60, 10

    import _instrument; _instrument.start("ladders3d")
    t_wall = time.time()

    fps = args.fps
    design = build_design(args.vol_um, args.depth_um, args.focal_um,
                          n_rungs=args.rungs, slot=args.slot,
                          tdt_label=args.tdt_label)
    n_designed = len(design["xyz"])
    nt = design["nt"]

    print("=" * 68)
    print("DESIGNED LADDERS  (depth / overlap / expression)  + tdTomato")
    print("=" * 68)
    print(f"  volume {args.vol_um:.0f} x {args.vol_um:.0f} x {args.depth_um:.0f} um"
          f"   vres={args.vres}   focal={args.focal_um:.0f} um")
    print(f"  ladder cells: {n_designed}  ({args.rungs} depth, "
          f"{design['n_pairs']} overlap pairs, {args.rungs} expression)")
    print(f"  filler neurons: {args.n_filler}   frames: {nt} @ {fps:.0f} Hz "
          f"({nt/fps:.1f} s)   motion={args.motion}")

    # ---------------- Phase 1 (designed anatomy) ----------------
    import calcia.pipeline as _pipe
    from calcia import simulate_neural_volume
    from calcia.config.params import VolumeParams, VascParams
    from calcia.traces.expression import expression_variation
    from calcia.volume import fill_nuclei

    _pipe.sample_dense_neurons = make_designed_sampler(
        design, args.n_filler, args.keepout_um, args.seed)

    vol_params = VolumeParams(
        vol_sz=(args.vol_um, args.vol_um, args.depth_um), vres=args.vres,
        vol_depth=0, region="striatum", N_neur=n_designed + args.n_filler)
    # Vasculature OFF: a vessel crossing a ladder rung is a confound, and this
    # demo is about depth/overlap/expression. The production volumes keep it on.
    vasc_params = VascParams(flag=False)

    print("\n[Phase 1] anatomy")
    t0 = time.time()
    vol_out = simulate_neural_volume(vol_params=vol_params,
                                     vasc_params=vasc_params,
                                     seed=args.seed, verbose=1)
    vol_params = vol_out.params["vol_params"]
    print(f"  Phase 1 done in {time.time()-t0:.1f}s")

    fill_nuclei(vol_out)
    normalize_designed_brightness(vol_out, n_designed)

    K = len(vol_out.gp_vals)
    n_neur = int(vol_params.N_neur)
    locs = np.asarray(vol_out.locs)
    placed = locs[:n_designed]
    max_err = float(np.abs(placed - design["xyz"]).max())
    print(f"  components K={K}  neurons={n_neur}  bg_proc={len(vol_out.bg_proc)}")
    print(f"  designed positions preserved to {max_err:.3f} um")
    if max_err > 1e-3:
        raise SystemExit("designed positions were not preserved through Phase 1")

    # ---------------- expression (mod_vals) per channel ----------------
    # mod_vals is calcia's per-component expression multiplier: it scales that
    # component's soma, dendrite and axon traces. It is the exact knob the
    # expression ladder needs, and (set to 0) the exact knob for "this cell does
    # not express tdTomato at all".
    rng = np.random.default_rng(args.seed + 77)
    base_mod = expression_variation(K, 0.0, (0.4, 2.53)).astype(np.float32)

    mod_g = base_mod.copy()
    mod_g[:n_designed] = design["expr"].astype(np.float32)

    tdt_filler = (rng.random(n_neur - n_designed) < args.tdt_filler_frac)
    mod_t = base_mod.copy()
    mod_t[:n_designed] = design["tdt_expr"].astype(np.float32)
    mod_t[n_designed:n_neur] *= tdt_filler.astype(np.float32)
    # Background dendrite / axon components (rows >= n_neur) belong to processes
    # from cells outside the imaged block; label them at the same rate so the red
    # neuropil is as sparse as the red cell population.
    if K > n_neur:
        mod_t[n_neur:] *= (rng.random(K - n_neur) < args.tdt_filler_frac
                           ).astype(np.float32)
    n_tdt_cells = int((mod_t[:n_neur] > 0).sum())
    print(f"  tdTomato+: {n_tdt_cells}/{n_neur} cells "
          f"({int((design['tdt_expr'] > 0).sum())}/{n_designed} ladder cells)")

    # ---------------- designed spike schedule ----------------
    from calcia.config.params import SpikeParams
    proto = SpikeParams(K=K, nt=nt, dt=1 / fps, N_bg=0, axonflag=True,
                        prot="GCaMP6f", smod_flag="burst", burst_mean=0,
                        verbose=0)
    s_times, spikes_binned = build_spike_matrix(
        design, K, n_designed, nt, fps, args.rate, args.spikes_per_event,
        args.seed, proto)
    print(f"  spike schedule: {int(spikes_binned[:n_designed].sum())} designed "
          f"spikes, {int(spikes_binned[n_designed:].sum())} filler/background")

    # ---------------- render the channels ----------------
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "ladders3d_smoke" if args.smoke else f"ladders3d_{int(args.vol_um)}um"
    root = os.path.join(OUTPUT_ROOT, f"{tag}_{ts}")
    os.makedirs(root, exist_ok=True)

    channels = [("gcamp", "gcamp", args.halo_weight)]
    if not args.no_crisp:
        channels.append(("gcamp_crisp", "gcamp", 0.0))
    if not args.no_tdt:
        channels.append(("tdt", "tdt", args.halo_weight))

    sep_focus = not args.no_separate_focus
    summary = {}
    for name, kind, halo_w in channels:
        print(f"\n[Phase 2-4] channel '{name}'  ({kind}, halo_weight={halo_w})")
        t0 = time.time()
        out = render_channel(
            vol_out, vol_params, kind=kind, nt=nt, fps=fps, seed=args.seed,
            mod_vals=(mod_t if kind == "tdt" else mod_g),
            s_times=s_times, spikes_binned=spikes_binned,
            halo_um=args.halo_um, halo_weight=halo_w, focal_um=args.focal_um,
            motion=args.motion, motion_seed=args.seed + 3,
            bg_scale=args.bg_scale, pavg=args.pavg, sfrac=args.sfrac,
            scan_buff=(30 if args.motion == "physio" else 10),
            separate_focus=sep_focus)
        noisy, clean = out["noisy"], out["clean"]
        d99, med = dff_stats(noisy)
        print(f"    scanned in {time.time()-t0:.1f}s  movie {noisy.shape}  "
              f"dff_p99={d99:.3f}  median={med:.0f}")

        sc = out["params_dict"]["scan_params"]
        # Alignment is measured on the transient, so it only means anything on a
        # dynamic channel; the static tdTomato movie has no transients to find.
        if kind == "tdt":
            ok = float("nan")
        else:
            pred, resid, amp = verify_alignment(clean, design, n_designed,
                                                args.vres, sc.sfrac, sc.scan_buff)
            ok = np.nanmax(resid) if np.isfinite(resid).any() else np.nan
            n_ok = int(np.isfinite(resid).sum())
            print(f"    GT<->movie alignment: max residual {ok:.1f} px "
                  f"(median {np.nanmedian(resid):.1f}) over {n_ok}/{n_designed} "
                  f"cells bright enough to localise")

        run_dir = os.path.join(root, name)
        os.makedirs(run_dir, exist_ok=True)
        meta = dict(
            kind=f"designed_ladders_{name}", demo="ladders3d", channel=name,
            region="striatum", prot=("tdt" if kind == "tdt" else "GCaMP6f"),
            dyn_type=("static" if kind == "tdt" else "Ca_DE"),
            optics_method=("two-scale-psf" if halo_w > 0 else "diffraction"),
            halo_um=float(args.halo_um), halo_weight=float(halo_w),
            focal_depth_um=float(args.focal_um), composite=False,
            illum_grad=False, vessels=False,
            motion_model=args.motion, motion_seed=int(args.seed + 3),
            seed=int(args.seed), nt=int(nt), dt=float(1 / fps),
            vol_sz=[float(args.vol_um), float(args.vol_um), float(args.depth_um)],
            vres=int(args.vres), N_neur=int(n_neur),
            n_designed=int(n_designed), n_filler=int(args.n_filler),
            n_soma=int(n_neur), N_soma_traces=int(out["time_out"].soma.shape[0]),
            total_spikes=int(spikes_binned.sum()),
            n_tdt_cells=(int(n_tdt_cells) if kind == "tdt" else None),
            movie_shape=[int(v) for v in noisy.shape],
            dff_p99=d99, median=med,
            align_resid_px_max=float(ok),
            config=dict(sfrac=int(sc.sfrac), scan_buff=int(sc.scan_buff),
                        motion_model=args.motion, bg_scale=float(args.bg_scale),
                        pavg=float(args.pavg)),
            timestamp=_dt.datetime.now().isoformat())
        C.save_full_bundle(run_dir, noisy=noisy, clean=clean, vol_out=vol_out,
                           vol_params=vol_params, opt_out=out["opt_out"],
                           time_out=out["time_out"], scan_out=out["scan_out"],
                           params_dict=out["params_dict"], metadata=meta,
                           dt=1 / fps,
                           make_viz=(args.viz_cache and name == "gcamp"))
        # The in-focus / out-of-focus split is this demo's own extra ground
        # truth: it is what lets a figure say "this much of the blob is the cell
        # and this much is haze from everything else in the column".
        so = out["scan_out"]
        if sep_focus and getattr(so, "mov_infocus", None) is not None:
            np.savez_compressed(
                os.path.join(run_dir, "focus_split.npz"),
                mov_infocus=np.transpose(so.mov_infocus, (2, 0, 1)).astype(np.float32),
                mov_oof=np.transpose(so.mov_oof, (2, 0, 1)).astype(np.float32),
                axes=np.array("THW"))
        summary[name] = dict(dff_p99=d99, median=med,
                             resid_max=float(ok), shape=list(noisy.shape))

    # ---------------- the design itself (what every figure reads) ----------------
    np.savez_compressed(
        os.path.join(root, "design.npz"),
        xyz=design["xyz"], group=design["group"], rung=design["rung"],
        pair_side=design["pair_side"], sep_um=design["sep_um"],
        expr=design["expr"], tdt_expr=design["tdt_expr"],
        fire_frame=design["fire_frame"], label=design["label"],
        filler_xyz=locs[n_designed:n_neur].astype(np.float32),
        filler_tdt=tdt_filler,
        mod_gcamp=mod_g[:n_neur], mod_tdt=mod_t[:n_neur],
        nt=np.int64(nt), slot=np.int64(design["slot"]),
        pre=np.int64(design["pre"]), finale_frame=np.int64(design["finale_frame"]),
        n_rungs=np.int64(design["n_rungs"]), n_pairs=np.int64(design["n_pairs"]),
        lane_y=design["lane_y"], focal_um=np.float64(args.focal_um),
        vol_um=np.float64(args.vol_um), depth_um=np.float64(args.depth_um),
        vres=np.int64(args.vres), sfrac=np.int64(args.sfrac),
        scan_buff=np.int64(30 if args.motion == "physio" else 10),
        fps=np.float64(fps), n_designed=np.int64(n_designed),
        spikes_designed=spikes_binned[:n_designed])

    with open(os.path.join(root, "design.json"), "w", encoding="utf-8") as f:
        json.dump(dict(
            cells=[dict(id=int(i), group=str(design["group"][i]),
                        rung=int(design["rung"][i]),
                        pair_side=int(design["pair_side"][i]),
                        x=float(design["xyz"][i, 0]), y=float(design["xyz"][i, 1]),
                        z=float(design["xyz"][i, 2]),
                        sep_um=(None if not np.isfinite(design["sep_um"][i])
                                else float(design["sep_um"][i])),
                        expr=float(design["expr"][i]),
                        tdt_expr=float(design["tdt_expr"][i]),
                        fire_frame=int(design["fire_frame"][i]),
                        label=str(design["label"][i]))
                   for i in range(n_designed)],
            nt=int(nt), fps=float(fps), slot=int(design["slot"]),
            finale_frame=int(design["finale_frame"]),
            focal_um=float(args.focal_um), vol_um=float(args.vol_um),
            depth_um=float(args.depth_um), n_filler=int(args.n_filler),
            channels={n: summary[n] for n in summary},
            args=vars(args)), f, indent=2)

    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Designed ladders — depth / overlap / expression (+ tdTomato)

Produced by `examples/demo_ladders3d.py`. A controlled experiment inside a
widefield simulation: {n_designed} neurons placed on three ladders that each vary
exactly one property, in a {args.vol_um:.0f} x {args.vol_um:.0f} x {args.depth_um:.0f} um
block with {args.n_filler} randomly placed filler neurons supplying a normal
neuropil. Focal plane {args.focal_um:.0f} um. {nt} frames @ {fps:.0f} Hz.
Sample motion: {args.motion}.

| path | what it is |
| --- | --- |
| `design.json` / `design.npz` | the ladder table:每个细胞的 x/y/z、表达量、tdTomato 标记、发放帧 |
| `gcamp/` | GCaMP6f channel, tissue-scatter halo ON — what a real 1P rig records |
| `gcamp_crisp/` | the SAME tissue, halo OFF (diffraction limited) — the only difference is one PSF parameter |
| `tdt/` | tdTomato structural channel, static, co-registered cell-for-cell |
| `*/focus_split.npz` | the scan's exact in-focus / out-of-focus split (20 um slab) |
| `figures/index.html` | **start here** — the 3D scene, the animations and one panel per ladder |

Each channel folder is a standard calcia run bundle (movies.npz, traces.npz,
optics.npz, params.pkl, metadata.json, cell_footprints.pkl, movie.gif).
Cell `i` is the same cell in all three channels and in `design.npz`.

Disk: ~2 GB, most of it `cell_footprints.pkl` (~0.6 GB per channel). All three
channels share ONE Phase-1 volume, so those three files are byte-identical —
delete two of them if you need the space; only `calcia.viz` reads them.

Rebuild the figures, or explore the 3D scene interactively:

```
python examples/viz_ladders3d.py {os.path.basename(root)}
python examples/viz_ladders3d.py {os.path.basename(root)} --interactive
```
""")

    print(f"\nTotal wall time: {time.time()-t_wall:.1f}s")
    print(f"Output: {root}")
    for n, s in summary.items():
        print(f"  {n:<12s} {s['shape']}  dff_p99={s['dff_p99']:.3f}  "
              f"align<={s['resid_max']:.1f}px")

    if not args.no_figs:
        print("\n[figures] running viz_ladders3d.py ...")
        rc = subprocess.call([sys.executable,
                              os.path.join(HERE, "viz_ladders3d.py"), root])
        if rc != 0:
            print(f"  viz_ladders3d.py exited {rc} — re-run it manually:\n"
                  f"    python examples/viz_ladders3d.py {root}")


if __name__ == "__main__":
    main()
