"""Shared config + IO/rendering/metrics for the striatum widefield scripts.

Reused across:
  - demo_widefield_striatum_v1.py
  - striatum_demix_dataset.py
  - striatum_param_sweep.py

Two things live here, both genuinely shared across the three scripts:

  1. `StriatumConfig` — the SINGLE SOURCE OF TRUTH for every tunable simulation
     parameter. Previously these were scattered across three layers (calcia
     defaults, per-factory literals, inline post-processing constants in the
     demo). Now one dataclass holds them all; a script overrides only the fields
     it needs and the `build_*` methods turn it into the concrete calcia
     dataclasses. `dataclasses.asdict(cfg)` gives a complete reproducible record
     for metadata.json.

  2. Stateless plumbing (load a cached Phase-1 volume, save TIFFs, build the
     preview GIF, image-quality metrics) — utilities, not tunable knobs.

Heavy deps (calcia, tifffile, matplotlib) are imported lazily inside the
methods/functions so `import _striatum_common` stays cheap.
"""
import pickle
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np


# ======================================================================
# Configuration (single source of truth)
# ======================================================================
@dataclass
class IllumConfig:
    """Non-uniform widefield illumination profile (bright centre -> dark edges).

    Real 1P widefield / miniscope data is dominated by a large-scale
    illumination gradient (LED/beam Gaussian + imaging-window vignetting) that
    NAOMi's idealised uniform Koehler illumination lacks. `illum_map` returns an
    (H, W) multiplier in [floor, 1] applied to the signal BEFORE the camera bias
    pedestal (excitation non-uniformity scales photons, not the bias).
    """
    enable: bool = True
    cx: float = 0.48
    cy: float = 0.45
    sx: float = 0.40
    sy: float = 0.45
    floor: float = 0.05

    def illum_map(self, shape_hw):
        H, W = shape_hw
        yy, xx = np.mgrid[0:H, 0:W]
        gx = (xx - self.cx * W) / (self.sx * W)
        gy = (yy - self.cy * H) / (self.sy * H)
        return (self.floor + (1.0 - self.floor)
                * np.exp(-(gx ** 2 + gy ** 2))).astype(np.float32)


@dataclass
class StriatumConfig:
    """All tunable parameters for one striatum widefield simulation."""

    # ---- geometry (Phase 1) ----
    vol_um: int = 1000          # lateral FOV (square), um
    depth_um: int = 60          # imaged depth, um (1P scatter-limited ~50-80 um)
    vres: int = 1               # voxels / um (1 keeps the 1-1.7 mm FOV in RAM)
    vol_depth: int = 0
    region: str = "striatum"
    n_neur: Optional[int] = None  # None -> use neur_density below
    # Neuron density in neurons/mm^3. 1e5 = full anatomical striatum density
    # (the original default). Only used when n_neur is None; override per-run
    # via the demo's --neur-density (e.g. 70000 for a 70%-density volume).
    neur_density: float = 1e5
    seed: int = 42

    # ---- time ----
    nt: int = 200               # frames
    fps: float = 20.0           # matches the real striatum window recordings

    # ---- spikes / traces (Phase 3) ----
    rate: float = 0.02          # burst-mode ~rate*96 Hz (0.02 -> ~2 Hz)
    prot: str = "GCaMP6f"
    smod: str = "burst"         # decorrelated per-neuron Poisson (good for demix)
    burst_mean: int = 0
    # bg_scale=1.0 = raw NAOMi wash (DENSE/washed mode: bright neuropil fills the
    # field into a smooth cloud like the real samples). 0.1 = de-washed sparse
    # mode where individual cells are visible (demix / sweep use .dewashed()).
    bg_scale: float = 1.0

    # ---- optics / PSF (Phase 2) ----
    obj_na: float = 0.8
    lambda_em_um: float = 0.52
    n_index: float = 1.35
    psf_sz: Tuple[float, float, float] = (12.0, 12.0, 20.0)
    # Focus at ~mid-depth of the cell layer (median soma z ~30 um): focusing at
    # the surface (0) put only top cells in focus while most (0-60 um) defocused.
    focal_depth_um: float = 30.0
    scatter_length_um_wf: Optional[float] = None

    # ---- widefield / camera (Phase 4) ----
    pavg: float = 5.0           # low for DENSE mode: bright neuropil dominates,
    lambda_ex_um: float = 0.488  # keeps mean-image median near the real ~1372 ADU
    qe_det: float = 0.8
    bias: float = 470.0         # real striatum tiffs sit at a ~470 ADU pedestal
    dark_rate: float = 0.3
    read_noise: float = 1.6
    gain_e_per_adu: float = 1.0

    # ---- scan ----
    scan_buff: int = 10
    motion: bool = True
    # Sample-motion model: 'randomwalk' (legacy bounded +/-1 voxel walk) or
    # 'physio' (realistic AR(1) drift+jitter + heavy-tailed jumps + intra-frame
    # blur, fit to real NoRMCorre striatum shifts). physio needs a larger crop
    # margin (real range ~+/-26 um); build_scan bumps scan_buff to >=30 for it.
    motion_model: str = "randomwalk"
    sfrac: int = 2

    # ---- post-processing (demo "make it look real" cosmetics; the raw
    #      physical pipeline leaves these off — demix/sweep never run them) ----
    soma_gain: float = 1.0      # multiply soma-voxel fluorescence (values-only)
    solid_soma: bool = True     # fill nucleus -> solid cells, not rings
    bright_frac: float = 0.2    # fraction of somata boosted (clear cells on wash)
    bright_gain: float = 3.0
    oof_blur_um: float = 5.0    # extra out-of-focus Gaussian (smooth the field)
    illum: IllumConfig = field(default_factory=IllumConfig)

    # ------------------------------------------------------------------
    # Preset constructors
    # ------------------------------------------------------------------
    @classmethod
    def dewashed(cls, **overrides):
        """De-washed sparse mode (individual cells visible; no cosmetics).

        Used by the demix dataset and the parameter sweep: raw physical
        widefield with bg_scale=0.1 and NO illumination gradient / soma
        boosting / blur post-processing.
        """
        base = dict(bg_scale=0.1, soma_gain=1.0, solid_soma=False,
                    bright_frac=0.0, oof_blur_um=0.0,
                    illum=IllumConfig(enable=False))
        base.update(overrides)
        return cls(**base)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def vol_sz(self) -> Tuple[int, int, int]:
        return (self.vol_um, self.vol_um, self.depth_um)

    def as_dict(self) -> dict:
        """Full reproducible record (for metadata.json)."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Builders -> concrete calcia dataclasses
    # ------------------------------------------------------------------
    def build_vol_params(self):
        from calcia.config.params import VolumeParams
        return VolumeParams(vol_sz=self.vol_sz, vres=self.vres,
                            vol_depth=self.vol_depth, region=self.region,
                            N_neur=self.n_neur, neur_density=self.neur_density)

    def build_psf(self):
        from calcia.config.params import PsfParams
        kw = dict(imaging_mode="widefield", psf_type="gaussian_analytical",
                  lambda_em_um=self.lambda_em_um, obj_na=self.obj_na,
                  n=self.n_index, psf_sz=self.psf_sz,
                  wf_focal_depth_um=self.focal_depth_um)
        if self.scatter_length_um_wf is not None:
            kw["scatter_length_um_wf"] = self.scatter_length_um_wf
        return PsfParams(**kw)

    def build_spike(self, K, has_axons, verbose=1):
        from calcia.config.params import SpikeParams
        return SpikeParams(K=K, nt=self.nt, dt=self.dt, N_bg=0,
                           axonflag=has_axons, rate=self.rate, prot=self.prot,
                           smod_flag=self.smod, burst_mean=self.burst_mean,
                           bg_scale=self.bg_scale, verbose=verbose)

    def build_scan(self, verbose=1):
        from calcia.config.params import ScanParams
        buff = self.scan_buff
        # physio motion spans ~+/-26 um; the crop margin must allow it or the
        # trajectory is clipped to the (tiny) legacy bound and the realism is lost.
        if self.motion_model == "physio" and buff < 30:
            buff = 30
        return ScanParams(scan_buff=buff, motion=self.motion,
                          sfrac=self.sfrac, verbose=verbose)

    def build_motion(self):
        """MotionParams for scan_widefield, or None for the legacy walk.

        Only the 'physio' model needs an explicit MotionParams; the legacy
        random walk is the scanner default when motion_params is None.
        """
        if self.motion_model != "physio":
            return None
        from calcia.config.params import MotionParams
        return MotionParams(model="physio", seed=self.seed + 3)

    def build_wf(self):
        from calcia.config.params import WidefieldParams
        return WidefieldParams(pavg=self.pavg, lambda_ex_um=self.lambda_ex_um,
                               qe_det=self.qe_det)

    def build_cam(self):
        from calcia.config.params import CameraNoiseParams
        return CameraNoiseParams(qe=1.0, dark_rate=self.dark_rate,
                                 t_exp=self.dt, read_noise=self.read_noise,
                                 gain_e_per_adu=self.gain_e_per_adu,
                                 bias=self.bias)

    def build_cal(self):
        from calcia.config.params import CalciumParams
        return CalciumParams(prot_type=self.prot.lower())

    def illum_map(self, shape_hw):
        return self.illum.illum_map(shape_hw)


# ======================================================================
# Volume: solid somata (fill the dark nucleus)
# ======================================================================
def fill_nuclei(vol_out, verbose=True):
    """Fill each neuron's nucleus with its soma fluorescence so cells render as
    SOLID bright blobs, not dark-centred rings.

    NAOMi gives the nucleus zero fluorescence (nuc_fluorsc=0), leaving a dark
    centre — the cytoplasmic-GCaMP/tdT "ring". But real washed 1P striatum cells
    are SOLID light blobs (nuclear exclusion is not resolved through scattering
    tissue, and the indicator is not perfectly excluded). This is a PHYSICAL
    correction (match real), NOT a brightness cosmetic. Values-only edit of a
    freshly-loaded volume: merges each nucleus's voxels into its soma footprint
    with the soma's median fluorescence + soma_mask, and zeros gp_nuc.
    """
    import numpy as np
    if not getattr(vol_out, "gp_nuc", None):
        return 0
    n_solid = 0
    for i in range(min(len(vol_out.gp_vals), len(vol_out.gp_nuc))):
        nuc_idx = np.asarray(vol_out.gp_nuc[i][0])
        cfd = vol_out.gp_vals[i]
        sm = np.asarray(cfd.soma_mask)
        if len(nuc_idx) == 0 or not sm.any():
            continue
        fill = float(np.median(cfd.fluorescence[sm]))
        cfd.indices = np.concatenate([cfd.indices, nuc_idx])
        cfd.fluorescence = np.concatenate(
            [cfd.fluorescence,
             np.full(len(nuc_idx), fill, cfd.fluorescence.dtype)])
        cfd.soma_mask = np.concatenate([sm, np.ones(len(nuc_idx), bool)])
        n_solid += 1
    vol_out.gp_nuc = [(np.array([], dtype=np.int64), 0.0)
                      for _ in vol_out.gp_nuc]
    if verbose:
        print(f"  solid somata: filled {n_solid} nuclei (no dark-centre rings)")
    return n_solid


# ======================================================================
# Optics: lateral tissue-scatter PSF broadening
# ======================================================================
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
    import numpy as np
    from scipy.ndimage import gaussian_filter
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
    import numpy as np
    from scipy.ndimage import gaussian_filter
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


# ======================================================================
# IO
# ======================================================================
def load_phase1(cache_path):
    """Load a cached Phase-1 (vol_out, vol_params) tuple."""
    with open(cache_path, "rb") as f:
        vol_out, vol_params = pickle.load(f)
    return vol_out, vol_params


# ======================================================================
# Run printout log — tee stdout+stderr to a file
# ======================================================================
# Saves a run's FULL console output (calcia's [1/7]..[7/7] progress, timing
# prints, library warnings, and the pyinstrument text report) to
# examples/output/logs/<name>_<timestamp>.log, so a long background run can be
# reviewed after the fact even though its live stdout would otherwise be lost.
# Complements pyinstrument (intra-run flamegraph) and per-run metadata.json
# (static config).
class _Tee:
    """Write-through stream: mirrors everything to the real stream AND a file.
    A console-encoding error (Windows cp1252 vs unicode) never blocks the file
    write — the log always gets the full text."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._fh.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno, ... -> real stream
        return getattr(self._stream, name)


def tee_stdout(log_name, output_dir=None):
    """Redirect sys.stdout+sys.stderr through a tee that also writes to
    ``examples/output/logs/<log_name>_<timestamp>.log``. Call ONCE at the very
    top of a script (right after imports). Returns the log path (or None).

    The whole console output of the run is saved there. File is line-buffered so
    a background run flushes live to disk. Never raises."""
    import datetime as _dt
    import os
    import sys
    try:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "output")
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(logs_dir, f"{log_name}_{stamp}.log")
        fh = open(path, "w", encoding="utf-8", buffering=1)  # line-buffered
        fh.write(f"# {log_name}  started "
                 f"{_dt.datetime.now().isoformat(timespec='seconds')}\n")
        fh.flush()
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)
        print(f"[run-log] console output -> {path}")
        return path
    except Exception as e:  # logging must never take down a real run
        print(f"[run-log] WARN could not set up tee log: {e}")
        return None


# ======================================================================
# Full reproducible run bundle (parity with demo_widefield_striatum_v1.py)
# ======================================================================
def save_full_bundle(run_dir, *, noisy, clean, vol_out, vol_params, opt_out,
                     time_out, scan_out, params_dict, metadata, dt,
                     make_gif=True, make_viz=True, verbose=True):
    """Write the same rich artifact set the main striatum demo produces.

    Writes: movies.npz (clean+noisy THW + mot_hist/blur_hist), movie_noisy.tif,
    movie_clean.tif, optics.npz, cell_footprints.pkl, params.pkl, traces.npz
    (soma/dend/bg/spikes + soma_neurons/locs ground truth), metadata.json,
    report.md, movie.gif, and the viz_cache bundle. Channel-specific ground truth
    (e.g. tdtomato_expression.npz) is saved by the caller separately.
    """
    import json
    import os
    import pickle

    import traceback

    os.makedirs(run_dir, exist_ok=True)
    saved, failed = [], []

    def _step(name, fn, critical=False):
        """Run one save step in isolation. A failure is logged LOUDLY (the old
        code hid viz/report errors behind a verbose-only message, so a persistent
        failure went unnoticed) but does NOT abort the remaining steps."""
        try:
            fn()
            saved.append(name)
        except Exception as e:
            failed.append(name)
            print(f"  [save] {'CRITICAL ' if critical else ''}FAILED {name}: {e}")
            traceback.print_exc()

    # Order matters for crash-safety: write the CHEAP + IRREPLACEABLE artifacts
    # first (the scanned movie, metadata, ground-truth traces), then the HEAVY /
    # OPTIONAL tail (multi-GB footprints, viz meshes, gif). If the tail dies or
    # fills the disk, the run dir still holds a usable movie + metadata + traces.

    def _save_movies():  # the irreplaceable scan output -> first
        movies = dict(
            mov_clean=np.transpose(clean, (2, 0, 1)).astype(np.float32),
            mov_noisy=np.transpose(noisy, (2, 0, 1)).astype(np.float32),
            axes=np.array("THW"))
        if getattr(scan_out, "mot_hist", None) is not None:
            movies["mot_hist"] = scan_out.mot_hist
        if getattr(scan_out, "blur_hist", None) is not None:
            movies["blur_hist"] = scan_out.blur_hist
        np.savez_compressed(os.path.join(run_dir, "movies.npz"), **movies)
    _step("movies.npz", _save_movies, critical=True)

    _step("movie_noisy.tif",
          lambda: save_tif(noisy, os.path.join(run_dir, "movie_noisy.tif")))
    _step("movie_clean.tif",
          lambda: save_tif(clean, os.path.join(run_dir, "movie_clean.tif")))

    def _save_metadata():
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    _step("metadata.json", _save_metadata)

    def _save_params():
        with open(os.path.join(run_dir, "params.pkl"), "wb") as f:
            pickle.dump(params_dict, f)
    _step("params.pkl", _save_params)

    def _save_traces():  # ground truth -> before the heavy tail
        gp = vol_out.gp_vals
        n_soma = sum(1 for g in gp if getattr(g, "soma_mask", None) is not None
                     and np.any(np.asarray(g.soma_mask)))
        locs = np.asarray(vol_out.locs)
        soma = np.asarray(time_out.soma, dtype=np.float32)
        traces = dict(n_soma=np.int64(n_soma), trace_axes=np.array("KT"),
                      locs_axes=np.array("Kxyz"), soma=soma, locs=locs,
                      soma_neurons=soma[:n_soma], soma_locs=locs[:n_soma])
        if time_out.spikes is not None:
            traces["spikes"] = time_out.spikes
            if time_out.spikes.shape[0] == soma.shape[0]:
                traces["spikes_neurons"] = time_out.spikes[:n_soma]
        if time_out.dend is not None:
            traces["dend"] = np.asarray(time_out.dend, dtype=np.float32)
            traces["dend_neurons"] = traces["dend"][:n_soma]
        if time_out.bg is not None:
            traces["bg"] = np.asarray(time_out.bg, dtype=np.float32)
        np.savez_compressed(os.path.join(run_dir, "traces.npz"), **traces)
    _step("traces.npz", _save_traces)

    def _save_optics():
        optics = dict(psf=opt_out.psf)
        if getattr(opt_out, "mask", None) is not None:
            optics["mask"] = opt_out.mask
        if getattr(opt_out, "col_mask", None) is not None:
            optics["col_mask"] = opt_out.col_mask
        np.savez_compressed(os.path.join(run_dir, "optics.npz"), **optics)
    _step("optics.npz", _save_optics)

    # --- heavy / optional tail ---
    def _save_footprints():  # multi-GB -> after everything critical is on disk
        with open(os.path.join(run_dir, "cell_footprints.pkl"), "wb") as f:
            pickle.dump(dict(gp_vals=vol_out.gp_vals, bg_proc=vol_out.bg_proc,
                             locs=vol_out.locs,
                             neur_vol_shape=vol_out.neur_vol.shape), f)
    _step("cell_footprints.pkl", _save_footprints)

    _step("report.md", lambda: write_summary_report(
        run_dir, neur_ves=getattr(vol_out, "neur_ves", None), verbose=False))

    if make_viz:
        def _save_viz():
            from calcia.viz.prep import prep_run
            prep_run(run_dir, neur_ves=getattr(vol_out, "neur_ves", None),
                     verbose=verbose)
        _step("viz_cache", _save_viz)

    if make_gif:
        _step("movie.gif", lambda: make_video(
            noisy, clean, os.path.join(run_dir, "movie.gif"), dt=dt, fps=30))

    if verbose:
        msg = f"  saved full bundle -> {os.path.basename(os.path.normpath(run_dir))}"
        if failed:
            msg += f"   [FAILED: {', '.join(failed)}]"
        print(msg)


# ======================================================================
# Summary report
# ======================================================================
def write_summary_report(run_dir, *, neur_ves=None, write=True, verbose=True):
    """Human-readable one-page summary of a finished striatum run.

    Reads the run's own saved artifacts (``metadata.json``, ``traces.npz``,
    ``cell_footprints.pkl``) so it works both inline (called by the demo after
    the run is saved) and standalone on any existing run dir. Reports the FOV /
    pixel geometry and the component inventory (neurons, background processes,
    spikes, and — when ``neur_ves`` is supplied — blood-vessel voxels).

    Parameters
    ----------
    run_dir : str
        A finished run directory.
    neur_ves : np.ndarray, optional
        In-memory vessel voxel volume (``vol_out.neur_ves``) so the report can
        state vessel voxel count + volume fraction. Omit (standalone use) and
        the vessel line falls back to noting the viz mesh, since counting
        voxels would otherwise require reloading the multi-GB Phase-1 volume.

    Returns
    -------
    str : the report text (also written to ``<run_dir>/report.md`` if ``write``).
    """
    import json
    import os

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    cfg = meta.get("config", {})

    # --- geometry ---
    vol_sz = meta["vol_sz"]                       # [x, y, z] um
    vres = meta["vres"]                           # vox / um
    sfrac = cfg.get("sfrac", 2)
    nt = meta["nt"]
    dt = meta["dt"]
    fov_x, fov_y, depth = vol_sz
    grid = (fov_x * vres, fov_y * vres, depth * vres)
    n_grid = grid[0] * grid[1] * grid[2]
    mov_shape = meta["movie_shape"]               # [H, W, T]
    H, W = mov_shape[0], mov_shape[1]
    um_per_px = sfrac / vres

    # --- components (prefer traces.npz for exact per-row counts) ---
    n_soma = int(meta.get("n_soma", meta.get("N_neur", 0)))
    n_comp = int(meta.get("N_soma_traces", n_soma))
    n_bg = n_comp - n_soma
    neuron_spikes = None
    tp = os.path.join(run_dir, "traces.npz")
    if os.path.exists(tp):
        z = np.load(tp)
        if "spikes_neurons" in z:
            neuron_spikes = int(np.asarray(z["spikes_neurons"]).sum())
    total_spikes = int(meta.get("total_spikes", 0))

    # --- vessels ---
    if neur_ves is not None:
        ves = np.asarray(neur_ves)
        n_ves = int((ves > 0).sum())
        ves_line = (f"  Blood vessels:         {n_ves:>12,} voxels "
                    f"({100.0 * n_ves / n_grid:.1f}% of volume)")
    elif os.path.exists(os.path.join(run_dir, "viz_cache")):
        ves_line = ("  Blood vessels:         vascular network present "
                    "(see viz_cache/vessels_*.vtp; voxel count needs the "
                    "Phase-1 volume)")
    else:
        ves_line = "  Blood vessels:         (not recorded)"

    dur = nt * dt
    spike_hz = (neuron_spikes / n_soma / dur) if (neuron_spikes and n_soma) else None

    L = []
    L.append("STRIATUM WIDEFIELD SIMULATION — SUMMARY REPORT")
    L.append(f"run:       {os.path.basename(os.path.normpath(run_dir))}")
    L.append(f"region:    {meta.get('region','?')}   indicator: {meta.get('prot','?')}"
             f"   motion: {cfg.get('motion_model','randomwalk')}")
    L.append("")
    L.append("FIELD OF VIEW")
    L.append(f"  Lateral FOV:           {fov_x} x {fov_y} um  "
             f"({fov_x/1000:.2f} x {fov_y/1000:.2f} mm)")
    L.append(f"  Imaged depth:          {depth} um")
    L.append(f"  Voxel resolution:      {vres} vox/um  ->  grid "
             f"{grid[0]} x {grid[1]} x {grid[2]} = {n_grid/1e6:.1f} M voxels")
    L.append("")
    L.append("OUTPUT MOVIE")
    L.append(f"  Frame size:            {H} x {W} px  ({H*W:,} px/frame)")
    L.append(f"  Pixel size:            {um_per_px:g} um/px  "
             f"(sfrac={sfrac} downsample / vres={vres})")
    L.append(f"  Frames:                {nt}  @ {1/dt:.0f} Hz  ->  {dur:.1f} s")
    L.append(f"  Total pixels:          {H*W*nt/1e6:.1f} M  ({H} x {W} x {nt})")
    L.append("")
    L.append("COMPONENTS")
    L.append(f"  Fluorescing neurons:   {n_soma:>12,}  (labelled somata)")
    L.append(f"  Background processes:  {n_bg:>12,}  (bg dendrites + axons)")
    L.append(f"  Total trace components:{n_comp:>12,}")
    L.append(ves_line)
    if neuron_spikes is not None:
        L.append(f"  Neuron spikes ({dur:.0f}s):    {neuron_spikes:>12,}"
                 + (f"  (~{spike_hz:.2f} Hz/neuron mean)" if spike_hz else ""))
    L.append(f"  Total spikes (all rows):{total_spikes:>11,}")
    text = "\n".join(L)

    if write:
        import os as _os
        with open(_os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")
    if verbose:
        print("\n" + text + "\n")
    return text


# ======================================================================
# Metrics
# ======================================================================
def brightest_frame(mov):
    """Index of the frame with the highest mean intensity. mov: (H,W,T)."""
    return int(mov.reshape(-1, mov.shape[2]).mean(0).argmax())


def cv_bright(mov):
    """Spatial coefficient of variation of the brightest frame."""
    fr = mov[:, :, brightest_frame(mov)]
    return float(fr.std() / (fr.mean() + 1e-12))


def dF(mov, pct=10):
    """dF over a static per-pixel baseline (pct-th temporal percentile)."""
    f0 = np.percentile(mov, pct, axis=2, keepdims=True)
    return mov - f0


# ======================================================================
# Rendering
# ======================================================================
def save_tif(arr, path, clip0=False):
    """Save (H,W,T) as a self-normalized 16-bit TIFF stack (0.5/99.5 pct)."""
    import tifffile
    a = arr.astype(np.float64)
    if clip0:
        a = np.clip(a, 0, None)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = (np.clip(a, 0, 1) * 65535).astype(np.uint16)
    tifffile.imwrite(str(path), np.transpose(a, (2, 0, 1)), imagej=True)


# Backwards-compatible alias (demo used this name).
save_tiff_normalized = save_tif


def make_video(mov_noisy, mov_clean, path, dt, contrast=0.995, fps=30):
    """Side-by-side noisy|clean GIF (each self-normalized)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    nt = mov_noisy.shape[2]
    vmin_n = np.percentile(mov_noisy, (1 - contrast) * 100)
    vmax_n = np.percentile(mov_noisy, contrast * 100)
    vmin_c = np.percentile(mov_clean, (1 - contrast) * 100)
    vmax_c = np.percentile(mov_clean, contrast * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.90, bottom=0.02)
    im1 = ax1.imshow(mov_noisy[:, :, 0], cmap="gray",
                     vmin=vmin_n, vmax=vmax_n, aspect="equal")
    im2 = ax2.imshow(mov_clean[:, :, 0], cmap="gray",
                     vmin=vmin_c, vmax=vmax_c, aspect="equal")
    ax1.set_title("Noisy (widefield)", fontsize=10)
    ax2.set_title("Clean (widefield)", fontsize=10)
    ax1.axis("off"); ax2.axis("off")
    time_text = fig.suptitle("t = 0.000 s", fontsize=11)

    def update(frame):
        im1.set_data(mov_noisy[:, :, frame])
        im2.set_data(mov_clean[:, :, frame])
        time_text.set_text(f"t = {frame * dt:.3f} s")
        return im1, im2, time_text

    anim = FuncAnimation(fig, update, frames=nt, blit=True, interval=1)
    anim.save(str(path), writer="pillow", fps=fps)
    plt.close(fig)
