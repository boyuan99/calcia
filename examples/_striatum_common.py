"""Shared config + IO/rendering/metrics for the striatum widefield scripts.

Reused across:
  - demo_widefield_striatum_v1.py
  - striatum_demix_dataset.py
  - striatum_param_sweep.py

What actually lives here now is only the DEMO CONFIG layer:

  `StriatumConfig` / `IllumConfig` — the SINGLE SOURCE OF TRUTH for every tunable
  knob of the striatum widefield demo (including "make it look real" cosmetics
  like `bright_frac` / `solid_soma` / `oof_blur_um`). A script overrides only the
  fields it needs and the `build_*` methods turn it into the concrete calcia
  dataclasses. `asdict(cfg)` gives a reproducible record for metadata.json. This
  stays out of core on purpose: it carries demo defaults + cosmetics that would
  pollute `calcia.config`.

Everything else that used to live here was promoted to calcia core and is
re-exported below so existing call sites keep working unchanged:
  - PSF scatter broadening      -> calcia.optics
  - nucleus fill                -> calcia.volume
  - image metrics               -> calcia.diagnostics
  - TIFF/GIF render, cache load -> calcia.io
  - full run bundle + report    -> calcia.io (save_full_bundle/write_summary_report)
  - run console log (Tee)       -> calcia.utils.logging  (tee_stdout wraps it)
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np

# Primitives promoted to calcia core — re-exported for backwards compatibility.
from calcia.optics import broaden_psf_scatter, broaden_psf_two_scale  # noqa: F401
from calcia.volume import fill_nuclei  # noqa: F401
from calcia.diagnostics import brightest_frame, cv_bright, dF  # noqa: F401
from calcia.io import (  # noqa: F401
    load_phase1, make_video, save_full_bundle, save_tif, save_tiff_normalized,
    write_summary_report)
from calcia.utils.logging import Tee, run_log_stem, tee_stdio  # noqa: F401


def tee_stdout(log_name, output_dir=None):
    """Tee stdout+stderr to ``examples/output/logs/<log_name>_<ts>.log``.

    Thin wrapper over :func:`calcia.utils.logging.tee_stdio` that applies the
    examples' ``output/`` directory + timestamped-stem convention. Call ONCE at
    the very top of a script. Returns the log path (or None). Never raises.
    """
    import datetime as _dt
    import os
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
    base = run_log_stem(output_dir, log_name)
    fh = tee_stdio(base + ".log",
                   header=f"# {log_name}  started "
                          f"{_dt.datetime.now().isoformat(timespec='seconds')}")
    if fh is None:
        return None
    print(f"[run-log] console output -> {base}.log")
    return base + ".log"


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
