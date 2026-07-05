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
    n_neur: Optional[int] = None  # None -> anatomical density (~1e5/mm^3)
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
                            N_neur=self.n_neur)

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
        return ScanParams(scan_buff=self.scan_buff, motion=self.motion,
                          sfrac=self.sfrac, verbose=verbose)

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
# IO
# ======================================================================
def load_phase1(cache_path):
    """Load a cached Phase-1 (vol_out, vol_params) tuple."""
    with open(cache_path, "rb") as f:
        vol_out, vol_params = pickle.load(f)
    return vol_out, vol_params


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
