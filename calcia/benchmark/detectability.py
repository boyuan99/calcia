"""Per-neuron detectability characterisation of a simulation ground truth.

Answers: *which* neurons in the data can realistically be segmented, which are
hard, and why — from the physics that governs how many photons each cell
delivers to the detector:

    optical_brightness = molecular_brightness      (expression x fluorophore)
                       x depth_attenuation         (exp(-2 z / L), 1p scatter)
                       x illumination_weight        (excitation gradient)
                       x collection_weight          (detection gradient)

Cells that were never infected by the AAV (expression modulation == 0) are
flagged separately — they carry no signal by construction and must be excluded
from any recall denominator.

Every neuron gets a continuous ``score`` in ``[0, 1]`` (its optical-brightness
rank within the infected, in-FOV population) and a discrete ``category``:

    ``uninfected`` < ``invisible`` < ``hard`` < ``detectable`` < ``easy``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .gt import GroundTruth

CATEGORIES = ["uninfected", "out_of_fov", "invisible", "hard", "detectable", "easy"]


@dataclass
class DetectabilityConfig:
    infected_eps: float = 1e-3        # trace max below this => not infected
    # optical-brightness percentile band edges within the infected, in-FOV pool.
    # < p_invisible -> invisible ; [p_invisible, p_hard) -> hard ;
    # [p_hard, p_easy) -> detectable ; >= p_easy -> easy.
    p_invisible: float = 25.0
    p_hard: float = 50.0
    p_easy: float = 90.0
    # "detectable" denominator = infected & in-FOV & optical >= this percentile.
    detectable_percentile: float = 50.0


@dataclass
class Detectability:
    """Per-neuron detectability arrays + category labels (all length N)."""

    infected: np.ndarray          # bool, AAV expressed
    in_fov: np.ndarray            # bool, soma centre inside the movie
    depth_um: np.ndarray          # z
    depth_from_focus_um: np.ndarray
    mol_brightness: np.ndarray    # expression x fluorophore (mean trace)
    depth_atten: np.ndarray       # exp(-2z/L)
    illum_weight: np.ndarray
    collection_weight: np.ndarray
    optical_brightness: np.ndarray  # photons-to-detector proxy
    score: np.ndarray             # [0,1] rank among infected in-FOV
    category: np.ndarray          # str per neuron (CATEGORIES)
    detectable: np.ndarray        # bool, the fair recall denominator
    cfg: DetectabilityConfig
    counts: Dict[str, int] = field(default_factory=dict)

    def mask(self, *cats: str) -> np.ndarray:
        """Boolean mask selecting neurons in any of the given categories."""
        m = np.zeros(len(self.category), bool)
        for c in cats:
            m |= self.category == c
        return m

    def summary(self) -> str:
        lines = [f"Detectability of {len(self.category)} neurons "
                 f"({int(self.in_fov.sum())} in-FOV, {int(self.infected.sum())} infected):"]
        for c in CATEGORIES:
            n = int((self.category == c).sum())
            lines.append(f"  {c:12s} {n:6d}  ({100*n/len(self.category):5.1f}%)")
        lines.append(f"  -> detectable pool (fair recall denom): {int(self.detectable.sum())}")
        return "\n".join(lines)


def characterize(gt: GroundTruth, cfg: DetectabilityConfig | None = None) -> Detectability:
    cfg = cfg or DetectabilityConfig()
    tr = gt.traces
    amp = tr.max(1)
    infected = amp > cfg.infected_eps
    mol = tr.mean(1).astype(np.float64)

    z = gt.z
    atten = np.exp(-2.0 * z / gt.scatter_length_um)
    illum = gt.sample_mask(gt.illum_mask) if gt.illum_mask is not None else np.ones(gt.n)
    colw = gt.sample_mask(gt.col_mask) if gt.col_mask is not None else np.ones(gt.n)
    optical = mol * atten * illum * colw

    base = gt.base_px()
    H, W = gt.movie_shape
    # base px are (a, b); either assignment lands in [0, H/W); FOV test is symmetric.
    in_fov = ((base[:, 0] >= 0) & (base[:, 0] < W) & (base[:, 1] >= 0) & (base[:, 1] < H)) | \
             ((base[:, 0] >= 0) & (base[:, 0] < H) & (base[:, 1] >= 0) & (base[:, 1] < W))

    pool = infected & in_fov
    score = np.zeros(gt.n)
    category = np.array(["uninfected"] * gt.n, dtype=object)
    category[infected & ~in_fov] = "out_of_fov"
    if pool.sum() > 0:
        ref = optical[pool]
        order = np.argsort(np.argsort(ref))
        score[pool] = order / max(len(ref) - 1, 1)
        e_inv = np.percentile(ref, cfg.p_invisible)
        e_hard = np.percentile(ref, cfg.p_hard)
        e_easy = np.percentile(ref, cfg.p_easy)
        ov = optical
        category[pool & (ov < e_inv)] = "invisible"
        category[pool & (ov >= e_inv) & (ov < e_hard)] = "hard"
        category[pool & (ov >= e_hard) & (ov < e_easy)] = "detectable"
        category[pool & (ov >= e_easy)] = "easy"
    category = category.astype("<U12")

    thr_det = np.percentile(optical[pool], cfg.detectable_percentile) if pool.sum() else np.inf
    detectable = pool & (optical >= thr_det)

    counts = {c: int((category == c).sum()) for c in CATEGORIES}
    return Detectability(
        infected=infected, in_fov=in_fov, depth_um=z,
        depth_from_focus_um=np.abs(z - gt.focal_depth_um),
        mol_brightness=mol, depth_atten=atten, illum_weight=illum,
        collection_weight=colw, optical_brightness=optical, score=score,
        category=category, detectable=detectable, cfg=cfg, counts=counts,
    )
