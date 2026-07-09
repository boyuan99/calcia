"""
Conceptual / teaching visualization of widefield PSF strategies.

This script is PURELY CONCEPTUAL. It does NOT scan any real volume, does NOT
touch the striatum pipeline, and does NOT compare against real sim frames.
Everything below is a synthetic, idealized toy problem whose only job is to
build intuition about *how* each PSF works and *why* it succeeds or fails.

Four PSF strategies are compared:
  1. Narrow diffraction-limited Gaussian (~1 um FWHM)  -- what calcia uses now.
  2. Single wide Gaussian (~18 um FWHM, "scatter").
  3. Dual-scale = narrow core + wide scatter halo (weighted sum, w_core=0.6).
  4. Fresnel-like wave-optics PSF, approximated by its *shape*: a sharp
     diffraction core sitting inside a broad, heavy-tailed (exponential)
     scattering halo. (The real thing is a heavy per-z Fresnel propagation;
     here we only mimic the radial shape so the physics is legible.)

Output: one annotated multi-panel figure, examples/psf_strategy_concept.png
"""

import numpy as np
from scipy.signal import fftconvolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm, PowerNorm

# --------------------------------------------------------------------------- #
#  Consistent identity: one color per strategy, reused across every panel.
#  (dataviz categorical slots; blue / orange / green / violet -- validated,
#   every panel is also directly labeled so identity is never color-alone.)
# --------------------------------------------------------------------------- #
C_NARROW = "#2a78d6"   # blue    -- sharp but holey (calcia today)
C_WIDE = "#eb6834"     # orange  -- the wash-out failure
C_DUAL = "#008300"     # green   -- best of both (engineered)
C_FRESNEL = "#4a3aa7"  # violet  -- best of both (from physics)
INK = "#0b0b0b"
INK2 = "#52514e"

PX_UM = 0.5            # micrometers per pixel

# --------------------------------------------------------------------------- #
#  1. Build the four PSFs
#     Two physical scales throughout:
#       core  <- diffraction / numerical aperture   (~1 um)
#       halo  <- tissue scattering                   (~10-20 um, heavy tail)
# --------------------------------------------------------------------------- #
FWHM_NARROW_UM = 1.0                       # diffraction-limited core
FWHM_WIDE_UM = 18.0                        # scatter blur
SIG_NARROW = FWHM_NARROW_UM / 2.3548 / PX_UM   # sigma, in pixels
SIG_WIDE = FWHM_WIDE_UM / 2.3548 / PX_UM
HALO_L_PX = 6.0 / PX_UM                    # exponential halo decay length (~6 um)
W_CORE = 0.6                               # weight on the sharp core
W_HALO = 0.4                               # weight on the scatter halo

KHALF = int(round(60.0 / PX_UM))           # kernel half-width (60 um) -> capture tails
ky, kx = np.mgrid[-KHALF:KHALF + 1, -KHALF:KHALF + 1]
kr = np.hypot(kx, ky)                       # radius in pixels


def _unit_sum(a):
    return a / a.sum()


g_narrow = _unit_sum(np.exp(-kr**2 / (2 * SIG_NARROW**2)))
g_wide = _unit_sum(np.exp(-kr**2 / (2 * SIG_WIDE**2)))
h_exp = _unit_sum(np.exp(-kr / HALO_L_PX))          # heavy-tailed scatter halo

# Unit-SUM kernels (energy conserving) -- used for convolution.
psf_conv = {
    "narrow": g_narrow,
    "wide": g_wide,
    "dual": W_CORE * g_narrow + W_HALO * g_wide,        # core + Gaussian halo
    "fresnel": W_CORE * g_narrow + W_HALO * h_exp,      # core + heavy-tail halo
}
# Unit-PEAK kernels -- used for display and radial profiles (cores align at 1).
psf_disp = {k: v / v.max() for k, v in psf_conv.items()}

STRATS = ["narrow", "wide", "dual", "fresnel"]
TITLES = {
    "narrow": "1. Narrow Gaussian",
    "wide": "2. Wide Gaussian",
    "dual": "3. Dual-scale (core+halo)",
    "fresnel": "4. Fresnel-like (physics)",
}
SUBTITLE = {
    "narrow": "~1 um  ·  diffraction only",
    "wide": "~18 um  ·  scatter only",
    "dual": "0.6·core + 0.4·wide halo",
    "fresnel": "core + heavy exp. tail",
}
COLOR = {"narrow": C_NARROW, "wide": C_WIDE, "dual": C_DUAL, "fresnel": C_FRESNEL}

# 1-D radial profiles (central row, r >= 0) of the unit-peak kernels.
cslice = KHALF
r_um = np.arange(0, KHALF + 1) * PX_UM
radial = {k: psf_disp[k][cslice, cslice:] for k in STRATS}

# --------------------------------------------------------------------------- #
#  2. Build the synthetic toy scene
#     (i)   a few bright "soma" disks
#     (ii)  a thin dark "vessel" line
#     (iii) a small "neuropil" dot lattice with a few-um gaps
# --------------------------------------------------------------------------- #
FIELD_UM = 160.0
N = int(round(FIELD_UM / PX_UM))            # 320 px
yy, xx = np.mgrid[0:N, 0:N] * PX_UM          # coords in um

scene = np.zeros((N, N), dtype=float)

# (iii) neuropil lattice: fine dots on a 6 um grid, radius 1.5 um, dark gaps.
DOT_SPACING = 6.0
DOT_R = 1.5
DOT_LVL = 0.55
grid = np.arange(4.0, FIELD_UM, DOT_SPACING)
for cy in grid:
    for cx in grid:
        scene[(xx - cx) ** 2 + (yy - cy) ** 2 <= DOT_R**2] = DOT_LVL

# (i) somata: bright disks laid on top of the neuropil (avoid the metric ROI).
SOMA_R = 6.0
SOMA_LVL = 1.0
soma_centers = [(95, 35), (120, 95), (55, 120), (30, 100), (118, 142)]
for cx, cy in soma_centers:
    scene[(xx - cx) ** 2 + (yy - cy) ** 2 <= SOMA_R**2] = SOMA_LVL

# (ii) vessel: a thin, slightly slanted dark line carved through the tissue.
xc_of_y = 72.0 + 0.06 * (yy - 80.0)
scene[np.abs(xx - xc_of_y) <= 1.5] = 0.0

# Convolve the toy scene with each energy-conserving PSF.
conv = {k: fftconvolve(scene, psf_conv[k], mode="same") for k in STRATS}

# --------------------------------------------------------------------------- #
#  3. Two scalar metrics that separate the four strategies
#       gap_fill  : how bright the gaps are vs typical neuropil.
#                   0 = gaps are black holes  ->  1 = gaps fully filled/smooth
#       soma_C    : does a soma still stand out from neuropil? (Michelson)
#                   high = sharp cell  ->  low = washed out
# --------------------------------------------------------------------------- #
def _michelson(hi, lo):
    return (hi - lo) / max(hi + lo, 1e-9)


# Clean lattice ROI (no soma, no vessel): x,y in [12, 48] um.
roi = (xx >= 12) & (xx <= 48) & (yy >= 12) & (yy <= 48)
# Soma ROI: the (95, 35) soma; center disk vs a neuropil annulus around it.
sc_x, sc_y = 95.0, 35.0
rr = np.hypot(xx - sc_x, yy - sc_y)
soma_core = rr <= 1.5
soma_ann = (rr >= 9.0) & (rr <= 13.0)


def metrics(img):
    patch = img[roi]
    gap_fill = float(np.clip(np.percentile(patch, 2) / max(patch.mean(), 1e-9), 0, 1))
    soma = _michelson(img[soma_core].mean(), img[soma_ann].mean())
    return gap_fill, soma


met = {k: metrics(conv[k]) for k in STRATS}
met_orig = metrics(scene)

# --------------------------------------------------------------------------- #
#  4. Figure
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": "#c3c2b7",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
})

# magma with an explicit black floor so PSFs that underflow to 0 stay black
KMAP = cm.magma.copy()
KMAP.set_under("#000004")
KMAP.set_bad("#000004")


def scalebar(ax, length_um=20.0, y_frac=0.06, x_frac=0.06, color="white"):
    x0 = x_frac * N
    y0 = y_frac * N
    ax.plot([x0, x0 + length_um / PX_UM], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0, y0 + 0.018 * N, f"{int(length_um)} um", color=color, fontsize=8, va="bottom")


fig = plt.figure(figsize=(17.5, 21.0))
subfigs = fig.subfigures(5, 1, height_ratios=[1.15, 0.92, 1.08, 0.60, 0.32], hspace=0.02)
fig.suptitle(
    "Widefield PSF strategies -- how each one works, and why it wins or fails",
    fontsize=18, fontweight="bold", y=0.997,
)
fig.text(0.5, 0.981,
         "Purely conceptual: synthetic toy inputs only. No real volume is scanned.",
         ha="center", fontsize=11, color=INK2, style="italic")

# ---- Section A1: the four PSF kernels (log-stretched to reveal the halo) ----
sfa1 = subfigs[0]
sfa1.suptitle("A1  |  The PSF itself  --  each kernel, log-stretched so the faint scatter halo is visible",
              fontsize=13.5, fontweight="bold", y=0.90)
gsa1 = sfa1.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.08,
                         left=0.035, right=0.965, top=0.72, bottom=0.06)
win = int(round(28.0 / PX_UM))             # display window: +/- 28 um
sl = slice(KHALF - win, KHALF + win + 1)
ext = [-win * PX_UM, win * PX_UM, -win * PX_UM, win * PX_UM]
im = None
for j, k in enumerate(STRATS):
    ax = sfa1.add_subplot(gsa1[0, j])
    im = ax.imshow(np.clip(psf_disp[k][sl, sl], 1e-7, None), origin="lower", extent=ext,
                   cmap=KMAP, norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_title(TITLES[k], color=COLOR[k], fontsize=12, fontweight="bold", pad=6)
    ax.text(0.5, 0.045, SUBTITLE[k], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.45, ec="none"))
    ax.set_xticks([-20, 0, 20]); ax.set_yticks([-20, 0, 20])
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.set_ylabel("um", fontsize=8)
    # annotate the two scales on the composite kernels
    if k in ("dual", "fresnel"):
        ax.add_patch(plt.Circle((0, 0), 1.2, fill=False, ec="#7cf7d0", lw=1.4))
        ax.add_patch(plt.Circle((0, 0), 16.0, fill=False, ec="#f79cff", lw=1.4, ls=(0, (4, 3))))
        ax.annotate("core (NA)", xy=(0.8, 0.8), xytext=(6, 6), fontsize=7.5,
                    color="#7cf7d0", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#7cf7d0", lw=1))
        ax.text(11.3, 12.0, "halo\n(scatter)", color="#f79cff", fontsize=7.5, fontweight="bold")
cax = sfa1.add_subplot(gsa1[0, 4])
cb = sfa1.colorbar(im, cax=cax)
cb.set_label("normalized intensity (log)", fontsize=8)
cb.ax.tick_params(labelsize=7)

# ---- Section A2: overlaid radial profiles, linear + log ----
sfa2 = subfigs[1]
sfa2.suptitle("A2  |  Radial profiles overlaid  --  LINEAR shows the core width; LOG reveals the tail",
              fontsize=13.5, fontweight="bold", y=0.97)
gsa2 = sfa2.add_gridspec(1, 2, wspace=0.15, left=0.06, right=0.97, top=0.80, bottom=0.17)

ax_lin = sfa2.add_subplot(gsa2[0, 0])
for k in STRATS:
    ax_lin.plot(r_um, radial[k], color=COLOR[k], lw=2.2, label=TITLES[k])
ax_lin.set_xlim(0, 28); ax_lin.set_ylim(-0.02, 1.05)
ax_lin.set_xlabel("radius (um)", fontsize=9)
ax_lin.set_ylabel("intensity (peak = 1)", fontsize=9)
ax_lin.set_title("linear y-axis", fontsize=11)
ax_lin.axvspan(0, FWHM_NARROW_UM, color=C_NARROW, alpha=0.12)
ax_lin.text(1.4, 0.88, "diffraction\ncore (NA)", fontsize=8, color=C_NARROW)
ax_lin.annotate("wide Gaussian = one broad scale\n(no sharp core)",
                xy=(9, 0.55), xytext=(12.5, 0.72), fontsize=8, color=C_WIDE,
                arrowprops=dict(arrowstyle="->", color=C_WIDE, lw=1.2))
ax_lin.text(3.0, 0.30, "narrow / dual / fresnel\nshare the same sharp core",
            fontsize=8, color=INK2)
ax_lin.grid(True, color="#e1e0d9", lw=0.7)
ax_lin.legend(fontsize=8, frameon=False, loc="upper right")

ax_log = sfa2.add_subplot(gsa2[0, 1])
for k in STRATS:
    ax_log.semilogy(r_um, np.clip(radial[k], 1e-6, None), color=COLOR[k], lw=2.2,
                    label=TITLES[k])
ax_log.set_xlim(0, 48); ax_log.set_ylim(1e-4, 1.4)
ax_log.set_xlabel("radius (um)", fontsize=9)
ax_log.set_ylabel("intensity (log)", fontsize=9)
ax_log.set_title("log y-axis  --  the tail is the whole story", fontsize=11)
ax_log.axvspan(5, 48, color="#f79cff", alpha=0.10)
ax_log.text(26, 4e-1, "scatter halo / tail", fontsize=8.5, color="#a23bb0", ha="center")
ax_log.annotate("Gaussian halos curve\ndown and die",
                xy=(20, 6e-4), xytext=(7.5, 1.6e-4), fontsize=8, color=C_DUAL,
                arrowprops=dict(arrowstyle="->", color=C_DUAL, lw=1.2))
ax_log.annotate("Fresnel: heavy tail = near-straight\nline, reaches far into the gaps",
                xy=(41, 3.5e-4), xytext=(19, 4e-3), fontsize=8, color=C_FRESNEL,
                arrowprops=dict(arrowstyle="->", color=C_FRESNEL, lw=1.2))
ax_log.grid(True, which="both", color="#e1e0d9", lw=0.6)
ax_log.legend(fontsize=8, frameon=False, loc="upper right")

# ---- Section B: the toy scene, and each PSF acting on it ----
sfb = subfigs[2]
sfb.suptitle("B  |  Effect on a synthetic toy scene  --  bright somata, a thin dark vessel, "
             "a neuropil dot-lattice with few-um gaps",
             fontsize=13.5, fontweight="bold", y=0.96)
gsb = sfb.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.06,
                       left=0.02, right=0.975, top=0.83, bottom=0.05)

# shared display norm across ground-truth + convolved (so wash-out reads as it is)
vmax = max(conv[k].max() for k in STRATS)
pnorm = PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)

# ground truth
ax0 = sfb.add_subplot(gsb[0, 0])
ax0.imshow(scene, origin="lower", cmap="gray", norm=pnorm)
ax0.set_title("ground truth (toy scene)", fontsize=11.5, fontweight="bold", pad=4, color=INK)
ax0.annotate("soma", xy=(sc_x / PX_UM, (sc_y + SOMA_R) / PX_UM), xytext=(150, 40),
             color="#ffd27f", fontsize=8, ha="center",
             arrowprops=dict(arrowstyle="->", color="#ffd27f"))
ax0.annotate("dark vessel", xy=(144, 210), xytext=(210, 250), color="#8fd0ff", fontsize=8,
             arrowprops=dict(arrowstyle="->", color="#8fd0ff"))
ax0.annotate("neuropil lattice\n(few-um gaps)", xy=(60, 100), xytext=(20, 300),
             color="#a0ffb0", fontsize=8, arrowprops=dict(arrowstyle="->", color="#a0ffb0"))
scalebar(ax0)
ax0.set_xticks([]); ax0.set_yticks([])

VERD = {
    "narrow": "black holes remain,\nsomata stay sharp",
    "wide": "gaps filled BUT\nsomata washed out",
    "dual": "gaps lifted from black\nAND somata kept",
    "fresnel": "same as dual --\nbut from physics",
}
imb = None
for j, k in enumerate(STRATS):
    ax = sfb.add_subplot(gsb[0, j + 1])
    imb = ax.imshow(conv[k], origin="lower", cmap="gray", norm=pnorm)
    ax.set_title(TITLES[k], color=COLOR[k], fontsize=11.5, fontweight="bold", pad=4)
    g, s = met[k]
    ax.text(0.035, 0.035,
            f"{VERD[k]}\n\ngap fill: {g:.2f}   soma C: {s:.2f}",
            transform=ax.transAxes, fontsize=8.3, color="white", va="bottom",
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.35", fc="black", alpha=0.55, ec="none"))
    ax.set_xticks([]); ax.set_yticks([])
caxb = sfb.add_subplot(gsb[0, 5])
cbb = sfb.colorbar(imb, cax=caxb)
cbb.set_label("intensity (gamma-stretched)", fontsize=8)
cbb.ax.tick_params(labelsize=7)

# ---- Section C: one-line verdict table ----
sfc = subfigs[3]
sfc.suptitle("C  |  One-line summary per strategy", fontsize=13.5, fontweight="bold", y=0.95)
axc = sfc.add_subplot(111)
axc.axis("off")
col_labels = ["Strategy", "Two scales?", "Black holes\nin gaps?",
              "Cells still\nvisible?", "gap fill /\nsoma C", "Relative\ncompute cost"]
rows = [
    ("1. Narrow Gaussian", "no (one narrow)", "YES -- gaps go black",
     "yes, sharp", f"{met['narrow'][0]:.2f} / {met['narrow'][1]:.2f}", "~1x  (cheap)"),
    ("2. Wide Gaussian", "no (one broad)", "no -- filled",
     "NO -- washed out", f"{met['wide'][0]:.2f} / {met['wide'][1]:.2f}", "~1x  (cheap)"),
    ("3. Dual-scale", "YES (core+halo)", "no -- lifted",
     "yes, sharp", f"{met['dual'][0]:.2f} / {met['dual'][1]:.2f}", "~2x  (cheap)"),
    ("4. Fresnel-like", "YES (physics)", "no -- lifted",
     "yes, sharp", f"{met['fresnel'][0]:.2f} / {met['fresnel'][1]:.2f}",
     "~100-1000x  (heavy)"),
]
tbl = axc.table(cellText=rows, colLabels=col_labels, loc="center",
                cellLoc="center", colLoc="center",
                colWidths=[0.20, 0.15, 0.18, 0.14, 0.13, 0.20])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e1e0d9")
    if r == 0:
        cell.set_facecolor("#f2f1ec")
        cell.set_text_props(fontweight="bold", color=INK)
    if c == 0 and r > 0:
        cell.set_text_props(fontweight="bold", color=list(COLOR.values())[r - 1])

# ---- Insight paragraph ----
sfd = subfigs[4]
axd = sfd.add_subplot(111)
axd.axis("off")
insight = (
    "The core+halo insight.   A single Gaussian has exactly ONE width, so it is "
    "forced to choose: make it narrow and the neuropil gaps survive as cell-sized "
    "black holes (strategy 1); make it broad and the gaps fill in but every soma is "
    "smeared into the neuropil (strategy 2).  You cannot get both from one scale.  "
    "A real 1-photon PSF is not one Gaussian -- it is a sharp diffraction core "
    "(set by the NA) sitting inside a broad, heavy-tailed scattering halo.  The "
    "narrow core keeps somata and the vessel crisp; the wide halo simultaneously "
    "pours diffuse light into the gaps so they are no longer black.  Strategy 3 "
    "engineers this as a weighted sum of two Gaussians; strategy 4 (Fresnel wave "
    "optics) produces the same behavior from first principles -- with an even "
    "heavier tail (visible only on the log axis) and a far larger compute cost.  "
    "Two scales in one PSF: that is why it lifts the gaps out of black AND keeps the cells."
)
axd.text(0.5, 0.92, insight, ha="center", va="top", fontsize=11.5, color=INK,
         wrap=True, linespacing=1.5,
         bbox=dict(boxstyle="round,pad=0.8", fc="#f7f6f2", ec="#d8d7d0"))

out = "examples/psf_strategy_concept.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved:", out)
print("metrics (gap_fill, soma_contrast):")
print("  ground truth:", tuple(round(v, 3) for v in met_orig))
for k in STRATS:
    print(f"  {k:8s}:", tuple(round(v, 3) for v in met[k]))
