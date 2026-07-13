"""Tuned presets for STATIC (non-calcium) structural indicators — tdTomato / BFP.

Opt-in named parameter bundles (not defaults — nothing here changes any core
dataclass default). Each ``STATIC_PRESETS`` entry carries the optics (emission
wavelength, tissue scatter length), widefield excitation, label localization
(cytoplasmic vs nuclear), the brightness/photon split (pavg + gain), the
diffuse-background scale, the out-of-focus blur, and the camera bias pedestal.
The values are intensive (CV / ratios / per-pixel ADU) so they hold across FOV
size. ``REAL_TARGETS`` gives the matching real-recording summary-stat ranges for
scoring a run (see ``calcia.diagnostics.image_metrics.print_comparison``).

Tuned against real striatum window recordings (tdt-bfp, 1152x1152 @ 20 Hz).
Match quality at a medium 1000 um FOV:
  tdt  5/5 summary stats in range (excellent) + the washed-cloud texture and
       intensity histogram both match.
  bfp  temporal_cv and p999 in range, median ~within 10%, and the intensity
       HISTOGRAM overlaps the real one across the full bright-nuclei tail. Two
       gaps remain, both rooted in Phase 1 (not the static trace / scan model):
         - spatial_cv (~0.55 vs 0.43): the MSN phase-1 gives ~6000 small
           (~5.5 um) nuclei, so even at nuc_frac=0.5 the labelled nuclei are
           denser and smaller than the real sparse, larger BFP+ nuclei.
         - floor_frac (~0.5 vs 0.13-0.30): the real mean image includes the
           dark window vignette and DARK BLOOD VESSELS punching the field; the
           striatum preset deliberately thins vessels and this regime adds no
           vignette, so the sim floor cannot drop as low.
"""

STATIC_PRESETS = {
    "tdt": dict(
        # Cytoplasmic tdTomato: green-LED excited red emission, dense fill.
        lambda_em_um=0.581, lambda_ex_um=0.555, scatter_length_um_wf=150.0,
        sigma_abs=4.5e-16, phi=0.69, qe_det=0.8,
        nuclear=False, dendflag=True, axonflag=True,
        bg_scale=2.5, soma_gain=1.0, nuc_fl=0.0, gamma=None,
        pavg=0.09, gain=0.04, oof_blur_um=12.0,
        bias=470.0, dark_rate=0.3, read_noise=1.6,
    ),
    "bfp": dict(
        # Nuclear-enriched BFP: violet-LED excited blue emission. Bright
        # punctate nuclei (nuc_fl, heavy-tailed expression) over a dim
        # cytoplasm/neuropil background (bg_scale, soma_gain).
        lambda_em_um=0.457, lambda_ex_um=0.405, scatter_length_um_wf=55.0,
        sigma_abs=2.5e-16, phi=0.55, qe_det=0.75,
        nuclear=True, dendflag=True, axonflag=True,
        bg_scale=1.0, soma_gain=1.0, nuc_fl=5.0, nuc_frac=0.5,
        gamma=(0.5, 1.6),
        pavg=0.5, gain=0.045, oof_blur_um=2.0,
        bias=250.0, dark_rate=0.3, read_noise=1.6,
    ),
}

# Real-recording summary-stat targets (min, max) across the 4 real files.
REAL_TARGETS = {
    "tdt": dict(spatial_cv=(0.08, 0.11), temporal_cv=(0.08, 0.12),
                median=(2260, 4000), floor_frac=(0.70, 0.82),
                p999_over_med=(1.30, 1.70)),
    "bfp": dict(spatial_cv=(0.35, 0.43), temporal_cv=(0.11, 0.14),
                median=(1500, 1700), floor_frac=(0.13, 0.30),
                p999_over_med=(5.00, 6.50)),
}
