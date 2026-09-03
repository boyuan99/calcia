"""Viewer for a ``demo_ladders3d.py`` run: a 3D scene that puts the tissue and
the movie it produced in the SAME picture, plus one panel per ladder.

THE IDEA
A widefield movie is a projection: every cell in a 150 um column lands on one 2D
frame, and once it is flattened you cannot tell what came from where. So don't
flatten it. This viewer draws the tissue block in 3D with every ladder cell at its
true (x, y, z), hangs the actual camera frames underneath as planes in the same
world coordinates, and drops a vertical line from each cell to its own pixel. A
cell and its image are then literally connected, and the three questions answer
themselves by looking:

  * DEPTH      — walk along lane A: same cell, deeper each time. Watch the blob
                 under it fade AND spread.
  * OVERLAP    — lane B: pairs closing in. Watch two blobs become one, and note
                 the separation at which it happens (far larger than the cells).
  * EXPRESSION — lane C: same depth, brightness sweeping ~18x. Watch the dim end
                 sink into the neuropil.

Rendering uses PARALLEL projection on purpose: a cell is then drawn exactly above
its own pixel, so the vertical lines are honest and not a perspective illusion.

The image planes show ACTIVITY (frame minus its own baseline), not the raw frame.
A raw 1P frame is a bright neuropil wash in which nothing is visible — which is
itself true and is shown in ``21_scatter_vs_crisp.png`` — but the point of the 3D
scene is the correspondence between a cell and its blob, and the activity image is
where the blob lives. The static tdTomato plane shows its time-mean instead, since
a static channel has no activity to difference.

Outputs (all into ``<run>/figures/``)

    00_design_map.png        what was built and when each cell fires
    01_scene3d_depth.png     3D hero still, somata coloured by depth
    02_scene3d_activity.png  3D still at the finale, somata coloured by dF/F
    03_orbit.gif             the 3D scene rotating
    04_activity.gif          fixed camera, time running: somata blink and the
                             frame below blinks with them
    10_depth_ladder.png      per-rung crops + amplitude / width / dF-F0 vs depth
    11_overlap_ladder.png    per-pair crops + A-alone/B-alone/both profiles
    12_expression_ladder.png per-rung crops + amplitude vs expression vs noise
    13_focus_split.png       how much of a frame is the in-focus layer at all
    20_two_color.png         GCaMP next to tdTomato, labelled cells marked
    21_scatter_vs_crisp.png  halo on/off, same tissue, same photons
    index.html               all of the above with captions

Run:
    conda run -n calcia python examples/viz_ladders3d.py <run_dir>
    conda run -n calcia python examples/viz_ladders3d.py <run_dir> --interactive
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(HERE, "output")

LANE_COLOR = {"depth": "#28c8ff", "overlap": "#ff5ad0", "expression": "#ffd23f"}
LANE_TITLE = {"depth": "A - DEPTH", "overlap": "B - OVERLAP",
              "expression": "C - EXPRESSION"}
CH_LABEL = {"gcamp": "WITH scatter halo\n(what a real rig records)",
            "gcamp_crisp": "NO halo\n(diffraction limited)",
            "tdt": "tdTomato (static)"}
CROP_UM = 46.0          # analysis half-window around a cell
FLOOR_ANNULUS = (30.0, 45.0)   # where the local neuropil floor is measured


# ======================================================================
# Loading
# ======================================================================
class Demo:
    """A finished ladders3d run: the design plus every rendered channel.

    Holds the movies in memory (small by construction) and owns the one piece of
    geometry every figure needs: the exact map from a cell's microns to its pixel.
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        d = np.load(os.path.join(self.root, "design.npz"), allow_pickle=True)
        self.d = {k: d[k] for k in d.files}
        self.n = int(self.d["n_designed"])
        self.vres = int(self.d["vres"])
        self.sfrac = int(self.d["sfrac"])
        self.scan_buff = int(self.d["scan_buff"])
        self.fps = float(self.d["fps"])
        self.nt = int(self.d["nt"])
        self.vol_um = float(self.d["vol_um"])
        self.depth_um = float(self.d["depth_um"])
        self.focal_um = float(self.d["focal_um"])
        self.um_per_px = self.sfrac / self.vres

        self.ch = {}
        for name in ("gcamp", "gcamp_crisp", "tdt"):
            p = os.path.join(self.root, name)
            if not os.path.isdir(p):
                continue
            m = np.load(os.path.join(p, "movies.npz"))
            e = dict(clean=np.asarray(m["mov_clean"], dtype=np.float32),
                     noisy=np.asarray(m["mov_noisy"], dtype=np.float32),
                     meta=json.load(open(os.path.join(p, "metadata.json"))),
                     dir=p, static=(name == "tdt"))
            tr = np.load(os.path.join(p, "traces.npz"), allow_pickle=True)
            e["soma"] = np.asarray(tr["soma_neurons"], dtype=np.float32)
            try:
                pk = pickle.load(open(os.path.join(p, "params.pkl"), "rb"))
                e["bias"] = float(getattr(pk["cam_params"], "bias", 0.0))
            except Exception:
                e["bias"] = 0.0
            # F0 = the resting level each pixel sits at (10th percentile over
            # time of the NOISE-FREE movie). Everything downstream is measured
            # against this, so dF and dF/F0 mean what they normally mean.
            e["f0"] = np.percentile(e["clean"], 10, axis=0).astype(np.float32)
            e["act"] = (e["noisy"] - e["f0"]).astype(np.float32)
            e["act_clean"] = (e["clean"] - e["f0"]).astype(np.float32)
            fs = os.path.join(p, "focus_split.npz")
            if os.path.isfile(fs):
                z = np.load(fs)
                e["infocus"] = np.asarray(z["mov_infocus"], dtype=np.float32)
                e["oof"] = np.asarray(z["mov_oof"], dtype=np.float32)
            self.ch[name] = e
        if not self.ch:
            raise SystemExit(f"no channel directories found under {self.root}")
        self.H, self.W = self.ch[list(self.ch)[0]]["clean"].shape[1:]

    # ---- geometry -------------------------------------------------------
    def px(self, xyz):
        """(N,3) um -> (N,2) movie pixel (a, b); a runs along volume x, b along y.

        calcia's analytic ``base_px`` map. ``demo_ladders3d`` measures its
        residual against the real transients and records it in each channel's
        metadata (``align_resid_px_max``), so this is a checked fact.
        """
        s, off = self.vres / self.sfrac, self.scan_buff / self.sfrac
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        return np.column_stack([xyz[:, 0] * s - off, xyz[:, 1] * s - off])

    def cell_px(self, i):
        return self.px(self.d["xyz"][i:i + 1])[0]

    def lane(self, group):
        return np.where(self.d["group"] == group)[0]

    def pair_ids(self, rung):
        m = self.lane("overlap")
        m = m[self.d["rung"][m] == rung]
        a = m[self.d["pair_side"][m] == 0][0]
        b = m[self.d["pair_side"][m] == 1][0]
        return int(a), int(b)

    # ---- crops ----------------------------------------------------------
    def crop_at(self, img, centre_um, half_um=CROP_UM):
        """NaN-padded square crop of a frame centred exactly on a micron point.

        NaN padding (rather than clamping the slice) matters: it keeps the crop
        centred and the extent honest even at the edge of the frame, so a marker
        drawn at a cell's offset lands where the cell actually is.
        """
        a, b = self.px(np.atleast_2d(centre_um))[0]
        h = int(round(half_um / self.um_per_px))
        a0, b0 = int(round(a)) - h, int(round(b)) - h
        out = np.full((2 * h + 1, 2 * h + 1), np.nan, dtype=np.float32)
        sa0, sb0 = max(0, a0), max(0, b0)
        sa1, sb1 = min(img.shape[0], a0 + 2 * h + 1), min(img.shape[1], b0 + 2 * h + 1)
        if sa1 > sa0 and sb1 > sb0:
            out[sa0 - a0:sa1 - a0, sb0 - b0:sb1 - b0] = img[sa0:sa1, sb0:sb1]
        # sub-pixel residual of the rounding, so the extent stays exact
        ea = (int(round(a)) - a) * self.um_per_px
        eb = (int(round(b)) - b) * self.um_per_px
        ext = [-h * self.um_per_px + ea, h * self.um_per_px + ea,
               -h * self.um_per_px + eb, h * self.um_per_px + eb]
        return out, ext

    def crop(self, img, i, half_um=CROP_UM):
        return self.crop_at(img, self.d["xyz"][i], half_um)

    def max_half_um(self, centre_um):
        """Largest crop half-width that still fits inside the frame.

        A clipped crop is not just ugly: the local-floor annulus in
        ``blob_metrics`` would be measured on a partial ring, so the reported
        peak and width would be wrong. Panels clamp to the smallest value that
        works for every cell they show, keeping all their crops comparable.
        """
        a, b = self.px(np.atleast_2d(centre_um))[0]
        h = min(a, self.H - 1 - a, b, self.W - 1 - b)
        return max(0.0, float(h) * self.um_per_px)

    def lane_half_um(self, ids, want=CROP_UM):
        """One crop size shared by a whole lane (see max_half_um)."""
        return float(min([want] + [self.max_half_um(self.d["xyz"][i])
                                   for i in ids]))

    # ---- per-cell measurement ------------------------------------------
    def dff_image(self, ch, i, win=9, nbase=4, use="clean"):
        """The frame difference that isolates cell ``i``'s own transient.

        Baseline = the frames just before it fires; signal = the peak over its
        own slot. Because the schedule gives each rung its own slot, this is that
        cell's contribution and nothing else — no demixing, no assumptions.
        """
        mov = self.ch[ch][use]
        f = int(self.d["fire_frame"][i])
        b0 = max(0, f - nbase)
        base = mov[b0:f].mean(0) if f > b0 else mov[0]
        hi = min(mov.shape[0], f + 1 + win)
        return mov[f + 1:hi].max(0) - base

    def diff(self, ch, frame, base_frames, use="clean"):
        mov = self.ch[ch][use]
        b0, b1 = base_frames
        return mov[min(frame, mov.shape[0] - 1)] - mov[b0:b1].mean(0)

    def noise_sigma(self, ch):
        """Per-pixel detector noise of the movie the network sees."""
        c = self.ch[ch]
        return float(np.std(c["noisy"] - c["clean"]))

    def f0_at(self, ch, i):
        """Resting fluorescence above the camera pedestal at a cell's pixel."""
        a, b = np.round(self.cell_px(i)).astype(int)
        a = int(np.clip(a, 0, self.H - 1)); b = int(np.clip(b, 0, self.W - 1))
        return float(self.ch[ch]["f0"][a, b] - self.ch[ch]["bias"])


def blob_metrics(crop, um_per_px, r_peak_um=7.0, annulus=FLOOR_ANNULUS,
                 min_peak=0.0):
    """Peak, FWHM and local floor of a blob, measured AT THE KNOWN CENTRE.

    Deliberately not an argmax hunt: on a faint deep cell the brightest pixel in
    a crop belongs to whatever else is nearby, and an argmax-based width silently
    reports the neighbour's. Here the centre is known from the design, the peak is
    the max inside a small disc around it (tolerant of the ~1 px scan-phase
    residual), the floor is the median of a surrounding annulus (the local
    neuropil the cell sits on), and the width comes from a radial profile of
    ``crop - floor``. Returns (peak, fwhm_um, floor).

    Two widths are returned because they answer different questions. FWHM is the
    width of the CORE — what a segmentation algorithm would draw. ``r50`` is the
    radius containing half of the blob's total light, which also counts the broad
    low tail the scattering halo and defocus put outside the core. A cell can
    keep its FWHM while its r50 doubles, and that difference is precisely the
    light that lands on its neighbours.

    ``min_peak`` (set to a few times the pixel noise) suppresses the WIDTHS only:
    a cell buried in the noise has no measurable width, and reporting one would
    dress noise up as a physical result. Its peak is still returned, because "the
    peak is below the noise" is exactly the finding.

    Returns (peak, fwhm_um, floor, r50_um).
    """
    v = np.asarray(crop, dtype=np.float64)
    n = v.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c) * um_per_px

    ann = (rr >= annulus[0]) & (rr < annulus[1]) & np.isfinite(v)
    if ann.sum() <= 8:                       # crop too small for a floor ring
        ann = (rr >= 0.65 * rr.max()) & np.isfinite(v)
    floor = float(np.median(v[ann])) if ann.any() else 0.0
    w = v - floor

    core = (rr <= r_peak_um) & np.isfinite(w)
    if not core.any():
        return 0.0, np.nan, floor, np.nan
    peak = float(np.nanmax(w[core]))
    if peak <= 0 or peak < min_peak:
        return peak, np.nan, floor, np.nan

    # radial profile in 1-pixel rings
    nb = int(np.floor(rr.max() / um_per_px))
    edges = np.arange(nb + 1) * um_per_px
    prof, rad, ring_sum = [], [], []
    for k in range(nb):
        m = (rr >= edges[k]) & (rr < edges[k + 1]) & np.isfinite(w)
        if m.sum() >= 2:
            prof.append(float(np.mean(w[m])))
            rad.append(0.5 * (edges[k] + edges[k + 1]))
            ring_sum.append(float(np.sum(np.clip(w[m], 0, None))))
    if len(prof) < 3:
        return peak, np.nan, floor, np.nan
    prof, rad = np.asarray(prof), np.asarray(rad)

    # FWHM: 2 x the first radius where the radial profile falls below half max
    half = 0.5 * prof[0]
    below = np.where(prof < half)[0]
    if below.size == 0:
        fwhm = np.nan
    elif int(below[0]) == 0:
        fwhm = float(2 * rad[0])
    else:
        k = int(below[0])
        r0, r1, p0, p1 = rad[k - 1], rad[k], prof[k - 1], prof[k]
        r = r0 + (half - p0) * (r1 - r0) / (p1 - p0) if p1 != p0 else r1
        fwhm = float(2 * r)

    # r50: the radius holding half of the blob's total light — unlike FWHM this
    # counts the broad low tail, which is where the halo and defocus put it.
    cum = np.cumsum(np.asarray(ring_sum))
    r50 = np.nan
    if cum[-1] > 0:
        j = int(np.searchsorted(cum, 0.5 * cum[-1]))
        r50 = float(rad[min(j, len(rad) - 1)])
    return peak, fwhm, floor, r50


def radial_profile(crop, um_per_px, annulus=FLOOR_ANNULUS):
    """(radius_um, mean value) of a crop about its centre, floor-subtracted.

    The same rings ``blob_metrics`` uses, returned as a curve. Plotting the curve
    beats reporting a single width number in a crowded field: a neighbouring cell
    inside the crop shows up as a visible bump you can discount by eye, where it
    would silently corrupt a scalar like r50.
    """
    v = np.asarray(crop, dtype=np.float64)
    n = v.shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c) * um_per_px
    ann = (rr >= annulus[0]) & (rr < annulus[1]) & np.isfinite(v)
    if ann.sum() <= 8:
        ann = (rr >= 0.65 * rr.max()) & np.isfinite(v)
    w = v - (float(np.median(v[ann])) if ann.any() else 0.0)
    nb = int(np.floor(rr.max() / um_per_px))
    edges = np.arange(nb + 1) * um_per_px
    rad, prof = [], []
    for k in range(nb):
        m = (rr >= edges[k]) & (rr < edges[k + 1]) & np.isfinite(w)
        if m.sum() >= 2:
            rad.append(0.5 * (edges[k] + edges[k + 1]))
            prof.append(float(np.mean(w[m])))
    return np.asarray(rad), np.asarray(prof)


# ======================================================================
# 3D scene (PyVista)
# ======================================================================
def _image_plane(img, demo, z_world):
    """A movie frame as a plane in VOLUME coordinates.

    Built as a VTK image whose cells sit exactly where their pixels' microns are
    (origin and spacing derived from the same base_px map), so with parallel
    projection a cell at (x, y, z) is drawn directly above its own pixel.
    """
    import pyvista as pv
    H, W = img.shape
    px = demo.um_per_px
    x0 = demo.scan_buff / demo.vres - 0.5 * px
    grid = pv.ImageData(dimensions=(H + 1, W + 1, 1), spacing=(px, px, 1.0),
                        origin=(x0, x0, z_world))
    grid.cell_data["img"] = np.ascontiguousarray(
        np.nan_to_num(img).ravel(order="F").astype(np.float32))
    return grid


class Scene3D:
    """Tissue above, the frames it produced below, wired together.

    One implementation shared by the stills, the orbit and the time animation, so
    they cannot drift apart.
    """

    def __init__(self, demo, plotter, *, planes=("gcamp",), colour_by="depth",
                 show_filler=True, plane_gap=0.34, plane_step=0.44):
        import pyvista as pv
        self.demo, self.pl = demo, plotter
        self.planes_spec = [p for p in planes if p in demo.ch]
        self.colour_by = colour_by
        V, Dp = demo.vol_um, demo.depth_um
        pl = plotter
        pl.set_background("#07080d")
        pl.enable_parallel_projection()      # cell sits exactly above its pixel

        box = pv.Box(bounds=(0, V, 0, V, -Dp, 0))
        pl.add_mesh(box, style="wireframe", color="#556170", line_width=1.8,
                    opacity=0.95)

        fp = pv.Plane(center=(V / 2, V / 2, -demo.focal_um), direction=(0, 0, 1),
                      i_size=V, j_size=V)
        pl.add_mesh(fp, color="#2ee6c8", opacity=0.10, lighting=False)
        pl.add_mesh(fp.extract_feature_edges(), color="#2ee6c8", line_width=2.5,
                    opacity=0.75)

        if show_filler and len(demo.d.get("filler_xyz", [])):
            f = np.asarray(demo.d["filler_xyz"], dtype=float).copy()
            f[:, 2] *= -1.0
            pl.add_mesh(pv.PolyData(f).glyph(
                geom=pv.Sphere(radius=4.0, theta_resolution=8, phi_resolution=8),
                scale=False, orient=False),
                color="#8892a4", opacity=0.30, show_scalar_bar=False)

        # --- image planes below the block, in acquisition order ---
        # The display range is set from the FINALE frame (all ladder cells up),
        # not from the whole stack: the stack is dominated by quiet frames, and a
        # percentile over it would clip every blob to white.
        ffr = int(demo.d["finale_frame"])
        self.plane_meshes, self.plane_z = {}, {}
        for k, name in enumerate(self.planes_spec):
            z = -Dp - plane_gap * V - k * plane_step * V
            ref = self.plane_image(name, min(ffr + 2, demo.nt - 1))
            lo, hi = np.percentile(ref, [45.0, 99.6] if not demo.ch[name]["static"]
                                   else [2.0, 99.6])
            g = _image_plane(self.plane_image(name, 0), demo, z)
            pl.add_mesh(g, scalars="img",
                        cmap=("afmhot" if name.startswith("gcamp") else "gist_heat"),
                        clim=(float(lo), float(hi)), lighting=False,
                        show_scalar_bar=False, name=f"plane_{name}")
            self.plane_meshes[name] = g
            self.plane_z[name] = z
            pl.add_point_labels(
                np.array([[V * 0.5, -26.0, z]]),
                [{"gcamp": "camera frame - GCaMP activity  (scatter halo ON)",
                  "gcamp_crisp": "same tissue, NO scatter  (diffraction limited)",
                  "tdt": "camera frame - tdTomato  (structural, time-mean)"}[name]],
                font_size=14, text_color="#e6edf3", shape=None,
                always_visible=True, show_points=False)

        bottom = min(self.plane_z.values()) if self.plane_z else -Dp
        for i in range(demo.n):
            x, y, z = demo.d["xyz"][i]
            pl.add_mesh(pv.Line((x, y, -z), (x, y, bottom)),
                        color=LANE_COLOR[str(demo.d["group"][i])],
                        line_width=1.6, opacity=0.55)

        # Lane labels sit beside each lane's own ROW on the first image plane, so
        # the label, its drop lines and the blobs it explains all end up in the
        # same part of the picture at any camera azimuth.
        z_lab = (self.plane_z[self.planes_spec[0]] if self.planes_spec
                 else -Dp - plane_gap * V)
        for g in ("depth", "overlap", "expression"):
            idx = demo.lane(g)
            if not len(idx):
                continue
            y0 = float(np.mean(demo.d["xyz"][idx, 1]))
            pl.add_point_labels(np.array([[-0.14 * V, y0, z_lab]]),
                                [LANE_TITLE[g]], font_size=15,
                                text_color=LANE_COLOR[g], shape=None,
                                always_visible=True, show_points=False)

        self.sphere = pv.Sphere(radius=7.5, theta_resolution=18, phi_resolution=18)
        self.pts = demo.d["xyz"].astype(float).copy()
        self.pts[:, 2] *= -1.0
        self._glyph, self._per_pt = None, 0
        self._add_ladder(np.zeros(demo.n))

    def plane_image(self, name, frame):
        c = self.demo.ch[name]
        if c["static"]:
            return c["noisy"].mean(0)
        return c["act"][int(np.clip(frame, 0, c["act"].shape[0] - 1))]

    def _add_ladder(self, dff):
        """Build the ladder-soma actor ONCE.

        The glyph is created a single time and afterwards only its scalar array
        is rewritten (see :meth:`set_activity`). Re-glyphing every frame was fast
        enough to write a GIF but far too slow to keep up with a slider being
        dragged, which is the whole point of the live window.
        """
        import pyvista as pv
        cloud = pv.PolyData(self.pts)
        if self.colour_by == "depth":
            cloud["scalar"] = self.demo.d["xyz"][:, 2]
            cmap, clim = "viridis_r", (0.0, self.demo.depth_um)
            title, fmt = "depth (um)", "%.0f"
        else:
            cloud["scalar"] = np.asarray(dff, dtype=float)
            cmap, clim = "inferno", (0.0, 1.0)
            title, fmt = "cell dF/F0 (of its own max)", "%.2f"
        gl = cloud.glyph(geom=self.sphere, scale=False, orient=False)
        self._glyph = gl
        n = len(self.pts)
        self._per_pt = (gl.n_points // n) if n and gl.n_points % n == 0 else 0
        self.pl.add_mesh(gl, scalars="scalar", cmap=cmap, clim=clim,
                         name="ladder", show_scalar_bar=True,
                         scalar_bar_args=dict(title=title, color="#e6edf3",
                                              n_labels=4, width=0.22, height=0.042,
                                              position_x=0.72, position_y=0.045,
                                              title_font_size=13,
                                              label_font_size=11, fmt=fmt))

    def set_activity(self, dff):
        """Recolour the somata in place — no geometry rebuilt."""
        if self.colour_by == "depth" or self._glyph is None:
            return
        v = np.asarray(dff, dtype=float)
        if self._per_pt:
            self._glyph["scalar"][:] = np.repeat(v, self._per_pt)
        else:                       # glyph point count not a clean multiple
            self._add_ladder(v)

    def set_frame(self, frame, dff=None):
        for name, mesh in self.plane_meshes.items():
            mesh.cell_data["img"][:] = np.nan_to_num(
                self.plane_image(name, frame)).ravel(order="F")
        if self.colour_by != "depth":
            self.set_activity(np.zeros(self.demo.n) if dff is None else dff)

    def camera(self, azimuth=38.0, elevation=20.0, zoom=1.0):
        pl = self.pl
        pl.camera_position = "yz"
        pl.camera.azimuth = azimuth
        pl.camera.elevation = elevation
        pl.reset_camera()
        pl.camera.zoom(zoom)


def ladder_dff_matrix(demo, ch="gcamp"):
    """(n_designed, nt) per-cell dF/F0 from the ground-truth traces — what the 3D
    somata are coloured by. Deliberately from the traces, not the movie: the movie
    cannot separate overlapping cells, which is the very thing being shown."""
    if ch not in demo.ch:
        ch = list(demo.ch)[0]
    s = demo.ch[ch]["soma"][:demo.n]
    f0 = np.percentile(s, 20, axis=1, keepdims=True)
    dff = (s - f0) / np.maximum(f0, 1e-6)
    return np.clip(dff / np.maximum(dff.max(axis=1, keepdims=True), 1e-6), 0, 1)


def render_3d(demo, figdir, *, orbit=True, anim=True, anim_stride=1,
              window=(1240, 900), verbose=True):
    import pyvista as pv
    pv.OFF_SCREEN = True
    made = []
    planes = [n for n in ("gcamp", "tdt") if n in demo.ch]
    ff = int(demo.d["finale_frame"])
    dffm = ladder_dff_matrix(demo)

    pl = pv.Plotter(off_screen=True, window_size=window)
    sc = Scene3D(demo, pl, planes=planes, colour_by="depth")
    sc.set_frame(min(ff + 2, demo.nt - 1))
    sc.camera(azimuth=36, elevation=22, zoom=1.02)
    pl.add_text("tissue in 3D  ->  the frames it produces\n"
                "somata at their true (x,y,z), coloured by DEPTH.\n"
                "every line ends on that cell's own pixel (parallel projection)",
                position="upper_left", font_size=11, color="#e6edf3")
    p = os.path.join(figdir, "01_scene3d_depth.png")
    pl.screenshot(p); pl.close(); made.append(p)
    if verbose:
        print(f"  {os.path.basename(p)}")

    pl = pv.Plotter(off_screen=True, window_size=window)
    sc = Scene3D(demo, pl, planes=planes, colour_by="activity")
    sc.set_frame(min(ff + 2, demo.nt - 1), dffm[:, min(ff + 2, demo.nt - 1)])
    sc.camera(azimuth=36, elevation=22, zoom=1.02)
    pl.add_text("FINALE: every ladder cell fires at once\n"
                "the realistic case — all depths land on one frame and nothing\n"
                "in that frame tells you which blob came from where",
                position="upper_left", font_size=11, color="#e6edf3")
    p = os.path.join(figdir, "02_scene3d_activity.png")
    pl.screenshot(p); pl.close(); made.append(p)
    if verbose:
        print(f"  {os.path.basename(p)}")

    if orbit:
        pl = pv.Plotter(off_screen=True, window_size=(980, 780))
        sc = Scene3D(demo, pl, planes=planes, colour_by="depth")
        sc.set_frame(min(ff + 2, demo.nt - 1))
        sc.camera(azimuth=36, elevation=22, zoom=0.95)
        p = os.path.join(figdir, "03_orbit.gif")
        pl.open_gif(p, fps=14)
        path = pl.generate_orbital_path(n_points=48, shift=demo.vol_um * 0.6,
                                        factor=2.2)
        pl.orbit_on_path(path, write_frames=True, step=0.0)
        pl.close(); made.append(p)
        if verbose:
            print(f"  {os.path.basename(p)}  ({os.path.getsize(p)/1e6:.1f} MB)")

    if anim:
        pl = pv.Plotter(off_screen=True, window_size=(1020, 790))
        sc = Scene3D(demo, pl, planes=planes, colour_by="activity")
        sc.camera(azimuth=34, elevation=20, zoom=1.00)
        txt = pl.add_text("", position="upper_left", font_size=11, color="#e6edf3")
        p = os.path.join(figdir, "04_activity.gif")
        pl.open_gif(p, fps=12)
        for t in range(0, demo.nt, max(1, anim_stride)):
            sc.set_frame(t, dffm[:, t])
            active = np.where(dffm[:, t] > 0.25)[0]
            who = ", ".join(f"{str(demo.d['group'][i])[:4]} {str(demo.d['label'][i])}"
                            for i in active[:3])
            pl.remove_actor(txt)
            txt = pl.add_text(
                f"frame {t:3d}/{demo.nt}   t = {t/demo.fps:5.2f} s\n"
                f"firing: {who if who else '-'}",
                position="upper_left", font_size=11, color="#e6edf3")
            pl.write_frame()
        pl.close(); made.append(p)
        if verbose:
            print(f"  {os.path.basename(p)}  ({os.path.getsize(p)/1e6:.1f} MB)")
    return made


def interactive(demo, fps=None, play=True):
    """Live window: rotate the scene, scrub the movie, or let it play itself.

    Two things a default PyVista slider does not do, and both matter here:

    * ``interaction_event='always'`` — VTK's slider fires its callback only when
      you RELEASE it by default, so dragging shows nothing until you let go. With
      'always' the scene follows the handle continuously, which is the only way
      scrubbing is useful for finding the moment a particular cell fires.
    * a timer event driving playback, so you do not have to drag at all. SPACE
      toggles it, the arrow keys step one frame, and the slider handle tracks
      whatever the animation is doing.

    Keeping up with a dragged slider is why ``Scene3D`` updates scalars in place
    rather than rebuilding the soma glyph per frame.
    """
    import pyvista as pv
    pv.OFF_SCREEN = False
    fps = float(fps or demo.fps)
    dffm = ladder_dff_matrix(demo)
    pl = pv.Plotter(window_size=(1300, 920), title="calcia - designed ladders")
    planes = [n for n in ("gcamp", "tdt") if n in demo.ch]
    sc = Scene3D(demo, pl, planes=planes, colour_by="activity")
    sc.camera(azimuth=36, elevation=22, zoom=1.1)

    hud = pl.add_text("", position="upper_left", font_size=11, color="#e6edf3")
    state = {"t": 0, "playing": bool(play), "rep": None}

    def _hud(t):
        active = np.where(dffm[:, t] > 0.25)[0]
        who = ", ".join(f"{str(demo.d['group'][i])[:4]} {str(demo.d['label'][i])}"
                        for i in active[:3])
        msg = (f"frame {t}/{demo.nt}   t = {t/demo.fps:5.2f} s   "
               f"[{'PLAYING' if state['playing'] else 'PAUSED'}]\n"
               f"firing: {who if who else '-'}\n"
               f"SPACE play/pause   <- -> step   R restart   drag slider to scrub")
        try:                       # vtkCornerAnnotation: corner 2 = upper left
            hud.SetText(2, msg)
        except Exception:
            pl.add_text(msg, position="upper_left", font_size=11,
                        color="#e6edf3", name="hud")

    def _goto(t, move_handle=True):
        t = int(t) % demo.nt
        state["t"] = t
        sc.set_frame(t, dffm[:, t])
        _hud(t)
        if move_handle and state["rep"] is not None:
            state["rep"].SetValue(float(t))

    def on_slider(value):
        if int(round(value)) != state["t"]:
            _goto(round(value), move_handle=False)

    w = pl.add_slider_widget(on_slider, [0, demo.nt - 1], value=0, title="frame",
                             pointa=(0.08, 0.93), pointb=(0.45, 0.93),
                             style="modern", color="#e6edf3",
                             interaction_event="always")
    state["rep"] = w.GetRepresentation()

    def _tick(step):
        if state["playing"]:
            _goto(state["t"] + 1)

    pl.add_timer_event(max_steps=10 ** 9, duration=int(round(1000.0 / fps)),
                       callback=_tick)

    def _toggle():
        state["playing"] = not state["playing"]
        _hud(state["t"])

    pl.add_key_event("space", _toggle)
    pl.add_key_event("Right", lambda: (_goto(state["t"] + 1)))
    pl.add_key_event("Left", lambda: (_goto(state["t"] - 1)))
    pl.add_key_event("r", lambda: (_goto(0)))

    _goto(0)
    print(f"  playing at {fps:.0f} fps.  SPACE = play/pause,  <- -> = step one "
          f"frame,  R = restart")
    print("  drag to rotate, scroll to zoom, drag the slider to scrub "
          "(updates live); close the window to exit")
    pl.show()


# ======================================================================
# 2D panels (matplotlib)
# ======================================================================
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117", "text.color": "#e6edf3",
        "axes.labelcolor": "#e6edf3", "xtick.color": "#9fb3c8",
        "ytick.color": "#9fb3c8", "axes.edgecolor": "#30363d",
        "axes.titlecolor": "#e6edf3", "font.size": 9,
        "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
        "legend.framealpha": 0.9, "grid.color": "#21262d"})
    return plt


def _show(ax, img, extent=None, cmap="inferno", clim=None):
    """imshow with x horizontal and y vertical.

    Movie arrays are indexed [x, y] (calcia's volume order), so every panel
    transposes exactly once — here, and nowhere else.
    """
    kw = dict(cmap=cmap, origin="lower", interpolation="nearest", aspect="equal")
    if extent is not None:
        kw["extent"] = extent
    if clim is not None:
        kw["vmin"], kw["vmax"] = clim
    return ax.imshow(np.asarray(img).T, **kw)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def _centre_mark(ax, color="#7CFC00"):
    """Mark where the cell IS on a crop, whether or not it can be seen.

    Without this a panel showing nothing is ambiguous — is the cell invisible, or
    is the crop off-target? With it, "nothing at the cross" is a readable result.
    """
    ax.plot([0], [0], marker="+", ms=9, mew=1.0, color=color, alpha=0.85)


def _plot_with_floor(ax, x, y, sig, label, color=None):
    """Plot a measured series on a LOG axis when some points are below the noise.

    A peak measured on a cell you cannot see comes out near zero or negative, and
    a log axis simply drops it — which reads as missing data rather than as the
    finding it is. Sub-noise points are parked on a floor line and drawn HOLLOW,
    so "we looked and there was nothing" stays visible and stays distinguishable
    from a real measurement.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > sig)
    yy = np.where(np.isfinite(y), np.maximum(y, 0.45 * sig), 0.45 * sig)
    line, = ax.plot(x, yy, "-", lw=1.7, label=label,
                    **({"color": color} if color else {}))
    c = line.get_color()
    ax.plot(np.asarray(x)[ok], yy[ok], "o", ms=5.5, color=c)
    ax.plot(np.asarray(x)[~ok], yy[~ok], "o", ms=5.5, mfc="none", mec=c, mew=1.4)
    return c


def fig_design_map(demo, figdir):
    plt = _mpl()
    d = demo.d
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 0.8],
                          hspace=0.40, wspace=0.20)

    ax = fig.add_subplot(gs[0, 0])
    if len(d.get("filler_xyz", [])):
        f = d["filler_xyz"]
        ax.scatter(f[:, 0], f[:, 1], s=5, c="#39424e", label="filler neurons")
    for g in ("depth", "overlap", "expression"):
        idx = demo.lane(g)
        ax.scatter(d["xyz"][idx, 0], d["xyz"][idx, 1], s=64, facecolors="none",
                   edgecolors=LANE_COLOR[g], lw=1.8, label=LANE_TITLE[g])
    # Alternate the label above / below its marker: consecutive rungs are close
    # enough that a single offset makes the text collide and become unreadable.
    for i in range(demo.n):
        side = int(d["pair_side"][i])
        up = (side == 1) if side >= 0 else (int(d["rung"][i]) % 2 == 0)
        ax.annotate(str(d["label"][i]), (d["xyz"][i, 0], d["xyz"][i, 1]),
                    textcoords="offset points", xytext=(0, 9 if up else -15),
                    fontsize=6, ha="center", color="#9fb3c8")
    ax.set_xlim(0, demo.vol_um); ax.set_ylim(0, demo.vol_um)
    ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title("designed layout - top view (the imaged FOV)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
              fontsize=7.5)

    ax = fig.add_subplot(gs[0, 1])
    if len(d.get("filler_xyz", [])):
        f = d["filler_xyz"]
        ax.scatter(f[:, 0], f[:, 2], s=5, c="#39424e")
    for g in ("depth", "overlap", "expression"):
        idx = demo.lane(g)
        ax.scatter(d["xyz"][idx, 0], d["xyz"][idx, 2], s=64, facecolors="none",
                   edgecolors=LANE_COLOR[g], lw=1.8)
    ax.axhline(demo.focal_um, color="#2ee6c8", lw=1.5, ls="--",
               label=f"focal plane ({demo.focal_um:.0f} um)")
    ax.set_xlim(0, demo.vol_um); ax.set_ylim(demo.depth_um, -5)
    ax.set_xlabel("x (um)"); ax.set_ylabel("depth z (um)   ->  deeper")
    ax.set_title("side view - only lane A varies in depth")
    ax.legend(fontsize=7.5, loc="lower right")

    ax = fig.add_subplot(gs[1, :])
    sp = d["spikes_designed"]
    for i in range(demo.n):
        t = np.where(sp[i] > 0)[0]
        ax.scatter(t / demo.fps, np.full_like(t, i, dtype=float), s=30,
                   marker="|", color=LANE_COLOR[str(d["group"][i])], lw=2.4)
    ax.axvline(int(d["finale_frame"]) / demo.fps, color="#e6edf3", lw=1.0,
               ls=":", label="finale - everyone at once")
    ax.set_yticks(range(demo.n))
    ax.set_yticklabels([f"{str(d['group'][i])[:4]} {str(d['label'][i])}"
                        for i in range(demo.n)], fontsize=6.5)
    ax.set_xlabel("time (s)"); ax.set_ylim(-1, demo.n)
    ax.set_title("activation schedule - one rung per slot, so every blob in the "
                 "movie has exactly one owner", pad=8)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="x", alpha=0.35)

    p = os.path.join(figdir, "00_design_map.png")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    return p


def fig_depth(demo, figdir):
    plt = _mpl()
    idx = demo.lane("depth")
    zs = demo.d["xyz"][idx, 2]
    chans = [c for c in ("gcamp", "gcamp_crisp") if c in demo.ch]
    n = len(idx)
    half = demo.lane_half_um(idx)
    sig = demo.noise_sigma("gcamp")

    fig = plt.figure(figsize=(2.3 * n + 2.0, 2.5 * len(chans) + 4.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0 * len(chans), 1.25],
                             hspace=0.26, top=0.90, bottom=0.06,
                             left=0.06, right=0.985)
    gtop = outer[0].subgridspec(len(chans), n, hspace=0.10, wspace=0.08)
    gbot = outer[1].subgridspec(1, 3, wspace=0.34)

    m = {c: [] for c in chans}
    prof_by_ch = {c: [] for c in chans}
    crops_by_ch = {}
    for r, c in enumerate(chans):
        crops, exts = [], []
        for i in idx:
            cr, ex = demo.crop(demo.dff_image(c, i), i, half)
            crops.append(cr); exts.append(ex)
        vmax = float(np.nanpercentile(np.stack(crops), 99.9))
        crops_by_ch[c] = crops
        for k, i in enumerate(idx):
            ax = fig.add_subplot(gtop[r, k])
            _show(ax, crops[k], extent=exts[k], clim=(0, vmax))
            _centre_mark(ax)
            _clean(ax)
            pk, fw, _, r50 = blob_metrics(crops[k], demo.um_per_px,
                                          min_peak=3 * sig)
            m[c].append((pk, fw, pk / max(demo.f0_at(c, i), 1e-6), r50))
            prof_by_ch[c].append(radial_profile(crops[k], demo.um_per_px))
            if r == 0:
                ax.set_title(f"z = {zs[k]:.0f} um"
                             + ("\n(focal plane)" if abs(zs[k] - demo.focal_um) < 1
                                else ""), fontsize=9.5)
            if k == 0:
                ax.set_ylabel(CH_LABEL[c], fontsize=8.5)
            ax.text(0.04, 0.05,
                    f"pk {pk:,.0f}\n" + ("FWHM -" if not np.isfinite(fw)
                                         else f"FWHM {fw:.0f}um"),
                    transform=ax.transAxes, fontsize=7.5, color="#e6edf3",
                    va="bottom")

    # --- 1: peak amplitude ---
    ax = fig.add_subplot(gbot[0, 0])
    for c in chans:
        _plot_with_floor(ax, zs, [s[0] for s in m[c]], sig, c)
    ax.axhline(sig, color="#ff6b6b", ls="--", lw=1.2,
               label=f"1 px noise ({sig:.0f} ADU)")
    ax.axvline(demo.focal_um, color="#2ee6c8", ls=":", lw=1.2)
    ax.set_yscale("log"); ax.set_xlabel("cell depth z (um)", fontsize=8.5)
    ax.set_ylabel("peak dF of its own transient (ADU)", fontsize=8.5)
    ax.set_title("amplitude falls off steeply with depth", fontsize=9.5)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6.5)
    ax.text(0.02, 0.04, "hollow marker = below the noise\n(parked on the floor "
            "line; nothing measurable there)", transform=ax.transAxes,
            fontsize=6.5, color="#6e7681")

    # --- 2: the actual radial profiles, one curve per rung ---
    # A curve rather than a width number: in a field this crowded a neighbouring
    # cell inside the crop appears as a visible bump you can discount by eye,
    # where it would silently corrupt any single width statistic.
    #
    # These are measured on the NOISE-FREE movie, so the gate is the crop's own
    # structural scatter (other cells and neuropil), not camera noise. That is
    # the right gate for a SHAPE question — a rung can be far too faint for a
    # detector and still have a perfectly well-defined profile — and it keeps
    # more rungs on the plot than the detectability gate used for amplitude.
    ax = fig.add_subplot(gbot[0, 1])
    import matplotlib.pyplot as _p
    cmap = _p.get_cmap("viridis_r")
    ref = chans[0]
    n_shown = 0
    for k, i in enumerate(idx):
        cr = crops_by_ch[ref][k]
        rad, pr = prof_by_ch[ref][k]
        if pr.size < 3 or pr[0] <= 0:
            continue
        outer = cr[np.isfinite(cr)]
        scat = float(np.std(outer[outer < np.nanpercentile(outer, 80)])) \
            if outer.size > 20 else np.inf
        if not np.isfinite(scat) or pr[0] < 5 * scat:
            continue
        ax.plot(rad, pr / pr[0], lw=1.7,
                color=cmap(zs[k] / max(demo.depth_um, 1e-6)),
                label=f"z={zs[k]:.0f} um")
        n_shown += 1
    if len(chans) > 1:
        rad, pr = prof_by_ch[chans[1]][0]
        if pr.size and pr[0] > 0:
            ax.plot(rad, pr / pr[0], lw=1.5, ls="--", color="#9fb3c8",
                    label=f"{chans[1]}, z={zs[0]:.0f} um")
    ax.set_xlabel("distance from the cell centre (um)", fontsize=8.5)
    ax.set_ylabel("dF, normalised to the centre", fontsize=8.5)
    ax.set_title(f"radial profile of each rung ({ref}, noise-free movie)\n"
                 "log axis: the cores nearly coincide, the TAILS do not",
                 fontsize=9.5)
    # Log y: the interesting part of these curves is the TAIL, two decades below
    # the peak and invisible on a linear axis. The x range stops well short of
    # the crop edge, where a neighbouring cell dominates and the curve stops
    # being about this cell at all.
    ax.set_yscale("log")
    ax.set_xlim(0, min(half, 32.0)); ax.set_ylim(3e-3, 1.8)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, loc="lower left")

    # --- 3: dF/F0 ---
    ax = fig.add_subplot(gbot[0, 2])
    for c in chans:
        _plot_with_floor(ax, zs, [s[2] for s in m[c]],
                         sig / max(demo.f0_at(c, idx[0]), 1e-6), c)
    ax.axvline(demo.focal_um, color="#2ee6c8", ls=":", lw=1.2)
    ax.set_yscale("log"); ax.set_xlabel("cell depth z (um)", fontsize=8.5)
    ax.set_ylabel("peak dF / F0 at its own pixel", fontsize=8.5)
    ax.set_title("dF/F0 falls faster still: the haze raises F0 too",
                 fontsize=9.5)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6.5)

    fig.suptitle("LANE A - DEPTH   identical cells, identical expression, "
                 "only z differs", fontsize=13, y=0.98)
    p = os.path.join(figdir, "10_depth_ladder.png")
    fig.savefig(p, dpi=135); plt.close(fig)
    return p


def pair_images(demo, ch, a, b):
    """Isolate each cell of an overlap pair, and their true merged image.

    The two cells of a pair fire half a slot apart, so at B's peak A's transient
    has not fully decayed and a naive "B alone" difference would carry a negative
    ghost of A. Three images are built instead, none of them a fudge:

      A alone  measured directly — B has not fired yet at that frame.
      B alone  ``I(at B's peak) - r * A_alone``, where ``r`` is A's own decay
               ratio between the two frames, read off its ground-truth trace.
               Exact: the scan is linear in the traces, so A's image at any
               moment is its image at its peak scaled by its trace.
      both     measured directly from the FINALE, where every ladder cell fires
               simultaneously — a real frame with both cells at full amplitude,
               not a sum of two.

    Returns dict(A alone, B alone, both).
    """
    fa, fb = int(demo.d["fire_frame"][a]), int(demo.d["fire_frame"][b])
    ff = int(demo.d["finale_frame"])
    ka, kb = fa + 3, min(fb + 3, demo.nt - 1)
    pre = (max(0, fa - 4), fa)

    IA = demo.diff(ch, ka, pre)
    Iab = demo.diff(ch, kb, pre)
    tr = demo.ch[ch]["soma"][a]
    base_a = float(np.percentile(tr, 20))
    num, den = tr[kb] - base_a, tr[ka] - base_a
    r = float(num / den) if abs(den) > 1e-9 else 0.0
    IB = Iab - np.clip(r, 0.0, 1.0) * IA

    both = demo.diff(ch, min(ff + 3, demo.nt - 1), (max(0, ff - 5), ff))
    return {"A alone": IA, "B alone": IB, "both": both}


def fig_overlap(demo, figdir):
    plt = _mpl()
    idx = demo.lane("overlap")
    rungs = sorted(set(int(r) for r in demo.d["rung"][idx]))
    chans = [c for c in ("gcamp", "gcamp_crisp") if c in demo.ch]
    n = len(rungs)
    fig = plt.figure(figsize=(2.4 * n + 2.2, 4.6 * len(chans) + 4.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.4 * len(chans), 1.0],
                             hspace=0.16, top=0.93, bottom=0.05,
                             left=0.07, right=0.985)
    gtop = outer[0].subgridspec(2 * len(chans), n, hspace=0.16, wspace=0.12,
                                height_ratios=[1.0, 0.72] * len(chans))
    gbot = outer[1].subgridspec(1, 1)

    dip = {c: [] for c in chans}
    seps = []
    for ci, c in enumerate(chans):
        for k, r in enumerate(rungs):
            a, b = demo.pair_ids(r)
            sep = float(demo.d["sep_um"][a])
            if ci == 0:
                seps.append(sep)
            mid = 0.5 * (demo.d["xyz"][a] + demo.d["xyz"][b])
            half = min(max(CROP_UM, 1.6 * sep), demo.max_half_um(mid))
            imgs = pair_images(demo, c, a, b)
            crops = {kk: demo.crop_at(v, mid, half)[0] for kk, v in imgs.items()}
            _, ext = demo.crop_at(imgs["both"], mid, half)

            ax = fig.add_subplot(gtop[2 * ci, k])
            _show(ax, crops["both"], extent=ext,
                  clim=(0, float(np.nanpercentile(crops["both"], 99.8))))
            _clean(ax)
            for cell, mk, col in ((a, "o", "#28c8ff"), (b, "s", "#ff5ad0")):
                ax.scatter([demo.d["xyz"][cell, 0] - mid[0]],
                           [demo.d["xyz"][cell, 1] - mid[1]], s=95, marker=mk,
                           facecolors="none", edgecolors=col, lw=1.4)
            if ci == 0:
                ax.set_title(f"separation {sep:.0f} um", fontsize=9.5)
            if k == 0:
                ax.set_ylabel(CH_LABEL[c] + "\nboth cells firing", fontsize=8)

            # --- profiles along the separation axis (volume y) ---
            axp = fig.add_subplot(gtop[2 * ci + 1, k])
            row = crops["both"].shape[0] // 2
            yy = np.linspace(ext[2], ext[3], crops["both"].shape[1])
            profs = {kk: np.nanmean(cr[max(0, row - 1):row + 2, :], axis=0)
                     for kk, cr in crops.items()}
            for kk, col, lw in (("A alone", "#28c8ff", 1.1),
                                ("B alone", "#ff5ad0", 1.1),
                                ("both", "#e6edf3", 1.8)):
                axp.plot(yy, profs[kk], color=col, lw=lw, label=kk)
            ya = demo.d["xyz"][a, 1] - mid[1]
            yb = demo.d["xyz"][b, 1] - mid[1]
            # Resolvability: the valley between the two peaks, relative to the
            # WEAKER one (the standard two-point criterion). Clamped to [0, 1] —
            # a "negative valley" is not a thing, it just means fully merged.
            pr = np.nan_to_num(profs["both"])
            lo, hi = sorted((int(np.argmin(np.abs(yy - ya))),
                             int(np.argmin(np.abs(yy - yb)))))
            if hi - lo >= 2:
                floor = float(np.nanmin(pr))
                weak = min(float(pr[lo]), float(pr[hi])) - floor
                valley = float(pr[lo:hi + 1].min()) - floor
                dd = 0.0 if weak <= 1e-6 else 1.0 - valley / weak
            else:
                dd = 0.0
            dd = float(np.clip(dd, 0.0, 1.0))
            dip[c].append(dd)
            axp.axvline(ya, color="#28c8ff", lw=0.8, ls="--")
            axp.axvline(yb, color="#ff5ad0", lw=0.8, ls="--")
            axp.set_xticks([]); axp.set_yticks([])
            axp.text(0.03, 0.76, f"dip {100*dd:.0f}%", transform=axp.transAxes,
                     fontsize=8, color=("#7CFC00" if dd > 0.19 else "#ff6b6b"))
            if k == 0:
                axp.set_ylabel("dF profile\nalong y", fontsize=7.5)
                axp.legend(fontsize=6, loc="upper right")

    ax = fig.add_subplot(gbot[0, 0])
    for c in chans:
        ax.plot(seps, 100 * np.asarray(dip[c]), "o-", lw=1.9, ms=6, label=c)
    ax.axhline(19, color="#ff6b6b", ls="--", lw=1.2,
               label="Rayleigh-like limit (19% dip)")
    ax.invert_xaxis(); ax.set_ylim(-3, 103)
    ax.set_xlabel("lateral separation between the two cells (um)   ->  closer")
    ax.set_ylabel("valley between the two peaks\n(% of the weaker one)",
                  fontsize=8.5)
    ax.set_title("where two cells stop being two cells - note that a soma is "
                 "only ~16 um across, so cells merge long before they touch",
                 fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("LANE B - OVERLAP   identical cells at the focal plane, only "
                 "the separation differs", fontsize=13, y=0.98)
    p = os.path.join(figdir, "11_overlap_ladder.png")
    fig.savefig(p, dpi=135); plt.close(fig)
    return p


def fig_expression(demo, figdir):
    plt = _mpl()
    idx = demo.lane("expression")
    ex = demo.d["expr"][idx]
    chans = [c for c in ("gcamp", "gcamp_crisp") if c in demo.ch]
    has_tdt = "tdt" in demo.ch
    n = len(idx)
    nrow = len(chans) + (1 if has_tdt else 0)
    half = demo.lane_half_um(idx)
    sig = demo.noise_sigma("gcamp")
    fig = plt.figure(figsize=(2.3 * n + 2.0, 2.5 * nrow + 3.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0 * nrow, 1.15],
                             hspace=0.22, top=0.90, bottom=0.06,
                             left=0.07, right=0.985)
    gtop = outer[0].subgridspec(nrow, n, hspace=0.10, wspace=0.08)
    gbot = outer[1].subgridspec(1, 1)

    peaks = {c: [] for c in chans}
    r = 0
    for c in chans:
        crops, exts = [], []
        for i in idx:
            cr, e = demo.crop(demo.dff_image(c, i), i, half)
            crops.append(cr); exts.append(e)
        vmax = float(np.nanpercentile(np.stack(crops), 99.9))
        for k, i in enumerate(idx):
            ax = fig.add_subplot(gtop[r, k])
            _show(ax, crops[k], extent=exts[k], clim=(0, vmax))
            _centre_mark(ax)
            _clean(ax)
            pk, _, _, _ = blob_metrics(crops[k], demo.um_per_px,
                                       min_peak=3 * sig)
            peaks[c].append(pk)
            if r == 0:
                ax.set_title(f"expression x{ex[k]:.2f}", fontsize=9.5)
            if k == 0:
                ax.set_ylabel("GCaMP dF\n" + CH_LABEL[c], fontsize=8.5)
            ax.text(0.04, 0.05, f"pk {pk:,.0f}", transform=ax.transAxes,
                    fontsize=7.5, color="#e6edf3")
        r += 1

    if has_tdt:
        # High-pass the structural image before cropping. A 1P tdTomato mean
        # image is a large smooth neuropil wash with the cells as a small ripple
        # on top; on a shared colour scale the wash alone spans the whole range
        # and every crop reads as a gradient. Removing the large-scale term is a
        # DISPLAY choice (the row says so) and keeps cell-to-cell amplitudes
        # comparable, which is what this row is for.
        from scipy.ndimage import gaussian_filter
        raw = demo.ch["tdt"]["noisy"].mean(0)
        tdt_mean = raw - gaussian_filter(raw, 40.0 / demo.um_per_px)
        crops, exts = [], []
        for i in idx:
            cr, e = demo.crop(tdt_mean, i, half)
            crops.append(cr); exts.append(e)
        vlo, vhi = np.nanpercentile(np.stack(crops), [3, 99.6])
        for k, i in enumerate(idx):
            ax = fig.add_subplot(gtop[r, k])
            _show(ax, crops[k], extent=exts[k], cmap="gist_heat", clim=(vlo, vhi))
            _clean(ax)
            lab = demo.d["tdt_expr"][i]
            ax.text(0.04, 0.05, f"tdT+ x{lab:.2f}" if lab > 0 else "tdT-",
                    transform=ax.transAxes, fontsize=8,
                    color=("#ffe0a0" if lab > 0 else "#9fb3c8"))
            if k == 0:
                ax.set_ylabel("tdTomato (static)\nbackground removed",
                              fontsize=8.5)
        r += 1

    ax = fig.add_subplot(gbot[0, 0])
    for c in chans:
        _plot_with_floor(ax, ex, peaks[c], sig, c)
    ax.axhline(sig, color="#ff6b6b", ls="--", lw=1.2,
               label=f"1 px noise sigma ({sig:.0f} ADU)")
    ax.text(0.02, 0.86, "hollow marker = below the noise floor",
            transform=ax.transAxes, fontsize=6.5, color="#6e7681")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("designed expression multiplier "
                  "(same depth, same everything else)")
    ax.set_ylabel("peak dF (ADU)")
    ax.set_title("the response is linear in expression, so expression changes\n"
                 "nothing about the SHAPE of a cell's image - it only decides "
                 "where you cross the noise", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    fig.suptitle("LANE C - EXPRESSION   identical cells at the focal plane, "
                 "brightness swept", fontsize=13, y=0.98)
    p = os.path.join(figdir, "12_expression_ladder.png")
    fig.savefig(p, dpi=135); plt.close(fig)
    return p


def fig_focus_split(demo, figdir):
    """How much of a frame is the layer you focused on, and how much is fog.

    The scan can render the same movie twice: once from volume planes inside a
    20 um slab around the focal plane, once from everything else. The split is
    exact (``mov_raw == mov_infocus + mov_oof``), so this is a measurement, not
    an estimate.

    Two different questions get two different rows, because they have different
    answers and conflating them would be misleading:

      the PICTURE   the time-mean frame — what a still image of this tissue is
                    made of. Dominated by the neuropil of the entire column, so
                    the out-of-focus share is large. This is the honest "how much
                    of my image is fog" number.
      the TRANSIENT the activity image at the finale. Here the split reflects
                    WHERE THE FIRING CELLS ARE, and in this demo two of the three
                    lanes sit exactly at the focal plane by design — so a high
                    in-focus share is a property of the experiment, not of 1P
                    imaging. Labelled as such.
    """
    if "gcamp" not in demo.ch or "infocus" not in demo.ch["gcamp"]:
        return None
    plt = _mpl()
    c = demo.ch["gcamp"]
    ff = int(demo.d["finale_frame"])
    t = min(ff + 2, demo.nt - 1)
    b0, b1 = max(0, ff - 6), ff - 1

    mean_tot, mean_in = c["clean"].mean(0), c["infocus"].mean(0)
    mean_of = c["oof"].mean(0)
    d_tot = c["clean"][t] - c["clean"][b0:b1].mean(0)
    d_in = c["infocus"][t] - c["infocus"][b0:b1].mean(0)
    d_of = c["oof"][t] - c["oof"][b0:b1].mean(0)

    slab, col = 20.0, demo.depth_um
    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.18,
                          hspace=0.22, top=0.90, bottom=0.06, left=0.05,
                          right=0.98)
    pxs = demo.px(demo.d["xyz"])

    for row, (tot, inf, oof, tag, cmap) in enumerate([
            (mean_tot, mean_in, mean_of, "time-mean frame (the picture)", "gray"),
            (d_tot, d_in, d_of, "activity image at the finale (the transient)",
             "inferno")]):
        vmax = float(np.percentile(tot, 99.7))
        vmin = float(np.percentile(tot, 1.0)) if row == 0 else 0.0
        for j, (img, ttl) in enumerate([
                (tot, f"everything (the whole {col:.0f} um column)"),
                (inf, f"only the {slab:.0f} um in-focus slab"),
                (oof, f"only the out-of-focus {col - slab:.0f} um")]):
            ax = fig.add_subplot(gs[row, j])
            _show(ax, img, cmap=cmap, clim=(vmin, vmax))
            ax.scatter(pxs[:, 0], pxs[:, 1], s=30, facecolors="none",
                       edgecolors="#7CFC00", lw=0.6)
            ax.set_title(ttl, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(tag, fontsize=9)
            _clean(ax)

        ax = fig.add_subplot(gs[row, 3])
        e_in = float(np.clip(inf, 0, None).sum())
        e_of = float(np.clip(oof, 0, None).sum())
        frac = 100 * e_of / max(e_in + e_of, 1e-9)
        ax.bar([0, 1], [100 - frac, frac], color=["#2ee6c8", "#ff8a5c"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"in focus\n({slab:.0f} um slab)",
                            f"out of focus\n(the other {col - slab:.0f} um)"],
                           fontsize=8.5)
        ax.set_ylabel("share of the signal (%)", fontsize=8.5)
        for x, v in ((0, 100 - frac), (1, frac)):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11,
                    color="#e6edf3")
        ax.set_ylim(0, 136); ax.grid(axis="y", alpha=0.3)
        if row == 0:
            ax.axhline(100 * slab / col, color="#9fb3c8", ls=":", lw=1.2)
            ax.text(0.5, 100 * slab / col + 3.0,
                    f"{100*slab/col:.0f}% = the slab's share of the column "
                    f"thickness", fontsize=7, color="#9fb3c8", ha="center")
            ax.text(0.5, 116, "a still image of this tissue is mostly light\n"
                              "from planes you are NOT focused on",
                    fontsize=8.5, ha="center", va="bottom", color="#e6edf3")
        else:
            ax.text(0.5, 116,
                    "for the TRANSIENT this says where the firing cells are -\n"
                    "and two of the three lanes sit at the focal plane by\n"
                    "design, so this number is about the experiment, not\n"
                    "about 1P imaging",
                    fontsize=7.5, ha="center", va="bottom", color="#9fb3c8")

    fig.suptitle("IN FOCUS vs OUT OF FOCUS - the scan's own exact split "
                 "(mov_raw = mov_infocus + mov_oof)", fontsize=12.5, y=0.965)
    p = os.path.join(figdir, "13_focus_split.png")
    fig.savefig(p, dpi=135); plt.close(fig)
    return p


def fig_two_color(demo, figdir):
    if "tdt" not in demo.ch:
        return None
    plt = _mpl()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.20, wspace=0.14)

    from scipy.ndimage import gaussian_filter
    g_act = demo.ch["gcamp"]["clean"].std(0)
    # Same display high-pass as the expression panel's red row: a 1P tdTomato
    # mean image is a large smooth neuropil wash with the cells as a ripple on
    # top, and on a raw scale the wash alone spans the entire colour range.
    # Removing the large-scale term is a DISPLAY choice, stated in the titles.
    t_raw = demo.ch["tdt"]["noisy"].mean(0)
    t_img = t_raw - gaussian_filter(t_raw, 40.0 / demo.um_per_px)
    pxs = demo.px(demo.d["xyz"])
    tdt_pos = demo.d["tdt_expr"] > 0

    ax = fig.add_subplot(gs[0, 0])
    _show(ax, g_act, cmap="magma", clim=tuple(np.percentile(g_act, [1, 99.6])))
    ax.set_title("GCaMP - temporal std (where the activity is)", fontsize=10)
    for i in range(demo.n):
        ax.scatter(pxs[i, 0], pxs[i, 1], s=55, facecolors="none",
                   edgecolors=LANE_COLOR[str(demo.d["group"][i])], lw=1.1)
    _clean(ax)

    ax = fig.add_subplot(gs[0, 1])
    _show(ax, t_img, cmap="gist_heat", clim=tuple(np.percentile(t_img, [2, 99.5])))
    ax.set_title("tdTomato - mean, background removed (structure, no activity)",
                 fontsize=10)
    for i in range(demo.n):
        ax.scatter(pxs[i, 0], pxs[i, 1], s=55, facecolors="none",
                   edgecolors=("#7CFC00" if tdt_pos[i] else "#6e7681"),
                   lw=1.3 if tdt_pos[i] else 0.7)
    _clean(ax)

    ax = fig.add_subplot(gs[0, 2])
    def _n(v):
        lo, hi = np.percentile(v, [2, 99.5])
        return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.zeros(g_act.shape + (3,), dtype=np.float32)
    rgb[..., 1] = _n(g_act); rgb[..., 0] = _n(t_img)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
              interpolation="nearest", aspect="equal")
    ax.set_title("merge - green = activity, red = tdTomato label", fontsize=10)
    _clean(ax)

    idx = demo.lane("overlap")
    r_tight = int(demo.d["rung"][idx][np.argmin(demo.d["sep_um"][idx])])
    a, b = demo.pair_ids(r_tight)
    mid = 0.5 * (demo.d["xyz"][a] + demo.d["xyz"][b])
    # For the zoom, use the PAIR'S OWN activity image rather than the whole-movie
    # temporal std: the std mixes in every other cell that ever fired, and the
    # question here is what these two cells look like when they fire.
    g_pair = pair_images(demo, "gcamp", a, b)["both"]
    for j, (img, cmap, ttl) in enumerate([
            (g_pair, "magma",
             f"closest pair ({demo.d['sep_um'][a]:.0f} um apart) - GCaMP, "
             f"both firing"),
            (t_img, "gist_heat", "same spot - tdTomato"), (None, None, "")]):
        ax = fig.add_subplot(gs[1, j])
        if img is None:
            ax.axis("off")
            ax.text(0.0, 0.95,
                    "The green channel shows ONE blob where there are TWO cells.\n"
                    "The red channel says only one of them carries the label.\n\n"
                    "Assigning the green transient to the red cell is then a guess\n"
                    "- which is exactly the failure a two-colour setup is meant to\n"
                    "resolve, and the reason separation matters far more than the\n"
                    "size of a soma does. Here the correspondence is known by\n"
                    "construction (design.json), so the error is measurable.",
                    transform=ax.transAxes, fontsize=9.5, va="top", color="#e6edf3")
            continue
        cr, ext = demo.crop_at(img, mid, min(52.0, demo.max_half_um(mid)))
        _show(ax, cr, extent=ext, cmap=cmap)
        for cell, mk in ((a, "o"), (b, "s")):
            lab = demo.d["tdt_expr"][cell] > 0
            ax.scatter([demo.d["xyz"][cell, 0] - mid[0]],
                       [demo.d["xyz"][cell, 1] - mid[1]], s=120, marker=mk,
                       facecolors="none",
                       edgecolors=("#7CFC00" if lab else "#e6edf3"), lw=1.6)
        ax.set_title(ttl, fontsize=9.5); _clean(ax)

    fig.suptitle("TWO COLOUR - same volume, same cells, correspondence known "
                 "(green ring = tdTomato+)", fontsize=13, y=0.97)
    p = os.path.join(figdir, "20_two_color.png")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    return p


def fig_scatter_vs_crisp(demo, figdir):
    if "gcamp_crisp" not in demo.ch:
        return None
    plt = _mpl()
    fig = plt.figure(figsize=(15, 10.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.0, 1.0], hspace=0.30,
                          wspace=0.16)

    ff = int(demo.d["finale_frame"])
    t = min(ff + 2, demo.nt - 1)
    base = (max(0, ff - 6), ff - 1)
    imgs = {c: demo.diff(c, t, base) for c in ("gcamp_crisp", "gcamp")}
    vmax = float(np.percentile(imgs["gcamp_crisp"], 99.8))
    pxs = demo.px(demo.d["xyz"])
    for j, (c, ttl) in enumerate([
            ("gcamp_crisp", "NO scatter halo (diffraction limited)"),
            ("gcamp", "WITH scatter halo - what the rig records")]):
        ax = fig.add_subplot(gs[0, j])
        _show(ax, imgs[c], clim=(0, vmax))
        ax.scatter(pxs[:, 0], pxs[:, 1], s=38, facecolors="none",
                   edgecolors="#7CFC00", lw=0.7)
        ax.set_title(ttl + "\n(activity image, finale frame)", fontsize=10)
        _clean(ax)

    ax = fig.add_subplot(gs[0, 2])
    psf = np.load(os.path.join(demo.ch["gcamp"]["dir"], "optics.npz"))["psf"]
    psfc = np.load(os.path.join(demo.ch["gcamp_crisp"]["dir"], "optics.npz"))["psf"]
    zc, kx = psf.shape[2] // 2, psf.shape[0] // 2
    rr = (np.arange(psf.shape[0]) - kx) / demo.vres
    for arr, lab, col in ((psfc, "no halo", "#28c8ff"),
                          (psf, "with halo", "#ff8a5c")):
        pr = arr[:, arr.shape[1] // 2, zc]
        ax.semilogy(rr, np.maximum(pr / pr.max(), 1e-6), color=col, lw=1.8,
                    label=lab)
    ax.set_xlabel("lateral distance (um)"); ax.set_ylabel("PSF (normalised)")
    ax.set_title("the cause: the emission PSF", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(1e-5, 1.5)

    # raw frames: what you actually get before any dF processing
    for j, c in enumerate(("gcamp_crisp", "gcamp")):
        ax = fig.add_subplot(gs[1, j])
        raw = demo.ch[c]["noisy"][t]
        _show(ax, raw, cmap="gray", clim=tuple(np.percentile(raw, [1, 99.5])))
        ax.set_title(f"{c} - the RAW frame (no dF)", fontsize=9.5)
        _clean(ax)
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0.0, 0.95,
            "The raw frames are what the camera writes to disk.\n"
            "Neither of them looks like the activity image above:\n"
            "a bright neuropil wash dominates both, and with the\n"
            "halo on there is essentially no visible cell structure\n"
            "at all.\n\n"
            "This is why 1P analysis starts with a temporal\n"
            "operation (dF, dF/F0, PCA/ICA, a network) rather than\n"
            "with segmenting a picture: the spatial information a\n"
            "single frame carries has already been spread out by\n"
            "the tissue.",
            transform=ax.transAxes, fontsize=9.5, va="top", color="#e6edf3")

    ax = fig.add_subplot(gs[2, :])
    idx = demo.lane("depth")
    row = int(round(demo.px(demo.d["xyz"][idx[:1]])[0, 1]))
    xx = (np.arange(demo.H) + demo.scan_buff / demo.sfrac) * demo.um_per_px
    for c, col in (("gcamp_crisp", "#28c8ff"), ("gcamp", "#ff8a5c")):
        prof = np.nanmean(imgs[c][:, max(0, row - 2):row + 3], axis=1)
        ax.plot(xx, prof, color=col, lw=1.7, label=c)
    for i in idx:
        ax.axvline(demo.d["xyz"][i, 0], color="#7CFC00", lw=0.8, ls=":", alpha=0.6)
        ax.annotate(f"z={demo.d['xyz'][i,2]:.0f}", (demo.d["xyz"][i, 0], 0),
                    xytext=(0, -16), textcoords="offset points", fontsize=7,
                    ha="center", color="#9fb3c8", annotation_clip=False)
    ax.set_xlabel("x (um)  -  a cut along lane A, all depths firing together")
    ax.set_ylabel("dF (ADU)")
    ax.set_title("same cells, same photons: the halo does not remove signal, it "
                 "SPREADS it - peaks drop and the floor between cells rises",
                 fontsize=10.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("TISSUE SCATTER - the only difference between these two runs "
                 "is one PSF parameter", fontsize=13, y=0.985)
    p = os.path.join(figdir, "21_scatter_vs_crisp.png")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    return p


# ======================================================================
# index.html
# ======================================================================
CAPTIONS = [
    ("00_design_map.png", "What was built / 实验设计",
     "Three lanes, one factor each, plus the firing schedule. Every ladder cell "
     "gets its own time slot, so a blob in the movie has exactly one owner and "
     "can be measured without demixing anything."),
    ("01_scene3d_depth.png", "The 3D scene / 三维场景",
     "Somata at their true (x, y, z), coloured by depth; below the block, the "
     "actual camera frames in the same world coordinates. Parallel projection, "
     "so each vertical line really does end on that cell's own pixel."),
    ("02_scene3d_activity.png", "The finale / 全部同时发放",
     "The same scene at the moment every ladder cell fires: the realistic case, "
     "where all depths land on one frame and nothing in the frame tells you "
     "which blob came from where."),
    ("03_orbit.gif", "Rotating / 旋转", "The same scene from every angle."),
    ("04_activity.gif", "Time running / 时间动画",
     "Somata blink with their own dF/F0 and the frame below blinks with them. "
     "Watch a deep cell fire and produce almost nothing."),
    ("10_depth_ladder.png", "Lane A - depth / 深度",
     "Identical cells, only z differs. Amplitude falls off steeply, the blob gets "
     "WIDER, and dF/F0 falls faster still because the out-of-focus haze from the "
     "rest of the column raises F0 as well."),
    ("11_overlap_ladder.png", "Lane B - overlap / 重叠",
     "Identical pairs closing in. Each profile panel shows A alone, B alone and "
     "both together, so you can watch the valley between two peaks fill up. The "
     "summary curve says at what separation two cells stop being two cells."),
    ("12_expression_ladder.png", "Lane C - expression / 表达量",
     "Same depth, brightness swept ~18x. The response is linear in expression, so "
     "expression does not change the shape of anything - it only decides where "
     "you cross the noise floor."),
    ("13_focus_split.png", "In focus vs out of focus / 焦内与焦外",
     "The scan's exact linear split of the same frame. Most of what the camera "
     "records is not the plane you focused on."),
    ("20_two_color.png", "Two colour / 双色",
     "GCaMP and tdTomato of the same cells, with the labelled subset known by "
     "construction. In the closest pair, one green blob covers two cells and only "
     "one of them is red."),
    ("21_scatter_vs_crisp.png", "Tissue scatter / 组织散射",
     "The two GCaMP runs differ in exactly one PSF parameter. The halo does not "
     "remove signal, it spreads it: peaks drop and the floor between cells rises. "
     "The raw frames show why 1P analysis has to start in the time domain."),
]


def write_index(demo, figdir, made):
    have = {os.path.basename(p) for p in made if p}
    meta = demo.ch.get("gcamp", {}).get("meta", {})
    rows = []
    for fn, title, cap in CAPTIONS:
        if fn not in have:
            continue
        rows.append(f"""
  <section>
    <h2>{title}</h2>
    <p>{cap}</p>
    <a href="{fn}"><img src="{fn}" alt="{title}"></a>
  </section>""")
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>calcia - designed ladders - {os.path.basename(demo.root)}</title>
<style>
 body {{ background:#0d1117; color:#e6edf3; font:15px/1.65 -apple-system,
        "Segoe UI","Microsoft YaHei",sans-serif; margin:0 auto; padding:34px;
        max-width:1180px; }}
 h1 {{ font-size:26px; margin:0 0 6px; }}
 h2 {{ font-size:19px; margin:38px 0 6px; color:#7ee0ff; }}
 p  {{ color:#9fb3c8; max-width:82ch; }}
 img {{ width:100%; border:1px solid #21262d; border-radius:8px; }}
 code {{ background:#161b22; padding:1px 6px; border-radius:4px; }}
 .meta {{ color:#6e7681; font-size:13px; }}
</style>
<h1>Designed ladders: depth · overlap · expression</h1>
<p class="meta">{os.path.basename(demo.root)} &nbsp;·&nbsp;
 {demo.vol_um:.0f}×{demo.vol_um:.0f}×{demo.depth_um:.0f} µm,
 focal plane {demo.focal_um:.0f} µm,
 {demo.nt} frames @ {demo.fps:.0f} Hz,
 {demo.um_per_px:g} µm/px,
 {demo.n} ladder cells + {meta.get('n_filler', '?')} filler,
 motion {meta.get('motion_model', '?')},
 GT↔movie alignment ≤ {meta.get('align_resid_px_max', float('nan')):.1f} px</p>
<p>A controlled experiment inside a widefield simulation: three lanes of neurons
 differing in exactly one property each, a firing schedule that gives every cell
 its own moment, the same tissue rendered with and without tissue scatter, and a
 co-registered tdTomato channel. Click any figure for full resolution.</p>
{''.join(rows)}
<h2>Explore it yourself</h2>
<p><code>python examples/viz_ladders3d.py {os.path.basename(demo.root)} --interactive</code>
 opens the 3D scene live — drag to rotate, scroll to zoom, slider to scrub time.</p>
"""
    p = os.path.join(figdir, "index.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p


# ======================================================================
# main
# ======================================================================
def resolve_root(arg):
    """Accept an absolute path, a path relative to the CWD, or a bare run name.

    All three are things a person actually types (the docstring above suggests
    ``examples/output/<run>``, which is relative to the repo root, not to
    ``examples/output``), so try them in that order instead of assuming one.
    """
    if arg:
        for p in (arg, os.path.join(OUTPUT_ROOT, arg)):
            if os.path.isdir(p):
                return os.path.abspath(p)
        raise SystemExit(f"not a directory: {arg}")
    cands = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "ladders3d_*")),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit("no ladders3d_* run found under examples/output/")
    return cands[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Figures + 3D scene for a demo_ladders3d.py run")
    ap.add_argument("run", nargs="?", default=None,
                    help="run directory (default: the most recent ladders3d_*)")
    ap.add_argument("--interactive", action="store_true",
                    help="open the live 3D window instead of writing files "
                         "(SPACE play/pause, arrows step, slider scrubs live)")
    ap.add_argument("--play-fps", type=float, default=None, dest="play_fps",
                    help="playback speed in the live window (default: the "
                         "movie's own frame rate, i.e. real time)")
    ap.add_argument("--paused", action="store_true",
                    help="open the live window paused instead of playing")
    ap.add_argument("--no-3d", action="store_true", dest="no_3d")
    ap.add_argument("--no-anim", action="store_true", dest="no_anim",
                    help="skip the two animated GIFs (the slow part)")
    ap.add_argument("--anim-stride", type=int, default=1, dest="anim_stride")
    args = ap.parse_args(argv)

    root = resolve_root(args.run)
    print(f"run: {root}")
    demo = Demo(root)
    print(f"  {demo.n} ladder cells, channels: {', '.join(demo.ch)}, "
          f"{demo.nt} frames, {demo.H}x{demo.W} px @ {demo.um_per_px:g} um/px")

    if args.interactive:
        interactive(demo, fps=args.play_fps, play=not args.paused)
        return 0

    figdir = os.path.join(root, "figures")
    os.makedirs(figdir, exist_ok=True)
    made = []
    print("[figures]")
    for fn in (fig_design_map, fig_depth, fig_overlap, fig_expression,
               fig_focus_split, fig_two_color, fig_scatter_vs_crisp):
        try:
            p = fn(demo, figdir)
            if p:
                made.append(p)
                print(f"  {os.path.basename(p)}")
        except Exception as e:
            import traceback
            print(f"  FAILED {fn.__name__}: {e}")
            traceback.print_exc()

    if not args.no_3d:
        print("[3D scene]")
        try:
            made += render_3d(demo, figdir, orbit=not args.no_anim,
                              anim=not args.no_anim, anim_stride=args.anim_stride)
        except Exception as e:
            import traceback
            print(f"  FAILED 3D: {e}")
            traceback.print_exc()

    p = write_index(demo, figdir, made)
    print(f"\n{len(made)} figures -> {figdir}")
    print(f"open: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
