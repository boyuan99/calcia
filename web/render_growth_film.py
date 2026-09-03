"""
Cinematic "a neuron grows from nothing" film -- rendered OFFLINE, shipped as video.

WHY THIS EXISTS
    The web showcase must not make a visitor's laptop do 3D work.  So nothing is
    rendered in the browser: this script renders the whole thing here, once, at
    whatever quality we like, and the site plays an H.264/VP9 file.  Video decode
    is hardware-accelerated on every device made in the last decade; a WebGL
    scene with a few thousand tubes is not.

WHAT IS ACTUALLY ON SCREEN
    Not an illustration -- the real generator.  `space_colonization` scatters
    attractor points through the tissue and grows every neuron's tree toward
    them simultaneously; an attractor is consumed by whichever tip reaches it
    first, so the trees partition space by competition alone.  The algorithm
    builds an explicit node/parent forest and then rasterizes it into voxels.
    `capture_growth=True` keeps that forest (calcia/volume/dendrites.py), which
    is what lets us replay growth iteration by iteration:

      * the drifting dust  = the live attractor pool
      * dust vanishing     = attractors being consumed, one growth step at a time
      * branch girth       = Rall's law on the real subtree tip counts
      * the finale flashes = real Poisson burst spikes convolved with calcia's
                             own calcium impulse response

    Node order is a valid growth order for free: nodes are created
    parent-before-child, so `node_parent[i] < i` always holds and every prefix is
    a connected forest.

THIS IS NOT WHAT THE WEBSITE PLAYS
    The site renders the same scene live in the browser from ~330 KB of geometry
    (web/export_growth_web.py + web/site/growth3d.js).  What this script is for
    is the poster the page falls back to without WebGL, plus a real video file
    for places that need one -- a talk, a tweet, a grant figure.

DELIBERATELY NO BURNED-IN TEXT
    Captions belong in HTML on top of the video -- they stay selectable,
    translatable and sharp at any size.

OUTPUTS (under --out, default web/site/assets/growth/)
    growth.mp4 / growth.webm   the film
    growth_poster.jpg          first-paint poster
    turntable/NNN.webp         drag-to-rotate stills of the finished tree
    turntable.mp4              same, as a preview loop
    growth_film.json           duration, fps, beats, scene stats

Run:
    conda run -n calcia python web/render_growth_film.py --smoke
    conda run -n calcia python web/render_growth_film.py
    conda run -n calcia python web/render_growth_film.py --width 1920 --height 1080 --fps 30
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "site", "assets", "growth")

# Restrained, premium palette -- five hues that stay distinct against black
# without reading as a rainbow chart.
PALETTE = [
    (0.31, 0.85, 0.91),   # cyan
    (0.60, 0.49, 1.00),   # violet
    (0.31, 0.89, 0.65),   # mint
    (1.00, 0.71, 0.33),   # amber
    (1.00, 0.42, 0.60),   # rose
]
DUST_COLOR = (0.58, 0.72, 0.98)
AXON_COLOR = (0.30, 0.40, 0.62)

# Beat structure as fractions of total duration.  Overlaps are intentional --
# a cut that lands exactly on a beat boundary reads as a stutter.
BEATS = {
    "dust_in":  (0.000, 0.075),
    "bloom":    (0.070, 0.215),
    "grow":     (0.195, 0.720),
    "axons_in": (0.690, 0.830),
    "pulse":    (0.780, 0.968),
    "fade_out": (0.972, 1.000),
}

AGE_SPAN_ITERS = 5.0     # how many growth iterations a tip stays "hot"
TURNTABLE_SPIN = 360.0
VIEW_ANGLE = 28.0        # slightly long lens; less perspective distortion


# ===================================================================== easing
def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def ease_in_out_sine(x):
    return 0.5 - 0.5 * math.cos(math.pi * clamp01(x))


def beat(t01, name):
    """Local 0..1 progress inside a named beat (0 before it, 1 after it)."""
    a, b = BEATS[name]
    return clamp01((t01 - a) / max(b - a, 1e-9))


# ============================================================== scene assembly
def build_scene(args):
    """Grow a handful of neurons and keep every intermediate we want to draw."""
    from calcia.config.params import VolumeParams, NeuronParams, DendParams
    from calcia.volume.neurons import sample_dense_neurons
    from calcia.volume.neural_volume import generate_neural_volume
    from calcia.volume.dendrites import grow_neuron_dendrites

    np.random.seed(args.seed)
    vol_sz = (args.vol_um, args.vol_um, args.depth_um)
    vp = VolumeParams(vol_sz=vol_sz, vres=args.vres, N_neur=args.n_neurons,
                      min_dist=args.min_dist, verbose=0)
    npar = NeuronParams(neur_type=args.neuron_type, n_samps=args.n_samps)
    # dtParams[1:3] are the horizontal/vertical radii of the dendritic field.
    # The library default (150 um) is a cortical pyramidal spread; at portrait
    # scale it overflows the block and the four trees fuse into one hairball, so
    # the film uses a tighter field where the competition is actually legible.
    base = DendParams()
    dt = list(base.dtParams)
    dt[1], dt[2] = args.field_um, args.field_depth_um
    dp = DendParams(dtParams=tuple(dt), sc_taper="rall", sc_thickness=3,
                    sc_attractors_per_neuron=args.attractors,
                    sc_influence_um=args.influence_um,
                    sc_max_iter=args.max_iter)

    t0 = time.time()
    neurons, angles, positions = sample_dense_neurons(vp, npar, verbose=0)
    # sample_dense_neurons can return fewer than requested when the volume
    # saturates; downstream loops index by vp.N_neur, so re-sync it.
    vp.N_neur = len(neurons)
    vol = generate_neural_volume(neurons, positions, vp, npar, verbose=0)
    dend = grow_neuron_dendrites(vp, dp, vol, positions=positions,
                                 rotation_angles=angles,
                                 strategy="space_colonization",
                                 seed=args.seed, capture_growth=True, verbose=0)
    g = dend.growth_graph
    print(f"[scene] {vp.N_neur} neurons, {len(g.node_pos)} dendrite nodes, "
          f"{g.n_iters} growth iterations, {len(g.attractors)} attractors "
          f"({time.time() - t0:.1f}s)")

    # ---- everything below is in MICRONS; the graph comes in full-res voxels --
    vres = float(g.vres)
    node_pos = g.node_pos / vres
    attractors = g.attractors / vres

    # Continuous Rall radius.  The voxelizer rounds this to whole voxels because
    # it has to splat balls; for rendering we keep the real number, which is
    # both prettier and more faithful than the rounded copy in node_r.
    parent = g.node_parent
    M = len(parent)
    child_count = np.zeros(M)
    np.add.at(child_count, parent[parent >= 0], 1)
    subtips = (child_count == 0).astype(np.float64)
    for i in range(M - 1, -1, -1):          # children precede parents in reverse
        p = parent[i]
        if p >= 0:
            subtips[p] += subtips[i]
    r = subtips ** (1.0 / float(dp.rallexp))
    rmax_per_neuron = np.zeros(int(g.node_nid.max()) + 1)
    np.maximum.at(rmax_per_neuron, g.node_nid, r)
    denom = rmax_per_neuron[g.node_nid]
    denom[denom <= 0] = 1.0
    node_radius = (args.tip_um + (args.trunk_um - args.tip_um)
                   * (r / denom) ** 0.7)

    somata = [(np.asarray(vc, dtype=np.float64), np.asarray(f, dtype=np.int64))
              for (vc, _vn, f) in neurons]

    axons = build_axons(args, vp, vres) if args.axons else []

    # The forest genuinely grows in lockstep (one shared attractor pool is the
    # whole point). Delaying each neuron's REVEAL is a cut, not a change to the
    # simulation -- it just keeps four identical bursts from cancelling out.
    stagger = (np.arange(vp.N_neur) * args.stagger_iters)[g.node_nid - 1]
    node_iter = g.node_iter + stagger
    n_iters = int(g.n_iters + args.stagger_iters * max(vp.N_neur - 1, 0))

    # Frame on the cast, not on the block. The block's bounding sphere is much
    # larger than the tissue the neurons actually occupy, and composing to it
    # leaves the subject small and off-centre.
    centre = node_pos.mean(axis=0)
    radius = 0.95 * float(np.percentile(
        np.linalg.norm(node_pos - centre, axis=1), 99))
    return dict(
        n_neurons=vp.N_neur, vol_sz=vol_sz, centre=centre, radius=radius,
        node_pos=node_pos, node_parent=parent, node_nid=g.node_nid,
        node_iter=node_iter, node_radius=node_radius, n_iters=n_iters,
        attractors=attractors, kill_um=float(dp.sc_kill_um),
        somata=somata, axons=axons,
    )


def build_axons(args, vp, vres):
    """Wispy background processes -- the same random walk the simulator uses."""
    from calcia.algorithms.random_walk import dendrite_random_walk

    shape = tuple(int(s * vres) for s in vp.vol_sz)
    rng = np.random.default_rng(args.seed + 101)
    cost = rng.random(shape).astype(np.float32)
    hi = np.array(shape) - 1
    walks = []
    for _ in range(args.n_axons):
        root = (rng.random(3) * hi).astype(np.int32)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-9
        end = np.clip(root + direction * hi.min() * 0.8, 0, hi).astype(np.int32)
        path = dendrite_random_walk(cost, root, end, distsc=0.5, maxlength=200,
                                    fillweight=0.05, maxel=int(1e7), minlength=25)
        if path is not None and len(path) >= 8:
            walks.append(np.asarray(path, dtype=np.float64) / vres)
    print(f"[scene] {len(walks)} background axon wisps")
    return walks


def build_calcium(args, n_neurons, n_frames, fps):
    """Finale flashes: real burst spikes through calcia's calcium impulse."""
    from calcia.config.params import SpikeParams
    from calcia.traces.spikes import gen_burst_spike_times
    from calcia.traces.calcium import make_calcium_impulse

    np.random.seed(args.seed + 7)
    sp = SpikeParams(K=n_neurons, nt=n_frames, rate=args.fire_rate, dt=1.0 / fps,
                     smod_flag="poisson", burst_mean=2.0, rate_dist="uniform")
    spikes = gen_burst_spike_times(sp)
    h = make_calcium_impulse(0.8, 1.0 / fps)
    ca = np.array([np.convolve(s, h)[:n_frames] for s in spikes])
    peak = ca.max(axis=1, keepdims=True)
    peak[peak <= 0] = 1.0
    return ca / peak


# ==================================================================== geometry
def vtk_faces(faces):
    n = len(faces)
    return np.hstack([np.full((n, 1), 3, dtype=np.int64), faces]).ravel()


def revealed_tree(scene, it_f):
    """The forest as it looked at (fractional) growth iteration `it_f`.

    Nodes born on the current iteration are drawn part-way along their parent
    edge, so branches visibly extend instead of popping in whole.
    """
    node_iter = scene["node_iter"]
    parent = scene["node_parent"]
    cur = int(math.floor(it_f))
    frac = it_f - cur

    done = node_iter < cur
    edge = node_iter == cur
    sel = done | edge
    if not sel.any():
        return None

    pos = scene["node_pos"].copy()
    if edge.any() and frac > 0.0:
        e = np.flatnonzero(edge)
        p = parent[e]
        pos[e] = pos[p] + frac * (pos[e] - pos[p])
    elif edge.any():
        sel = done
        if not sel.any():
            return None

    age = np.clip((it_f - node_iter) / AGE_SPAN_ITERS, 0.0, 1.0)
    return pos, sel, age


def tree_polydata(scene, pos, sel, age, nid_wanted):
    """One neuron's revealed sub-tree as a tube, or None if it has no edges."""
    import pyvista as pv

    parent = scene["node_parent"]
    mine = sel & (scene["node_nid"] == nid_wanted)
    idx = np.flatnonzero(mine)
    if idx.size < 2:
        return None
    remap = np.full(len(parent), -1, dtype=np.int64)
    remap[idx] = np.arange(idx.size)

    has_parent = parent[idx] >= 0
    kids = idx[has_parent]
    pars = parent[kids]
    ok = remap[pars] >= 0
    kids, pars = kids[ok], pars[ok]
    if kids.size == 0:
        return None

    lines = np.column_stack([np.full(kids.size, 2, dtype=np.int64),
                             remap[pars], remap[kids]]).ravel()
    poly = pv.PolyData(pos[idx], lines=lines)
    # Tips are thin and thicken as they mature -- growth you can see.
    maturity = 0.30 + 0.70 * age[idx]
    poly["radius"] = scene["node_radius"][idx] * maturity
    poly["age"] = age[idx]
    return poly.tube(scalars="radius", absolute=True, n_sides=8, capping=True)


def neuron_cmap(base):
    """White-hot growth front cooling into the neuron's own hue."""
    from matplotlib.colors import LinearSegmentedColormap
    base = np.array(base)
    light = np.clip(base * 0.45 + 0.55, 0, 1)
    deep = base * 0.55
    return LinearSegmentedColormap.from_list(
        "front", [(1.0, 1.0, 1.0), tuple(light), tuple(base), tuple(deep)], N=192)


# ====================================================================== camera
def camera_at(scene, t01, spin_deg=None):
    c = scene["centre"]
    R = scene["radius"]
    if spin_deg is None:
        az = math.radians(-40.0 + 205.0 * ease_in_out_sine(t01))
    else:
        az = math.radians(spin_deg)
    el = math.radians(14.0 + 24.0 * smoothstep((t01 - 0.05) / 0.85))
    # Multiples of the block's bounding radius. At a 28-degree lens the whole
    # block only clears the frame past ~3R, so the opening deliberately sits
    # inside the tissue and the film pulls out as the trees fill it.
    dist = float(np.interp(t01, [0.0, 0.20, 0.72, 1.0],
                           [2.40, 2.70, 3.35, 3.10])) * R
    eye = c + dist * np.array([math.cos(el) * math.cos(az),
                               math.cos(el) * math.sin(az),
                               math.sin(el)])
    return [tuple(eye), tuple(c), (0.0, 0.0, 1.0)]


# =============================================================== post-process
def grade(img, strength, exposure=1.55, vignette=0.28):
    """Bloom + lift + vignette.  This is what turns a plot into a film.

    Bloom is computed on a quarter-scale copy: at 1600x900 the wide blur is the
    single most expensive operation in the frame, and nobody can see the
    difference in a glow.
    """
    from scipy.ndimage import gaussian_filter, zoom

    f = img.astype(np.float32) / 255.0
    lum = f @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mask = np.clip((lum - 0.55) / 0.45, 0.0, 1.0)[..., None] * f

    small = mask[::4, ::4]
    glow = np.zeros_like(small)
    for sigma, weight in ((2.0, 0.50), (6.0, 0.32), (16.0, 0.22)):
        glow += weight * gaussian_filter(small, (sigma, sigma, 0))
    zf = (f.shape[0] / small.shape[0], f.shape[1] / small.shape[1], 1)
    glow = zoom(glow, zf, order=1)[:f.shape[0], :f.shape[1]]

    out = f + strength * glow
    # Cool the shadows a touch so black reads as tissue, not as a hole.
    out += (1.0 - np.clip(out, 0, 1)) * np.array([0.008, 0.014, 0.030],
                                                 dtype=np.float32)
    if vignette > 0:
        h, w = out.shape[:2]
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        rad = np.sqrt(xx * xx + yy * yy) / math.sqrt(2.0)
        out *= (1.0 - vignette * rad[..., None] ** 2.2)
    out *= exposure
    out = out / (1.0 + 0.30 * out)          # gentle filmic shoulder
    return np.clip(out * 255.0 * 1.30, 0, 255).astype(np.uint8)


# ====================================================================== render
class Film:
    def __init__(self, args, scene):
        import pyvista as pv
        pv.OFF_SCREEN = True

        self.args = args
        self.scene = scene
        self.pv = pv
        self.pl = pv.Plotter(off_screen=True,
                             window_size=(args.width, args.height))
        self.pl.set_background("black")
        self.pl.enable_anti_aliasing("ssaa")
        if not args.no_depth_peel:
            try:
                self.pl.enable_depth_peeling(number_of_peels=4)
            except Exception as exc:              # pragma: no cover - driver dep
                print(f"[render] depth peeling unavailable ({exc}); continuing")
        self.pl.enable_3_lights()

        self.cmaps = [neuron_cmap(PALETTE[i % len(PALETTE)])
                      for i in range(scene["n_neurons"])]
        self.tree_actors = [None] * scene["n_neurons"]
        self._add_static()

    # -- actors that never change topology -------------------------------
    def _add_static(self):
        pv = self.pv
        sc = self.scene

        if self.args.box:
            lo = np.zeros(3)
            hi = np.array(sc["vol_sz"], dtype=float)
            box = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
            self.box_actor = self.pl.add_mesh(box, style="wireframe",
                                              color=(0.14, 0.20, 0.36),
                                              line_width=1, opacity=0.0,
                                              lighting=False)
        else:
            self.box_actor = None

        self.soma_meshes, self.soma_actors = [], []
        self.shell_meshes, self.shell_actors = [], []
        self.soma_base = []
        for i, (verts, faces) in enumerate(sc["somata"]):
            f = vtk_faces(faces)
            centroid = verts.mean(axis=0)
            self.soma_base.append((verts.copy(), centroid))
            colour = PALETTE[i % len(PALETTE)]

            mesh = pv.PolyData(verts.copy(), f)
            actor = self.pl.add_mesh(mesh, color=colour, smooth_shading=True,
                                     opacity=0.0, ambient=0.30, diffuse=0.80,
                                     specular=0.45, specular_power=28)
            self.soma_meshes.append(mesh)
            self.soma_actors.append(actor)

            shell = pv.PolyData(centroid + 1.16 * (verts - centroid), f)
            sactor = self.pl.add_mesh(shell, color=colour, opacity=0.0,
                                      lighting=False, smooth_shading=True)
            self.shell_meshes.append(shell)
            self.shell_actors.append(sactor)

        if sc["axons"]:
            pts, lines = [], []
            off = 0
            for w in sc["axons"]:
                pts.append(w)
                n = len(w)
                lines.append(np.column_stack([
                    np.full(n - 1, 2, dtype=np.int64),
                    np.arange(off, off + n - 1), np.arange(off + 1, off + n)]))
                off += n
            poly = pv.PolyData(np.vstack(pts),
                               lines=np.vstack(lines).ravel())
            self.axon_actor = self.pl.add_mesh(
                poly, color=AXON_COLOR, line_width=1.4, opacity=0.0,
                lighting=False, render_lines_as_tubes=True)
        else:
            self.axon_actor = None

        self.dust_actor = None
        self.spark_actor = None

    # -- per-frame -------------------------------------------------------
    def set_dust(self, front_pts, opacity):
        """Attractors still unclaimed by anything currently on screen.

        Consumption is re-derived from the revealed geometry rather than replayed
        from the recorded kill iteration, so it stays exact even though the four
        trees are revealed on staggered clocks.
        """
        from scipy.spatial import cKDTree

        sc = self.scene
        if self.dust_actor is not None:
            self.pl.remove_actor(self.dust_actor, render=False)
            self.dust_actor = None
        if opacity <= 0.01:
            return
        attr = sc["attractors"]
        if front_pts is not None and len(front_pts) > 0:
            d, _ = cKDTree(front_pts).query(attr, k=1)
            alive = d > sc["kill_um"]
        else:
            alive = np.ones(len(attr), dtype=bool)
        if not alive.any():
            return
        cloud = self.pv.PolyData(attr[alive])
        self.dust_actor = self.pl.add_mesh(
            cloud, color=DUST_COLOR, point_size=4.6, opacity=opacity,
            render_points_as_spheres=True, lighting=False)

    def set_sparks(self, pts, opacity):
        """The growth front itself -- bright specks the bloom turns into light."""
        if self.spark_actor is not None:
            self.pl.remove_actor(self.spark_actor, render=False)
            self.spark_actor = None
        if opacity <= 0.02 or pts is None or len(pts) == 0:
            return
        self.spark_actor = self.pl.add_mesh(
            self.pv.PolyData(pts), color=(1.0, 1.0, 1.0), point_size=6.0,
            opacity=opacity, render_points_as_spheres=True, lighting=False)

    def set_trees(self, pos, sel, age):
        for i in range(self.scene["n_neurons"]):
            if self.tree_actors[i] is not None:
                self.pl.remove_actor(self.tree_actors[i], render=False)
                self.tree_actors[i] = None
            tube = tree_polydata(self.scene, pos, sel, age, i + 1)
            if tube is None:
                continue
            self.tree_actors[i] = self.pl.add_mesh(
                tube, scalars="age", cmap=self.cmaps[i], clim=(0.0, 1.0),
                show_scalar_bar=False, smooth_shading=True, ambient=0.34,
                diffuse=0.78, specular=0.40, specular_power=26)

    def set_somata(self, scale, opacity, flash):
        for i, (verts, centroid) in enumerate(self.soma_base):
            s = scale[i]
            if s <= 0.001:
                self.soma_actors[i].prop.opacity = 0.0
                self.shell_actors[i].prop.opacity = 0.0
                continue
            self.soma_meshes[i].points = centroid + s * (verts - centroid)
            self.shell_meshes[i].points = (centroid
                                           + s * 1.16 * (verts - centroid))
            f = float(flash[i])
            self.soma_actors[i].prop.opacity = float(opacity[i])
            self.soma_actors[i].prop.ambient = 0.30 + 0.55 * f
            self.shell_actors[i].prop.opacity = float(opacity[i]) * (0.06
                                                                    + 0.42 * f)

    def frame(self, t01, calcium):
        sc = self.scene
        n = sc["n_neurons"]

        # ---- growth ----------------------------------------------------
        gprog = ease_in_out_sine(beat(t01, "grow"))
        it_f = gprog * sc["n_iters"]
        tree = revealed_tree(sc, it_f) if it_f > 0 else None
        if tree is not None:
            pos, sel, age = tree
            self.set_trees(pos, sel, age)
            front = pos[sel & (age < 1.0 / AGE_SPAN_ITERS)]
            revealed = pos[sel]
        else:
            self.set_trees(sc["node_pos"], np.zeros(len(sc["node_pos"]), bool),
                           np.zeros(len(sc["node_pos"])))
            front, revealed = None, None

        growing = beat(t01, "grow")
        self.set_sparks(front, 0.85 * min(1.0, growing * 12.0)
                        * (1.0 - smoothstep((growing - 0.9) / 0.1)))

        # ---- dust ------------------------------------------------------
        dust_op = 0.90 * smoothstep(beat(t01, "dust_in"))
        dust_op *= 1.0 - 0.70 * smoothstep(beat(t01, "grow"))
        self.set_dust(revealed, dust_op)

        # ---- somata ----------------------------------------------------
        bl = beat(t01, "bloom")
        stagger = np.linspace(0.0, 0.55, n)
        local = np.clip((bl - stagger) / max(1e-6, 1.0 - stagger.max()), 0, 1)
        eased = np.array([smoothstep(v) for v in local])
        scale = 0.25 + 0.75 * eased
        opacity = eased

        pulse = beat(t01, "pulse")
        if pulse > 0.0 and calcium is not None:
            k = min(calcium.shape[1] - 1, int(pulse * (calcium.shape[1] - 1)))
            flash = calcium[:, k]
        else:
            flash = np.zeros(n)
        self.set_somata(scale, opacity, flash)

        # ---- ambience --------------------------------------------------
        if self.axon_actor is not None:
            self.axon_actor.prop.opacity = 0.22 * smoothstep(
                beat(t01, "axons_in"))
        if self.box_actor is not None:
            self.box_actor.prop.opacity = 0.055 * smoothstep(
                beat(t01, "dust_in"))

        self.pl.camera_position = camera_at(sc, t01)
        self.pl.camera.view_angle = VIEW_ANGLE
        img = self._capture()

        fade = 1.0 - smoothstep(beat(t01, "fade_out"))
        if fade < 1.0:
            img = (img.astype(np.float32) * fade).astype(np.uint8)
        return img

    def _capture(self):
        """Screenshot, then atmospheric depth fade, then the film grade.

        The depth buffer is free (~5 ms) and buys the single strongest cue the
        renderer otherwise lacks: without it every branch is equally bright and
        the tree reads flat, like a wiring diagram.
        """
        img = self.pl.screenshot(return_img=True)
        if self.args.fog > 0:
            depth = self.pl.get_image_depth()
            dist = -depth.astype(np.float32)
            eye = np.asarray(self.pl.camera_position[0], dtype=np.float64)
            centre_dist = float(np.linalg.norm(eye - self.scene["centre"]))
            near = centre_dist - self.scene["radius"] * 0.85
            far = centre_dist + self.scene["radius"] * 1.05
            t = np.clip((dist - near) / max(far - near, 1e-6), 0.0, 1.0)
            t = np.nan_to_num(t, nan=1.0)
            img = (img.astype(np.float32)
                   * (1.0 - self.args.fog * t[..., None] ** 1.4))
            img = img.astype(np.uint8)
        return grade(img, self.args.bloom, self.args.exposure)

    def still(self, spin_deg, calcium_level=0.0):
        sc = self.scene
        n = sc["n_neurons"]
        pos, sel, age = revealed_tree(sc, float(sc["n_iters"]))
        self.set_trees(pos, sel, age)
        self.set_dust(None, 0.0)
        self.set_sparks(None, 0.0)
        self.set_somata(np.ones(n), np.ones(n), np.full(n, calcium_level))
        if self.axon_actor is not None:
            self.axon_actor.prop.opacity = 0.22
        if self.box_actor is not None:
            self.box_actor.prop.opacity = 0.055
        self.pl.camera_position = camera_at(sc, 0.85, spin_deg=spin_deg)
        self.pl.camera.view_angle = VIEW_ANGLE
        return self._capture()

    def close(self):
        self.pl.close()


# ===================================================================== encode
def run_ffmpeg(cmd, label):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout.decode("utf-8", "replace")[-3000:])
        raise RuntimeError(f"ffmpeg failed for {label}")


def encode_video(frame_glob, out_base, fps, crf_h264=20, crf_vp9=33):
    mp4 = out_base + ".mp4"
    webm = out_base + ".webm"
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps), "-i", frame_glob,
                "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-crf", str(crf_h264), "-preset", "slow",
                "-movflags", "+faststart", mp4], "h264")
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps), "-i", frame_glob,
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", str(crf_vp9),
                "-row-mt", "1", "-pix_fmt", "yuv420p", webm], "vp9")
    return mp4, webm


def downscale_variant(src_mp4, out_base, width):
    """A phone should not download the 1600 px master to fill a 390 px screen."""
    vf = f"scale={width}:-2:flags=lanczos"
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_mp4, "-vf", vf, "-c:v", "libx264", "-profile:v", "high",
                "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow",
                "-movflags", "+faststart", out_base + ".mp4"], "h264 small")
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", src_mp4, "-vf", vf, "-c:v", "libvpx-vp9", "-b:v", "0",
                "-crf", "38", "-row-mt", "1", "-pix_fmt", "yuv420p",
                out_base + ".webm"], "vp9 small")
    return out_base


def write_png(path, img):
    import imageio.v2 as imageio
    imageio.imwrite(path, img)


# ======================================================================= main
def add_scene_args(p):
    """Flags that define WHAT is grown, shared with web/export_growth_web.py.

    The browser renderer and this film must portray the same tissue, so the
    scene description lives in one place and both front ends import it.
    """
    p.add_argument("--seed", type=int, default=11,
                   help="Master seed. Change it for a different-looking cast.")
    p.add_argument("--n-neurons", dest="n_neurons", type=int, default=4,
                   help="Neurons on screen. Keep it small -- this is a portrait.")
    p.add_argument("--neuron-type", dest="neuron_type", default="spherical",
                   choices=["spherical", "pyramidal"],
                   help="Soma morphology. 'spherical' = striatal MSN-like.")
    p.add_argument("--vol-um", dest="vol_um", type=float, default=260.0,
                   help="Lateral extent of the tissue block, microns.")
    p.add_argument("--depth-um", dest="depth_um", type=float, default=150.0,
                   help="Depth of the tissue block, microns (rounds up to 10).")
    p.add_argument("--vres", type=int, default=2,
                   help="Voxels per micron. Only sets the growth lattice here.")
    p.add_argument("--min-dist", dest="min_dist", type=float, default=85.0,
                   help="Minimum soma separation, microns. Bigger = airier.")
    p.add_argument("--n-samps", dest="n_samps", type=int, default=2200,
                   help="Soma surface samples. Below ~2000 the facets show at hero scale; cost is O(n^2), ~1s per neuron at 2200.")
    p.add_argument("--field-um", dest="field_um", type=float, default=70.0,
                   help="Horizontal radius of each dendritic field, microns "
                        "(DendParams.dtParams[1]). The 150 um library default "
                        "fuses the trees at portrait scale.")
    p.add_argument("--field-depth-um", dest="field_depth_um", type=float,
                   default=48.0, help="Vertical radius of the dendritic field.")
    p.add_argument("--attractors", type=int, default=340,
                   help="Attractors per neuron -- the density of the growth field.")
    p.add_argument("--influence-um", dest="influence_um", type=float, default=26.0,
                   help="Radius a tip sees attractors within, microns.")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=150,
                   help="Growth iterations available (growth stops when the "
                        "attractor pool is exhausted, usually before this).")
    p.add_argument("--stagger-iters", dest="stagger_iters", type=int, default=6,
                   help="Reveal delay between neurons, in growth iterations. "
                        "A cut, not a simulation change: the forest really does "
                        "grow in lockstep.")
    p.add_argument("--trunk-um", dest="trunk_um", type=float, default=0.95,
                   help="Rendered radius of a trunk, microns.")
    p.add_argument("--tip-um", dest="tip_um", type=float, default=0.10,
                   help="Rendered radius of a terminal tip, microns.")

    p.add_argument("--axons", dest="axons", action="store_true", default=True,
                   help="Draw faint background axon wisps (default on).")
    p.add_argument("--no-axons", dest="axons", action="store_false",
                   help="Drop the background wisps.")
    p.add_argument("--n-axons", dest="n_axons", type=int, default=90,
                   help="How many background wisps to walk.")
    p.add_argument("--fire-rate", dest="fire_rate", type=float, default=1.6,
                   help="Finale firing rate, Hz, for the calcium flashes.")
    return p


def parse_args():
    p = argparse.ArgumentParser(
        description="Render the cinematic neuron-growth film (poster + social "
                    "export). The live web hero is rendered in the browser -- "
                    "see web/export_growth_web.py.")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output directory for video + turntable + manifest.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny fast render to check the pipeline end to end.")
    add_scene_args(p)

    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=22.0,
                   help="Film duration. Every beat is a fraction of this.")
    p.add_argument("--bloom", type=float, default=0.55,
                   help="Bloom strength. 0 disables the glow entirely.")
    p.add_argument("--fog", type=float, default=0.42,
                   help="Depth fade, 0..1. Reads as tissue depth; 0 disables.")
    p.add_argument("--exposure", type=float, default=1.55,
                   help="Gain before the filmic shoulder. Raise to brighten.")
    p.add_argument("--box", dest="box", action="store_true", default=True,
                   help="Faint wireframe of the tissue block (default on).")
    p.add_argument("--no-box", dest="box", action="store_false")
    p.add_argument("--no-depth-peel", dest="no_depth_peel", action="store_true",
                   help="Skip depth peeling if the driver misbehaves.")
    p.add_argument("--turntable-frames", dest="turntable_frames", type=int,
                   default=72, help="Drag-to-rotate stills. 0 disables.")
    p.add_argument("--turntable-size", dest="turntable_size", type=int,
                   default=1000, help="Square edge of each turntable still, px.")
    p.add_argument("--keep-frames", dest="keep_frames", action="store_true",
                   help="Keep the intermediate PNG frames.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.width, args.height, args.fps = 640, 360, 24
        args.seconds, args.n_samps, args.attractors = 6.0, 350, 160
        args.turntable_frames, args.turntable_size = 12, 400
        args.n_axons, args.vol_um, args.depth_um = 30, 200.0, 90.0
        args.min_dist, args.field_um = 62.0, 55.0

    out_dir = os.path.abspath(args.out)
    frames_dir = os.path.join(out_dir, "frames")
    turn_dir = os.path.join(out_dir, "turntable")
    for d in (out_dir, frames_dir, turn_dir):
        os.makedirs(d, exist_ok=True)
    for stale in glob.glob(os.path.join(frames_dir, "*.png")):
        os.remove(stale)

    print("=" * 68)
    print(f"Neuron growth film -> {out_dir}")
    print(f"  {args.width}x{args.height} @ {args.fps}fps, {args.seconds}s, "
          f"seed {args.seed}")
    print("=" * 68)

    scene = build_scene(args)
    n_frames = int(round(args.seconds * args.fps))
    calcium = build_calcium(args, scene["n_neurons"], n_frames, args.fps)

    film = Film(args, scene)
    t_start = time.time()
    for i in range(n_frames):
        t01 = i / max(n_frames - 1, 1)
        img = film.frame(t01, calcium)
        write_png(os.path.join(frames_dir, f"f{i:05d}.png"), img)
        if i % 30 == 0 or i == n_frames - 1:
            done = i + 1
            rate = done / (time.time() - t_start)
            eta = (n_frames - done) / max(rate, 1e-6)
            print(f"  frame {done}/{n_frames}  {rate:.1f} fps  eta {eta:5.0f}s")
    print(f"[render] film frames in {time.time() - t_start:.0f}s")

    mp4, webm = encode_video(os.path.join(frames_dir, "f%05d.png"),
                             os.path.join(out_dir, "growth"), args.fps)
    small = downscale_variant(mp4, os.path.join(out_dir, "growth_sm"), 960)
    poster_src = os.path.join(frames_dir, f"f{int(n_frames * 0.62):05d}.png")
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", poster_src, "-q:v", "3",
                os.path.join(out_dir, "growth_poster.jpg")], "poster")

    turntable = 0
    if args.turntable_frames > 0:
        turntable = render_turntable(args, scene, turn_dir, out_dir)

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    manifest = {
        "duration_s": args.seconds,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "beats": {k: {"start_s": round(a * args.seconds, 3),
                      "end_s": round(b * args.seconds, 3)}
                  for k, (a, b) in BEATS.items()},
        "scene": {
            "n_neurons": scene["n_neurons"],
            "n_dendrite_nodes": int(len(scene["node_pos"])),
            "n_growth_iterations": int(scene["n_iters"]),
            "n_attractors": int(len(scene["attractors"])),
            "n_axon_wisps": len(scene["axons"]),
            "volume_um": list(scene["vol_sz"]),
            "seed": args.seed,
            "strategy": "space_colonization",
        },
        "turntable": {"frames": turntable, "size": args.turntable_size},
        "files": {"mp4": "growth.mp4", "webm": "growth.webm",
                  "mp4_small": "growth_sm.mp4", "webm_small": "growth_sm.webm",
                  "poster": "growth_poster.jpg",
                  "turntable_dir": "turntable", "turntable_mp4": "turntable.mp4"},
    }
    # NOT growth.json -- that name belongs to web/export_growth_web.py, whose
    # payload the live page loads. This file is only a record of the render.
    with open(os.path.join(out_dir, "growth_film.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    for name in ("growth.mp4", "growth.webm", "growth_sm.mp4", "growth_sm.webm",
                 "growth_poster.jpg", "turntable.mp4"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            print(f"  {name:20s} {os.path.getsize(path) / 1e6:7.2f} MB")
    print(f"Output: {out_dir}")


def render_turntable(args, scene, turn_dir, out_dir):
    """Drag-to-rotate stills: 3D that costs the client one image decode."""
    for stale in glob.glob(os.path.join(turn_dir, "*")):
        os.remove(stale)
    size = args.turntable_size
    args_w, args_h = args.width, args.height
    args.width = args.height = size

    film = Film(args, scene)
    t0 = time.time()
    tmp = os.path.join(out_dir, "_turn_tmp")
    os.makedirs(tmp, exist_ok=True)
    n = args.turntable_frames
    for i in range(n):
        img = film.still(spin_deg=-40.0 + TURNTABLE_SPIN * i / n,
                         calcium_level=0.0)
        write_png(os.path.join(tmp, f"t{i:03d}.png"), img)
    film.close()
    args.width, args.height = args_w, args_h

    # -start_number 0: the image2 muxer counts from 1 by default, and the page
    # indexes frames modulo n, so a 1-based strip 404s on frame 0.
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", os.path.join(tmp, "t%03d.png"), "-vcodec", "libwebp",
                "-lossless", "0", "-q:v", "76", "-compression_level", "6",
                "-start_number", "0",
                os.path.join(turn_dir, "%03d.webp")], "turntable webp")
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", "24", "-i", os.path.join(tmp, "t%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                "-movflags", "+faststart",
                os.path.join(out_dir, "turntable.mp4")], "turntable mp4")
    shutil.rmtree(tmp, ignore_errors=True)

    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(turn_dir, "*")))
    print(f"[turntable] {n} stills, {total / 1e6:.2f} MB total "
          f"({time.time() - t0:.0f}s)")
    return n


if __name__ == "__main__":
    main()
