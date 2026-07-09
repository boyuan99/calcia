"""Backend-agnostic 3D scene.

:class:`Scene3D` operates on a plain ``pyvista.Plotter`` and uses only generic
PyVista API (``add_mesh`` / ``add_points`` / actor scalar updates).  The exact
same class is therefore hosted by:

  * the desktop app  -> a ``pyvistaqt.QtInteractor`` (a Plotter subclass), and
  * the browser app  -> an off-screen ``pyvista.Plotter`` streamed by trame.

That separation is the whole reason the browser path stays open: no Qt symbol
ever leaks into the scene logic.

Per-frame cost is O(N): only the soma point-scalars (dF/F) are updated; no
geometry is rebuilt.  Selecting a neuron lazily adds its soma surface +
dendrite mesh.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyvista as pv

from .geometry import GeometryCache
from .linkage import NeuronTable
from .model import SimRun


class Scene3D:
    def __init__(
        self,
        plotter: pv.Plotter,
        run: SimRun,
        geom: Optional[GeometryCache] = None,
        table: Optional[NeuronTable] = None,
        *,
        dff_clim=(0.0, 2.0),
        soma_point_size: float = 12.0,
        dendrite_mode: str = "points",
    ):
        self.p = plotter
        self.run = run
        self.geom = geom or GeometryCache(run)
        self.table = table or NeuronTable(run)
        self.dff = run.dff()
        self.dff_clim = dff_clim
        self.soma_point_size = soma_point_size
        self.dendrite_mode = dendrite_mode

        self.frame = 0
        self.selected: Optional[int] = None
        self.pick_callback: Optional[Callable[[int], None]] = None

        self._soma_poly: Optional[pv.PolyData] = None
        self._soma_actor = None
        self._soma_mesh_poly: Optional[pv.PolyData] = None
        self._soma_mesh_actor = None
        self._soma_mesh_nid = None
        self._soma_mesh_tree = None          # KDTree over mesh vertices (picking)
        # a pick within this many voxels of a soma-mesh vertex is treated as a
        # direct hit on that neuron's surface (exact nid); further away we fall
        # back to nearest-soma-centre (vessels / dendrites / bare point cloud).
        self.pick_mesh_tol: float = 3.0
        self._vessel_actor = None
        self._volume_actor = None
        self._sel_actors: list = []
        self._highlight_actor = None

    # -------------------------------------------------------- build static
    def build(self, show_vessels: bool = True, vessel_downsample: int = 1):
        """Populate the scene with vessels + soma point cloud."""
        self.p.set_background("black")
        if show_vessels and self.run.neur_ves is not None:
            ves = self.geom.vessels(downsample=vessel_downsample)
            if ves.n_points:
                self._vessel_actor = self.p.add_mesh(
                    ves, color="#c0303a", opacity=0.35, smooth_shading=True,
                    name="vessels", specular=0.3,
                )

        self._soma_poly = self.geom.soma_points()
        self._soma_poly["dff"] = self.dff[:, self.frame].astype(np.float32)
        self._soma_actor = self.p.add_points(
            self._soma_poly, scalars="dff", cmap="hot", clim=self.dff_clim,
            render_points_as_spheres=True, point_size=self.soma_point_size,
            name="somas", show_scalar_bar=True, scalar_bar_args={"title": "dF/F"},
        )
        self.p.add_axes()
        return self

    # -------------------------------------------------------------- frames
    def set_frame(self, t: int, render: bool = True):
        self.frame = int(t)
        # clamp to available trace samples (movie may have more frames than
        # the traces have time points)
        c = min(self.frame, self.dff.shape[1] - 1)
        if self._soma_poly is not None:
            self._soma_poly["dff"][:] = self.dff[:, c].astype(np.float32)
            self._soma_poly.Modified()  # nudge pipeline + redraw
        if self._soma_mesh_poly is not None and self._soma_mesh_nid is not None:
            self._soma_mesh_poly["dff"][:] = self.dff[self._soma_mesh_nid, c]
            self._soma_mesh_poly.Modified()
        if render:
            self.p.render()

    # ------------------------------------------------------- soma layers
    def set_soma_points_visible(self, on: bool):
        if self._soma_actor is not None:
            self._soma_actor.SetVisibility(bool(on))
            self.p.render()

    def attach_soma_mesh(self, mesh=None):
        """Add a prebuilt all-soma mesh to the scene (main-thread / GPU part).

        ``mesh`` defaults to the geometry cache's merged mesh.  The expensive
        marching-cubes *build* is done by :meth:`GeometryCache.all_soma_surfaces`
        (safe to call off the GUI thread); only this ``add_mesh`` must run on the
        main thread.  Returns False if there is nothing to show.
        """
        if self._soma_mesh_actor is not None:
            return True
        if mesh is None:
            mesh = self.geom.all_soma_surfaces()
        if not mesh.n_points:
            return False
        self._soma_mesh_poly = mesh
        self._soma_mesh_nid = np.asarray(mesh["nid"])
        # index the surface vertices so a 3D pick can be mapped straight to the
        # neuron whose soma was clicked (mesh vertices carry 'nid').
        try:
            from scipy.spatial import cKDTree
            self._soma_mesh_tree = cKDTree(np.asarray(mesh.points))
        except Exception:
            self._soma_mesh_tree = None
        c = min(self.frame, self.dff.shape[1] - 1)
        mesh["dff"] = self.dff[self._soma_mesh_nid, c].astype(np.float32)
        self._soma_mesh_actor = self.p.add_mesh(
            mesh, scalars="dff", cmap="hot", clim=self.dff_clim,
            name="soma_mesh", show_scalar_bar=False, smooth_shading=True,
            reset_camera=False)
        self.p.render()
        return True

    def remove_soma_mesh(self):
        if self._soma_mesh_actor is not None:
            self.p.remove_actor("soma_mesh")
            self._soma_mesh_actor = None
            self._soma_mesh_poly = None
            self._soma_mesh_nid = None
            self._soma_mesh_tree = None
            self.p.render()

    def show_all_soma_mesh(self, on: bool):
        """Synchronous toggle (used by the browser/trame backend).

        Builds + attaches (or removes) in one call -- blocks while building, so
        the desktop app uses a background thread + :meth:`attach_soma_mesh`
        instead.  Returns False if there was nothing to build."""
        if on:
            return self.attach_soma_mesh()
        self.remove_soma_mesh()
        return True

    # ------------------------------------------------------------ selection
    def select(self, i: Optional[int], render: bool = True):
        self._clear_selection(render=False)
        self.selected = i
        if i is not None:
            # reset_camera=False everywhere: selecting must never move the view
            soma = self.geom.soma_surface(i)
            if soma.n_points:
                self._sel_actors.append(self.p.add_mesh(
                    soma, color="#39d353", opacity=0.9, name="sel_soma",
                    smooth_shading=True, reset_camera=False))
            dend = self.geom.dendrite(i, mode=self.dendrite_mode)
            if dend.n_points:
                if self.dendrite_mode == "points":
                    self._sel_actors.append(self.p.add_points(
                        dend, color="#7ee787", point_size=4.0,
                        render_points_as_spheres=True, name="sel_dend",
                        reset_camera=False))
                else:
                    self._sel_actors.append(self.p.add_mesh(
                        dend, color="#7ee787", name="sel_dend",
                        reset_camera=False))
            # ring highlight on the selected soma glyph
            loc = self.run.soma_grid_locs()[i]
            self._highlight_actor = self.p.add_points(
                pv.PolyData(loc[None, :].astype(np.float32)),
                color="cyan", point_size=self.soma_point_size + 10,
                render_points_as_spheres=True, name="sel_ring", opacity=0.6,
                reset_camera=False)
            self._sel_actors.append(self._highlight_actor)
        if render:
            self.p.render()

    def _clear_selection(self, render: bool = True):
        for name in ("sel_soma", "sel_dend", "sel_ring"):
            try:
                self.p.remove_actor(name)
            except Exception:
                pass
        self._sel_actors.clear()
        self._highlight_actor = None
        if render:
            self.p.render()

    # ------------------------------------------------------ optional layers
    def set_vessels_visible(self, on: bool):
        if self._vessel_actor is not None:
            self._vessel_actor.SetVisibility(bool(on))
            self.p.render()

    def show_volume(self, on: bool, downsample: int = 4):
        """Optional dense fluorescence 'fog' (GPU volume, LOD via downsample)."""
        if on and self._volume_actor is None and self.run.neur_vol is not None:
            vol = self.run.neur_vol
            if downsample > 1:
                vol = vol[::downsample, ::downsample, ::downsample]
            grid = pv.ImageData(dimensions=np.array(vol.shape) + 1)
            grid.spacing = (downsample, downsample, downsample)
            grid.cell_data["f"] = vol.flatten(order="F")
            self._volume_actor = self.p.add_volume(
                grid, scalars="f", cmap="bone", opacity="linear", name="fog",
                reset_camera=False)
        elif not on and self._volume_actor is not None:
            self.p.remove_actor("fog")
            self._volume_actor = None
        self.p.render()

    # ------------------------------------------------------------- picking
    def _pick_neuron(self, point) -> int:
        """Map a picked 3D world point to a neuron index.

        When the soma **mesh** layer is on, a pick that lands on a soma surface
        is resolved to *that* neuron exactly by reading the ``nid`` off the
        nearest surface vertex -- so clicking a blob selects the blob you
        clicked, not merely the neuron whose centre happens to be closest to the
        surface point.  Picks that miss the mesh (bare point cloud, vessels,
        dendrites) fall back to the nearest soma centre.
        """
        p = np.asarray(point, dtype=np.float32)[:3]
        tree = self._soma_mesh_tree
        if tree is not None and self._soma_mesh_nid is not None:
            d, j = tree.query(p)
            if d <= self.pick_mesh_tol:
                return int(self._soma_mesh_nid[int(j)])
        return self.table.pick_from_3d(p)

    def enable_picking(self):
        """Wire 3D picking -> neuron index -> pick_callback.

        Uses point picking to find the frontmost surface point under the cursor,
        then :meth:`_pick_neuron` turns it into a neuron id (exact when the soma
        mesh is shown; nearest-centre otherwise).
        """
        def _cb(point, *args):
            if point is None:
                return
            i = self._pick_neuron(point)
            if self.pick_callback:
                self.pick_callback(i)
            else:
                self.select(i)
        try:
            self.p.enable_point_picking(
                callback=_cb, use_picker="point", show_message=False,
                show_point=False)
        except TypeError:  # older/newer signature fallback
            self.p.enable_point_picking(callback=_cb, show_message=False)

    # -------------------------------------------------------------- camera
    def reset_view(self, render: bool = True):
        """Restore the default isometric perspective framing the whole scene."""
        self.p.disable_parallel_projection()
        self.p.camera_position = "iso"
        self.p.reset_camera()
        if render:
            self.p.render()

    def focus_on(self, i: Optional[int], radius: float = 120.0,
                 render: bool = True):
        """Recentre the camera on neuron ``i`` (keeps the current view angle).

        Frames a ``radius``-voxel box around the soma so the selected cell fills
        the view, without otherwise reorienting the scene -- works the same in
        the isometric and the top-down (:meth:`sync_top_down`) views.
        """
        if i is None:
            return
        loc = np.asarray(self.run.soma_grid_locs()[i], dtype=float)
        gx, gy, gz = self.run.grid_shape
        r = float(radius)
        bounds = (max(loc[0] - r, 0.0), min(loc[0] + r, float(gx)),
                  max(loc[1] - r, 0.0), min(loc[1] + r, float(gy)),
                  max(loc[2] - r, 0.0), min(loc[2] + r, float(gz)))
        self.p.reset_camera(bounds=bounds)
        if render:
            self.p.render()

    def sync_top_down(self, render: bool = True):
        """Look straight down z, oriented + zoomed to match the 2D movie.

        grid-x runs *down* the screen and grid-y runs *right* -- the movie
        panel's convention (row down, col right) -- and parallel projection
        frames the full ``(gx, gy)`` field, so a soma sits at the same relative
        screen position in the 3D and 2D views at the same scale.
        """
        gx, gy, gz = self.run.grid_shape
        cam = self.p.camera
        cx, cy, cz = gx / 2.0, gy / 2.0, gz / 2.0
        cam.focal_point = (cx, cy, cz)
        cam.position = (cx, cy, cz + max(gx, gy))   # above, looking down -z
        cam.up = (-1.0, 0.0, 0.0)                    # grid +x -> screen down
        self.p.enable_parallel_projection()          # orthographic, like the movie
        self.p.reset_camera(bounds=(0.0, float(gx), 0.0, float(gy),
                                    0.0, float(gz)))
        if render:
            self.p.render()
