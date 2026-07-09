"""Browser app (the path we kept open): the SAME Scene3D, served by trame.

    python -m calcia.viz.app_trame <run_dir> [--port 8080]

This reuses :class:`~calcia.viz.scene3d.Scene3D` unchanged on an off-screen
``pyvista.Plotter`` and streams it to the browser with trame's PyVista widget.
A frame slider and a "select neuron" control drive the identical shared state
used by the desktop app -- demonstrating that no rewrite is needed to go from
desktop to web.

Server-side VTK rendering (streamed pixels/geometry) keeps VTK's efficiency
while delivering to a browser, which is exactly the trade-off discussed: the
multi-GB data never has to be shipped to the client.
"""

from __future__ import annotations

import argparse
import os

import pyvista as pv

from . import model, runs
from .geometry import GeometryCache
from .linkage import NeuronTable
from .scene3d import Scene3D


def build_app(run_dir=None, vessel_ds=1, dendrites="points"):
    from trame.app import get_server
    from trame.ui.vuetify import SinglePageLayout
    from trame.widgets import vuetify, vtk as vtk_widgets

    pv.OFF_SCREEN = True
    all_runs = runs.discover_runs()
    run_dir = runs.resolve(run_dir)
    if not run_dir:
        raise SystemExit("no runs found under " + runs.default_root())
    # normalise so it compares equal to discover_runs()' absolute item values
    run_dir = os.path.abspath(run_dir)

    plotter = pv.Plotter(off_screen=True)
    cur = {}  # holds the live scene/run so the reload closure can swap them

    def load(path):
        """(Re)build the scene on the SAME plotter -> ren_win stays valid."""
        run = model.load(path, load_vessels=True)
        plotter.clear()
        table = NeuronTable(run)
        geom = GeometryCache(run)
        scene = Scene3D(plotter, run, geom, table, dendrite_mode=dendrites)
        scene.build(show_vessels=True, vessel_downsample=vessel_ds)
        cur.update(run=run, scene=scene)
        return run

    try:
        load(run_dir)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"could not load run {run_dir!r}: "
                         f"{type(exc).__name__}: {exc}")

    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller
    state.frame = 0
    state.neuron = 0
    state.run_path = cur["run"].run_dir   # absolute, matches item values
    state.frame_max = cur["run"].nt - 1
    state.load_error = ""
    state.soma_pts = True
    state.soma_mesh = False
    state.run_items = [{"text": r.summary(), "value": r.path} for r in all_runs]

    @state.change("soma_pts")
    def _on_soma_pts(soma_pts, **_):
        cur["scene"].set_soma_points_visible(bool(soma_pts))
        ctrl.view_update()

    @state.change("soma_mesh")
    def _on_soma_mesh(soma_mesh, **_):
        cur["scene"].show_all_soma_mesh(bool(soma_mesh))
        ctrl.view_update()

    @state.change("run_path")
    def _on_run(run_path, **_):
        if not run_path or run_path == cur["run"].run_dir:
            return
        try:
            run = load(run_path)
        except Exception as exc:  # keep the old scene, tell the user, roll back
            state.load_error = f"{type(exc).__name__}: {exc}"
            state.run_path = cur["run"].run_dir
            return
        state.load_error = ""
        state.frame = 0
        state.neuron = 0
        state.frame_max = run.nt - 1
        ctrl.view_reset_camera()
        ctrl.view_update()

    @state.change("frame")
    def _on_frame(frame, **_):
        cur["scene"].set_frame(int(frame), render=False)
        ctrl.view_update()

    @state.change("neuron")
    def _on_neuron(neuron, **_):
        try:
            cur["scene"].select(int(neuron), render=False)
        except Exception:
            pass
        ctrl.view_update()

    with SinglePageLayout(server) as layout:
        layout.title.set_text("calcia viz (browser)")
        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtk_widgets.VtkRemoteView(plotter.ren_win)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera
        with layout.toolbar:
            vuetify.VSelect(v_model=("run_path",), items=("run_items",),
                            label="run", hide_details=True, dense=True,
                            style="max-width: 320px")
            vuetify.VChip("{{ load_error }}", v_if=("load_error",),
                          color="red", text_color="white", small=True,
                          classes="ml-2")
            vuetify.VSpacer()
            vuetify.VSwitch(v_model=("soma_pts",), label="soma pts",
                            hide_details=True, dense=True, classes="mr-3")
            vuetify.VSwitch(v_model=("soma_mesh",), label="soma mesh",
                            hide_details=True, dense=True, classes="mr-3")
            vuetify.VSlider(v_model=("frame", 0), min=0, max=("frame_max", 1),
                            step=1, label="frame", hide_details=True,
                            style="max-width: 260px")
            vuetify.VTextField(v_model=("neuron", 0), label="neuron #",
                               type="number", hide_details=True,
                               style="max-width: 110px")
    return server


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run directory; omit to use the most recent run")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--vessel-ds", type=int, default=1)
    ap.add_argument("--dendrites", choices=["points", "tube"], default="points")
    args = ap.parse_args(argv)
    server = build_app(args.run_dir, args.vessel_ds, args.dendrites)
    server.start(port=args.port)


if __name__ == "__main__":
    main()
