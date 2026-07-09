"""Desktop app: 3D geometry + 2D movie + calcium traces, fully linked.

    python -m calcia.viz.app_qt [run_dir] [--no-vessels] [--vessel-ds N]
                                            [--dendrites points|tube]

``run_dir`` is optional: launch with no argument and an in-program run picker
lists every viewable run under ``examples/output`` (plus a "Browse..." button
for any folder).  Once open, **File -> Open run...** switches runs at runtime.

The three views share exactly two pieces of state -- ``current_frame`` and
``selected_neuron`` -- carried by :class:`SharedState` Qt signals.  Any view can
set them; all views react.  The 3D scene is the backend-agnostic
:class:`~calcia.viz.scene3d.Scene3D`, hosted here by a ``pyvistaqt``
interactor; the identical scene can instead be served to a browser by
``app_trame`` (nothing Qt leaks into ``Scene3D``).
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

from . import model, runs
from .geometry import GeometryCache
from .linkage import NeuronTable
from .panels import MoviePanel, TracesPanel
from .scene3d import Scene3D

# keep top-level windows alive across run switches (prevent GC)
_WINDOWS: list = []


def default_opts(**over):
    """Default rendering options; overridable by CLI or the picker."""
    opts = SimpleNamespace(no_vessels=False, vessel_ds=1,
                           dendrites="points", load_volume=False)
    opts.__dict__.update(over)
    return opts


class RunPicker(QDialog):
    """In-program run chooser: lists viewable runs + a Browse button.

    Returns the chosen run directory via :meth:`selected_path` after ``exec``.
    """

    COLS = ["run", "region", "kind", "neurons", "frames"]

    def __init__(self, root=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open simulation run")
        self.resize(720, 420)
        self._root = root or runs.default_root()
        self._runs = []
        self._path = None

        lay = QVBoxLayout(self)
        self.root_lbl = QLabel()
        lay.addWidget(self.root_lbl)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self.accept)
        lay.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        browse = QPushButton("Browse folder...")
        browse.clicked.connect(self._browse)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.refresh)
        btn_row.addWidget(browse)
        btn_row.addWidget(rescan)
        btn_row.addStretch(1)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        btn_row.addWidget(self.buttons)
        lay.addLayout(btn_row)

        self.refresh()

    def refresh(self):
        self.root_lbl.setText(f"runs under: {self._root}")
        self._runs = runs.discover_runs(self._root)
        self.table.setRowCount(len(self._runs))
        for r, info in enumerate(self._runs):
            vals = [info.name, info.region or "?",
                    "smoke" if info.smoke else "run",
                    str(info.n_neur), str(info.nt)]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        if self._runs:
            self.table.selectRow(0)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select a run folder",
                                             self._root)
        if d:
            if runs.is_run_dir(d):
                self._path = d
                self.accept()
            else:
                QMessageBox.warning(
                    self, "Not a run",
                    "That folder is missing required files "
                    "(metadata.json / traces.npz / movies.npz / "
                    "cell_footprints.pkl).")

    def selected_path(self):
        if self._path:
            return self._path
        row = self.table.currentRow()
        if 0 <= row < len(self._runs):
            return self._runs[row].path
        return None


class _SomaMeshWorker(QThread):
    """Builds the all-soma mesh off the GUI thread (compute + disk cache).

    Only the marching-cubes build / file read happens here; the ``add_mesh``
    (GPU) call is done back on the main thread in the ``done`` slot, so the
    window stays responsive instead of freezing during a long build.
    """
    done = Signal(object)

    def __init__(self, geom, parent=None):
        super().__init__(parent)
        self.geom = geom

    def run(self):
        try:
            mesh = self.geom.all_soma_surfaces()
        except Exception:
            mesh = None
        self.done.emit(mesh)


class SharedState(QObject):
    """The two variables every view subscribes to."""
    frameChanged = Signal(int)
    neuronSelected = Signal(int)

    def __init__(self, nt: int):
        super().__init__()
        self.nt = nt
        self._frame = 0
        self._neuron = None
        self._locked = False

    def set_frame(self, t: int):
        t = max(0, min(int(t), self.nt - 1))
        if t != self._frame:
            self._frame = t
            self.frameChanged.emit(t)

    def set_locked(self, on: bool):
        """When locked, new picks are ignored so the selection stays put."""
        self._locked = bool(on)

    def set_neuron(self, i):
        if self._locked:                 # selection frozen -> ignore new picks
            return
        if i != self._neuron:
            self._neuron = i
            self.neuronSelected.emit(-1 if i is None else int(i))


class MainWindow(QMainWindow):
    def __init__(self, run, opts):
        super().__init__()
        self.run = run
        self.opts = opts
        self.setWindowTitle(f"calcia viz — {run.metadata.get('tag', '')} "
                            f"[{run.n_neur} neurons, {run.nt} frames]")
        self.state = SharedState(run.nt)

        self.table = NeuronTable(run)
        self.geom = GeometryCache(run)

        # --- 3D view (pyvistaqt) -------------------------------------------
        # The QtInteractor holds a VTK render window / GL context registered in
        # pyvista's global _ALL_PLOTTERS; if the build fails after this point we
        # must close it explicitly or it leaks (accumulating across retries).
        self.interactor = QtInteractor(self)
        try:
            self.scene = Scene3D(self.interactor, run, self.geom, self.table,
                                 dendrite_mode=opts.dendrites)
            self.scene.build(show_vessels=not opts.no_vessels,
                             vessel_downsample=opts.vessel_ds)
            self.scene.pick_callback = self.state.set_neuron
            self.scene.enable_picking()

            # --- 2D + traces ------------------------------------------------
            self.movie = MoviePanel(run, self.table)
            self.traces = TracesPanel(run)
        except Exception:
            self.interactor.close()  # finalize the orphaned render window
            raise

        # menu bar + top toolbar (options live at the top of the app)
        self._build_menu()
        self._build_toolbar()

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.movie)
        right.addWidget(self.traces)
        right.setSizes([500, 300])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.interactor)
        split.addWidget(right)
        split.setSizes([900, 600])

        self.setCentralWidget(split)

        # --- wiring: shared state -> views ---------------------------------
        self.state.frameChanged.connect(self._on_frame)
        self.state.neuronSelected.connect(self._on_neuron)
        self.movie.neuronPicked.connect(self.state.set_neuron)
        self.traces.neuronPicked.connect(self.state.set_neuron)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # --------------------------------------------------------------- menu
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act_open = QAction("&Open run...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_run_dialog)
        m.addAction(act_open)
        act_reload = QAction("&Reload", self)
        act_reload.setShortcut("Ctrl+R")
        act_reload.triggered.connect(lambda: self._switch_to(self.run.run_dir))
        m.addAction(act_reload)
        m.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)

        # --- View: display-layer toggles ----------------------------------
        v = self.menuBar().addMenu("&View")
        self.act_vessels = self._toggle_action(
            v, "Blood &vessels", self.scene._vessel_actor is not None,
            self.scene.set_vessels_visible)
        self.act_pts = self._toggle_action(
            v, "Soma &points", True, self.scene.set_soma_points_visible)
        self.mesh_act = self._toggle_action(
            v, "Soma &mesh (3D)", False, self._toggle_soma_mesh)
        v.addSeparator()
        self.act_allcells = self._toggle_action(
            v, "All cell positions (2D)", False,
            self.movie.set_show_all_positions)
        self.act_outlines = self._toggle_action(
            v, "Soma &outlines (2D)", False, self._toggle_outlines)
        self.act_stab = self._toggle_action(
            v, "&Stabilize movie (align to cells)", False, self._toggle_stabilize)
        if self.run.neur_vol is not None:
            v.addSeparator()
            self._toggle_action(v, "&Fog (dense volume)", False,
                                lambda on: self.scene.show_volume(on))

    def _toggle_action(self, menu, label, checked, slot):
        """A checkable menu action wired to ``slot(bool)`` (no fire on init)."""
        act = QAction(label, self, checkable=True)
        act.setChecked(checked)
        act.toggled.connect(slot)      # connect AFTER setChecked -> no init fire
        menu.addAction(act)
        return act

    def _open_run_dialog(self):
        dlg = RunPicker(root=os.path.dirname(os.path.normpath(self.run.run_dir)),
                        parent=self)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            path = dlg.selected_path()
            if path and os.path.normpath(path) != os.path.normpath(self.run.run_dir):
                self._switch_to(path)

    def _switch_to(self, run_dir):
        """Open ``run_dir`` in a fresh window and close this one.

        A new window (rather than in-place rebuild) sidesteps partial actor
        teardown / signal-rewire leaks in the pyvistaqt interactor.
        """
        try:
            win = open_run_window(run_dir, self.opts)
        except Exception as exc:  # keep current window on failure
            QMessageBox.critical(self, "Load failed", f"{run_dir}\n\n{exc}")
            return
        win.resize(self.size())
        win.move(self.pos())
        self.close()

    def closeEvent(self, event):
        self._timer.stop()
        try:
            self.interactor.close()  # finalize VTK render window / GL context
        except Exception:
            pass
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        super().closeEvent(event)

    # ------------------------------------------------------------- toolbar
    def _build_toolbar(self):
        """Top toolbar: playback + movie source + outline width.

        Display-layer toggles live in the View menu; this row holds the
        continuous controls."""
        tb = QToolBar("Playback")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        tb.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.run.nt - 1)
        self.slider.setMinimumWidth(320)
        self.slider.valueChanged.connect(self.state.set_frame)
        tb.addWidget(self.slider)

        self.frame_lbl = QLabel(" 0 ")
        self.frame_lbl.setMinimumWidth(48)
        tb.addWidget(self.frame_lbl)

        tb.addSeparator()
        tb.addWidget(QLabel(" movie: "))
        src = QComboBox(); src.addItems(["clean", "noisy"])
        src.currentTextChanged.connect(self.movie.set_source)
        tb.addWidget(src)

        tb.addSeparator()
        tb.addWidget(QLabel(" outline width: "))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.5, 8.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(1.5)
        self.width_spin.valueChanged.connect(self.movie.set_outline_width)
        tb.addWidget(self.width_spin)

        # --- selection lock + 3D camera controls ---------------------------
        tb.addSeparator()
        self.lock_btn = QPushButton("Lock")
        self.lock_btn.setCheckable(True)
        self.lock_btn.setToolTip("Freeze the current selection (ignore new picks)")
        self.lock_btn.toggled.connect(self._toggle_lock)
        tb.addWidget(self.lock_btn)

        self.locate_btn = QPushButton("Locate")
        self.locate_btn.setToolTip("Centre the 3D camera on the selected neuron")
        self.locate_btn.clicked.connect(self._locate_selected)
        tb.addWidget(self.locate_btn)

        self.sync_btn = QPushButton("Sync view (top-down)")
        self.sync_btn.setToolTip(
            "Match the 2D movie: look straight down, same orientation and zoom")
        self.sync_btn.clicked.connect(lambda: self.scene.sync_top_down())
        tb.addWidget(self.sync_btn)

        self.reset_view_btn = QPushButton("Reset view")
        self.reset_view_btn.setToolTip("Restore the default 3D isometric view")
        self.reset_view_btn.clicked.connect(lambda: self.scene.reset_view())
        tb.addWidget(self.reset_view_btn)

    def _busy(self, fn, *a):
        """Run a possibly-slow toggle under a wait cursor."""
        from qtpy.QtCore import Qt as _Qt
        from qtpy.QtGui import QCursor
        QApplication.setOverrideCursor(QCursor(_Qt.WaitCursor))
        try:
            fn(*a)
        finally:
            QApplication.restoreOverrideCursor()

    def _toggle_soma_mesh(self, on):
        if not on:
            self.scene.remove_soma_mesh()
            return
        # already in memory -> attach instantly on the main thread
        if getattr(self.geom, "_all_soma", None) is not None:
            self._busy(self.scene.attach_soma_mesh)
            return
        # otherwise build/load off the GUI thread so the window stays alive
        self.mesh_act.setEnabled(False)
        self.mesh_act.setText("Soma mesh (building…)")
        self._mesh_worker = _SomaMeshWorker(self.geom, self)
        self._mesh_worker.done.connect(self._on_soma_mesh_built)
        self._mesh_worker.start()

    def _on_soma_mesh_built(self, mesh):
        self.mesh_act.setEnabled(True)
        self.mesh_act.setText("Soma &mesh (3D)")
        if mesh is None or not getattr(mesh, "n_points", 0):
            self.mesh_act.blockSignals(True)
            self.mesh_act.setChecked(False)
            self.mesh_act.blockSignals(False)
            return
        if self.mesh_act.isChecked():       # user may have unticked while building
            self.scene.attach_soma_mesh(mesh)

    def _toggle_outlines(self, on):
        self._busy(self.movie.set_show_all_outlines, on)

    def _toggle_stabilize(self, on):
        # first enable measures per-frame shifts (phase correlation) -> may pause
        self._busy(self.movie.set_stabilize, on)

    # ------------------------------------------------- lock / camera controls
    def _toggle_lock(self, on):
        self.state.set_locked(on)
        self.lock_btn.setText("Locked" if on else "Lock")

    def _locate_selected(self):
        """Centre the 3D camera on the currently selected neuron (if any)."""
        i = self.state._neuron
        if i is None:
            self.statusBar().showMessage("Locate: no neuron selected", 2000)
            return
        self.scene.focus_on(int(i))

    # -------------------------------------------------------------- reactions
    def _on_frame(self, t):
        self.frame_lbl.setText(str(t))
        if self.slider.value() != t:
            self.slider.blockSignals(True); self.slider.setValue(t)
            self.slider.blockSignals(False)
        self.scene.set_frame(t)
        self.movie.set_frame(t)
        self.traces.set_frame(t)

    def _on_neuron(self, i):
        sel = None if i < 0 else int(i)
        self.scene.select(sel)
        self.movie.set_selected(sel)
        self.traces.set_selected(sel)

    # ------------------------------------------------------------- playback
    def _toggle_play(self):
        if self._timer.isActive():
            self._timer.stop(); self.play_btn.setText("Play")
        else:
            self._timer.start(max(20, int(self.run.dt * 1000)))
            self.play_btn.setText("Pause")

    def _advance(self):
        self.state.set_frame((self.state._frame + 1) % self.run.nt)


def open_run_window(run_dir, opts, size=(1600, 950)):
    """Load ``run_dir`` and show a :class:`MainWindow` for it.

    Registered in ``_WINDOWS`` so it survives run switches without being GC'd.
    A ``QApplication`` must already exist.
    """
    run = model.load(run_dir, load_vessels=not opts.no_vessels,
                     load_volume=opts.load_volume)
    win = MainWindow(run, opts)
    win.resize(*size)
    _WINDOWS.append(win)
    win.show()
    win.raise_()
    win.activateWindow()
    return win


def _pick_run(parent=None):
    dlg = RunPicker(parent=parent)
    if not (dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()):
        return None
    return dlg.selected_path()


def run_app(run_dir=None, opts=None):
    """Launch the app.  If ``run_dir`` is None, show the run picker first.

    A load failure (bad path, corrupt/old-format run) shows an error and falls
    back to the picker instead of crashing with a traceback.
    """
    opts = opts or default_opts()
    app = QApplication.instance() or QApplication(sys.argv)

    run_dir = runs.resolve(run_dir) if run_dir else _pick_run()
    while run_dir:
        try:
            open_run_window(run_dir, opts)
            break
        except Exception as exc:
            QMessageBox.critical(None, "Could not open run",
                                 f"{run_dir}\n\n{type(exc).__name__}: {exc}")
            run_dir = _pick_run()  # let the user choose another
    else:
        return 0  # cancelled / nothing to open

    return app.exec_() if hasattr(app, "exec_") else app.exec()


def main(argv=None):
    ap = argparse.ArgumentParser(description="calcia simulation visualizer")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run directory; omit to pick one in-program")
    ap.add_argument("--no-vessels", action="store_true")
    ap.add_argument("--vessel-ds", type=int, default=1,
                    help="downsample factor for vessel marching cubes")
    ap.add_argument("--dendrites", choices=["points", "tube"], default="points")
    ap.add_argument("--load-volume", action="store_true",
                    help="also load dense volume for the optional fog layer")
    args = ap.parse_args(argv)
    opts = default_opts(no_vessels=args.no_vessels, vessel_ds=args.vessel_ds,
                        dendrites=args.dendrites, load_volume=args.load_volume)
    return run_app(args.run_dir, opts)


if __name__ == "__main__":
    sys.exit(main())
