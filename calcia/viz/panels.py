"""Qt side-panels: the 2D movie view and the calcium-trace plot.

Both are thin Qt wrappers; they hold no application state of their own beyond
what they display.  They *emit* ``neuronPicked(int)`` when the user clicks, and
expose ``set_frame`` / ``set_selected`` slots driven by the shared state.

Qt is imported through ``qtpy`` so the binding (PyQt5 / PySide6) is chosen by
the ``QT_API`` environment variable rather than hard-coded.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import pyqtgraph as pg  # noqa: E402

pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")


# ============================================================== movie panel ==
class MoviePanel(QWidget):
    """2D final-image view (pyqtgraph) with a selected-neuron marker."""

    neuronPicked = Signal(int)

    # qualitative palette so neighbouring soma outlines get distinct colours
    OUTLINE_PALETTE = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
        "#9a6324", "#e0ac00", "#f7a1c4", "#aaffc3",
    ]

    def __init__(self, run, table, parent=None):
        super().__init__(parent)
        self.run = run
        self.table = table
        self._source = "clean"
        self._frame = 0
        self._selected = None

        self._show_all = False
        self._outlines_built = False
        self._outline_width = 1.5
        self._stabilize = False
        self._shifts = None            # (T,2) per-frame (row,col) jitter shifts

        self.iv = pg.ImageView()
        self.iv.ui.roiBtn.hide()
        self.iv.ui.menuBtn.hide()
        vb = self.iv.getView()
        # all-neuron positions (hidden until toggled)
        self.all_pts = pg.ScatterPlotItem(size=6, pen=None)
        self.all_pts.setVisible(False)
        vb.addItem(self.all_pts)
        # all soma outlines: one curve per palette colour, built lazily
        self.outline_curves = []
        # selected-neuron outline (drawn bold, on top) + centre marker
        self.sel_outline = pg.PlotCurveItem(
            pen=pg.mkPen("#39d353", width=self._sel_width()), connect="finite")
        vb.addItem(self.sel_outline)
        self.marker = pg.ScatterPlotItem(
            size=16, pen=pg.mkPen("cyan", width=2), brush=None)
        vb.addItem(self.marker)
        vb.invertY(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.iv)

        self.iv.getView().scene().sigMouseClicked.connect(self._on_click)
        self._set_image(auto_range=True)   # frame the image once, on startup

    # -- data ----------------------------------------------------------------
    def _movie(self):
        if self._source == "noisy" and self.run.mov_noisy is not None:
            return self.run.mov_noisy
        return self.run.mov_clean

    def _sel_width(self):
        # selected outline is drawn noticeably bolder than the rest
        return self._outline_width + 2.5

    def _frame_img(self):
        """Current frame, motion-stabilised to the fixed overlays if enabled."""
        img = self._movie()[self._frame]
        if self._stabilize and self._shifts is not None:
            dr, dc = self._shifts[self._frame]
            dr, dc = int(round(dr)), int(round(dc))
            if dr or dc:
                img = np.roll(img, (-dr, -dc), axis=(0, 1))
        return img

    def _set_image(self, auto_range=False):
        # auto_range=False preserves the user's zoom/pan across frames/sources
        self.iv.setImage(self._frame_img(), autoLevels=False,
                         levels=self._levels(), autoHistogramRange=False,
                         autoRange=auto_range)

    def _levels(self):
        mv = self._movie()
        return float(np.percentile(mv, 1)), float(np.percentile(mv, 99.5))

    # -- slots ---------------------------------------------------------------
    def set_frame(self, t: int):
        self._frame = int(t)
        self._set_image(auto_range=False)   # keep current zoom during playback
        if self._show_all:
            self._update_all_pts()

    def set_source(self, source: str):
        self._source = source
        self._set_image(auto_range=False)

    def set_selected(self, i):
        self._selected = i
        if i is None:
            self.marker.setData([], [])
            self.sel_outline.setData([], [])
        else:
            r, c = self.table.soma_movie_xy(i)
            self.marker.setData([c], [r])  # (x=col, y=row)
            polys = self.table.soma_contours(i)
            if polys:
                nan = np.array([np.nan])
                xs = np.concatenate([np.append(p[:, 0], nan) for p in polys])
                ys = np.concatenate([np.append(p[:, 1], nan) for p in polys])
                self.sel_outline.setData(xs, ys)
            else:
                self.sel_outline.setData([], [])

    # -- overlays ------------------------------------------------------------
    def set_show_all_positions(self, on: bool):
        """Scatter every neuron's soma position (coloured by dF/F)."""
        self._show_all = bool(on)
        self.all_pts.setVisible(self._show_all)
        if self._show_all:
            self._update_all_pts()

    def _update_all_pts(self):
        xy = self.table._soma_movie          # (N, 2) as (row, col)
        dff = self.run.dff()
        c = min(self._frame, dff.shape[1] - 1)
        vals = np.clip(dff[:, c] / 2.0, 0, 1)          # dF/F 0..2 -> 0..1
        brushes = [pg.mkBrush(int(255 * v), int(160 * v), 40, 220) for v in vals]
        self.all_pts.setData(x=xy[:, 1], y=xy[:, 0], brush=brushes, pen=None)

    def _build_outline_curves(self):
        """One PlotCurveItem per palette colour so overlapping outlines are
        distinguishable (neurons bucketed by id modulo the palette size)."""
        k = len(self.OUTLINE_PALETTE)
        buckets = self.table.soma_contour_buckets(k)
        vb = self.iv.getView()
        for (x, y), color in zip(buckets, self.OUTLINE_PALETTE):
            curve = pg.PlotCurveItem(
                x=x, y=y, connect="finite",
                pen=pg.mkPen(color, width=self._outline_width))
            curve.setVisible(False)
            vb.addItem(curve)
            self.outline_curves.append(curve)

    def set_show_all_outlines(self, on: bool):
        """Draw every neuron's soma outline (lazy, cached, multi-colour)."""
        if on and not self._outlines_built:
            self._build_outline_curves()
            self._outlines_built = True
        for c in self.outline_curves:
            c.setVisible(bool(on))

    def set_outline_width(self, w: float):
        """Set pen width for all soma outlines (and the bolder selected one)."""
        self._outline_width = float(w)
        for curve, color in zip(self.outline_curves, self.OUTLINE_PALETTE):
            curve.setPen(pg.mkPen(color, width=self._outline_width))
        self.sel_outline.setPen(pg.mkPen("#39d353", width=self._sel_width()))

    # -- motion stabilisation -----------------------------------------------
    def set_stabilize(self, on: bool):
        """Undo the movie's frame-to-frame jitter so cell content lines up with
        the fixed soma outlines/positions.  Per-frame shifts are measured once
        (phase cross-correlation to the mean frame) and cached."""
        self._stabilize = bool(on)
        if self._stabilize and self._shifts is None:
            self._compute_shifts()
        self._set_image(auto_range=False)

    def _compute_shifts(self):
        from skimage.registration import phase_cross_correlation
        mov = self.run.mov_clean          # measure on the clean movie
        ref = mov.mean(axis=0)            # mean frame ~ rest (overlay) position
        shifts = np.zeros((mov.shape[0], 2), dtype=np.float32)
        for t in range(mov.shape[0]):
            try:
                res = phase_cross_correlation(ref, mov[t], upsample_factor=1)
                s = res[0] if isinstance(res, (tuple, list)) else res
                shifts[t] = np.asarray(s)[:2]   # (row, col)
            except Exception:
                shifts[t] = 0
        self._shifts = shifts

    # -- interaction ---------------------------------------------------------
    def _on_click(self, ev):
        if not ev.double() and ev.button() != Qt.LeftButton:
            return
        vb = self.iv.getView()
        pt = vb.mapSceneToView(ev.scenePos())
        col, row = pt.x(), pt.y()
        i = self.table.pick_from_movie(row, col)
        self.neuronPicked.emit(i)


# ============================================================== trace panel ==
class TracesPanel(QWidget):
    """Calcium-trace plot: selected neuron (bold) + population context."""

    neuronPicked = Signal(int)

    MAX_LINES = 64  # draw every trace only when N is small

    def __init__(self, run, parent=None):
        super().__init__(parent)
        self.run = run
        self.dff = run.dff()
        # time axis follows the traces' own length (may differ from movie nt)
        self.n_t = self.dff.shape[1]
        self.t = np.arange(self.n_t) * run.dt
        self._frame = 0
        self._selected = None

        self.fig = Figure(figsize=(5, 3), facecolor="#111")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, facecolor="#111")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._cursor = None
        self._draw_static()

    def _draw_static(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#111")
        n = self.run.n_neur
        if n <= self.MAX_LINES:
            for k in range(n):
                ax.plot(self.t, self.dff[k], lw=0.5, color="#3a3a3a", zorder=1)
        else:
            mean = self.dff.mean(0)
            ax.plot(self.t, mean, lw=1.0, color="#4a5a6a",
                    label="population mean", zorder=1)
        self._sel_line, = ax.plot([], [], lw=1.8, color="#39d353",
                                   zorder=3, label="selected")
        self._spk = ax.scatter([], [], marker="|", s=120, color="#f778ba",
                               zorder=4)
        self._cursor = ax.axvline(0, color="cyan", lw=1.0, zorder=5)
        ax.set_xlabel("time (s)", color="w")
        ax.set_ylabel("dF/F", color="w")
        ax.tick_params(colors="w")
        for s in ax.spines.values():
            s.set_color("#444")
        ax.set_xlim(self.t[0], self.t[-1] if self.n_t > 1 else 1)
        ax.set_title("click a neuron in any view", color="#aaa", fontsize=9)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # -- slots ---------------------------------------------------------------
    def set_frame(self, t: int):
        self._frame = int(t)
        fi = min(self._frame, self.n_t - 1)  # movie may outrun the traces
        self._cursor.set_xdata([self.t[fi], self.t[fi]])
        self.canvas.draw_idle()

    def set_selected(self, i):
        self._selected = i
        if i is None:
            self._sel_line.set_data([], [])
            self._spk.set_offsets(np.empty((0, 2)))
            self.ax.set_title("click a neuron in any view", color="#aaa",
                              fontsize=9)
        else:
            self._sel_line.set_data(self.t, self.dff[i])
            sp = self.run.spikes_neurons
            if sp is not None:
                idx = np.flatnonzero(sp[i] > 0)
                if idx.size:
                    y = np.full(idx.size, self.dff[i].max() * 1.05)
                    self._spk.set_offsets(np.column_stack([self.t[idx], y]))
                else:
                    self._spk.set_offsets(np.empty((0, 2)))
            self.ax.set_title(f"neuron #{i}", color="#39d353", fontsize=9)
            self.ax.relim(); self.ax.autoscale_view(scalex=False)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or self.run.n_neur > self.MAX_LINES:
            return
        # pick nearest trace by value at the clicked time
        ti = int(np.clip(event.xdata / self.run.dt, 0, self.n_t - 1))
        col = self.dff[:, ti]
        i = int(np.argmin(np.abs(col - event.ydata)))
        self.neuronPicked.emit(i)
