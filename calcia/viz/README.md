# calcia simulation visualizer

Interactive visualization for NAOMi/calcia outputs. Links **calcium traces** to
individual **neurons**, shows **blood vessels** and **neuron morphology** in a
**3D** view, and shows the **2D final image** — all driven by one shared
timeline and one selected neuron.

`calcia.viz` ships inside the `calcia` package. Install its (optional) deps
once with `pip install -e ".[viz]"`; `import calcia` never needs them.

## Run it

`run_dir` is **optional** everywhere — omit it and the app finds runs for you
(discovered under `examples/output`, or `$CALCIA_OUTPUT` if set).

```bash
# desktop: launch with NO argument -> in-program run picker (lists every
# viewable run, plus a "Browse folder..." button).
conda run -n calcia python -m calcia.viz
# ...or pass a run directly:
conda run -n calcia python -m calcia.viz examples/output/<run_dir>

# headless end-to-end smoke test (omit path -> most recent run) -> viz_cache/render_check.png
conda run -n calcia python -m calcia.viz.render_check

# browser (same 3D scene, served by trame; toolbar has a run dropdown)
conda run -n calcia python -m calcia.viz.app_trame --port 8080
```

After `pip install -e ".[viz]"` these are also available as console commands:
`calcia-viz` (desktop), `calcia-viz-web` (browser), `calcia-viz-check`
(headless).

In the desktop window, **File → Open run…** (Ctrl+O) switches runs at runtime;
**File → Reload** (Ctrl+R) reloads the current one. Run discovery
(`runs.discover_runs`) treats a folder as viewable when it has
`metadata.json`, `traces.npz`, `movies.npz` and `cell_footprints.pkl`, and
orders by `metadata.json` mtime (robust against `viz_cache/` writes).

Useful flags: `--no-vessels` (skip the multi-GB phase-1 pickle),
`--vessel-ds N` (downsample vessel marching cubes), `--dendrites tube|points`,
`--load-volume` (enable the optional dense "fog" layer).

### Controls layout

All options sit at the **top** of the window:

* **Menu bar** — *File* (Open run / Reload / Quit) and *View* (checkable display
  toggles, below).
* **Toolbar** — Play/Pause, the frame slider + frame number, the movie source
  (clean / noisy), and an **outline width** spinner.

The 3D camera and the 2D zoom/pan are **preserved during playback and on
selection** (playing the movie no longer resets the view).

### View-menu display toggles

| toggle | effect |
|--------|--------|
| **Blood vessels** | show/hide the blood-vessel mesh |
| **Soma points** | show/hide the soma point cloud (on by default) |
| **Soma mesh (3D)** | render *all* somas as marching-cubes surfaces, recoloured per frame by dF/F. Built on a **background thread** (window stays responsive), decimated + cached to `soma_mesh_dec*.vtu`; first enable takes a moment on big runs |
| **All cell positions (2D)** | scatter every neuron's position on the 2D movie, coloured by dF/F |
| **Soma outlines (2D)** | draw every neuron's soma outline on the 2D movie, **each neuron in a distinct colour** (16-colour palette) so overlapping outlines stay legible; width set by the toolbar spinner |
| **Stabilize movie (align to cells)** | undo the movie's frame-to-frame jitter so cell content lines up with the fixed outlines/positions. Per-frame shifts are measured once (phase cross-correlation to the mean frame) and cached |
| **Fog (dense volume)** | optional dense fluorescence volume (only with `--load-volume`) |

The selected neuron gets a **bold** green soma outline + cyan centre marker on
the 2D view (thicker than the other outlines so it stands out). The browser app
mirrors the two 3D soma toggles.

## Design

* **Geometry-first.** Sparse voxel structures → PyVista meshes built once and
  cached to disk under `viz_cache/` (`vessels_*.vtp`; the all-soma marching-cubes
  mesh as `soma_mesh.vtu`) so later launches load instead of recomputing. Each
  frame only updates soma scalars (dF/F), never rebuilds geometry. Direct volume
  rendering is an opt-in, downsampled layer.
* **One `neuron_id` spine.** `soma_neurons[i]` ↔ `neur_num==i+1` ↔ `gp_vals[i]`
  ↔ `soma_locs[i]`. A click in any view resolves to `i`; all views react.
* **Backend-agnostic scene.** `scene3d.Scene3D` only touches a
  `pyvista.Plotter`, so the desktop (`app_qt`, pyvistaqt) and browser
  (`app_trame`, trame) hosts share the *same* scene code — the web path stays
  open with no rewrite. Server-side VTK keeps efficiency; large data never ships
  to the client.

## Modules

| file | role |
|------|------|
| `runs.py`       | discover viewable runs on disk; `resolve` (explicit or latest) |
| `model.py`      | load a run dir (+ optional phase-1 pickle) → `SimRun`; coords, dF/F |
| `linkage.py`    | `NeuronTable`: id ↔ trace ↔ voxels ↔ 2D footprint; click→id |
| `geometry.py`   | `GeometryCache`: vessels (marching cubes), soma points, per-neuron soma surface + dendrite (points/tube) |
| `scene3d.py`    | `Scene3D`: backend-agnostic PyVista scene (build / set_frame / select / picking) |
| `panels.py`     | `MoviePanel` (pyqtgraph) + `TracesPanel` (matplotlib) |
| `app_qt.py`     | desktop shell: `SharedState` signals wire the three views |
| `app_trame.py`  | browser shell reusing `Scene3D` |
| `render_check.py` | headless render + linkage sanity check |

## Coordinates

`neur_num` / `gp_vals[k].indices` are C-order linear indices into
`grid_shape = vol_sz * vres`. `soma_locs` are in microns; a voxel coordinate is
`um * vres`. All 3D meshes live in grid/voxel coordinates.
