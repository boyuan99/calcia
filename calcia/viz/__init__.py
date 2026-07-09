"""Interactive visualization for NAOMi/calcia simulation outputs.

Links calcium traces to individual neurons and shows blood vessels, neuron
morphology (soma / dendrites), a 3D geometry view and the 2D final image.

Design principles (see README.md):
  * Geometry-first: sparse voxel structures are converted to cached meshes
    once, not volume-raycast every frame.
  * Backend-agnostic 3D scene (:class:`~calcia.viz.scene3d.Scene3D` operates
    on a plain ``pyvista.Plotter``) so the same scene can be hosted by a Qt
    desktop app *or* streamed to a browser via trame.
  * A single ``neuron_id`` threads trace <-> 3D geometry <-> 2D footprint.

Entry points:
  * ``python -m calcia.viz.app_qt   <run_dir>``   desktop (PyQt5 + pyvistaqt)
  * ``python -m calcia.viz.app_trame <run_dir>``  browser  (trame)
  * ``python -m calcia.viz.render_check <run_dir>`` headless PNG smoke test
"""

from .model import SimRun, load
from .linkage import NeuronTable

__all__ = ["SimRun", "load", "NeuronTable"]
