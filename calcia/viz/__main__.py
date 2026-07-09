"""``python -m calcia.viz`` -> launch the desktop visualizer.

Equivalent to ``python -m calcia.viz.app_qt``.  Pass a run directory or omit
it to pick one in-program:

    python -m calcia.viz
    python -m calcia.viz examples/output/<run_dir> --dendrites tube
"""

import sys

from .app_qt import main

if __name__ == "__main__":
    sys.exit(main())
