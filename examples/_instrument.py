"""Standard one-call run instrumentation for calcia examples.

Put this as the FIRST line inside a script's ``if __name__ == "__main__":`` block
(or at the top of its ``main()``):

    import _instrument; _instrument.start()

It does BOTH, mandated for every example, with zero further code:
  1. RUN LOG   — tees stdout+stderr to ``examples/output/logs/<script>_<ts>.log``
                 so the full console output is saved for every run.
  2. PYINSTRUMENT — starts a profiler and, at process exit (atexit), writes
                 ``examples/output/logs/<script>_<ts>.html`` + prints a text summary.

Call it in the ``__main__`` block (NOT at module top) so it does not fire when a
script is merely imported by another. Idempotent, line-buffered, never raises.
"""
import atexit
import datetime as _dt
import os
import sys

_STARTED = False


class _Tee:
    """Write-through stream: mirrors to the real stream AND a file; a console
    encoding error never blocks the file write."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def start(name=None, output_dir=None):
    """Enable run log + pyinstrument for the current run. Returns the log/profile
    path stem (or None). Safe to call more than once (only the first wins)."""
    global _STARTED
    if _STARTED:
        return None
    _STARTED = True
    try:
        if name is None:
            name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "run"
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "output")
        logs = os.path.join(output_dir, "logs")
        os.makedirs(logs, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(logs, f"{name}_{stamp}")
        fh = open(base + ".log", "w", encoding="utf-8", buffering=1)
        fh.write(f"# {name}  {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        fh.flush()
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)
        print(f"[instrument] run log -> {base}.log")
        try:
            from pyinstrument import Profiler
            prof = Profiler()
            prof.start()

            def _stop():
                try:
                    prof.stop()
                    with open(base + ".html", "w", encoding="utf-8") as f:
                        f.write(prof.output_html())
                    print(prof.output_text(unicode=True, color=False,
                                           show_all=False))
                    print(f"[instrument] profile -> {base}.html")
                except Exception as e:
                    print(f"[instrument] profile save failed: {e}")

            atexit.register(_stop)
            print("[instrument] pyinstrument profiling ON")
        except Exception as e:
            print(f"[instrument] pyinstrument unavailable: {e}")
        return base
    except Exception as e:
        print(f"[instrument] setup failed: {e}")
        return None
