"""Run-console logging: mirror stdout+stderr to a timestamped log file.

A long or background run's live console output would otherwise be lost; teeing
it to ``<output_dir>/logs/<name>_<timestamp>.log`` keeps calcia's ``[1/7]..``
progress prints, timing, warnings, and any profiler report reviewable after the
fact. This is the shared strategy behind the examples' run wrappers; it lives in
core so any driver (a demo, a batch script, a notebook) can use one Tee.
"""
import datetime as _dt
import os
import sys


class Tee:
    """Write-through stream: mirrors everything to the real stream AND a file.

    A console-encoding error (e.g. Windows cp1252 vs unicode) on the real stream
    never blocks the file write — the log always gets the full text.
    """

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

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

    def __getattr__(self, name):  # isatty, encoding, fileno, ... -> real stream
        return getattr(self._stream, name)


def run_log_stem(output_dir, name):
    """Return the ``<output_dir>/logs/<name>_<YYYYmmdd_HHMMSS>`` path stem
    (extension-less), creating the ``logs`` directory. Callers append ``.log``
    for the console log and e.g. ``.html`` for a profiler report so the two share
    a stem. This is the run-log naming convention shared by every run driver.
    """
    logs = os.path.join(output_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(logs, f"{name}_{stamp}")


def tee_stdio(log_path, header=None):
    """Redirect ``sys.stdout`` + ``sys.stderr`` through a :class:`Tee` that also
    writes to ``log_path`` (opened line-buffered so a background run flushes live
    to disk). Optionally write a ``header`` line first. Returns the open file
    handle, or ``None`` on failure. Never raises — logging must not take down a
    real run.
    """
    try:
        d = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(d, exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        if header:
            fh.write(header if header.endswith("\n") else header + "\n")
            fh.flush()
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)
        return fh
    except Exception as e:  # logging must never take down a real run
        print(f"[logging] WARN could not set up tee log: {e}")
        return None
