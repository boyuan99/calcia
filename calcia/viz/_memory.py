"""Memory-safety guards so the viewer never exhausts RAM and freezes the box.

The visualiser reads run artefacts that can be multiple GB (the phase-1 pickle
is ~7.5 GB, ``cell_footprints.pkl`` ~3 GB).  Loading one of those when free RAM
is already low can drive the machine into swap-thrash and hang it.  These
helpers refuse such a load with a clear error instead.

``available_ram_bytes`` works without any third-party dependency: it tries
``psutil`` first, then falls back to the native OS call (Win32
``GlobalMemoryStatusEx`` / POSIX ``sysconf``).
"""

from __future__ import annotations

import os


def available_ram_bytes():
    """Best-effort available (not total) physical RAM in bytes, or None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
        except Exception:
            return None
    try:  # POSIX
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


class MemoryBudgetError(RuntimeError):
    """Raised when an operation would use an unsafe fraction of free RAM."""


def guard_load(path, *, multiplier: float = 1.6, headroom_gb: float = 2.0):
    """Refuse to load ``path`` if it would likely exhaust RAM.

    A pickle/npz typically needs more RAM than its file size while
    deserialising, hence ``multiplier``.  We also keep ``headroom_gb`` free for
    the OS and other apps.  Raises :class:`MemoryBudgetError` instead of letting
    the load thrash the machine.  Silently allows the load if free RAM is
    unknown (never blocks on missing telemetry).
    """
    try:
        need = os.path.getsize(path) * multiplier
    except OSError:
        return
    avail = available_ram_bytes()
    if avail is None:
        return
    safe = avail - headroom_gb * 1e9
    if need > safe:
        raise MemoryBudgetError(
            f"Refusing to load {os.path.basename(path)}: needs ~"
            f"{need/1e9:.1f} GB but only {avail/1e9:.1f} GB RAM is free "
            f"(keeping {headroom_gb:.0f} GB headroom). Close other apps, or use "
            f"a smaller run / the cached mesh path.")


def enough_ram_for(bytes_needed, *, headroom_gb: float = 2.0) -> bool:
    """True if ``bytes_needed`` fits within free RAM minus headroom."""
    avail = available_ram_bytes()
    if avail is None:
        return True
    return bytes_needed < (avail - headroom_gb * 1e9)
