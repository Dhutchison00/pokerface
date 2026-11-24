"""Screen capture of a fixed rectangle (physical pixels, all monitors)."""
from __future__ import annotations

import ctypes
import sys

import numpy as np


def make_dpi_aware():
    """So screen coordinates = physical pixels on scaled (125%/150%) displays."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class Grabber:
    def __init__(self):
        import mss
        make_dpi_aware()
        self._mss = mss.mss()

    def virtual_screen(self):
        m = self._mss.monitors[0]
        return {"left": m["left"], "top": m["top"], "width": m["width"], "height": m["height"]}

    def grab(self, region=None) -> np.ndarray:
        """region = {left, top, width, height}; None = all monitors.  Returns BGR."""
        r = region or self.virtual_screen()
        shot = self._mss.grab({k: int(r[k]) for k in ("left", "top", "width", "height")})
        return np.asarray(shot)[:, :, :3].copy()
