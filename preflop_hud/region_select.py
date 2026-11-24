"""Snipping-tool style region picker.

Freezes the screen (all monitors), dims it, and lets you drag a rectangle.
Prints the chosen region as JSON on stdout: {"left","top","width","height"}
(or {"cancelled": true}).  Run standalone:  python -m preflop_hud.region_select
"""
from __future__ import annotations

import json
import sys


def select_region(auto_close_ms=None):
    from .capture import Grabber  # sets DPI awareness before Tk starts
    import tkinter as tk
    from PIL import Image, ImageEnhance, ImageTk

    g = Grabber()
    vs = g.virtual_screen()
    shot = Image.fromarray(g.grab(vs)[:, :, ::-1])
    dim = ImageEnhance.Brightness(shot).enhance(0.4)

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{vs['width']}x{vs['height']}+{vs['left']}+{vs['top']}")
    cv = tk.Canvas(root, width=vs["width"], height=vs["height"], highlightthickness=0, cursor="crosshair")
    cv.pack(fill="both", expand=True)
    bg = ImageTk.PhotoImage(dim)
    cv.create_image(0, 0, anchor="nw", image=bg)
    hint = "Drag a box around the whole poker table  ·  release to confirm  ·  Esc to cancel"
    # hint on every monitor
    for m in g._mss.monitors[1:]:
        x = m["left"] - vs["left"] + m["width"] // 2
        y = m["top"] - vs["top"] + 40
        cv.create_text(x + 1, y + 1, text=hint, fill="black", font=("Segoe UI", 16, "bold"))
        cv.create_text(x, y, text=hint, fill="white", font=("Segoe UI", 16, "bold"))

    state = {"start": None, "crop_item": None, "crop_img": None, "rect": None, "label": None, "result": None}

    def on_down(e):
        state["start"] = (e.x, e.y)

    def _draw(x0, y0, x1, y1):
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if state["crop_item"]:
            cv.delete(state["crop_item"])
        if x1 - x0 > 2 and y1 - y0 > 2:
            state["crop_img"] = ImageTk.PhotoImage(shot.crop((x0, y0, x1, y1)))
            state["crop_item"] = cv.create_image(x0, y0, anchor="nw", image=state["crop_img"])
        if state["rect"]:
            cv.delete(state["rect"])
            cv.delete(state["label"])
        state["rect"] = cv.create_rectangle(x0, y0, x1, y1, outline="#4fc3f7", width=2)
        state["label"] = cv.create_text(x0 + 4, y0 - 12 if y0 > 20 else y1 + 12, anchor="w",
                                        text=f"{x1 - x0} × {y1 - y0}", fill="#4fc3f7",
                                        font=("Segoe UI", 11, "bold"))

    def on_move(e):
        if state["start"]:
            _draw(*state["start"], e.x, e.y)

    def on_up(e):
        if not state["start"]:
            return
        x0, x1 = sorted((state["start"][0], e.x))
        y0, y1 = sorted((state["start"][1], e.y))
        if x1 - x0 < 80 or y1 - y0 < 60:
            state["start"] = None
            return
        state["result"] = {"left": vs["left"] + x0, "top": vs["top"] + y0,
                           "width": x1 - x0, "height": y1 - y0}
        root.destroy()

    def on_esc(_e):
        state["result"] = None
        root.destroy()

    cv.bind("<ButtonPress-1>", on_down)
    cv.bind("<B1-Motion>", on_move)
    cv.bind("<ButtonRelease-1>", on_up)
    root.bind("<Escape>", on_esc)
    root.after(50, lambda: (root.focus_force(), root.lift()))
    if auto_close_ms:  # self-test: close without a selection
        root.after(int(auto_close_ms), root.destroy)
    root.mainloop()
    return state["result"]


if __name__ == "__main__":
    ms = None
    if "--auto-close-ms" in sys.argv:
        ms = int(sys.argv[sys.argv.index("--auto-close-ms") + 1])
    r = select_region(ms)
    sys.stdout.write(json.dumps(r if r else {"cancelled": True}))
    sys.stdout.flush()
