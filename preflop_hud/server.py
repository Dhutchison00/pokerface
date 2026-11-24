"""Local web server + capture loop.

GET  /                 the chart page
GET  /api/state        latest parsed table + chart (JSON)
GET  /api/frame.jpg    last captured frame with the OCR overlay
POST /api/select-region   opens the drag-to-select overlay, saves the region
POST /api/pause           toggle pause
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .cards import hole_cards_to_class, HAND_INDEX
from .overrides import Overrides
from . import reader as reader_mod
from .reader import read_table, annotate
from .spot import derive_spot
from .strategy import Engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CONFIG = os.path.join(ROOT, "config.json")


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_config(cfg):
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _signature(spot, hero_class):
    v = spot.get("villain") or {}
    return json.dumps([spot.get("type"), spot.get("hero_pos"), round(spot.get("eff_bb") or 0, 1),
                       v.get("pos"), v.get("bet"), spot.get("n_players"), hero_class,
                       [p["pos"] for p in spot.get("behind", [])]])


class App:
    def __init__(self, image=None, interval=None):
        self.cfg = load_config()
        self.image = image
        self.interval = interval or float(self.cfg.get("interval", 0.5))
        self.engine = Engine()
        self.overrides = Overrides(os.path.join(ROOT, "ranges"))
        self.lock = threading.Lock()
        self.paused = False
        self.version = 0
        self.state = {"status": "starting", "version": 0}
        self.frame_jpg = None
        self._pending = None
        self._accepted = None
        self._last_data = None
        self._chart_pending = False
        self._last_resolve = 0.0
        self._last_small = None
        self._grabber = None
        self.selecting = False

    # ------------------------------------------------------------ frame source
    def _frame(self):
        if self.image:
            img = cv2.imread(self.image)
            if img is None:
                raise RuntimeError(f"Can't read image {self.image}")
            return img
        region = self.cfg.get("region")
        if not region:
            return None
        if self._grabber is None:
            from .capture import Grabber
            self._grabber = Grabber()
        return self._grabber.grab(region)

    # ------------------------------------------------------------ main loop
    def loop(self):
        while True:
            t0 = time.time()
            try:
                if not self.paused and not self.selecting:
                    self.tick()
            except Exception as e:
                traceback.print_exc()
                self._publish({"status": "error", "error": f"{type(e).__name__}: {e}"}, keep_chart=True)
            time.sleep(max(0.05, self.interval - (time.time() - t0)))

    def tick(self):
        img = self._frame()
        if img is None:
            self._publish({"status": "no_region", "error": "No capture region yet: click “Select region”."},
                          keep_chart=True)
            return
        # local change detector: count pixels that changed noticeably on a thumbnail
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (320, 220), interpolation=cv2.INTER_AREA)
        changed = True
        if self._last_small is not None and self._last_small.shape == small.shape:
            changed = int((cv2.absdiff(small, self._last_small) > 12).sum()) > 8
        self._last_small = small

        if not changed and self.image is None:
            if self._pending is None and self._accepted is not None:
                # nothing new on screen: skip OCR, but upgrade a fallback chart once its solve lands
                if self._chart_pending and self._last_data and time.time() - self._last_resolve > 2:
                    self._accept(self._last_data, force=True)
                return
            if self._pending is not None:   # screen held still since that read: confirmed
                data, self._pending = self._pending, None
                return self._accept(data)
        data = self._parse(img)
        if self.image is not None or data["sig"] == self._accepted or \
                (self._pending is not None and data["sig"] == self._pending["sig"]):
            self._pending = None
            return self._accept(data)
        self._pending = data                # new reading, wait for one confirming frame
        self._publish({"status": "settling", "table": data["table"], "parse_ms": data["ms"]}, keep_chart=True)

    def _parse(self, img):
        t0 = time.time()
        lay = self.cfg.get("layout")
        st = read_table(img, lay)
        spot = derive_spot(st)
        hc = st.hero_cards or {}
        hero_class = None
        if hc.get("complete"):
            r = hc["ranks"]
            hero_class = hole_cards_to_class(r[0], r[1], hc.get("suited"))
        frame = annotate(img, st, lay)
        h, w = frame.shape[:2]
        if w > 1100:
            frame = cv2.resize(frame, (1100, int(h * 1100 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.frame_jpg = buf.tobytes()
        table = {
            "blinds": [st.sb, st.bb, st.ante], "unit": st.unit, "pot": st.pot, "ocr_ms": st.ocr_ms,
            "seats": [{"idx": s.idx, "name": s.name, "present": s.present, "stack": s.stack,
                       "bet": round(s.bet, 3), "labels": s.labels, "has_cards": s.has_cards,
                       "acting": s.acting, "pos": spot.get("positions", {}).get(str(s.idx))}
                      for s in st.seats],
            "hero_cards": hc, "hero_class": hero_class, "buttons": st.hero_buttons,
            "error": st.error, "ocr_device": reader_mod.OCR_DEVICE,
        }
        return {"spot": spot, "hc": hc, "hero_class": hero_class, "table": table,
                "sig": _signature(spot, hero_class), "ms": int((time.time() - t0) * 1000)}

    def _accept(self, data, force=False):
        spot, hero_class, hc = data["spot"], data["hero_class"], data["hc"]
        self._last_data = data
        if data["sig"] == self._accepted and self.image is None and not force and not                 (self._chart_pending and time.time() - self._last_resolve > 2):
            self._publish({"status": "live", "table": data["table"], "parse_ms": data["ms"]}, keep_chart=True)
            return
        chart = self.overrides.find(spot) if spot.get("eff_bb") else None
        if chart is None:
            chart = self.engine.solve(spot)
        hero = None
        if hero_class and chart.get("freqs"):
            k = HAND_INDEX.get(hero_class)
            if k is not None:
                raw = chart.get("freqs_raw") or chart["freqs"]
                hero = {"hand": hero_class, "index": k, "suit_unsure": hc.get("suited") is None,
                        "freqs": dict(zip(chart["actions"], raw[k])),
                        "ev": (chart.get("ev") or [None] * 169)[k],
                        "unreached": bool((chart.get("unreached") or [False] * 169)[k])}
                if chart.get("ev_actions"):
                    hero["ev_actions"] = dict(zip(chart["actions"], chart["ev_actions"][k]))
        self._accepted = data["sig"]
        self._chart_pending = bool(chart.get("pending_solver"))
        self._last_resolve = time.time()
        self._publish({"status": "live", "table": data["table"], "spot": spot, "chart": chart, "hero": hero,
                       "parse_ms": data["ms"], "override_errors": self.overrides.errors})

    def _publish(self, update, keep_chart=False):
        with self.lock:
            new = dict(self.state) if keep_chart else {}
            if "error" not in update:
                new.pop("error", None)
            new.update(update)
            new["paused"] = self.paused
            new["region"] = self.cfg.get("region")
            new["image_mode"] = bool(self.image)
            new["updated"] = time.time()
            self.version += 1
            new["version"] = self.version
            self.state = _jsonable(new)

    # ------------------------------------------------------------ region picker
    def select_region(self):
        self.selecting = True
        try:
            out = subprocess.run([sys.executable, "-m", "preflop_hud.region_select"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=300)
            if out.returncode != 0:
                print(out.stderr, file=sys.stderr)
                return {"ok": False, "error": out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "picker crashed"}
            data = json.loads(out.stdout.strip()) if out.stdout.strip() else {}
            if not data or data.get("cancelled"):
                return {"ok": False, "cancelled": True}
            self.cfg["region"] = data
            save_config(self.cfg)
            self._accepted = self._pending = self._last_small = None
            self.image = None
            return {"ok": True, "region": data}
        finally:
            self.selecting = False


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with app.lock:
                    body = json.dumps(app.state).encode()
                self._send(200, body, "application/json")
            elif path == "/api/frame.jpg":
                if app.frame_jpg:
                    self._send(200, app.frame_jpg, "image/jpeg")
                else:
                    self._send(404, b"no frame yet", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            # only accept requests from our own page
            origin = self.headers.get("Origin", "")
            if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
                return self._send(403, b"forbidden", "text/plain")
            path = self.path.split("?")[0]
            if path == "/api/select-region":
                res = app.select_region()
                self._send(200, json.dumps(res).encode(), "application/json")
            elif path == "/api/pause":
                app.paused = not app.paused
                app._publish({"status": "paused" if app.paused else "live"}, keep_chart=True)
                self._send(200, json.dumps({"paused": app.paused}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return H


def serve(port=8765, image=None, open_browser=True, interval=None):
    app = App(image=image, interval=interval)
    threading.Thread(target=app.loop, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}/"
    print(f"Preflop HUD running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
