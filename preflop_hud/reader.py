"""Turn a screenshot of the table into a structured TableState.

Pipeline:  full-frame OCR  ->  find felt  ->  assign every text box to a seat
(stack / name / status label / bet) by nearest felt-relative anchor  ->
non-text detectors for card backs, the dealer button and hero hole cards.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from .layout import DEFAULT_LAYOUT

_ocr = None
OCR_DEVICE = None   # "cuda", "dml" or "cpu": what the OCR sessions actually run on, set by get_ocr()
_PROVIDERS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}


def _gpu_choice(want: str) -> str:
    """Pick the OCR device. want = "auto" (GPU if this onnxruntime build has one, else CPU),
    "cuda", "dml" or "cpu". Set it with the HUD_OCR_DEVICE environment variable."""
    if want == "cpu":
        return "cpu"
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return "cpu"
    for dev in ((want,) if want in ("cuda", "dml") else ("cuda", "dml")):
        if _PROVIDERS[dev] in avail:
            if dev == "cuda" and hasattr(ort, "preload_dlls"):
                try:   # load CUDA / cuDNN from the nvidia-* pip packages if they're installed
                    ort.preload_dlls()
                except Exception:
                    pass
            return dev
    return "cpu"


def _session_providers(ocr) -> list[str]:
    out = []
    for path in (("text_det", "infer"), ("text_cls", "infer"), ("text_rec", "session")):
        obj = ocr
        for attr in path + ("session",):
            obj = getattr(obj, attr, None)
        if obj is not None:
            out.append(obj.get_providers()[0])
    return out


def get_ocr():
    """RapidOCR on the GPU when available (CUDA, then DirectML), else the CPU.

    onnxruntime silently falls back to the CPU when a GPU provider can't start
    (missing CUDA/cuDNN DLLs, old driver), so the sessions are checked afterwards
    and OCR_DEVICE reports what is really in use.
    """
    global _ocr, OCR_DEVICE
    if _ocr is None:
        import logging
        import os
        from rapidocr_onnxruntime import RapidOCR
        dev = _gpu_choice(os.environ.get("HUD_OCR_DEVICE", "auto").strip().lower())
        kw = {} if dev == "cpu" else {f"{m}_use_{dev}": True for m in ("det", "cls", "rec")}
        try:
            ocr = RapidOCR(**kw)
        except Exception as e:   # GPU session failed outright: fall back to the CPU
            logging.getLogger(__name__).warning("OCR on %s failed (%s); using the CPU", dev, e)
            ocr, kw = RapidOCR(), {}
        used = _session_providers(ocr)
        if kw and used and any(p != _PROVIDERS[dev] for p in used):
            dev = "cpu" if all(p == "CPUExecutionProvider" for p in used) else f"{dev} (partly cpu)"
        OCR_DEVICE = dev
        print(f"OCR device: {dev}  ({', '.join(used)})", flush=True)
        # warm-up: the first run loads CUDA/cuDNN kernels (~2 s), keep that off the first real read
        warm = np.full((96, 320, 3), 255, np.uint8)
        cv2.putText(warm, "19.7 BB", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        try:
            ocr(warm)
        except Exception:
            pass
        _ocr = ocr
    return _ocr


# ----------------------------------------------------------------- parsing helpers
AMOUNT_RE = re.compile(r"^\$?(\d[\d,]*\.?\d*|\.\d+)([KM])?(BB)?$")
BLINDS_RE = re.compile(r"(\d[\d,.]*[KM]?)/(\d[\d,.]*[KM]?)(?:[(（](\d[\d,.]*[KM]?)[)）])?")
LABELS = {
    "FOLD": "fold", "FOLDED": "fold", "SB": "sb", "BB": "bb", "SITTINGOUT": "sitting_out",
    "SITOUT": "sitting_out", "ALLIN": "allin", "ALL-IN": "allin", "CALL": "call", "CHECK": "check",
    "RAISE": "raise", "BET": "bet", "D": "dealer",
}


def _num(tok: str) -> float:
    tok = tok.replace(",", "")
    mult = 1.0
    if tok.endswith("K"):
        mult, tok = 1e3, tok[:-1]
    elif tok.endswith("M"):
        mult, tok = 1e6, tok[:-1]
    return float(tok) * mult


def _norm_text(t: str) -> str:
    t = t.upper().replace(" ", "").replace("O.", "0.")
    return t


def parse_amount(t: str):
    """Returns (value, is_bb) or None.  '19.7BB' -> (19.7, True); '8,800' -> (8800, False)."""
    t = _norm_text(t)
    if t in ("ALLIN", "ALL-IN"):
        return None
    m = AMOUNT_RE.match(t)
    if not m:
        return None
    try:
        v = _num(m.group(1) + (m.group(2) or ""))
    except ValueError:
        return None
    return v, bool(m.group(3))


@dataclass
class Seat:
    idx: int
    present: bool = False
    name: str = ""
    stack: float | None = None   # in BB
    bet: float = 0.0             # in BB, chips in front of the seat this street
    labels: list = field(default_factory=list)
    has_cards: bool = False
    card_back_frac: float = 0.0
    acting: bool = False

    @property
    def folded(self):
        return "fold" in self.labels

    @property
    def sitting_out(self):
        return "sitting_out" in self.labels

    @property
    def all_in(self):
        return "allin" in self.labels or (self.stack is not None and self.stack <= 0.001 and self.present)


@dataclass
class TableState:
    ok: bool = False
    error: str = ""
    sb: float | None = None          # chips
    bb: float | None = None
    ante: float = 0.0
    unit: str = "BB"
    pot: float | None = None         # BB
    seats: list = field(default_factory=list)
    button_seat: int | None = None
    button_source: str = ""
    sb_seat: int | None = None
    bb_seat: int | None = None
    hero_cards: dict | None = None
    to_call_hint: float | None = None   # from the "Call X" action button
    hero_buttons: list = field(default_factory=list)
    felt: list | None = None
    ocr_ms: int = 0
    texts: list = field(default_factory=list)  # for the debug overlay

    def to_dict(self):
        d = asdict(self)
        for s, so in zip(d["seats"], self.seats):
            s["folded"], s["sitting_out"], s["all_in"] = so.folded, so.sitting_out, so.all_in
        return d


# ----------------------------------------------------------------- detectors
def find_felt(img, lay):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, tuple(lay["felt_hsv_lo"]), tuple(lay["felt_hsv_hi"]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, area = st[i]
    if area < 0.05 * img.shape[0] * img.shape[1]:
        return None
    return int(x), int(y), int(w), int(h)


def card_back_fraction(img, box):
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] < 60) & (hsv[..., 2] > 90) & (hsv[..., 2] < 235)
    return float(m.mean())


def find_dealer_buttons(img, felt):
    """White, round, roughly button-sized blobs near the felt."""
    fx, fy, fw, fh = felt
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (0, 0, 195), (180, 45, 255))
    n, _, st, cen = cv2.connectedComponentsWithStats(g)
    out = []
    rmin, rmax = 0.018 * fw, 0.050 * fw
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not (rmin < w < rmax and 0.8 < w / max(h, 1) < 1.25):
            continue
        circ = a / (math.pi * w * h / 4)
        if circ < 0.6:
            continue
        cx, cy = cen[k]
        u, v = (cx - fx) / fw, (cy - fy) / fh
        if -0.15 < u < 1.15 and -0.25 < v < 1.15:
            out.append((float(cx), float(cy), float(circ)))
    return out


def _ink_color(hsv, mask):
    colored = mask & (hsv[..., 1] > 90) & (hsv[..., 2] > 70)
    dark = mask & (hsv[..., 2] < 100)
    if colored.sum() > dark.sum():
        hue = float(np.median(hsv[..., 0][colored]))
        if 35 < hue < 90:
            return "green"
        if 90 <= hue <= 135:
            return "blue"
        return "red"
    return "black"


def _suit_info(card_img, rank_box, scale):
    """Colour (from the rank glyph) + binary shape of the suit pip below it.

    OCR boxes sometimes swallow the pip, so we look at the ink itself: inside a
    vertical strip under the rank, the top blob(s) are the rank glyph and the
    next blob down is the pip.
    """
    (x0, y0), (x1, y1) = rank_box
    # strip size from the glyph height too: OCR boxes around a narrow glyph ("J", "1") can be
    # tight horizontally (and differ between CPU and GPU runs), which would cut the pip off
    w = max(x1 - x0, 0.75 * (y1 - y0))
    x0, x1 = (x0 + x1 - w) / 2, (x0 + x1 + w) / 2
    H, W = card_img.shape[:2]
    sx0, sx1 = max(0, int(x0 - 0.25 * w)), min(W, int(x1 + 0.25 * w))
    # generous headroom: a box that starts below the glyph's top would cut the glyph at the
    # strip edge, leaving it outside the card-face outline (then the pip is taken for the rank)
    sy0, sy1 = max(0, int(y0 - 0.3 * (y1 - y0))), min(H, int(y0 + 3.2 * w + (y1 - y0)))
    if sx1 - sx0 < 4 or sy1 - sy0 < 4:
        return None
    crop = card_img[sy0:sy1, sx0:sx1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    card_face = (hsv[..., 1] < 50) & (hsv[..., 2] > 180)
    ink = ((hsv[..., 1] > 90) & (hsv[..., 2] > 70)) | (hsv[..., 2] < 100)
    # only ink enclosed by a card face (ignore felt/background around it)
    cnts, _ = cv2.findContours(card_face.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside = np.zeros(card_face.shape, np.uint8)
    for c in cnts:
        if cv2.contourArea(c) > 0.05 * card_face.size:
            cv2.drawContours(inside, [c], -1, 1, -1)
    ink &= inside.astype(bool) & ~card_face
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    comps = [k for k in range(1, n) if st[k, 4] > 0.004 * ink.size]
    if not comps:
        return None
    comps.sort(key=lambda k: st[k, 1])
    top = st[comps[0], 1]
    rank_c = [k for k in comps if st[k, 1] < top + 0.3 * st[comps[0], 3]]
    rank_bottom = max(st[k, 1] + st[k, 3] for k in rank_c)
    below = [k for k in comps if st[k, 1] >= rank_bottom - 2]
    color = _ink_color(hsv, np.isin(lab, rank_c))
    if not below:
        return {"color": color, "pip": None, "fill": None, "suit": None}
    k = max(below[:2], key=lambda c: st[c, 4]) if len(below) > 1 and st[below[1], 1] - st[below[0], 1] < 5 else below[0]
    x, y, ww, hh, a = st[k]
    pip = (lab[y:y + hh, x:x + ww] == k).astype(np.uint8) * 255
    pip = cv2.resize(pip, (24, 24), interpolation=cv2.INTER_AREA) > 127
    fill = a / float(ww * hh)
    suit = None
    if color == "red":
        suit = "h" if fill > 0.58 else "d"
    elif color == "blue":
        suit = "d"
    elif color == "green":
        suit = "c"
    return {"color": color, "pip": pip, "fill": round(float(fill), 3), "suit": suit}


FOUR_COLOR_SUIT = {"black": "s", "red": "h", "blue": "d", "green": "c"}
_seen_four_color = False   # set once a blue or green card shows up: the deck is four-colour


def _four_color_deck(lay, suits) -> bool:
    """layout "four_color_deck": true / false forces it; unset (default) = auto-detect.

    A two-colour deck never shows blue or green ink, so the first blue or green
    hero card proves the deck is four-colour for the rest of the session.
    """
    global _seen_four_color
    forced = lay.get("four_color_deck")
    if forced is not None:
        return bool(forced)
    if any(s and s["color"] in ("blue", "green") for s in suits):
        _seen_four_color = True
    return _seen_four_color


def decide_suited(suits, four_color: bool) -> bool | None:
    """Suited / offsuit from two _suit_info results. None = can't tell.

    Colour is checked first: different colours are always offsuit, and blue or
    green (or any colour in a known four-colour deck) maps to exactly one suit,
    so the same colour is suited without looking at the pips. Only red/red and
    black/black in a two-colour deck need the pip shapes compared.
    """
    if not (suits and len(suits) == 2 and all(suits)):
        return None
    a, b = suits
    if a["color"] != b["color"]:
        return False
    if four_color or a["color"] in ("blue", "green"):
        return True
    if a["pip"] is None or b["pip"] is None:
        return None
    iou = (a["pip"] & b["pip"]).sum() / max(1, (a["pip"] | b["pip"]).sum())
    a["iou"] = b["iou"] = round(float(iou), 3)
    return bool(iou > 0.72)


def read_hero_cards(img, felt, lay):
    fx, fy, fw, fh = felt
    u0, v0, u1, v1 = lay["hero_cards"]
    x0, y0 = int(fx + u0 * fw), int(fy + v0 * fh)
    x1, y1 = int(fx + u1 * fw), int(fy + v1 * fh)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if ((hsv[..., 1] < 40) & (hsv[..., 2] > 200)).mean() < 0.08:
        return None  # no white card faces
    scale = 3
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def ocr_ranks(im, s):
        res, _ = get_ocr()(im)
        out = []
        for box, txt, conf in res or []:
            t = _norm_text(txt).replace("O", "0").replace("I", "1")
            t = {"10": "T", "1O": "T", "0": "Q"}.get(t, t)
            if len(t) == 1 and t in "AKQJT98765432":
                xs = [p[0] * scale / s for p in box]
                ys = [p[1] * scale / s for p in box]
                out.append((min(xs), t, ((min(xs), min(ys)), (max(xs), max(ys))), float(conf)))
        out.sort()
        dedup = []  # one rank per card
        for r in out:
            if dedup and abs(r[0] - dedup[-1][0]) < 0.1 * up.shape[1]:
                continue
            dedup.append(r)
        return dedup[:2]

    ranks = ocr_ranks(up, scale)
    if len(ranks) < 2:  # single glyphs are easy to miss: retry other scales / greyscale
        gray = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        for s, src in ((2, crop), (4, crop), (3, gray)):
            alt = ocr_ranks(cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s)
            if len(alt) == 2:
                ranks = alt
                break
    if len(ranks) < 2:
        return {"ranks": [r[1] for r in ranks], "complete": False}
    suits = [_suit_info(up, r[2], scale) for r in ranks]
    four = _four_color_deck(lay, suits)
    suited = False if ranks[0][1] == ranks[1][1] else decide_suited(suits, four)
    out = {
        "ranks": [r[1] for r in ranks],
        "colors": [s["color"] if s else None for s in suits],
        "suits": [s["suit"] if s else None for s in suits],
        "suited": suited,
        "four_color": four,
        "complete": True,
    }
    if four:
        # four-colour deck: the colour is the suit
        out["suits"] = [FOUR_COLOR_SUIT.get(c) for c in out["colors"]]
    elif suits[0] and suits[1] and suits[0]["color"] == suits[1]["color"] in ("black", "red"):
        # same colour: the pip comparison decides; keep the display consistent with it
        a, b = ("s", "c") if suits[0]["color"] == "black" else (suits[0]["suit"] or "h", None)
        b = b or ("d" if a == "h" else "h")
        out["suits"] = [a, a if suited else b]
    return out


# ----------------------------------------------------------------- main entry
def read_table(img, layout=None, ocr_result=None) -> TableState:
    lay = dict(DEFAULT_LAYOUT)
    if layout:
        lay.update(layout)
    st = TableState()
    st.seats = [Seat(i) for i in range(len(lay["seats"]))]
    felt = find_felt(img, lay)
    if felt is None:
        st.error = "Table felt not found in the capture region"
        return st
    st.felt = list(felt)
    fx, fy, fw, fh = felt
    asp = fh / fw

    def uv(x, y):
        return (x - fx) / fw, (y - fy) / fh

    def dist(a, b):
        return math.hypot(a[0] - b[0], (a[1] - b[1]) * asp)

    t0 = time.time()
    if ocr_result is None:
        ocr_result, _ = get_ocr()(img)
    st.ocr_ms = int((time.time() - t0) * 1000)
    texts = []
    for box, txt, conf in ocr_result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        u, v = uv(cx, cy)
        texts.append({"t": txt, "n": _norm_text(txt), "x": cx, "y": cy, "u": u, "v": v,
                      "box": [min(xs), min(ys), max(xs), max(ys)], "conf": float(conf), "used": None})
    st.texts = texts

    # ---- blinds
    label = next((t for t in texts if t["n"].startswith("BLIND")), None)
    cands = []
    for t in texts:
        m = BLINDS_RE.search(t["n"])
        if m:
            try:
                sb, bb = _num(m.group(1)), _num(m.group(2))
                ante = _num(m.group(3)) if m.group(3) else 0.0
            except ValueError:
                continue
            score = 0.0
            if label is not None:
                score -= math.hypot(t["x"] - label["x"], t["y"] - label["y"])
            if not (1.4 <= bb / max(sb, 1e-9) <= 3.1):
                score -= 1e6
            cands.append((score, sb, bb, ante, t))
    if not cands:
        st.error = "Blinds text (e.g. 200/400(50)) not found"
        return st
    _, st.sb, st.bb, st.ante, bt = max(cands, key=lambda c: c[0])
    bt["used"] = "blinds"

    # ---- hero action buttons (below the table)
    for t in texts:
        if t["v"] > 1.2 and t["used"] is None:
            n = t["n"]
            for kw in ("FOLD", "CALL", "CHECK", "RAISE", "BET", "ALLIN"):
                if n.startswith(kw):
                    st.hero_buttons.append(n)
                    t["used"] = "button"
                    if kw == "CALL":
                        a = parse_amount(n[4:])
                        if a:
                            st.to_call_hint = a[0] if a[1] else a[0] / st.bb
                    break

    # ---- pot
    for t in texts:
        if t["n"].startswith("POT") and t["used"] is None:
            a = parse_amount(t["n"].split(":")[-1].replace("POT", ""))
            if a:
                st.pot = a[0] if a[1] else a[0] / st.bb
                t["used"] = "pot"

    # ---- hero card zone (ignore texts in it)
    hu0, hv0, hu1, hv1 = lay["hero_cards"]
    for t in texts:
        if t["used"] is None and hu0 <= t["u"] <= hu1 and hv0 <= t["v"] <= hv1:
            t["used"] = "hero_card_zone"

    # unit: are stacks shown in BB?
    bb_texts = sum(1 for t in texts if (a := parse_amount(t["n"])) and a[1])
    st.unit = "BB" if bb_texts >= 2 else "chips"

    def to_bb(a):
        v, is_bb = a
        return v if is_bb or st.unit == "BB" else v / st.bb

    R = lay["seat_radius"]
    seat_xy = [tuple(s["stack"]) for s in lay["seats"]]
    bet_xy = [tuple(s["bet"]) for s in lay["seats"]]

    # ---- stacks (amount nearest each stack anchor)
    for i, anc in enumerate(seat_xy):
        best = None
        for t in texts:
            if t["used"] is not None:
                continue
            n = t["n"]
            a = parse_amount(n)
            is_allin = n in ("ALLIN", "ALL-IN")
            if not a and not is_allin:
                continue
            if a and st.unit == "BB" and not a[1]:
                continue  # bare number in BB mode = timer, not a stack
            d = dist((t["u"], t["v"]), anc)
            if d < R and (best is None or d < best[0]):
                best = (d, t, a, is_allin)
        if best:
            _, t, a, is_allin = best
            seat = st.seats[i]
            seat.present = True
            seat.stack = 0.0 if is_allin else to_bb(a)
            if is_allin:
                seat.labels.append("allin")
            t["used"] = f"stack{i}"

    # ---- labels, names, timers
    for t in texts:
        if t["used"] is not None:
            continue
        d_all = [dist((t["u"], t["v"]), a) for a in seat_xy]
        i = int(np.argmin(d_all))
        if d_all[i] > R * 1.25:
            continue
        n = t["n"]
        if n in LABELS:
            st.seats[i].labels.append(LABELS[n])
            st.seats[i].present = True
            t["used"] = f"label{i}"
        elif n.isdigit() and int(n) <= 99 and st.unit == "BB":
            st.seats[i].acting = True
            t["used"] = f"timer{i}"
        elif t["v"] < seat_xy[i][1] and not parse_amount(n):
            if not st.seats[i].name:
                st.seats[i].name = t["t"]
                st.seats[i].present = True
                t["used"] = f"name{i}"

    # ---- bets: remaining amounts on/near the felt
    for t in texts:
        if t["used"] is not None:
            continue
        a = parse_amount(t["n"])
        if not a or not (-0.1 < t["u"] < 1.1 and -0.15 < t["v"] < 1.05):
            continue
        if st.unit == "BB" and not a[1]:
            continue
        d_all = [dist((t["u"], t["v"]), b) for b in bet_xy]
        i = int(np.argmin(d_all))
        if d_all[i] <= lay["bet_radius"]:
            st.seats[i].bet += to_bb(a)
            t["used"] = f"bet{i}"

    # ---- card backs
    du0, dv0, du1, dv1 = lay["card_back_box"]
    for i, (su, sv) in enumerate(seat_xy):
        if i == 0:
            continue
        box = (int(fx + (su + du0) * fw), int(fy + (sv + dv0) * fh),
               int(fx + (su + du1) * fw), int(fy + (sv + dv1) * fh))
        f = card_back_fraction(img, box)
        st.seats[i].card_back_frac = round(f, 3)
        st.seats[i].has_cards = f >= lay["card_back_min_frac"]

    # ---- hero cards
    st.hero_cards = read_hero_cards(img, felt, lay)
    st.seats[0].has_cards = bool(st.hero_cards and st.hero_cards.get("ranks"))

    # ---- blinds / button seats
    for s in st.seats:
        if "sb" in s.labels:
            st.sb_seat = s.idx
        if "bb" in s.labels:
            st.bb_seat = s.idx
    buttons = find_dealer_buttons(img, felt)
    if buttons:
        cu, cv = 0.5, 0.5
        bx, by, _ = max(buttons, key=lambda b: b[2])
        bu, bv = uv(bx, by)
        ang_b = math.atan2((bv - cv) * asp, bu - cu)
        best = None
        for s in st.seats:
            if not s.present:
                continue
            su, sv = seat_xy[s.idx]
            ang = math.atan2((sv - cv) * asp, su - cu)
            dd = abs((ang - ang_b + math.pi) % (2 * math.pi) - math.pi)
            if best is None or dd < best[0]:
                best = (dd, s.idx)
        if best:
            st.button_seat = best[1]
            st.button_source = "button"
    st.ok = True
    return st


def annotate(img, st: TableState, layout=None):
    """Debug overlay: felt, anchors and what each OCR box was used as."""
    lay = dict(DEFAULT_LAYOUT)
    if layout:
        lay.update(layout)
    out = img.copy()
    if st.felt:
        fx, fy, fw, fh = st.felt
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
        for i, s in enumerate(lay["seats"]):
            for key, col in (("stack", (255, 200, 0)), ("bet", (0, 140, 255))):
                u, v = s[key]
                p = (int(fx + u * fw), int(fy + v * fh))
                cv2.circle(out, p, 6, col, -1)
                cv2.putText(out, f"{i}", (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for t in st.texts:
        x0, y0, x1, y1 = map(int, t["box"])
        col = (0, 200, 0) if t["used"] else (0, 0, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 1)
        if t["used"]:
            cv2.putText(out, t["used"], (x0, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return out
