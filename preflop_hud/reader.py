"""Turn a screenshot of the table into a structured TableState.

Pipeline:  full-frame OCR  ->  find felt  ->  assign every text box to a seat
(stack / name / status label / bet) by nearest felt-relative anchor  ->
non-text detectors for card backs, the dealer button and hero hole cards.
"""
from __future__ import annotations

import math
import re
import sys
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
    board: list = field(default_factory=list)   # community cards, left to right: {rank, suit, color}
    to_call_hint: float | None = None   # from the "Call X" action button
    hero_buttons: list = field(default_factory=list)
    felt: list | None = None
    ocr_ms: int = 0
    texts: list = field(default_factory=list)  # for the debug overlay
    seat_layout: list | None = None            # the seat anchors this frame was read with
    clipped_seats: list = field(default_factory=list)   # {seat, edge}: stack text outside the capture

    def to_dict(self):
        d = asdict(self)
        for s, so in zip(d["seats"], self.seats):
            s["folded"], s["sitting_out"], s["all_in"] = so.folded, so.sitting_out, so.all_in
        return d


# ----------------------------------------------------------------- detectors
def find_felt(img, lay):
    """Bounding box of the felt: the largest blob of any of the felt colours (green, then the
    other themes in "felt_hsv_extra", e.g. the red table). Each colour is a list of HSV ranges
    (red wraps around hue 0, so it takes two)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colours = [[(lay["felt_hsv_lo"], lay["felt_hsv_hi"])]] + [
        [tuple(r) for r in c] for c in lay.get("felt_hsv_extra", [])]
    best = None
    for ranges in colours:
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, tuple(lo), tuple(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        if best is None or st[i, 4] > best[4]:
            best = st[i]
    if best is None:
        return None
    x, y, w, h, area = best
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
        # circular mean: red sits at both ends of OpenCV's 0..180 hue scale, and a plain median of
        # a red glyph's pixels (some near 0, some near 180, a few in between) can land on green
        # or blue: a 2 of hearts read as clubs, a 5 of hearts as diamonds
        ang = hsv[..., 0][colored].astype(np.float64) * (np.pi / 90.0)
        hue = float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
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
    box_x0, box_x1 = x0, x1                 # this card's own glyph lies within these columns
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

    def cx(k):
        return sx0 + st[k, 0] + st[k, 2] / 2      # blob centre, in card_img columns

    # the rank glyph is the top ink lined up with this card's OCR box: with overlapping hole cards
    # the next card's rank sits at the same height in the strip (a blue 4 beside a red 5) and
    # would otherwise lend the 5 its colour
    margin = 0.15 * (box_x1 - box_x0)
    own = [k for k in comps if box_x0 - margin <= cx(k) <= box_x1 + margin] or comps
    top_k = own[0]
    top = st[top_k, 1]
    rank_c = [k for k in own if st[k, 1] < top + 0.3 * st[top_k, 3]]
    rank_bottom = max(st[k, 1] + st[k, 3] for k in rank_c)
    gx0 = min(st[k, 0] for k in rank_c)
    gx1 = max(st[k, 0] + st[k, 2] for k in rank_c)
    gw = gx1 - gx0
    # the pip: the next ink down, under this glyph (not the neighbouring card's)
    below = [k for k in comps if st[k, 1] >= rank_bottom - 2
             and gx0 - 0.5 * gw <= st[k, 0] + st[k, 2] / 2 <= gx1 + 0.5 * gw]
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


# ----------------------------------------------------------------- rank glyphs
RANK_CHARS = "AKQJT98765432"


def _glyph_mask(bgr):
    """The rank glyph in a region around one OCR box: ink (saturated, or dark against the card face)
    that doesn't touch the region's edge (card borders, felt), keeping the biggest blob plus blobs
    of similar height beside it ("10" is two). Bool mask cropped to the glyph, or None."""
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(float)
    pale = hsv[..., 1] < 50
    face_v = np.percentile(v[pale], 90) if pale.sum() > 10 else 255.0
    ink = (hsv[..., 1] > 80) | (v < 0.62 * face_v)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    H, W = ink.shape
    inner = [k for k in range(1, n) if st[k, 0] > 0 and st[k, 1] > 0
             and st[k, 0] + st[k, 2] < W and st[k, 1] + st[k, 3] < H and st[k, 4] > 8]
    if not inner:
        return None
    big = max(inner, key=lambda k: st[k, 4])
    by, bh = st[big, 1], st[big, 3]
    keep = [k for k in inner if st[k, 3] > 0.6 * bh and abs(st[k, 1] - by) < 0.25 * bh]
    x0, y0 = min(st[k, 0] for k in keep), min(st[k, 1] for k in keep)
    x1 = max(st[k, 0] + st[k, 2] for k in keep)
    y1 = max(st[k, 1] + st[k, 3] for k in keep)
    return np.isin(lab, keep)[y0:y1, x0:x1]


def _glyph_image(mask):
    """Black-on-white, ~40px tall, padded: what the recogniser reads best for a single glyph."""
    g = np.where(mask, 0, 255).astype(np.uint8)
    g = cv2.resize(g, None, fx=40 / g.shape[0], fy=40 / g.shape[0], interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.copyMakeBorder(g, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)


def _decode_rank(glyph_bgr):
    """Read one glyph with the OCR recogniser restricted to rank characters (A K Q J 10 9..2), so an
    unusual glyph (this app's tall Q) can't come out as "n" or "?". Returns a rank or None."""
    rec = get_ocr().text_rec
    chars = rec.postprocess_op.character
    h, w = glyph_bgr.shape[:2]
    norm = rec.resize_norm_img(glyph_bgr, max(320 / 48, w / h))      # as rapidocr: room for "10"
    preds = rec.session(norm[None].astype(np.float32))[0][0]          # (time, classes)
    allowed = [0] + [i for i, c in enumerate(chars) if c in "AKQJ1098765432"]   # 0 = CTC blank
    sub = preds[:, allowed]
    idx = sub.argmax(1)
    out, prev = [], 0
    for k in idx:
        if k != 0 and k != prev:
            out.append(chars[allowed[k]])
        prev = k
    if not out:                    # nothing beat "blank": the most likely rank character
        j = int(np.unravel_index(sub[:, 1:].argmax(), sub[:, 1:].shape)[1]) + 1
        out = [chars[allowed[j]]]
    s = "".join(out)
    if s.startswith("1") and "0" not in s and _ink_blobs(glyph_bgr) < 2:
        # "10" is read as a 1 then a 0 (its digits can touch at small sizes, so it may be one shape);
        # a lone "1" on a single shape is a narrow 7 (or similar): the best rank that isn't 1 or 0
        keep = [k for k in range(1, len(allowed)) if chars[allowed[k]] not in "10"]
        j = keep[int(np.unravel_index(sub[:, keep].argmax(), sub[:, keep].shape)[1])]
        s = chars[allowed[j]]
    if s.startswith("1"):
        return "T"
    if s[:1] == "0":               # no 0 rank: a lone 0 is a Q
        return "Q"
    return s[:1] if s[:1] in RANK_CHARS else None


def _ink_blobs(glyph_bgr):
    """Separate dark shapes in a black-on-white glyph image (ignoring specks)."""
    ink = (glyph_bgr[..., 0] < 128).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    total = max(1, int(ink.sum()))
    return sum(1 for k in range(1, n) if st[k, 4] > 0.05 * total)


def _lowercase_word(txt):
    """OCR text that's only letters with a lowercase one: this app's tall Q comes back as "a" (and
    would be taken for an A once upper-cased). Ranks are capitals or digits, so it's unread."""
    t = txt.strip()
    return t.isalpha() and t != t.upper()


def _fix_2_7(rank, mask):
    """2 and 7 get confused; a 2 has a flat base, a 7 ends in a narrow stroke."""
    if rank not in ("2", "7") or mask is None or mask.shape[0] < 6:
        return rank
    base = mask[int(0.85 * mask.shape[0]):, :]
    cols = np.nonzero(base.any(axis=0))[0]
    width = (cols.max() - cols.min() + 1) / mask.shape[1] if cols.size else 0.0
    return "2" if width >= 0.55 else "7"


def _hero_rank_glyphs(up):
    """Rank glyphs of the hole cards found from the card faces, without OCR text detection (which
    misses this app's red A and 2, and its tall Q): the face union is the white blob; the ranks
    are the tallest ink blobs along its top edge ("10" = two blobs side by side). The suit pips
    sit lower and are left out. Returns [(x0, y0, x1, y1, mask)] left to right."""
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(float)
    pale = hsv[..., 1] < 45
    if not pale.any():
        return []
    face_v = np.percentile(v[pale], 90)
    face = (pale & (v > 0.80 * face_v)).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(face)
    if n < 2:
        return []
    k = 1 + int(np.argmax(st[1:, 4]))
    fy, fh = st[k, 1], st[k, 3]
    filled = np.zeros_like(face)
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    ink = (filled > 0) & (face == 0) & ((hsv[..., 1] > 80) | (v < 0.62 * face_v))
    m, l2, s2, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    cand = [j for j in range(1, m) if s2[j, 1] < fy + 0.35 * fh and s2[j, 3] >= 0.15 * fh and s2[j, 4] > 20]
    if not cand:
        return []
    top = min(s2[j, 1] for j in cand)
    gh = max(s2[j, 3] for j in cand if s2[j, 1] < top + 0.15 * fh)
    cand = [j for j in cand if s2[j, 1] < top + 0.4 * gh and s2[j, 3] >= 0.55 * gh]   # the rank row
    cand.sort(key=lambda j: s2[j, 0])
    groups = []
    for j in cand:
        if groups and s2[j, 0] - max(s2[i, 0] + s2[i, 2] for i in groups[-1]) < 0.2 * gh:
            groups[-1].append(j)
        else:
            groups.append([j])
    out = []
    for g in groups:
        x0, y0 = min(s2[i, 0] for i in g), min(s2[i, 1] for i in g)
        x1, y1 = max(s2[i, 0] + s2[i, 2] for i in g), max(s2[i, 1] + s2[i, 3] for i in g)
        out.append((x0, y0, x1, y1, np.isin(l2, g)[y0:y1, x0:x1]))
    return out


def _find_other_rank(up, found):
    """The second hole card's rank when OCR found only one: ink beside the found glyph, at its
    height and of a similar size. Returns a rank entry like ocr_ranks' (x, rank, box, conf) or None."""
    _, _, ((bx0, by0), (bx1, by1)), _ = found
    h = by1 - by0
    H, W = up.shape[:2]
    ry0, ry1 = max(0, int(by0 - 0.25 * h)), min(H, int(by1 + 0.25 * h))
    best = None
    for side in ("left", "right"):
        cx0, cx1 = (0, int(bx0 - 0.1 * h)) if side == "left" else (int(bx1 + 0.1 * h), W)
        if cx1 - cx0 < 0.3 * h:
            continue
        band = up[ry0:ry1, cx0:cx1]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2].astype(float)
        pale = hsv[..., 1] < 50
        face_v = np.percentile(v[pale], 90) if pale.sum() > 10 else 255.0
        ink = (hsv[..., 1] > 80) | (v < 0.62 * face_v)
        # felt shows through on the outer side of the cards: only ink with card face around it
        ink &= cv2.dilate(pale.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        n, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
        for k in range(1, n):
            x, y, w, hh, a = st[k]
            if not (0.6 * h <= hh <= 1.5 * h) or abs((ry0 + y) - by0) > 0.35 * h or a < 0.05 * w * hh:
                continue
            dist = (cx1 - (x + w)) if side == "left" else x        # distance from the found glyph
            if best is None or dist < best[0]:
                best = (dist, cx0 + x, ry0 + y, cx0 + x + w, ry0 + y + hh)
    if best is None:
        return None
    _, gx0, gy0, gx1, gy1 = best
    pad = 0.15 * (gy1 - gy0)
    mask = _glyph_mask(up[max(0, int(gy0 - pad)):int(gy1 + pad), max(0, int(gx0 - pad)):int(gx1 + pad)])
    if mask is None:
        return None
    r = _fix_2_7(_decode_rank(_glyph_image(mask)), mask)
    return (gx0, r, ((gx0, gy0), (gx1, gy1)), 0.0) if r else None


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
        cands = []
        for box, txt, conf in res or []:
            t = _norm_text(txt).replace("O", "0").replace("I", "1")
            if not t or len(t) > 2:
                continue                       # labels such as "HIGH CARD"
            r = {"10": "T", "1O": "T", "0": "Q"}.get(t, t)
            r = r if len(r) == 1 and r in RANK_CHARS and not _lowercase_word(txt) else None   # None: read the glyph
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            cands.append((min(xs) * scale / s, min(ys), r, (min(xs), min(ys), max(xs), max(ys)), float(conf)))
        cands.sort()
        groups = []                            # one box per card: boxes this close belong together
        for c in cands:
            if groups and c[0] - groups[-1][0][0] < 0.1 * up.shape[1]:
                groups[-1].append(c)
            else:
                groups.append([c])
        out = []
        for g in groups:
            # the rank rather than the suit pip under it: a readable rank first, then the top-most box
            x, _, r, (bx0, by0, bx1, by1), conf = min(g, key=lambda c: (c[2] is None, c[1]))
            if r is None or r in ("2", "7"):
                pad = 0.12 * (by1 - by0)
                mask = _glyph_mask(im[max(0, int(by0 - pad)):int(by1 + pad), max(0, int(bx0 - pad)):int(bx1 + pad)])
                if r is None and mask is not None:
                    r = _decode_rank(_glyph_image(mask))
                r = _fix_2_7(r, mask)
            if r is None:
                continue
            k = scale / s
            out.append((x, r, ((bx0 * k, by0 * k), (bx1 * k, by1 * k)), conf))
        return out[:2]

    # first: the glyphs found from the card faces, each read by the rank-only recogniser
    ranks = []
    for (gx0, gy0, gx1, gy1, mask) in _hero_rank_glyphs(up):
        r = _fix_2_7(_decode_rank(_glyph_image(mask)), mask)
        if r:
            ranks.append((gx0, r, ((gx0, gy0), (gx1, gy1)), 1.0))
    if len(ranks) != 2:                  # fall back to OCR text detection
        ranks = ocr_ranks(up, scale)
    if len(ranks) < 2:  # single glyphs are easy to miss: retry other scales / greyscale
        gray = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        for s, src in ((2, crop), (4, crop), (3, gray)):
            alt = ocr_ranks(cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), s)
            if len(alt) == 2:
                ranks = alt
                break
    if len(ranks) == 1:
        # text detection can miss a glyph entirely (this app's tall Q): both ranks sit at the
        # same height, so look beside the one we found for a glyph of the same size
        other = _find_other_rank(up, ranks[0])
        if other is not None:
            ranks = sorted(ranks + [other])
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


RANK_FIX = {"10": "T", "1O": "T", "IO": "T", "LO": "T", "0": "Q", "O": "Q", "1": "T"}


def _rank_token(txt):
    t = _norm_text(txt).replace("I", "1")
    t = RANK_FIX.get(t, t)
    return t if len(t) == 1 and t in "AKQJT98765432" else None


def _find_board_cards(hsv, felt_h):
    """Card-face boxes (x0, y0, x1, y1) in a board crop, left to right; touching cards are split."""
    face = ((hsv[..., 1] < 50) & (hsv[..., 2] > 185)).astype(np.uint8)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(face)
    cards = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if not (0.15 * felt_h < h < 0.40 * felt_h) or a < 0.45 * w * h:
            continue                                   # not card-sized / not a solid face (logo text)
        cw = 0.72 * h                                  # face width / height of these cards
        if w < 0.5 * cw:
            continue
        gap = 0.07 * h
        k_cards = max(1, int(round((w + gap) / (cw + gap))))
        step = (w + gap) / k_cards
        for j in range(k_cards):
            cards.append((int(x + j * step), int(y), int(x + j * step + step - gap), int(y + h)))
    cards.sort()
    return cards[:5]


def _rank_glyph(hsv_card):
    """The rank glyph of one card as black-on-white, ~40px tall: the top ink blob(s) of the corner
    ("10" is two blobs side by side), without the suit pip under it. None if there's no ink."""
    ch, cw = hsv_card.shape[:2]
    c = hsv_card[:int(0.5 * ch), :int(0.6 * cw)]
    ink = (~((c[..., 1] < 60) & (c[..., 2] > 170))).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    comps = [k for k in range(1, n) if st[k, 4] > 0.01 * ink.size and st[k, 0] > 0 and st[k, 1] > 0]
    if not comps:
        return None
    comps.sort(key=lambda k: st[k, 1])
    top, h0 = st[comps[0], 1], st[comps[0], 3]
    rank = [k for k in comps if st[k, 1] < top + 0.3 * h0]
    x0, y0 = min(st[k, 0] for k in rank), min(st[k, 1] for k in rank)
    x1 = max(st[k, 0] + st[k, 2] for k in rank)
    y1 = max(st[k, 1] + st[k, 3] for k in rank)
    g = np.where(np.isin(lab, rank), 0, 255).astype(np.uint8)[y0:y1, x0:x1]
    if g.size == 0:
        return None
    g = cv2.resize(g, None, fx=40 / g.shape[0], fy=40 / g.shape[0], interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cv2.copyMakeBorder(g, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255),
                        cv2.COLOR_GRAY2BGR)


def read_board(img, felt, lay):
    """Community cards: white faces in the middle of the felt. The rank comes from OCR, the suit
    from the colour of the rank glyph (four-colour deck: red hearts, blue diamonds, green clubs,
    black spades). Returns [{"rank", "suit", "color"}] left to right (rank None if unread)."""
    fx, fy, fw, fh = felt
    u0, v0, u1, v1 = lay.get("board", DEFAULT_LAYOUT["board"])
    H, W = img.shape[:2]
    x0, y0 = max(0, int(fx + u0 * fw)), max(0, int(fy + v0 * fh))
    x1, y1 = min(W, int(fx + u1 * fw)), min(H, int(fy + v1 * fh))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    boxes = _find_board_cards(hsv, fh)
    if not boxes:
        return []
    # each card's rank glyph, read by the rank-only recogniser. Over every recorded frame this got
    # 94 of 2173 board ranks wrong (nearly all mid-animation) against 213 for reading the board with
    # one OCR pass first: that pass's text-direction step turns a 6 into a 9 and reads a 4 as a 10
    ranks = []
    for (bx0, by0, bx1, by1) in boxes:
        r = None
        g = _rank_glyph(hsv[by0:by1, bx0:bx1])
        if g is not None:
            mask = g[..., 0] < 128
            ys, xs = np.nonzero(mask)
            mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else None
            r = _decode_rank(g)
            if r is not None:
                r = _fix_2_7(r, mask)
        ranks.append(r)
    if any(r is None for r in ranks):
        # a glyph that couldn't be isolated or read: one OCR pass over the board, upscaled so a single
        # glyph is ~100px tall, for those cards only
        card_h = boxes[0][3] - boxes[0][1]
        scale = max(1.0, min(4.0, 300.0 / max(card_h, 1)))
        up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        tokens = [[] for _ in boxes]
        for box, txt, conf in get_ocr()(up)[0] or []:
            cx = np.mean([p[0] for p in box]) / scale
            cy = np.mean([p[1] for p in box]) / scale
            for i, (bx0, by0, bx1, by1) in enumerate(boxes):
                if bx0 <= cx <= bx1 and by0 <= cy <= by0 + 0.5 * (by1 - by0) and not _lowercase_word(txt):
                    tokens[i].append(_norm_text(txt).replace("I", "1"))

        def pick(toks):
            # "10" can come back split as "1" + "0", and a lone "0" is otherwise a misread Q
            if any(t.startswith("1") for t in toks):
                return "T"
            return next((r for r in map(_rank_token, toks) if r), None)
        ranks = [r if r is not None else pick(t) for r, t in zip(ranks, tokens)]
    out = []
    for i, (bx0, by0, bx1, by1) in enumerate(boxes):
        corner = (slice(by0, by0 + int(0.45 * (by1 - by0))), slice(bx0, bx0 + int(0.5 * (bx1 - bx0))))
        h = hsv[corner]
        ink = ~((h[..., 1] < 50) & (h[..., 2] > 185))
        color = _ink_color(h, ink) if ink.any() else None
        out.append({"rank": ranks[i], "color": color,
                    "suit": {"red": "h", "blue": "d", "green": "c", "black": "s"}.get(color)})
    if any(c["color"] in ("blue", "green") for c in out):
        global _seen_four_color            # a blue or green board card proves a four-colour deck too
        _seen_four_color = True
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
    # which seat layout this table uses: the one with more stack amounts next to its anchors
    amounts = [(t["u"], t["v"]) for t in texts if parse_amount(t["n"]) and
               (st.unit != "BB" or parse_amount(t["n"])[1])]

    def fit(seats):
        return sum(1 for s in seats if any(dist(a, tuple(s["stack"])) < R for a in amounts))
    best = max([lay["seats"]] + list(lay.get("seats_alt") or []), key=fit)   # max keeps the first on a tie
    if best is not lay["seats"] and fit(best) > fit(lay["seats"]):
        lay = dict(lay, seats=best)
        st.seats = [Seat(i) for i in range(len(best))]
    st.seat_layout = [dict(s) for s in lay["seats"]]
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

    # ---- seats the capture region cuts off: a seat whose stack text lies outside the frame is never
    # read, so the table counts one player short and the seats acting before it are named a position
    # late. Past the edge (or on it) always counts; a stack-text half-width inside it counts when no
    # stack was read there, as the text is then partly out of frame
    img_h, img_w = img.shape[:2]
    for i, (su, sv) in enumerate(seat_xy):
        x, y = fx + su * fw, fy + sv * fh
        gaps = {"left": x, "top": y, "right": img_w - x, "bottom": img_h - y}
        edge = min(gaps, key=gaps.get)
        limit = lay["seat_edge_margin"] if st.seats[i].stack is not None else lay["seat_edge_margin_unread"]
        if gaps[edge] < limit * fw:
            st.clipped_seats.append({"seat": i, "edge": edge})

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

    # ---- community cards
    try:
        st.board = read_board(img, felt, lay)
    except Exception as e:          # never lose the table over it (the tracker waits for a second empty read)
        st.board = []
        print(f"board read failed: {type(e).__name__}: {e}", file=sys.stderr)

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
        for i, s in enumerate(st.seat_layout or lay["seats"]):
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
