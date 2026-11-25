"""Card ranks by template: this app's 13 rank glyphs, learned from recorded frames.

The general-purpose text recogniser keeps misreading this deck's glyphs (a narrow 7 as 10, the tall Q
as "a", a 6 as 9, a 4 as 10, a touching 10 as 3). The deck only ever shows 13 shapes, so each glyph is
compared with labelled examples of them instead: binarised, scaled into a fixed box (keeping its
aspect), blurred a little, and matched to the nearest example (1-NN). The recogniser stays as the
fallback when the nearest example isn't clearly closer than the nearest of another rank.

    python -m preflop_hud.rank_templates build      # harvest from recordings/ -> rank_templates.npz
    python -m preflop_hud.rank_templates check      # leave-one-recording-out accuracy vs the recogniser

Labels come from the hands' known cards (TRUTH below: boards and hole cards checked on screenshots); a
glyph is taken as an example only where today's reader agrees with that label, so a mid-animation
frame (a card one place over) can't teach a wrong shape.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATES = os.path.join(HERE, "rank_templates.npz")
BOX = 32
RANKS = "AKQJT98765432"
MIN_MARGIN = 0.15          # the nearest example of another rank must be this much further (relative)

# hand's known cards per recording: board left to right, hole cards (rank pairs, either order)
TRUTH = {
    "20261124-181826": {"hero": "A2"},
    "20261124-182807": {"board": "Kd Ad 5c Qs 2h", "hero": "KJ"},
    "20261124-183201": {"board": "2d 4s 4h Td Qd", "hero": "K9"},
    "20261124-183932": {"board": "Jd 6h 9h 4c Qd", "hero": "QT"},
    "20261124-184015": {"board": "9h 4h 5c 2s 3h", "hero": "KJ"},
    "20261125-132801": {"board": "Qc Kd 8c 8s 8d", "hero": "77"},
    "20261125-133413": {"hero": "53"},
    "20261125-133421": {"hero": "53"},
    "20261125-133537": {"hero": "87"},
    "20261125-133748": {"board": "Kh Kd 8d 2s 6d", "hero": "99"},
    "20261125-135016": {"board": "2h 4d Ks Kc 9c", "hero": "88"},
    "20261125-152108": {"board": "6c 9c Jh Qs 3h", "hero": "53"},
}

_cache = None


def glyph_vector(mask):
    """A glyph mask (bool, cropped to the ink) as a fixed-size float vector: scaled to fit BOX x BOX
    keeping its aspect (a "10" stays wide, a "1"-like 7 narrow), centred, lightly blurred."""
    import cv2
    m = np.asarray(mask, np.uint8) * 255
    ys, xs = np.nonzero(m)
    if not ys.size:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = (BOX - 2) / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((BOX, BOX), np.float32)
    y0, x0 = (BOX - nh) // 2, (BOX - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r / 255.0
    out = cv2.GaussianBlur(out, (3, 3), 0.8)
    return out.ravel()


def load():
    global _cache
    if _cache is None:
        if not os.path.exists(TEMPLATES):
            _cache = (np.zeros((0, BOX * BOX), np.float32), np.array([], dtype="<U1"))
        else:
            z = np.load(TEMPLATES)
            _cache = (z["vectors"].astype(np.float32) / 255.0, z["ranks"])
    return _cache


def classify(mask, examples=None):
    """(rank, distance, margin) of a glyph mask, or (None, None, None) with no templates. margin: how much
    further (relative) the nearest example of any other rank is."""
    vecs, ranks = examples if examples is not None else load()
    v = glyph_vector(mask)
    if v is None or not len(vecs):
        return None, None, None
    d = np.sqrt(((vecs - v) ** 2).sum(axis=1))
    i = int(d.argmin())
    r = str(ranks[i])
    other = d[ranks != r]
    margin = float((other.min() - d[i]) / max(other.min(), 1e-9)) if other.size else 1.0
    return r, float(d[i]), margin


def rank_of(mask, glyph_bgr=None):
    """The rank of one glyph: the nearest template when it's clear, else the recogniser (glyph_bgr given)."""
    r, _, margin = classify(mask)
    if r is not None and margin >= MIN_MARGIN:
        return r
    if glyph_bgr is not None:
        from .reader import _decode_rank, _fix_2_7
        return _fix_2_7(_decode_rank(glyph_bgr), mask)
    return r


# ---------------------------------------------------------------- harvesting labelled glyphs
def _board_glyphs(img, felt, lay):
    import cv2
    from .reader import _find_board_cards, _rank_glyph
    fx, fy, fw, fh = felt
    u0, v0, u1, v1 = lay["board"]
    H, W = img.shape[:2]
    crop = img[max(0, int(fy + v0 * fh)):min(H, int(fy + v1 * fh)), max(0, int(fx + u0 * fw)):min(W, int(fx + u1 * fw))]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    out = []
    for (bx0, by0, bx1, by1) in _find_board_cards(hsv, fh)[:5]:
        g = _rank_glyph(hsv[by0:by1, bx0:bx1])
        if g is None:
            out.append((None, None))
            continue
        mask = g[..., 0] < 128
        ys, xs = np.nonzero(mask)
        out.append((g, mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else None))
    return out


def _hero_glyphs(img, felt, lay):
    import cv2
    from .reader import _glyph_image, _hero_rank_glyphs
    fx, fy, fw, fh = felt
    u0, v0, u1, v1 = lay["hero_cards"]
    H, W = img.shape[:2]
    crop = img[max(0, int(fy + v0 * fh)):min(H, int(fy + v1 * fh)), max(0, int(fx + u0 * fw)):min(W, int(fx + u1 * fw))]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if ((hsv[..., 1] < 40) & (hsv[..., 2] > 200)).mean() < 0.08:
        return []
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return [(_glyph_image(m), m) for (_, _, _, _, m) in _hero_rank_glyphs(up)]


def harvest(recordings=None, verbose=True):
    """[(recording, frame, "board"|"hero", label, mask, recogniser's read)] from recorded frames with known cards."""
    import cv2
    from .layout import DEFAULT_LAYOUT
    from .reader import _decode_rank, _fix_2_7, find_felt
    out = []
    for rec, truth in TRUTH.items():
        if recordings is not None and rec not in recordings:
            continue
        frames = sorted(glob.glob(os.path.join(ROOT, "recordings", rec, "*.png")))
        board = [c[0] for c in truth.get("board", "").split()]
        hero = sorted(truth.get("hero", ""))
        for f in frames:
            img = cv2.imread(f)
            felt = find_felt(img, DEFAULT_LAYOUT)
            if felt is None:
                continue
            for i, (g, m) in enumerate(_board_glyphs(img, felt, DEFAULT_LAYOUT)):
                if m is not None and i < len(board):
                    out.append((rec, os.path.basename(f), "board", board[i], m, _fix_2_7(_decode_rank(g), m)))
            hg = _hero_glyphs(img, felt, DEFAULT_LAYOUT)
            if len(hg) == 2 and hero:
                reads = [_fix_2_7(_decode_rank(g), m) for g, m in hg]
                if sorted(reads) == hero:                 # labels by position, from a read that matches
                    for (g, m), r in zip(hg, reads):
                        out.append((rec, os.path.basename(f), "hero", r, m, r))
                else:
                    for (g, m), r in zip(hg, reads):
                        out.append((rec, os.path.basename(f), "hero?", None, m, r))
        if verbose:
            print(f"  {rec}: {sum(1 for x in out if x[0] == rec)} glyphs", flush=True)
    return out


def _examples(rows, cap_per_rank=60, seed=0):
    """Training examples: glyphs whose label the recogniser agreed with (clean shapes), capped per rank
    and spread over recordings, as vectors."""
    rng = np.random.default_rng(seed)
    by = {}
    for rec, f, kind, label, m, read in rows:
        if label is None or read != label:
            continue
        by.setdefault(label, []).append((rec, m))
    vecs, ranks = [], []
    for r, items in by.items():
        idx = rng.permutation(len(items))[:cap_per_rank]
        for i in idx:
            v = glyph_vector(items[i][1])
            if v is not None:
                vecs.append(v)
                ranks.append(r)
    return np.array(vecs, np.float32), np.array(ranks)


def build():
    print("harvesting labelled glyphs from recordings/ …")
    rows = harvest()
    vecs, ranks = _examples(rows)
    missing = [r for r in RANKS if r not in set(ranks)]
    np.savez_compressed(TEMPLATES, vectors=np.round(vecs * 255).astype(np.uint8), ranks=ranks)
    print(f"{len(vecs)} examples, ranks: " + " ".join(f"{r}:{int((ranks == r).sum())}" for r in RANKS)
          + (f"  (no examples of {''.join(missing)})" if missing else "") + f" -> {TEMPLATES}")


def check():
    """Leave one recording out: templates from the others, scored on its glyphs with a known label (board
    positions as the board shows them, hole cards whose read matched the hand), against the recogniser."""
    rows = harvest()
    labelled = [x for x in rows if x[3] is not None]
    recs = sorted({x[0] for x in labelled})
    tot = {"template": 0, "template+fallback": 0, "recogniser": 0}
    n = 0
    bad = {k: {} for k in tot}
    for rec in recs:
        ex = _examples([x for x in rows if x[0] != rec])
        for (r, f, kind, label, m, read) in (x for x in labelled if x[0] == rec):
            t, _, margin = classify(m, ex)
            tf = t if margin is not None and margin >= MIN_MARGIN else read
            n += 1
            for k, v in (("template", t), ("template+fallback", tf), ("recogniser", read)):
                if v != label:
                    tot[k] += 1
                    bad[k][(rec, kind, label, v)] = bad[k].get((rec, kind, label, v), 0) + 1
    print(f"{n} labelled glyphs in {len(recs)} recordings (board ones include mid-animation frames):")
    for k in tot:
        worst = sorted(bad[k].items(), key=lambda kv: -kv[1])[:6]
        print(f"  {k:18s} wrong {tot[k]:4d}  {worst}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"build": build, "check": check}[cmd]()
