"""Hand classes (the 169 starting hands), combo weights and range-string parsing.

Grid convention (same as GTO Wizard / most solvers):
    row i, col j over RANKS = "AKQJT98765432"
    i == j  -> pocket pair            (e.g. "TT")
    i <  j  -> suited, upper-right     (e.g. "AKs")
    i >  j  -> offsuit, lower-left     (e.g. "AKo")
Class index = i * 13 + j.
"""
from __future__ import annotations

import re
import numpy as np

RANKS = "AKQJT98765432"
RANK_VALUE = {r: 12 - i for i, r in enumerate(RANKS)}  # A=12 ... 2=0
SUITS = "shdc"


def hand_name(i: int, j: int) -> str:
    if i == j:
        return RANKS[i] * 2
    if i < j:
        return RANKS[i] + RANKS[j] + "s"
    return RANKS[j] + RANKS[i] + "o"


HAND_NAMES = [hand_name(i, j) for i in range(13) for j in range(13)]
HAND_INDEX = {h: k for k, h in enumerate(HAND_NAMES)}


def _weight(name: str) -> int:
    if len(name) == 2:
        return 6
    return 4 if name[2] == "s" else 12


COMBO_WEIGHTS = np.array([_weight(h) for h in HAND_NAMES], dtype=np.float64)
TOTAL_COMBOS = COMBO_WEIGHTS.sum()  # 1326


def card_id(rank: str, suit: str) -> int:
    """0..51 as rank_value*4 + suit_index (2c... As)."""
    return RANK_VALUE[rank] * 4 + SUITS.index(suit)


def class_combos(idx: int) -> list[tuple[int, int]]:
    """All concrete (card, card) combos for a hand class."""
    name = HAND_NAMES[idx]
    r1, r2 = RANK_VALUE[name[0]], RANK_VALUE[name[1]]
    out = []
    if len(name) == 2:
        for a in range(4):
            for b in range(a + 1, 4):
                out.append((r1 * 4 + a, r1 * 4 + b))
    elif name[2] == "s":
        for a in range(4):
            out.append((r1 * 4 + a, r2 * 4 + a))
    else:
        for a in range(4):
            for b in range(4):
                if a != b:
                    out.append((r1 * 4 + a, r2 * 4 + b))
    return out


def hole_cards_to_class(r1: str, r2: str, suited: bool | None) -> str | None:
    """('J','T',False) -> 'JTo'.  suited=None for pairs or when unknown."""
    r1, r2 = r1.upper(), r2.upper()
    if r1 == "1":
        r1 = "T"
    if r2 == "1":
        r2 = "T"
    if r1 not in RANK_VALUE or r2 not in RANK_VALUE:
        return None
    if r1 == r2:
        return r1 + r2
    if RANK_VALUE[r2] > RANK_VALUE[r1]:
        r1, r2 = r2, r1
    if suited is None:
        return r1 + r2 + "o"
    return r1 + r2 + ("s" if suited else "o")


# ---------------------------------------------------------------- range strings
_TOKEN = re.compile(r"^([AKQJT2-9]{2}[so]?)(\+|-([AKQJT2-9]{2}[so]?))?(?::([0-9.]+))?$", re.I)


def _norm(h: str) -> str:
    h = h.strip()
    if len(h) >= 2:
        h = h[0].upper() + h[1].upper() + h[2:].lower()
    # put higher rank first
    if len(h) >= 2 and h[0] != h[1] and RANK_VALUE[h[1]] > RANK_VALUE[h[0]]:
        h = h[1] + h[0] + h[2:]
    return h


def _expand_class(h: str) -> list[str]:
    """'AK' (no suffix) -> ['AKs','AKo']; pairs stay as is."""
    if len(h) == 2 and h[0] != h[1]:
        return [h + "s", h + "o"]
    return [h]


def parse_range(text: str) -> np.ndarray:
    """Parse 'AA,KK,AKs:0.5,A2s+,22+,KTo-K8o,T9s' -> 169 frequency vector (0..1).

    Also accepts the 'hand:freq' list that GTO Wizard / most range tools export,
    whitespace or comma separated.
    """
    freqs = np.zeros(169)
    if not text:
        return freqs
    for raw in re.split(r"[,\s]+", text.strip()):
        if not raw:
            continue
        m = _TOKEN.match(raw)
        if not m:
            raise ValueError(f"Can't parse range token {raw!r}")
        base, plus, end, f = _norm(m.group(1)), m.group(2), m.group(3), m.group(4)
        freq = float(f) if f is not None else 1.0
        hands: list[str] = []
        if plus == "+":
            if base[0] == base[1]:  # 77+
                v = RANK_VALUE[base[0]]
                hands = [r * 2 for r in RANKS if RANK_VALUE[r] >= v]
            else:  # A2s+ : kicker up to one below the top card
                hi, lo, suf = base[0], base[1], base[2:]
                for r in RANKS:
                    if RANK_VALUE[lo] <= RANK_VALUE[r] < RANK_VALUE[hi]:
                        hands += _expand_class(hi + r + suf)
        elif end:
            end = _norm(end)
            if base[0] == base[1]:  # 99-55
                a, b = sorted([RANK_VALUE[base[0]], RANK_VALUE[end[0]]])
                hands = [r * 2 for r in RANKS if a <= RANK_VALUE[r] <= b]
            elif base[0] == end[0]:  # KTo-K8o (fixed top card)
                a, b = sorted([RANK_VALUE[base[1]], RANK_VALUE[end[1]]])
                for r in RANKS:
                    if a <= RANK_VALUE[r] <= b:
                        hands += _expand_class(base[0] + r + base[2:])
            else:  # T9s-65s (connector run, fixed gap)
                gap = RANK_VALUE[base[0]] - RANK_VALUE[base[1]]
                a, b = sorted([RANK_VALUE[base[0]], RANK_VALUE[end[0]]])
                for v in range(a, b + 1):
                    hi = RANKS[12 - v]
                    lo = RANKS[12 - (v - gap)]
                    hands += _expand_class(hi + lo + base[2:])
        else:
            hands = _expand_class(base)
        for h in hands:
            freqs[HAND_INDEX[h]] = freq
    return freqs


def format_range(freqs: np.ndarray, eps: float = 0.005) -> str:
    parts = []
    for k, f in enumerate(freqs):
        if f > 1 - eps:
            parts.append(HAND_NAMES[k])
        elif f > eps:
            parts.append(f"{HAND_NAMES[k]}:{f:.2f}")
    return ",".join(parts)


def range_pct(freqs: np.ndarray) -> float:
    return float((freqs * COMBO_WEIGHTS).sum() / TOTAL_COMBOS * 100)
