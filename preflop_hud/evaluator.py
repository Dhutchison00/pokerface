"""Vectorised 7-card poker hand evaluator (numpy).

Cards are ints 0..51 = rank*4 + suit, rank 0 = deuce ... 12 = ace.
evaluate7(cards[N,7]) -> int32 score; higher beats lower, equal = split.
"""
from __future__ import annotations

import numpy as np

_B = 13
_B5 = _B ** 5


def _build_tables():
    straight_top = np.full(8192, -1, dtype=np.int32)
    top5 = np.zeros(8192, dtype=np.int32)
    highbit = np.full(8192, -1, dtype=np.int32)
    windows = [(0b11111 << (t - 4), t) for t in range(12, 3, -1)]  # A-high .. 6-high
    wheel = (1 << 12) | 0b1111  # A,2,3,4,5
    for m in range(8192):
        for w, t in windows:
            if m & w == w:
                straight_top[m] = t
                break
        else:
            if m & wheel == wheel:
                straight_top[m] = 3
        ranks = [r for r in range(12, -1, -1) if m >> r & 1]
        if ranks:
            highbit[m] = ranks[0]
        enc = 0
        for r in (ranks + [0] * 5)[:5]:
            enc = enc * _B + r
        top5[m] = enc
    return straight_top, top5, highbit


STRAIGHT_TOP, TOP5, HIGHBIT = _build_tables()


def evaluate7(cards: np.ndarray) -> np.ndarray:
    cards = np.asarray(cards)
    n = cards.shape[0]
    r = cards // 4
    s = cards % 4
    onehot = r[:, :, None] == np.arange(13)[None, None, :]
    counts = onehot.sum(1)                                    # (N,13)
    rankbits = (onehot.any(1) * (1 << np.arange(13))).sum(1)  # (N,)
    suit_counts = (s[:, :, None] == np.arange(4)).sum(1)      # (N,4)
    fsuit = suit_counts.argmax(1)
    has_flush = suit_counts.max(1) >= 5
    in_f = s == fsuit[:, None]
    fbits = ((1 << r) * in_f).astype(np.int64)
    flushmask = np.bitwise_or.reduce(fbits, axis=1)

    maxc = counts.max(1)
    npairs = (counts >= 2).sum(1)
    ntrips = (counts >= 3).sum(1)

    # generic ordering: by (count desc, rank desc)
    key = counts[np.arange(n)[:, None], r] * 16 + r
    srt = -np.sort(-key, axis=1) % 16
    enc_generic = ((srt[:, 0] * _B + srt[:, 1]) * _B + srt[:, 2]) * _B * _B + srt[:, 3] * _B + srt[:, 4]

    q = srt[:, 0]
    kick_q = HIGHBIT[rankbits & ~(1 << q)]
    enc_quads = ((((q * _B + q) * _B + q) * _B + q) * _B) + kick_q
    p1, p2 = srt[:, 0], srt[:, 2]
    kick_2p = HIGHBIT[rankbits & ~(1 << p1) & ~(1 << p2)]
    enc_2p = ((((p1 * _B + p1) * _B + p2) * _B + p2) * _B) + kick_2p

    st = STRAIGHT_TOP[rankbits]
    sf = np.where(has_flush, STRAIGHT_TOP[flushmask], -1)

    score = enc_generic.astype(np.int64)  # high card / pair / trips / FH use generic
    cat = np.zeros(n, dtype=np.int64)
    pair = (maxc == 2) & (npairs == 1)
    two = (maxc == 2) & (npairs >= 2)
    trips = (maxc == 3) & (npairs == 1)
    fh = (ntrips >= 1) & (npairs >= 2) & (maxc < 4)
    quads = maxc == 4
    cat[pair] = 1
    cat[two] = 2
    score[two] = enc_2p[two]
    cat[trips] = 3
    strt = st >= 0
    m = strt & (cat < 4)
    cat[m] = 4
    score[m] = st[m]
    m = has_flush & (cat < 5)
    cat[m] = 5
    score[m] = TOP5[flushmask[m]]
    m = fh & (cat < 6)
    cat[m] = 6
    score[m] = enc_generic[m]
    m = quads
    cat[m] = 7
    score[m] = enc_quads[m]
    m = sf >= 0
    cat[m] = 8
    score[m] = sf[m]
    return (cat * _B5 + score).astype(np.int64)
