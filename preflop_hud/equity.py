"""169x169 preflop all-in equity matrix (Monte Carlo, cached to data/equity169.npy).

EQ[a, b] = equity of hand class a against hand class b (ties count half).
Card removal between the two hands is respected; removal against third
players is ignored (standard for push/fold tools).
"""
from __future__ import annotations

import os
import time

import numpy as np

from .cards import HAND_NAMES, class_combos
from .evaluator import evaluate7

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(DATA_DIR, "equity169.npy")

_EQ = None


def _sample_matchup_row(a: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    """Equity of class a vs every class b (b >= a computed, rest filled later)."""
    combos = [np.array(class_combos(k)) for k in range(169)]
    out = np.zeros(169)
    bs = np.arange(a, 169)
    nb = len(bs)
    N = nb * samples
    ca = combos[a][rng.integers(0, len(combos[a]), N)]
    cb = np.empty((N, 2), dtype=np.int64)
    bidx = np.repeat(bs, samples)
    for b in bs:
        sl = slice((b - a) * samples, (b - a + 1) * samples)
        cb[sl] = combos[b][rng.integers(0, len(combos[b]), samples)]
    # resample conflicting combos for b
    for _ in range(200):
        bad = (cb[:, 0] == ca[:, 0]) | (cb[:, 0] == ca[:, 1]) | (cb[:, 1] == ca[:, 0]) | (cb[:, 1] == ca[:, 1])
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        for b in np.unique(bidx[idx]):
            sel = idx[bidx[idx] == b]
            cb[sel] = combos[b][rng.integers(0, len(combos[b]), len(sel))]
            ca[sel] = combos[a][rng.integers(0, len(combos[a]), len(sel))]
    wins = np.zeros(N)
    chunk = 150_000
    for st in range(0, N, chunk):
        sl = slice(st, min(N, st + chunk))
        n = sl.stop - sl.start
        keys = rng.random((n, 52))
        used = np.concatenate([ca[sl], cb[sl]], axis=1)
        np.put_along_axis(keys, used, 2.0, axis=1)
        board = np.argpartition(keys, 5, axis=1)[:, :5]
        sa = evaluate7(np.concatenate([ca[sl], board], axis=1))
        sb = evaluate7(np.concatenate([cb[sl], board], axis=1))
        wins[sl] = (sa > sb) + 0.5 * (sa == sb)
    res = wins.reshape(nb, samples).mean(1)
    out[a:] = res
    return out


def build(samples: int = 3000, seed: int = 7, verbose: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eq = np.full((169, 169), np.nan)
    t0 = time.time()
    for a in range(169):
        row = _sample_matchup_row(a, samples, rng)
        eq[a, a:] = row[a:]
        eq[a:, a] = 1 - row[a:]
        if verbose and a % 13 == 12:
            print(f"  equity rows {a + 1}/169  ({time.time() - t0:.0f}s)", flush=True)
    np.fill_diagonal(eq, 0.5)
    return eq


def load(build_if_missing: bool = True) -> np.ndarray:
    global _EQ
    if _EQ is None:
        if not os.path.exists(CACHE):
            if not build_if_missing:
                raise FileNotFoundError(CACHE)
            print("Building 169x169 equity table (one-time, a few minutes)...", flush=True)
            os.makedirs(DATA_DIR, exist_ok=True)
            np.save(CACHE, build())
        _EQ = np.load(CACHE)
    return _EQ


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    os.makedirs(DATA_DIR, exist_ok=True)
    eq = build(n)
    np.save(CACHE, eq)
    i = HAND_NAMES.index
    print("AA vs KK", eq[i("AA"), i("KK")], "AKo vs 22", eq[i("AKo"), i("22")], "AKs vs QJs", eq[i("AKs"), i("QJs")])
