"""Pilot data for a 3-way turn value net: solved turns, through the GTOpen server as it is.

    python -m preflop_hud.pilot_data --split test --count 300        (a fixed test set, its own seed)
    python -m preflop_hud.pilot_data --split train --hours 10        (overnight; stop early: create data/pilot/STOP)

Each example is one 3-way turn solved with the lean live menu (75% bets, all-in raises, two raises)
to --target % of the pot, on the GPU. In: a random four-card board, three ranges, three stacks (the
pot is always 100 chips: values scale with it). Out: the first player's value for every hand at the
root, the only per-hand root values the server returns, plus the three range-level EVs.

Ranges are meant to look like ranges after a flop was played, which nothing can compute yet: each
starts as a preflop chart range (a random opener and two callers at a random grid depth, the HUD's
own charts), and is then reshaped at random by how strong each hand is on the board ("chart" leaves
it alone; "strong" leans to the best hands, "capped" drops them, "polar" keeps both ends, "middle"
keeps the calling hands). Each example records its family, so a net's error can be split by it.

Shards of SHARD examples go to data/pilot/<split>_<n>.npz, written whole (temp file, then renamed);
an example depends only on (split, index), so a run can be stopped and resumed. A failed solve is
skipped and logged, a dead server restarted. The PC is kept awake while it runs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time

import numpy as np

from .cards import class_combos
from .evaluator import evaluate7
from .multiway import card_index, card_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pilot")
SHARD = 50
SEEDS = {"train": 20261125, "test": 99177}
POT = 100.0
SIZES = [{"bet": "75", "raise": "a"}] * 3
MAX_RAISES = 2
POSITIONS = ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]         # preflop order, 8-handed
POSTFLOP = ["SB", "BB", "UTG", "UTG1", "LJ", "HJ", "CO", "BTN"]
DEPTHS = [10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60]                    # the grid depths drawn from
FAMILIES = ["chart", "strong", "capped", "polar", "middle"]
FAMILY_P = [0.35, 0.2, 0.2, 0.1, 0.15]
MIN_COMBOS = 12.0           # a reshaped range thinner than this (weighted combos) falls back to its chart range
MIN_WEIGHT = 0.004

# the 1326 combos, as (low card, high card) with cards numbered rank * 4 + suit ("23456789TJQKA", "cdhs")
PAIRS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)])
PAIR_INDEX = np.full((52, 52), -1, int)
PAIR_INDEX[PAIRS[:, 0], PAIRS[:, 1]] = PAIR_INDEX[PAIRS[:, 1], PAIRS[:, 0]] = np.arange(len(PAIRS))


def combo_text(i):
    a, b = PAIRS[i]
    return card_name(int(b)) + card_name(int(a))


def combo_of(text):
    return int(PAIR_INDEX[card_index(text[:2]), card_index(text[2:])])


def class_to_combos(w169):
    """169 class weights -> 1326 combo weights."""
    out = np.zeros(len(PAIRS), np.float32)
    for k in range(169):
        if w169[k] > 0:
            for a, b in class_combos(k):
                out[PAIR_INDEX[a, b]] = w169[k]
    return out


def strength(board):
    """Every combo's strength on the board, 0..1: its rank among all combos, averaged over the cards
    still to come (roughly its equity against a random hand). NaN for combos touching the board."""
    board = list(board)
    live = [c for c in range(52) if c not in board]
    ok = ~np.isin(PAIRS, board).any(axis=1)
    idx = np.flatnonzero(ok)
    rows = np.empty((len(idx), len(live), 7), np.int64)
    rows[:, :, 0:2] = PAIRS[idx][:, None, :]
    rows[:, :, 2:2 + len(board)] = board
    rows[:, :, 6] = np.array(live)[None, :]
    score = evaluate7(rows.reshape(-1, 7)).reshape(len(idx), len(live)).astype(np.float64)
    blocked = (PAIRS[idx][:, :, None] == np.array(live)[None, None, :]).any(axis=1)      # the card is in the hand
    score[blocked] = np.nan
    pct = np.full(score.shape, np.nan)
    for j in range(len(live)):
        col = score[:, j]
        good = ~np.isnan(col)
        order = np.searchsorted(np.sort(col[good]), col[good], side="left") + \
            0.5 * (np.searchsorted(np.sort(col[good]), col[good], side="right")
                   - np.searchsorted(np.sort(col[good]), col[good], side="left"))
        pct[good, j] = order / good.sum()
    out = np.full(len(PAIRS), np.nan)
    out[idx] = np.nanmean(pct, axis=1)
    return out


def reshape(w, s, family, rng):
    """One range reshaped by strength s (see the module text). p is each hand's place inside the range."""
    live = (w > 0) & ~np.isnan(s)
    out = np.where(live, w, 0.0).astype(np.float64)
    if family == "chart" or not live.any():
        return out.astype(np.float32)
    order = np.argsort(s[live])
    cum = np.cumsum(out[live][order])
    p = np.empty(live.sum())
    p[order] = (cum - 0.5 * out[live][order]) / cum[-1]
    if family == "strong":
        f = p ** rng.uniform(1.0, 4.0)
    elif family == "capped":
        t, c = rng.uniform(0.65, 0.95), rng.uniform(0.6, 1.0)
        f = 1.0 - c * np.clip((p - t) / 0.08, 0.0, 1.0)
    elif family == "polar":
        lo, hi, m = rng.uniform(0.1, 0.4), rng.uniform(0.6, 0.9), rng.uniform(0.05, 0.4)
        f = np.where((p < lo) | (p > hi), 1.0, m)
    else:                                               # middle
        mu, sd = rng.uniform(0.4, 0.75), rng.uniform(0.12, 0.3)
        f = 0.05 + np.exp(-((p - mu) / sd) ** 2)
    f = f * rng.lognormal(0.0, 0.2, len(f))
    out[live] *= f / f.max()
    out[out < MIN_WEIGHT] = 0.0
    return out.astype(np.float32)


class Charts:
    """Preflop chart ranges for 3-way single-raised pots, read once from the cached grid points."""

    def __init__(self):
        from .solved import SolvedCharts
        sc = SolvedCharts()
        self.lines = []                                   # (depth, opener, caller, caller, {pos: combos[1326]})
        for depth in DEPTHS:
            got = {}
            for i, o in enumerate(POSITIONS[:-1]):
                node, _ = sc.node(8, depth, 1.0, None, f"RFI:{o}")
                if node is None or "raise" not in node["actions"]:
                    continue
                got[("open", o)] = self._range(node, "raise")
                for c in POSITIONS[i + 1:]:
                    node, _ = sc.node(8, depth, 1.0, None, f"VS_OPEN:{c}:{o}")
                    if node is not None and "call" in node["actions"]:
                        r = self._range(node, "call")
                        if r.sum() >= MIN_COMBOS:
                            got[("call", c, o)] = r
            for (kind, *rest) in list(got):
                if kind != "open":
                    continue
                o = rest[0]
                callers = [k[1] for k in got if k[0] == "call" and k[2] == o]
                for a in range(len(callers)):
                    for b in range(a + 1, len(callers)):
                        self.lines.append((depth, o, callers[a], callers[b],
                                           {o: got[("open", o)], callers[a]: got[("call", callers[a], o)],
                                            callers[b]: got[("call", callers[b], o)]}))

    @staticmethod
    def _range(node, action):
        strat = np.asarray(node["strategy"], float)[node["actions"].index(action)]
        w = np.clip(np.asarray(node["reach"], float) * strat, 0.0, 1.0)
        return class_to_combos(np.where(w < MIN_WEIGHT, 0.0, w))      # the charts' near-zero dust isn't a range


def make_example(charts, split, index):
    """The solver request and the stored inputs of one example (a function of split and index alone)."""
    rng = np.random.default_rng([SEEDS[split], index])
    for _ in range(50):                                   # redraw until every range has enough hands on the board
        depth, o, c1, c2, by_pos = charts.lines[rng.integers(len(charts.lines))]
        seats = sorted((o, c1, c2), key=POSTFLOP.index)   # postflop action order
        board = rng.choice(52, 4, replace=False)
        s = strength(board)
        ranges, fams = [], []
        for pos in seats:
            base = np.where(np.isnan(s), 0.0, by_pos[pos]).astype(np.float32)
            fam = FAMILIES[rng.choice(len(FAMILIES), p=FAMILY_P)]
            r = reshape(base, s, fam, rng)
            if r.sum() < MIN_COMBOS:
                r, fam = base, "chart"
            ranges.append(r)
            fams.append(fam)
        if min(float(r.sum()) for r in ranges) >= MIN_COMBOS:
            break
    # stacks behind, in pots: anchored on short tournament stacks (most of the weight under 3 pots)
    spr = float(np.exp(rng.uniform(np.log(0.4), np.log(6.0))))
    stacks = np.full(3, spr) if rng.random() < 0.5 else np.clip(spr * rng.uniform(0.6, 1.6, 3), 0.3, 10.0)
    req = {"board": "".join(card_name(int(c)) for c in board),
           "ranges": [",".join(f"{combo_text(i)}:{r[i]:.4f}" for i in np.flatnonzero(r)) for r in ranges],
           "starting_pot": POT, "stacks": [round(float(x) * POT, 2) for x in stacks], "sizes": SIZES,
           "max_raises": MAX_RAISES, "allin_threshold": 85, "rake_pct": 0.0, "rake_cap": 0.0}
    meta = {"index": index, "depth": depth, "seats": seats, "opener": o, "families": fams}
    return req, {"board": board.astype(np.int8), "ranges": np.stack(ranges), "stacks": stacks.astype(np.float32)}, meta


def solve(srv, req, target, secs):
    info = srv.api("/api/mw/spot", req, timeout=120)
    t0 = time.time()
    srv.api("/api/mw/solve", {"max_iterations": 20000, "target_exploit_pct": target, "check_every": 10,
                              "algorithm": "dcfr"}, timeout=120)
    while srv.api("/api/mw/status", timeout=120)["state"] == "running":
        if time.time() - t0 > secs:
            srv.api("/api/mw/stop", {}, timeout=120)
        time.sleep(0.1)
    st = srv.api("/api/mw/status", timeout=120)
    if st.get("error"):
        raise RuntimeError(st["error"])
    v = srv.api("/api/mw/node", {"path": [], "ev": True}, timeout=300)
    if v.get("ev_skipped") or v["actor"] != 0:
        raise RuntimeError("the root's values weren't returned")
    ev = np.full(len(PAIRS), np.nan, np.float32)
    for h in v["hands"]:
        if h["ev"] is not None and h["weight"] > 0:
            ev[combo_of(h["combo"])] = h["ev"] / POT
    return {"ev": ev, "range_ev": np.array(st["evs"], np.float32) / POT, "exploit": float(st["exploit_pct"]),
            "iterations": int(st["iteration"]), "nodes": int(info["nodes"]), "secs": round(time.time() - t0, 2),
            "engine": st.get("engine") or ""}


def write_shard(path, rows):
    arrays = {k: np.stack([r[k] for r in rows]) for k in ("board", "ranges", "stacks", "ev", "range_ev")}
    arrays.update(exploit=np.array([r["exploit"] for r in rows], np.float32),
                  secs=np.array([r["secs"] for r in rows], np.float32),
                  nodes=np.array([r["nodes"] for r in rows], np.int32),
                  index=np.array([r["meta"]["index"] for r in rows], np.int64),
                  meta=np.frombuffer(json.dumps([r["meta"] for r in rows]).encode(), np.uint8))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.pilot_data", description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=list(SEEDS), default="train")
    ap.add_argument("--count", type=int, default=10 ** 9, help="stop after this many examples in the split")
    ap.add_argument("--hours", type=float, default=1000.0, help="stop after this long")
    ap.add_argument("--target", type=float, default=0.15, help="%% of the pot each solve is taken to")
    ap.add_argument("--secs", type=float, default=30.0, help="time cap per solve (the exploitability reached is saved)")
    ap.add_argument("--port", type=int, default=3757)
    ap.add_argument("--gtopen-dir")
    a = ap.parse_args(argv)
    from .gtopen import Server, find_gtopen_dir
    os.makedirs(OUT, exist_ok=True)
    if os.name == "nt":       # keep the PC from idle-sleeping while this process runs (no settings change)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    gdir = find_gtopen_dir(a.gtopen_dir)
    charts = Charts()
    stop = os.path.join(OUT, "STOP")
    done = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(OUT)
                  if f.startswith(a.split + "_") and f.endswith(".npz") and ".tmp" not in f)
    shard = (done[-1] + 1) if done else 0
    index = shard * SHARD
    print(f"{len(charts.lines)} preflop lines; {a.split}: starting at example {index} (shard {shard})", flush=True)
    with open(os.path.join(OUT, f"{a.split}.pid"), "w") as f:
        f.write(str(os.getpid()))
    t_start, rows, srv, fails = time.time(), [], None, 0

    def start():
        s = Server(gdir, os.path.join(gdir, "saves", "pilot_data.log"), a.port).__enter__()
        with open(os.path.join(OUT, f"{a.split}_server.pid"), "w") as f:
            f.write(str(s.proc.pid))
        return s
    try:
        srv = start()
        while index < a.count and time.time() - t_start < a.hours * 3600 and not os.path.exists(stop):
            req, inputs, meta = make_example(charts, a.split, index)
            try:
                out = solve(srv, req, a.target, a.secs)
                fails = 0
            except Exception as e:                       # skip the example; a dead server is restarted
                fails += 1
                print(f"example {index}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                if srv.proc.poll() is not None or fails >= 3:
                    try:
                        srv.__exit__(None, None, None)
                    except Exception:
                        pass
                    if fails >= 10:
                        raise RuntimeError("ten failures in a row: giving up")
                    srv = start()
                out = None
            if out is not None:
                rows.append(dict(inputs, **out, meta=dict(meta, engine=out["engine"])))
            else:                                        # keep the shard's indices aligned: a placeholder
                rows.append(dict(inputs, ev=np.full(len(PAIRS), np.nan, np.float32),
                                 range_ev=np.full(3, np.nan, np.float32), exploit=float("nan"), secs=0.0, nodes=0,
                                 meta=dict(meta, failed=True)))
            index += 1
            if len(rows) == SHARD or index >= a.count:
                write_shard(os.path.join(OUT, f"{a.split}_{shard:05d}.npz"), rows)
                ok = [r for r in rows if r["nodes"]]
                el = time.time() - t_start
                print(f"shard {shard}: {len(ok)}/{len(rows)} solved, median {np.median([r['secs'] for r in ok]):.1f} s, "
                      f"median exploit {np.median([r['exploit'] for r in ok]):.3f}%, "
                      f"{sum(r['exploit'] > a.target * 1.05 for r in ok)} over target; "
                      f"{index} examples, {el / 3600:.2f} h", flush=True)
                shard, rows = shard + 1, []
    finally:
        if srv is not None:
            srv.__exit__(None, None, None)
    print("stopped: " + ("STOP file" if os.path.exists(stop) else "count or time reached"), flush=True)


if __name__ == "__main__":
    main()
