"""The gate for sparse runouts: on 3-way turns, does a tree that deals only some river cards play
the turn like the full tree, and leave the same ranges for the river?

    python -m preflop_hud.validate_sparse --exe C:\\path\\to\\gto-server.exe --spots 6

A 3-way turn is the one place the truth is cheap: its full tree solves in seconds, so every reduced
tree can be scored against it. Each synthetic hand (8-handed, the BTN opens, both blinds call, the
flop is checked through, as validate_multiway deals them) is solved, all on the CPU and with the same
bet menu, as:

  reference     the full tree, solved tight: the yardstick
  full          the full tree at the working target: how far two solves of the SAME tree sit apart
                (the noise floor: no reduced tree can be expected to do better)
  sparse N      N river cards dealt (--sizes), and one size again with another seed
  checked down  no river betting at all (what the 3-way flop estimator did, and what failed)
  no narrowing  the ranges the turn started with

and scored two ways against the reference, over the turn lines the reference takes most often:

  ranges   each player's range when the river starts, read through the same river card in both
           trees (total variation: the share of the range that would have to move)
  play     how each hand plays the turn's decisions (the mean share of a hand's strategy that
           differs, weighted by how often the reference gets there with it)

Pass, decided before running: sparse beats both "checked down" and "no narrowing" on ranges, beats
"checked down" on play, and gets closer to the reference as cards are added. The report goes to
data/validation/sparse_<time>.json.

The server must be a build with sparse runouts (the "runouts" field of /api/mw/spot); the sparse
tree is CPU-only, so every arm is solved on the CPU to keep them comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np

from . import multiway as mw
from . import postflop as pf
from .validate_multiway import OUT_DIR, POS8, _dist, synthetic_hand

MIN_LINE = 0.03          # turn lines (and decisions) the reference reaches less often aren't scored
MAX_LINES = 4            # street-ending lines scored per spot
RIVER_CARDS = 3          # river cards the end-of-turn ranges are read through, per arm


class Solve:
    """One gto-server holding one 3-way solve."""

    def __init__(self, srv):
        self.srv = srv
        self._views = {}

    def run(self, req, target, secs, iters=5000):
        self._views = {}
        info = self.srv.api("/api/mw/spot", req)
        t0 = time.time()
        self.srv.api("/api/mw/solve", {"max_iterations": iters, "target_exploit_pct": target, "check_every": 10,
                                       "engine": "cpu"})
        while self.srv.api("/api/mw/status")["state"] == "running":
            if time.time() - t0 > secs:
                self.srv.api("/api/mw/stop", {})
            time.sleep(0.2)
        st = self.srv.api("/api/mw/status")
        if st.get("error"):
            raise RuntimeError(st["error"])
        if st.get("engine") != "cpu":
            raise RuntimeError(f"solved on {st.get('engine')!r}, not the CPU")
        return {"nodes": info["nodes"], "iterations": st["iteration"], "exploit_pct": round(st["exploit_pct"], 3),
                "secs": round(time.time() - t0, 1)}

    def view(self, path):
        key = tuple(path)
        if key not in self._views:
            self._views[key] = self.srv.api("/api/mw/node", {"path": list(path), "ev": False})
        return self._views[key]


def turn_lines(ref):
    """The reference's turn: every decision it reaches at least MIN_LINE of the time ({path: view}) and
    its likeliest lines that end with a river to bet on ([(path, probability)])."""
    decisions, ends = {}, []

    def go(path, prob):
        v = ref.view(path)
        if v["kind"] == "chance":
            ends.append((path, prob))
            return
        if v["kind"] != "action":
            return
        if prob >= MIN_LINE:
            decisions[path] = v
        for i, a in enumerate(v["actions"]):
            if prob * a["freq"] >= MIN_LINE / 4:
                go(path + (i,), prob * a["freq"])
    go((), 1.0)
    ends.sort(key=lambda t: -t[1])
    return decisions, [e for e in ends if e[1] >= MIN_LINE][:MAX_LINES]


def river_ranges(solve, line, card):
    """Each live player's range when the river starts: read where they first act (checks between
    don't change the others' reach). {} when the river has no decisions (an all-in runout)."""
    out, step = {}, tuple(line) + (card,)
    for _ in range(3):
        v = solve.view(step)
        if v["kind"] != "action" or v["actor"] in out:
            break
        out[v["actor"]] = {h["combo"]: float(h["weight"]) for h in v["hands"]}
        kinds = [a["kind"] for a in v["actions"]]
        if "check" not in kinds:
            break
        step = step + (kinds.index("check"),)
    return out


def play_distance(ref_view, arm_view):
    """Mean share of a hand's strategy that differs at one decision, weighted by the reference's reach."""
    arm = {h["combo"]: h["strategy"] for h in arm_view["hands"]}
    num = den = 0.0
    for h in ref_view["hands"]:
        s = arm.get(h["combo"])
        if s is None or h["weight"] <= 0:
            continue
        num += h["weight"] * 0.5 * sum(abs(a - b) for a, b in zip(h["strategy"], s))
        den += h["weight"]
    return num / den if den else None


def freq_distance(ref_view, arm_view):
    """Share of the whole range that takes a different action at one decision (which hands do it aside)."""
    return 0.5 * sum(abs(a["freq"] - b["freq"]) for a, b in zip(ref_view["actions"], arm_view["actions"]))


def by_class(rng):
    """A combo range summed into the 169 hand classes ("AKs", "TT", "T9o")."""
    order = "23456789TJQKA"
    out = {}
    for c, w in rng.items():
        hi, lo = sorted((c[0], c[2]), key=order.index, reverse=True)
        k = hi + lo + ("" if hi == lo else "s" if c[1] == c[3] else "o")
        out[k] = out.get(k, 0.0) + w
    return out


def class_dist(a, b):
    a, b = by_class(a), by_class(b)
    ta, tb = sum(a.values()) or 1.0, sum(b.values()) or 1.0
    return 0.5 * sum(abs(a.get(k, 0.0) / ta - b.get(k, 0.0) / tb) for k in set(a) | set(b))


def spread(cards, n):
    """n of the cards, spread across the list."""
    if len(cards) <= n:
        return list(cards)
    return [cards[int((i + 0.5) * len(cards) / n)] for i in range(n)]


def start_ranges(ref):
    """Each player's range at the start of the turn, read where they first act after checks."""
    out, step = {}, ()
    for _ in range(3):
        v = ref.view(step)
        out[v["actor"]] = {h["combo"]: float(h["weight"]) for h in v["hands"]}
        step = step + ([a["kind"] for a in v["actions"]].index("check"),)
    return out


def without(rng, card):
    return {c: w for c, w in rng.items() if card not in (c[:2], c[2:])}


def score(ref, arm, decisions, ends, start):
    """One arm against the reference: (range distances, play distances, the no-narrowing distances)."""
    rd, base, cd = [], [], []
    for line, _ in ends:
        kept = arm.view(line)["cards"]
        for card in spread(kept, RIVER_CARDS):
            a, r = river_ranges(arm, line, card), river_ranges(ref, line, card)
            for p in r:
                if p in a:
                    rd.append(_dist(a[p], r[p]))
                    cd.append(class_dist(a[p], r[p]))
                    base.append(_dist(without(start[p], card), r[p]))
    pd = [d for path, v in decisions.items() if (d := play_distance(v, arm.view(path))) is not None]
    fd = [freq_distance(v, arm.view(path)) for path, v in decisions.items()]
    return rd, pd, base, cd, fd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.validate_sparse", description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True, help="a gto-server built with sparse runouts")
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--stacks", type=float, nargs="+", default=[20.0, 40.0], help="bb at the start of the hand")
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 10, 16, 24], help="river cards kept")
    ap.add_argument("--target", type=float, default=0.5, help="%% of the pot every arm is solved to")
    ap.add_argument("--ref-target", type=float, default=0.15, help="%% of the pot the reference is solved to")
    ap.add_argument("--secs", type=float, default=90.0, help="time cap per arm")
    ap.add_argument("--ref-secs", type=float, default=300.0, help="time cap for the reference")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gtopen-dir")
    ap.add_argument("--port", type=int, default=3747)
    a = ap.parse_args(argv)
    from .gtopen import Server, find_gtopen_dir
    from .solved import SolvedCharts
    gdir = find_gtopen_dir(a.gtopen_dir)
    sc = SolvedCharts()
    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, time.strftime("sparse_%Y%m%d-%H%M%S.json"))
    report = {"args": vars(a), "spots": []}
    mid = a.sizes[len(a.sizes) // 2] if len(a.sizes) > 1 else a.sizes[0]
    log = os.path.join(gdir, "saves", "validate_sparse.log")
    with Server(gdir, log, a.port, exe=a.exe) as s_ref, Server(gdir, log, a.port + 1, exe=a.exe) as s_arm:
        ref, arm = Solve(s_ref), Solve(s_arm)
        for i in range(a.spots):
            stack = a.stacks[i % len(a.stacks)]
            hand = synthetic_hand(rng, stack, "checked through")
            hand["start"]["turn"] = dict(hand["start"]["flop"])
            ranges, why = pf.build_ranges(hand, sc.node, None, multiway=True)
            if ranges is None:
                print(f"spot {i}: no ranges ({why})")
                continue
            spot, why = mw.build_mw_spot(hand, ranges, None, "turn")
            if spot is None:
                print(f"spot {i}: no spot ({why})")
                continue
            req = spot["request"]
            print(f"\nspot {i}: {' '.join(hand['board'])} | {stack:g}bb | pot {req['starting_pot']} "
                  f"stacks {req['stacks']}", flush=True)
            row = {"board": hand["board"], "stack": stack, "arms": {}}
            row["reference"] = ref.run(req, a.ref_target, a.ref_secs)
            decisions, ends = turn_lines(ref)
            start = start_ranges(ref)
            row["lines"] = [{"path": list(p), "prob": round(q, 3)} for p, q in ends]
            print(f"  reference: {row['reference']} | {len(decisions)} decisions, {len(ends)} lines scored", flush=True)
            if not ends:
                continue
            down = dict(req, sizes=[dict(r) for r in req["sizes"][:2]] + [{"bet": "", "raise": ""}])
            arms = [("full", req)] + [(f"sparse {k}", dict(req, runouts={"river": k, "seed": 1})) for k in a.sizes]
            arms += [(f"sparse {mid} (seed 2)", dict(req, runouts={"river": mid, "seed": 2})), ("checked down", down)]
            for name, r in arms:
                try:
                    res = arm.run(r, a.target, a.secs)
                    rd, pd, base, cd, fd = score(ref, arm, decisions, ends, start)
                except RuntimeError as e:
                    print(f"  {name:20s} failed: {e}")
                    continue
                res.update(range_dist=round(float(np.mean(rd)), 4) if rd else None,
                           play_dist=round(float(np.mean(pd)), 4) if pd else None,
                           class_dist=round(float(np.mean(cd)), 4) if cd else None,
                           freq_dist=round(float(np.mean(fd)), 4) if fd else None)
                row["arms"][name] = res
                row["no_narrowing"] = round(float(np.mean(base)), 4) if base else None
                print(f"  {name:20s} ranges {res['range_dist']} (by class {res['class_dist']})  play {res['play_dist']} "
                      f"(range-wide {res['freq_dist']})  ({res['nodes']} nodes, {res['iterations']} it, "
                      f"{res['exploit_pct']}% pot, {res['secs']} s)", flush=True)
            print(f"  {'no narrowing':20s} ranges {row.get('no_narrowing')}")
            report["spots"].append(row)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)
    rows = [r for r in report["spots"] if r.get("arms")]
    if not rows:
        return
    names = list(rows[0]["arms"])

    def mean(name, key):
        v = [r["arms"][name][key] for r in rows if name in r["arms"] and r["arms"][name].get(key) is not None]
        return float(np.mean(v)) if v else float("nan")
    print(f"\nmeans over {len(rows)} spots (distance from the reference; lower is closer)")
    print(f"  {'arm':20s} {'ranges':>8s} {'play':>8s} {'nodes':>9s} {'secs':>7s}")
    for n in names:
        print(f"  {n:20s} {mean(n, 'range_dist'):8.4f} {mean(n, 'play_dist'):8.4f} {mean(n, 'nodes'):9.0f} "
              f"{mean(n, 'secs'):7.1f}")
    nn = float(np.mean([r["no_narrowing"] for r in rows if r.get("no_narrowing") is not None]))
    print(f"  {'no narrowing':20s} {nn:8.4f}")
    # size by size: one size passing says little if the small trees, the ones worth having, don't
    verdicts = {}
    for k in a.sizes:
        sp = f"sparse {k}"
        verdicts[sp] = bool(mean(sp, "range_dist") < min(mean("checked down", "range_dist"), nn)
                            and mean(sp, "play_dist") < mean("checked down", "play_dist"))
        print(f"  {sp}: {'closer than' if verdicts[sp] else 'NOT closer than'} checked down and no narrowing")
    sp = "every sparse size"
    ok = all(verdicts.values()) and (mean(f"sparse {a.sizes[-1]}", "range_dist")
                                     <= mean(f"sparse {a.sizes[0]}", "range_dist"))
    report["by_size"] = verdicts
    report["summary"] = {n: {"range_dist": mean(n, "range_dist"), "play_dist": mean(n, "play_dist"),
                             "nodes": mean(n, "nodes"), "secs": mean(n, "secs")} for n in names}
    report["summary"]["no narrowing"] = {"range_dist": nn}
    report["pass"] = bool(ok)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\ngate ({sp} vs checked down and no narrowing, and closer with more cards): "
          f"{'PASS' if ok else 'FAIL'}  ->  {out_path}")


if __name__ == "__main__":
    main()
