"""Check the 3-way flop range estimator against full offline solves, on synthetic hands.

    python -m preflop_hud.validate_multiway --spots 6 --ref-minutes 40        (overnight: hours of GPU)
    python -m preflop_hud.validate_multiway --spots 1 --ref-minutes 1         (a quick smoke run)

Each synthetic hand: 8-handed at --stack bb, the BTN opens and both blinds call, a random flop, and
one of a few flop lines (checked through; the SB bets and both call; the BTN bets and one folds...).
Each player's range at the end of the flop is found three ways:

  estimator  the live one: a reduced flop tree (the size bet, turn and river checked down), time-capped
  preflop    no narrowing at all (what a checked flop assumed before the estimator)
  reference  a full 3-way flop tree (33/75% flop bets, all-in raises, 75% turn and river bets with
             all-in raises) solved for --ref-minutes: the yardstick

and scored against the reference: the distance between the two ranges (total variation: the share of
the range that would have to move, 0 = the same, 1 = nothing in common), and how the next street plays
from each (the first player's turn strategy over the whole range, from a lean 3-way turn solve). The
report goes to data/validation/multiway_<time>.json and is printed.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "validation")
POS8 = {"0": "BTN", "1": "SB", "2": "BB", "3": "UTG", "4": "UTG1", "5": "LJ", "6": "HJ", "7": "CO"}
REF_SIZES = [{"bet": "33 75", "raise": "a"}, {"bet": "75", "raise": "a"}, {"bet": "75", "raise": "a"}]
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def _act(street, seat, action, to=None):
    return {"street": street, "seat": seat, "pos": POS8[str(seat)], "action": action, "to": to, "inferred": False}


# flop lines (seat 1 = SB acts first, 2 = BB, 0 = BTN); bets as a fraction of the pot
LINES = {
    "checked through": [(1, "check"), (2, "check"), (0, "check")],
    "SB bets 33%, both call": [(1, "bet", 0.33), (2, "call"), (0, "call")],
    "SB bets 75%, BB folds, BTN calls": [(1, "bet", 0.75), (2, "fold"), (0, "call")],
    "checked to the BTN, bets 50%, SB calls, BB folds": [(1, "check"), (2, "check"), (0, "bet", 0.5), (1, "call"),
                                                        (2, "fold")],
}


def synthetic_hand(rng, stack, line):
    cards = rng.sample(DECK, 6)
    open_to = 2.2
    pot = 3 * open_to + 1.0
    flop = []
    bet = 0.0
    for step in LINES[line]:
        seat, kind = step[0], step[1]
        if kind == "bet":
            bet = round(step[2] * pot, 2)
            flop.append(_act("flop", seat, "bet", bet))
        elif kind == "call":
            flop.append(_act("flop", seat, "call", bet))
        else:
            flop.append(_act("flop", seat, kind))
    pre = [_act("preflop", s, "fold") for s in (3, 4, 5, 6, 7)] + [
        _act("preflop", 0, "raise", open_to), _act("preflop", 1, "call", open_to), _act("preflop", 2, "call", open_to)]
    behind = stack - open_to
    folded = {a["seat"] for a in flop if a["action"] == "fold"}
    return {"id": rng.randrange(10 ** 9), "hero": cards[0] + cards[1], "street": "turn", "board": cards[2:6],
            "notes": [], "positions": POS8, "allin": [], "hero_turn": False,
            "in_hand": [s for s in (0, 1, 2) if s not in folded], "actions": pre + flop,
            "start": {"preflop": {"pot": None, "dead": 1.0, "stacks": {0: stack, 1: stack, 2: stack}},
                      "flop": {"pot": pot, "dead": None, "stacks": {0: behind, 1: behind, 2: behind}}}}


class Api:
    """The 3-way endpoints of one gto-server, with the HUD's tree walk on top."""

    def __init__(self, srv):
        self.srv = srv

    def build_and_solve(self, req, target, secs, iters=100000):
        info = self.srv.api("/api/mw/spot", req)
        t0 = time.time()
        self.srv.api("/api/mw/solve", {"max_iterations": iters, "target_exploit_pct": target, "check_every": 10})
        while self.srv.api("/api/mw/status")["state"] == "running":
            if time.time() - t0 > secs:
                self.srv.api("/api/mw/stop", {})
            time.sleep(0.5)
        st = self.srv.api("/api/mw/status")
        return {"nodes": info["nodes"], "iterations": st["iteration"], "exploit_pct": st["exploit_pct"],
                "secs": round(time.time() - t0, 1)}

    def end_ranges(self, hand, spot):
        """Each player's reach at the end of the flop: walk the flop's actions, deal the turn, then read each
        player at their first turn node (checks between don't change the others' reach)."""
        start_put = spot["meta"]["start_put"]
        get = lambda p: mw.mw_view(self.srv.api("/api/mw/node", {"path": mw.mw_path(p), "ev": False}), start_put)  # noqa: E731
        meta = spot["meta"]
        view, path, why = pf.walk(get, [a for a in hand["actions"] if a["street"] == "flop"], meta["players"],
                                  meta["hero_player"], False)
        if view is None or view["node_type"] != "chance":
            raise RuntimeError(f"the flop line doesn't end in the tree: {why}")
        seat_of = {p: s for s, p in meta["players"].items()}
        out, step = {}, path + [{"type": "card", "card": hand["board"][3]}]
        for _ in range(3):
            v = get(step)
            if v["node_type"] != "action":
                break
            out[seat_of[v["player"]]] = {x["combo"]: float(x["reach"]) for x in v["players"][v["player"]]["hands"]}
            kinds = [a["kind"] for a in v["actions"]]
            if "check" not in kinds:
                break
            step = step + [{"type": "action", "index": kinds.index("check")}]
        return out


def _canon(r):
    """{combo text: weight} keyed by the two cards' numbers, low first (combo text orders vary)."""
    return {tuple(sorted((mw.card_index(c[:2]), mw.card_index(c[2:])))): w for c, w in r.items()}


def _dist(a, b):
    """Total variation distance between two ranges ({combo: weight}), each normalised."""
    a, b = _canon(a), _canon(b)
    keys = set(a) | set(b)
    ta, tb = sum(a.values()) or 1.0, sum(b.values()) or 1.0
    return round(0.5 * sum(abs(a.get(k, 0.0) / ta - b.get(k, 0.0) / tb) for k in keys), 3)


def _combos_of(weights169):
    from .cards import class_combos
    out = {}
    for k in range(169):
        if weights169[k] > 0:
            for c1, c2 in class_combos(k):
                out[mw.card_name(c2) + mw.card_name(c1)] = float(weights169[k])
    return out


def turn_play(api, hand, ranges, carry):
    """The first player's turn strategy over their whole range, from a lean 3-way turn solve."""
    live = mw.players_at(hand, "turn")
    if len(live) != 3:
        return None
    spot, why = mw.build_mw_spot(hand, ranges, None, "turn", carry)
    if spot is None:
        return None
    api.build_and_solve(spot["request"], 0.5, 30)
    v = api.srv.api("/api/mw/node", {"path": [], "ev": False})
    return {a["label"]: round(a["freq"], 3) for a in v["actions"]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.validate_multiway", description=__doc__.split("\n")[0])
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--stack", type=float, default=20.0, help="bb at the start of the hand (default 20)")
    ap.add_argument("--ref-minutes", type=float, default=40.0, help="time for each full reference solve")
    ap.add_argument("--ref-target", type=float, default=0.5, help="% of the pot the reference stops at (default 0.5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gtopen-dir")
    ap.add_argument("--port", type=int, default=3746)
    a = ap.parse_args(argv)
    os.environ.setdefault("SOLVER_MEM_MB", "20000")       # the full flop tree's arena is ~7 GB of RAM
    from .gtopen import Server, find_gtopen_dir
    from .solved import SolvedCharts
    gdir = find_gtopen_dir(a.gtopen_dir)
    sc = SolvedCharts()
    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, time.strftime("multiway_%Y%m%d-%H%M%S.json"))
    report = {"args": vars(a), "spots": []}
    with Server(gdir, os.path.join(gdir, "saves", "validate_multiway.log"), a.port) as srv:
        api = Api(srv)
        for i in range(a.spots):
            line = list(LINES)[i % len(LINES)]
            hand = synthetic_hand(rng, a.stack, line)
            ranges, why = pf.build_ranges(hand, sc.node, None, multiway=True)
            if ranges is None:
                print(f"spot {i}: no ranges ({why})")
                continue
            row = {"board": hand["board"], "line": line}
            print(f"\nspot {i}: {' '.join(hand['board'][:3])} | turn {hand['board'][3]} | {line}", flush=True)
            # the estimator, exactly as live (same tree, same time cap)
            est, why = mw.build_estimator_spot(hand, ranges, None)
            row["estimator_solve"] = api.build_and_solve(est["request"], mw.ESTIMATOR_PCT, mw.ESTIMATOR_SECS, 400)
            est_r = api.end_ranges(hand, est)
            # the reference: the full flop tree, solved long
            ref, why = mw.build_mw_spot(dict(hand, street="flop", board=hand["board"][:3], in_hand=[0, 1, 2],
                                             actions=[x for x in hand["actions"] if x["street"] in ("preflop", "flop")]),
                                        ranges, None, "flop", sizes=REF_SIZES)
            row["reference_solve"] = api.build_and_solve(ref["request"], a.ref_target, a.ref_minutes * 60)
            ref_r = api.end_ranges(hand, ref)
            pre_r = {s: _combos_of(ranges["weights"][POS8[str(s)]]) for s in ref_r}
            row["distance"] = {POS8[str(s)]: {"estimator": _dist(est_r.get(s, {}), ref_r[s]),
                                              "preflop": _dist(pre_r[s], ref_r[s])} for s in ref_r}
            # how the turn plays from each set of ranges (3-way turns only)
            turn_hand = dict(hand, start=dict(hand["start"], turn={
                "pot": hand["start"]["flop"]["pot"] + sum(x["to"] or 0 for x in hand["actions"]
                                                          if x["street"] == "flop" and x["action"] in ("bet", "call")),
                "dead": None, "stacks": {s: hand["start"]["flop"]["stacks"][s] - max(
                    [x["to"] or 0 for x in hand["actions"] if x["street"] == "flop" and x["seat"] == s] or [0])
                    for s in (0, 1, 2)}}), actions=hand["actions"])
            row["turn_play"] = {name: turn_play(api, turn_hand, ranges, {"hands": r, "notes": []})
                                for name, r in (("reference", ref_r), ("estimator", est_r))}
            row["turn_play"]["preflop"] = turn_play(api, turn_hand, ranges, None)
            for pos, d in row["distance"].items():
                print(f"  {pos:3s} range distance from the reference: estimator {d['estimator']:.3f}, "
                      f"no narrowing {d['preflop']:.3f}")
            print(f"  estimator: {row['estimator_solve']} | reference: {row['reference_solve']}")
            if row["turn_play"]["reference"]:
                for name, tp in row["turn_play"].items():
                    print(f"  turn, first player's range from the {name} ranges: {tp}")
            report["spots"].append(row)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)
    d = [r["distance"] for r in report["spots"]]
    if d:
        est = np.mean([v["estimator"] for x in d for v in x.values()])
        pre = np.mean([v["preflop"] for x in d for v in x.values()])
        print(f"\nmean range distance from the reference over {len(d)} spots: estimator {est:.3f}, "
              f"no narrowing {pre:.3f}  ->  {out_path}")


if __name__ == "__main__":
    main()
