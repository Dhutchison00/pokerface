"""Would a value net for 3-way turns have stable targets? A check that needs no solver changes.

    python -m preflop_hud.validate_targets --spots 6

A net that prices the start of a 3-way turn would be trained on turn solves: each hand's value at the
turn's root. Those values are only worth learning if they hold still when the solve is repeated in a
slightly different way. For each synthetic 3-way turn (8-handed, the BTN opens, both blinds call, the
flop is checked through), the baseline is the lean live menu solved tight; it is compared with

  convergence   the same tree stopped at the loose target a data generator would use (one solve
                stopped earlier, so this is drift along a path, not independent noise)
  algorithm     the same tree and target solved with PCFR+ instead of DCFR (another route to an
                equilibrium; agreement does not show there is only one)
  nudge         the pot 1% bigger: values should scale with it and nothing else should move
  menu          a richer bet menu: how far the cheap menu's values are from a fuller game's

on the first player's value for every hand at the root (the only per-hand root values the API
returns: the other players' are read after checks, so they are conditional), in % of the starting
pot, and on all three players' range-level root EVs.

The rule, fixed before running, per comparison: a reach-weighted mean difference of at most
STABLE_PCT of the pot is stable, at most MARGINAL_PCT marginal, more is unstable. Only arms that
reached their target are compared. The report goes to data/validation/targets_<time>.json.

Stable values are necessary, not sufficient: this says nothing about how much data a net needs,
whether it can fit them, or the ranges a real leaf would see (these are preflop ranges after a
checked flop).
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
from .validate_multiway import OUT_DIR, synthetic_hand

STABLE_PCT, MARGINAL_PCT = 0.5, 1.5
RICH = [{"bet": "33 75", "raise": "60 a"}, {"bet": "33 75", "raise": "60 a"}, {"bet": "33 75 a", "raise": "60 a"}]
RICH_MAX_RAISES = 3


def solve(srv, req, target, secs, algorithm):
    info = srv.api("/api/mw/spot", req)
    t0 = time.time()
    srv.api("/api/mw/solve", {"max_iterations": 20000, "target_exploit_pct": target, "check_every": 10,
                              "algorithm": algorithm})
    while srv.api("/api/mw/status")["state"] == "running":
        if time.time() - t0 > secs:
            srv.api("/api/mw/stop", {})
        time.sleep(0.2)
    st = srv.api("/api/mw/status")
    if st.get("error"):
        raise RuntimeError(st["error"])
    v = srv.api("/api/mw/node", {"path": [], "ev": True})
    if v.get("ev_skipped") or v["actor"] != 0:
        raise RuntimeError("the root's values weren't returned")
    hands = {h["combo"]: (float(h["weight"]), h["ev"]) for h in v["hands"]
             if h["weight"] > 0 and h["ev"] is not None and np.isfinite(h["ev"])}
    return {"nodes": info["nodes"], "iterations": st["iteration"], "exploit_pct": round(st["exploit_pct"], 3),
            "reached": st["exploit_pct"] <= target * 1.05, "secs": round(time.time() - t0, 1),
            "engine": st.get("engine"), "evs": st["evs"], "hands": hands}


def compare(base, other, pot, scale=1.0):
    """Per-hand value differences (other / scale against base), in % of the pot, reach-weighted."""
    w, d = [], []
    for c, (wt, ev) in base["hands"].items():
        if c in other["hands"]:
            w.append(wt)
            d.append(abs(other["hands"][c][1] / scale - ev) / pot * 100)
    w, d = np.array(w), np.array(d)
    order = np.argsort(d)
    cum = np.cumsum(w[order]) / w.sum()
    return {"mean": round(float((w * d).sum() / w.sum()), 3),
            "p90": round(float(d[order][np.searchsorted(cum, 0.9)]), 3), "worst": round(float(d.max()), 3),
            "range_ev": [round(abs(a / scale - b) / pot * 100, 3) for a, b in zip(other["evs"], base["evs"])]}


def verdict(mean):
    return "stable" if mean <= STABLE_PCT else "marginal" if mean <= MARGINAL_PCT else "unstable"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.validate_targets", description=__doc__.split("\n")[0])
    ap.add_argument("--spots", type=int, default=6)
    ap.add_argument("--stacks", type=float, nargs="+", default=[20.0, 40.0])
    ap.add_argument("--loose", type=float, default=0.5, help="%% of the pot a data generator would stop at")
    ap.add_argument("--tight", type=float, default=0.15, help="%% of the pot the baseline is solved to")
    ap.add_argument("--secs", type=float, default=180.0, help="time cap per solve")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--gtopen-dir")
    ap.add_argument("--port", type=int, default=3755)
    ap.add_argument("--pid-file", help="write the server's process id here")
    a = ap.parse_args(argv)
    from .gtopen import Server, find_gtopen_dir
    from .solved import SolvedCharts
    gdir = find_gtopen_dir(a.gtopen_dir)
    sc = SolvedCharts()
    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, time.strftime("targets_%Y%m%d-%H%M%S.json"))
    report = {"args": vars(a), "rule": {"stable": STABLE_PCT, "marginal": MARGINAL_PCT}, "spots": []}
    with Server(gdir, os.path.join(gdir, "saves", "validate_targets.log"), a.port) as srv:
        if a.pid_file:
            with open(a.pid_file, "w") as f:
                f.write(str(srv.proc.pid))
        for i in range(a.spots):
            stack = a.stacks[i % len(a.stacks)]
            hand = synthetic_hand(rng, stack, "checked through")
            hand["start"]["turn"] = dict(hand["start"]["flop"])
            ranges, why = pf.build_ranges(hand, sc.node, None, multiway=True)
            spot, why2 = (None, why) if ranges is None else mw.build_mw_spot(hand, ranges, None, "turn")
            if spot is None:
                print(f"spot {i}: skipped ({why or why2})")
                continue
            req = spot["request"]
            pot = req["starting_pot"]
            print(f"\nspot {i}: {' '.join(hand['board'])} | {stack:g}bb", flush=True)
            arms = {"baseline": (req, a.tight, "dcfr", 1.0),
                    "convergence": (req, a.loose, "dcfr", 1.0),
                    "algorithm": (req, a.tight, "pcfr+", 1.0),
                    "nudge": (dict(req, starting_pot=round(pot * 1.01, 4)), a.tight, "dcfr", 1.01),
                    "menu": (dict(req, sizes=RICH, max_raises=RICH_MAX_RAISES), a.tight, "dcfr", 1.0)}
            row, got = {"board": hand["board"], "stack": stack, "solves": {}, "compare": {}}, {}
            for name, (r, target, algo, scale) in arms.items():
                try:
                    got[name] = solve(srv, r, target, a.secs, algo)
                except RuntimeError as e:
                    print(f"  {name:12s} failed: {e}")
                    continue
                s = got[name]
                row["solves"][name] = {k: v for k, v in s.items() if k != "hands"}
                print(f"  {name:12s} {s['nodes']} nodes, {s['iterations']} it, {s['exploit_pct']}% pot"
                      f"{'' if s['reached'] else ' (TARGET NOT REACHED)'}, {s['secs']} s, {s['engine']}", flush=True)
            base = got.get("baseline")
            if base is None:
                continue
            w = np.array([x[0] for x in base["hands"].values()])
            ev = np.array([x[1] for x in base["hands"].values()])
            m = float((w * ev).sum() / w.sum())
            row["spread_pct"] = round(float(np.sqrt((w * (ev - m) ** 2).sum() / w.sum())) / pot * 100, 2)
            row["check"] = {"mean_value_minus_third_pot": round(m - pot / 3, 3), "status_ev0": round(base["evs"][0], 3)}
            print(f"  values spread across hands: {row['spread_pct']}% of the pot (sanity: mean value less a third "
                  f"of the pot {row['check']['mean_value_minus_third_pot']}, the solver's EV {row['check']['status_ev0']})")
            for name, (_, _, _, scale) in arms.items():
                if name == "baseline" or name not in got:
                    continue
                c = compare(base, got[name], pot, scale)
                c["both_reached"] = bool(base["reached"] and got[name]["reached"])
                c["verdict"] = verdict(c["mean"])
                row["compare"][name] = c
                print(f"  {name:12s} mean {c['mean']}%  p90 {c['p90']}%  worst {c['worst']}%  range EVs {c['range_ev']}"
                      f"  -> {c['verdict']}{'' if c['both_reached'] else ' (an arm missed its target)'}", flush=True)
            report["spots"].append(row)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)
    print(f"\nby comparison, over the spots where both arms reached their target (% of the pot)")
    summary = {}
    for name in ("convergence", "algorithm", "nudge", "menu"):
        cs = [r["compare"][name] for r in report["spots"] if name in r["compare"]]
        ok = [c for c in cs if c["both_reached"]]
        if not ok:
            print(f"  {name:12s} no spot where both arms reached their target ({len(cs)} solved)")
            continue
        mean = float(np.mean([c["mean"] for c in ok]))
        summary[name] = {"spots": len(ok), "of": len(cs), "mean": round(mean, 3),
                         "p90": round(float(np.mean([c["p90"] for c in ok])), 3),
                         "worst": round(float(max(c["worst"] for c in ok)), 3), "verdict": verdict(mean)}
        print(f"  {name:12s} {len(ok)}/{len(cs)} spots  mean {summary[name]['mean']}  p90 {summary[name]['p90']}  "
              f"worst {summary[name]['worst']}  -> {summary[name]['verdict']}")
    report["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
