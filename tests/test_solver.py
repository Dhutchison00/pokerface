"""Solver checks: python tests/test_solver.py

1. Jam/fold-only trees reproduce known Nash push/fold (HU 10bb: SB 58% / BB 37%).
2. Every precomputed 8-handed solution passes sanity checks:
   AA never folds where it's reached, opens widen UTG -> BTN, BB defends wider
   vs BTN than vs UTG, adjacent stack depths don't jump, and the best-response
   gain (convergence) is small.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preflop_hud.solver import (Tree, CFR, Solution, sol_path, solve_and_save, STACK_GRID, SolverStore,  # noqa: E402
                                 check_zero_sum)
from preflop_hud.cards import HAND_INDEX, range_pct, COMBO_WEIGHTS as W  # noqa: E402

# 1. known Nash
t = Tree(2, [10, 10], 0.0, jamfold_only=True)
c = CFR(t)
c.solve(1000, check_every=500, tol=0.0005)
push = range_pct(c.avg_strategy(t.dec[0])[t.nodes[t.dec[0]]["actions"].index("allin")])
bbn = next(i for i in t.dec if t.nodes[i]["key"] == "VS_JAM:BB:BTN")
call = range_pct(c.avg_strategy(bbn)[1])
assert abs(push - 58) < 1.5 and abs(call - 37) < 1.5, (push, call)
print(f"HU 10bb Nash: SB push {push:.1f}%  BB call {call:.1f}%  OK")

# 1b. every terminal is zero-sum (antes, blinds, squeeze share included)
for n, stacks, kw in [(8, [25] * 8, {}), (8, [60] * 8, {}), (2, [40, 40], {}), (6, [12, 30, 8, 50, 20, 15], {}),
                      (4, [9, 14, 6, 20], {"jamfold_only": True, "posted": [0, 0, 0.5, 1]})]:
    tr = Tree(n, stacks, 0.125 * n, **kw)
    err = check_zero_sum(tr)
    assert err < 1e-9, (n, stacks, err)
print("zero-sum check OK on 5 trees (incl. unequal stacks)")

# 2. grid sanity
dead = SolverStore.dead_key(0.125 * 8)
AA = HAND_INDEX["AA"]
prev = None
rows = []
for s in STACK_GRID:
    if not os.path.exists(sol_path(8, s, dead)):
        # a fresh clone has no cache: solve the grid point as the precompute would (seconds, once)
        solve_and_save(8, s, dead, verbose=False)
    sol = Solution(sol_path(8, s, dead))
    assert max(sol.meta["br_gain"]) < 0.01, (s, sol.meta["br_gain"])
    for key in sol.meta["spots"]:
        d = sol.spot(key)
        if d["reach"][AA] > 0.02 and "fold" in d["actions"]:
            f = d["strategy"][d["actions"].index("fold")][AA]
            assert f < 0.02, (s, key, "AA folds", f)

    def cont(key):
        d = sol.spot(key)
        return 100 - range_pct(d["strategy"][d["actions"].index("fold")])
    opens = [cont(f"RFI:{p}") for p in ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN"]]
    assert all(b >= a - 1.0 for a, b in zip(opens, opens[1:])), (s, opens)
    bb_utg, bb_btn = cont("VS_OPEN:BB:UTG"), cont("VS_OPEN:BB:BTN")
    assert bb_btn > bb_utg, (s, bb_utg, bb_btn)
    if prev is not None:
        jumps = [abs(a - b) for a, b in zip(opens, prev)]
        assert max(jumps) < 12, (s, opens, prev)
    prev = opens
    rows.append((s, opens, bb_utg, bb_btn, max(sol.meta["br_gain"])))

print(" stack | open % UTG UTG1  LJ   HJ   CO  BTN | BB def vs UTG / BTN | BR gain")
for s, o, a, b, g in rows:
    print(f" {s:4d}  | " + " ".join(f"{x:4.0f}" for x in o) + f" |      {a:4.0f} / {b:4.0f}    | {g:.4f}")
print("solver checks passed")
