"""Map a detected spot onto the preflop solver.

* Short-stack jam/fold spots are solved live (CFR, ~0.5 s) with the real stacks.
* Everything else is looked up in the precomputed grid (equal stacks, nearest
  stack depth).  Missing grid points are solved in the background and the
  caller falls back to the approximate model meanwhile.
"""
from __future__ import annotations

import numpy as np

from .cards import COMBO_WEIGHTS as W
from .solver import SolverStore, live_jamfold, node_export, DEFAULT_PARAMS, _size
from .spot import POSITION_NAMES

ACTION_ORDER = ["allin", "raise", "call", "check", "fold"]
LIVE_RFI_MAX = 9.5      # below this effective stack: pure jam/fold with real stacks
LIVE_VS_JAM_MAX = 20.0  # first-in jams up to this size: solved live with real stacks
REACH_MIN = 0.02        # hands reaching a node less often than this are shown empty

SPOT_LABEL = {"RFI": "first in", "VS_OPEN": "vs open", "VS_SHOVE": "vs all-in", "VS_3BET": "vs 3-bet",
              "LIMPED": "vs limp"}


def _chart_from_node(d, title, method, note=None):
    acts = d["actions"]
    strat = np.asarray(d["strategy"], float)        # (A, 169)
    ev = np.asarray(d["ev"], float)                 # (A, 169)
    reach = np.asarray(d["reach"], float)           # (169,)
    keys = [a for a in ACTION_ORDER if a in acts]
    idx = [acts.index(a) for a in keys]
    freqs = strat[idx].T.copy()                     # (169, A')
    evs = ev[idx].T.copy()
    unreached = reach < REACH_MIN
    freqs[freqs < 0.01] = 0
    freqs = freqs / np.maximum(freqs.sum(1, keepdims=True), 1e-9)
    shown = np.where(unreached[:, None], 0, freqs)
    w = W * reach * (~unreached)
    tot = max(w.sum(), 1e-9)
    pct = {a: round(float((freqs[:, i] * w).sum() / tot * 100), 1) for i, a in enumerate(keys)}
    best_ev = (evs * freqs).sum(1)
    return {"title": title, "method": method, "approx": False, "solver": True, "note": note,
            "actions": keys, "freqs": np.round(shown, 3).tolist(),
            "freqs_raw": np.round(freqs, 3).tolist(),
            "ev": np.round(best_ev, 2).tolist(), "ev_actions": np.round(evs, 2).tolist(),
            "unreached": unreached.tolist(), "pct": pct}


class SolvedCharts:
    def __init__(self, iters=1500):
        self.store = SolverStore(iters=iters)

    # ------------------------------------------------------------------ live
    def _live_rfi(self, spot):
        hero = spot["hero"]
        players = [hero] + list(spot["behind"])
        if len(players) < 2:
            return None
        stacks = [p["total"] for p in players]
        posted = [p["bet"] for p in players]
        dead = max(0.0, spot["pot"] - sum(posted))
        t, c = live_jamfold(stacks, posted, dead)
        d = node_export(c, t.dec[0])
        title = f"{spot['hero_pos']} first in · {spot['eff_bb']:g}bb · push/fold"
        return _chart_from_node(d, title, "Solver: live jam/fold equilibrium with the real stacks")

    def _live_vs_jam(self, spot):
        v, hero = spot["villain"], spot["hero"]
        players = [v, hero] + list(spot["behind"])
        stacks = [p["total"] for p in players]
        posted = [p["posted"] for p in players]
        dead = max(0.0, spot["pot"] - sum(p["bet"] for p in players))
        t, c = live_jamfold(stacks, posted, dead)
        key = f"VS_JAM:{t.names[1]}:{t.names[0]}"
        i = next(j for j in t.dec if t.nodes[j]["key"] == key)
        d = node_export(c, i)
        title = f"{spot['hero_pos']} vs {v['pos']} all-in · {spot['eff_bb']:g}bb"
        return _chart_from_node(d, title, "Solver: live jam/fold equilibrium with the real stacks")

    # ------------------------------------------------------------------ grid
    @staticmethod
    def _key(spot):
        """Solver node for this spot, or None if the tree doesn't contain it."""
        t = spot.get("type")
        h = spot.get("hero_pos")
        n = spot.get("n_players")
        v = spot.get("villain") or {}
        vp = v.get("pos")
        hero = spot.get("hero") or {}
        hero_bet = hero.get("bet", 0.0)
        raises = spot.get("raises") or []
        limpers = spot.get("limpers") or []
        order = POSITION_NAMES.get(n, [])

        def acts_before(x, y):
            return x in order and y in order and order.index(x) < order.index(y)

        sb_name = "BTN" if n == 2 else "SB"
        if t == "RFI":
            return f"RFI:{h}"
        if t == "LIMPED":
            if h == "BB" and len(limpers) == 1 and limpers[0]["pos"] == sb_name:
                return "BB_VS_LIMP"
            return None
        if limpers or len(raises) != 1:
            return None                         # multiway / squeeze: not in the tree
        hero_limped = h == sb_name and abs(hero_bet - 1.0) < 0.05
        if t == "VS_OPEN":
            if hero_limped and vp == "BB":
                return "SB_VS_ISO"
            if not spot.get("hero_opened") and hero_bet <= 1.0:
                return f"VS_OPEN:{h}:{vp}"
            return None
        if t == "VS_3BET":
            if spot.get("hero_opened") and acts_before(h, vp) or (h == sb_name and vp == "BB"):
                return f"VS_3BET:{h}:{vp}"
            return None
        if t == "VS_SHOVE":
            if hero_limped and vp == "BB":
                return "SB_VS_BBJAM"
            if spot.get("hero_opened"):
                if h == "BB" and vp == sb_name and hero_bet <= 5.0:
                    return "BB_VS_SBJAM_ISO"   # BB iso-raised an SB limp, SB jammed
                if acts_before(h, vp) or (h == sb_name and vp == "BB"):
                    return f"VS_3BETJAM:{h}:{vp}"  # hero opened, a later player jammed
                return f"VS_4BET:{h}:{vp}"         # hero 3-bet, the opener jammed
            if hero_bet <= 1.0:
                return f"VS_JAM:{h}:{vp}"
        return None

    def chart(self, spot):
        """Returns (chart or None, status message or None)."""
        t = spot.get("type")
        eff = spot.get("eff_bb")
        n = spot.get("n_players")
        if t not in SPOT_LABEL or eff is None or not n:
            return None, None
        raises = spot.get("raises") or []
        first_in_jam = (t == "VS_SHOVE" and len(raises) == 1 and not spot.get("limpers")
                        and not spot.get("hero_opened") and spot.get("hero", {}).get("bet", 0) <= 1.0)
        if t == "RFI" and eff < LIVE_RFI_MAX:
            return self._live_rfi(spot), None
        if first_in_jam and eff <= LIVE_VS_JAM_MAX:
            return self._live_vs_jam(spot), None
        key = self._key(spot)
        if key is None:
            return None, "spot isn't in the solver tree (multiway / squeeze / limp-raise)"
        dead = spot.get("dead", 0.0)   # antes/dead money measured from the pot
        sol, msg = self.store.get(n, eff, dead)
        if sol is None:
            return None, msg
        d = sol.spot(key)
        if d is None:
            return None, f"{key} isn't reachable in the {sol.meta['stack']}bb tree"
        grid = sol.meta["stack"]
        v = spot.get("villain") or {}
        label = {"RFI": "first in", "VS_OPEN": f"vs {v.get('pos')} open", "VS_3BET": f"vs {v.get('pos')} 3-bet",
                 "VS_SHOVE": f"vs {v.get('pos')} all-in", "LIMPED": "vs SB limp"}[t]
        if key == "SB_VS_ISO":
            label = "vs BB iso-raise"
        title = f"{spot['hero_pos']} {label} · {eff:g}bb"
        method = f"Solver (CFR) · {n}-handed, equal {grid}bb stacks · postflop modelled"
        note = None
        if t == "VS_OPEN" and key.startswith("VS_OPEN"):
            p = sol.meta.get("params") or DEFAULT_PARAMS
            tbl = p["sb_open_size"] if v.get("pos") == "SB" and n > 2 else p["open_size"]
            solved = _size(tbl, grid)
            if v.get("bet") and abs(v["bet"] - solved) / solved > 0.25:
                note = f"Solved for {solved:g}bb opens; this open is {v['bet']:g}bb"
        return _chart_from_node(d, title, method, note), None
