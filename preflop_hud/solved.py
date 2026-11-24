"""Map a detected spot onto the preflop solver.

* Short-stack jam/fold spots are solved live (CFR, ~0.5 s) with the real stacks.
* Everything else is looked up in the precomputed grid (equal stacks, nearest
  stack depth).  Missing grid points are solved in the background and the
  caller falls back to the approximate model meanwhile.
"""
from __future__ import annotations

import numpy as np

from . import modes
from .cards import COMBO_WEIGHTS as W
from .solver import SolverStore, live_jamfold, node_export, DEFAULT_PARAMS, _size
from .spot import POSITION_NAMES

ACTION_ORDER = ["allin", "raise", "call", "check", "fold"]
LIVE_RFI_MAX = 9.5      # below this effective stack: pure jam/fold with real stacks
LIVE_VS_JAM_MAX = 20.0  # first-in jams up to this size: solved live with real stacks
REACH_MIN = 0.02        # hands reaching a node less often than this are shown empty

SPOT_LABEL = {"RFI": "first in", "VS_OPEN": "vs open", "VS_SHOVE": "vs all-in", "VS_3BET": "vs 3-bet",
              "LIMPED": "vs limp"}


def _chart_from_node(d, title, method, note=None, source="hud"):
    """d: actions, strategy (A, 169), reach (169,), and ev (A, 169) or None (GTOpen exports none)."""
    acts = d["actions"]
    strat = np.asarray(d["strategy"], float)        # (A, 169)
    reach = np.asarray(d["reach"], float)           # (169,)
    keys = [a for a in ACTION_ORDER if a in acts]
    idx = [acts.index(a) for a in keys]
    freqs = strat[idx].T.copy()                     # (169, A')
    unreached = reach < REACH_MIN
    freqs[freqs < 0.01] = 0
    freqs = freqs / np.maximum(freqs.sum(1, keepdims=True), 1e-9)
    shown = np.where(unreached[:, None], 0, freqs)
    w = W * reach * (~unreached)
    tot = max(w.sum(), 1e-9)
    pct = {a: round(float((freqs[:, i] * w).sum() / tot * 100), 1) for i, a in enumerate(keys)}
    chart = {"title": title, "method": method, "approx": False, "solver": True, "source": source, "note": note,
             "actions": keys, "freqs": np.round(shown, 3).tolist(),
             "freqs_raw": np.round(freqs, 3).tolist(), "ev": None,
             "unreached": unreached.tolist(), "pct": pct}
    if d.get("ev") is not None:
        evs = np.asarray(d["ev"], float)[idx].T     # (169, A')
        chart["ev"] = np.round((evs * freqs).sum(1), 2).tolist()
        chart["ev_actions"] = np.round(evs, 2).tolist()
    return chart


class SolvedCharts:
    def __init__(self, iters=1500, gtopen_dir=None):
        from .gtopen import GtopenStore, OUT_DIR
        self.iters = iters
        self.stores = {}                                    # HUD solver cache per mode tag
        self.store = self._store(None)                      # tournament chip EV
        self.gtopen = GtopenStore(gtopen_dir or OUT_DIR)   # precomputed GTOpen charts, preferred

    def _store(self, game):
        tag = modes.solver_tag(game)
        if tag not in self.stores:
            self.stores[tag] = SolverStore(iters=self.iters, params=modes.solver_params(game), tag=tag)
        return self.stores[tag]

    # ------------------------------------------------------------------ live
    def _live_rfi(self, spot, game):
        hero = spot["hero"]
        players = [hero] + list(spot["behind"])
        if len(players) < 2:
            return None
        stacks = [p["total"] for p in players]
        posted = [p["bet"] for p in players]
        dead = max(0.0, spot["pot"] - sum(posted))
        t, c = live_jamfold(stacks, posted, dead, params=modes.solver_params(game))
        d = node_export(c, t.dec[0])
        title = f"{spot['hero_pos']} first in · {spot['eff_bb']:g}bb · push/fold"
        return _chart_from_node(d, title, "Solver: live jam/fold equilibrium with the real stacks")

    def _live_vs_jam(self, spot, game):
        v, hero = spot["villain"], spot["hero"]
        players = [v, hero] + list(spot["behind"])
        stacks = [p["total"] for p in players]
        posted = [p["posted"] for p in players]
        dead = max(0.0, spot["pot"] - sum(p["bet"] for p in players))
        t, c = live_jamfold(stacks, posted, dead, params=modes.solver_params(game))
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

    def chart(self, spot, game=None):
        """Returns (chart or None, status message or None). game: modes setting (None = tournament chip EV)."""
        game = modes.normalize(game)
        t = spot.get("type")
        eff = spot.get("eff_bb")
        n = spot.get("n_players")
        if t not in SPOT_LABEL or eff is None or not n:
            return None, None
        raises = spot.get("raises") or []
        first_in_jam = (t == "VS_SHOVE" and len(raises) == 1 and not spot.get("limpers")
                        and not spot.get("hero_opened") and spot.get("hero", {}).get("bet", 0) <= 1.0)
        chip_ev_note = ("ICM: no solve for this spot at this stage, showing chip EV"
                        if game["icm_stage"] else None)
        live = (t == "RFI" and eff < LIVE_RFI_MAX) or (first_in_jam and eff <= LIVE_VS_JAM_MAX)
        if live and game["icm_stage"]:
            # short-stack jam/fold spots are where ICM matters most: an ICM export beats the
            # chip-EV live solve with the real stacks
            key = self._key(spot)
            ch = self._grid_chart(spot, game, key, gtopen_only=True)[0] if key else None
            if ch is not None:
                return ch, None
        if t == "RFI" and eff < LIVE_RFI_MAX:
            ch = self._live_rfi(spot, game)
            if ch:
                ch["note"] = chip_ev_note
            return ch, None
        if first_in_jam and eff <= LIVE_VS_JAM_MAX:
            ch = self._live_vs_jam(spot, game)
            ch["note"] = chip_ev_note
            return ch, None
        key = self._key(spot)
        if key is None:
            return None, "spot isn't in the solver tree (multiway / squeeze / limp-raise)"
        return self._grid_chart(spot, game, key)

    def _grid_chart(self, spot, game, key, gtopen_only=False):
        """Chart from a precomputed grid point: a GTOpen export of this game type when there is one,
        else the HUD solver's cache (solved in the background when missing)."""
        t, eff, n = spot.get("type"), spot.get("eff_bb"), spot.get("n_players")
        dead = spot.get("dead", 0.0)   # antes/dead money measured from the pot
        v = spot.get("villain") or {}
        label = {"RFI": "first in", "VS_OPEN": f"vs {v.get('pos')} open", "VS_3BET": f"vs {v.get('pos')} 3-bet",
                 "VS_SHOVE": f"vs {v.get('pos')} all-in", "LIMPED": "vs SB limp"}[t]
        if key == "SB_VS_ISO":
            label = "vs BB iso-raise"
        title = f"{spot['hero_pos']} {label} · {eff:g}bb"

        gt = self.gtopen.get(n, SolverStore.grid_stack(eff), dead, modes.gtopen_tag(game))
        d = gt.spot(key) if gt else None
        mode_txt = modes.describe(game)
        note = None
        if d is not None:
            m = gt.meta
            sol, source = gt, "gtopen"
            antes = f", {m['dead']:g}bb antes" if m["dead"] else ""
            values = "balanced flop values" if game["icm_stage"] else "learned flop values"
            method = (f"GTOpen solver · {mode_txt} · {n}-handed, equal {m['stack']}bb stacks{antes}"
                      f"{' · limps' if m.get('limp') else ''} · {values}")
        elif gtopen_only:
            return None, None
        else:
            sol, msg = self._store(game).get(n, eff, dead)
            if sol is None:
                return None, msg
            d = sol.spot(key)
            if d is None:
                return None, f"{key} isn't reachable in the {sol.meta['stack']}bb tree"
            source = "hud"
            chip = " (chip EV)" if game["icm_stage"] else ""
            method = (f"Solver (CFR) · {mode_txt}{chip} · {n}-handed, equal {sol.meta['stack']}bb stacks"
                      " · postflop modelled")
            if game["icm_stage"]:
                note = "ICM: no solve for this spot at this stage, showing chip EV"
        grid = sol.meta["stack"]
        if t == "VS_OPEN" and key.startswith("VS_OPEN"):
            p = sol.meta.get("params") or DEFAULT_PARAMS
            tbl = p["sb_open_size"] if v.get("pos") == "SB" and n > 2 else p["open_size"]
            solved = _size(tbl, grid)
            if v.get("bet") and abs(v["bet"] - solved) / solved > 0.25:
                size_note = f"Solved for {solved:g}bb opens; this open is {v['bet']:g}bb"
                note = f"{note}. {size_note}" if note else size_note
        return _chart_from_node(d, title, method, note, source), None
