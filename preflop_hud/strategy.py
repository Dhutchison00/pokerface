"""Preflop chart generation (chip-EV).

Computed (equilibrium / exact EV, using the real stacks on screen):
  * first-in push/fold          - fictitious-play Nash vs every player left to act
  * calling an all-in           - best response to the shover's Nash pushing range
  * 3-bet jam vs an open        - jam EV vs opener's estimated range and calling response

Approximate (hand-authored heuristics, labelled as such on the page):
  * first-in raise ranges at deeper stacks
  * flat-call / 3-bet ranges vs an open at deeper stacks
  * vs limpers, vs 3-bets

Simplifications: one caller at most (no overcalls), card removal against
third players ignored, chip EV only (no ICM).
"""
from __future__ import annotations

import numpy as np

from . import equity
from .cards import HAND_NAMES, COMBO_WEIGHTS as W, TOTAL_COMBOS, RANK_VALUE, range_pct

ACTION_ORDER = ["allin", "raise", "call", "check", "fold"]

# Approximate first-in raise % by position for chip-EV MTT play with antes.
#            eff >40   25-40   17-25
OPEN_PCT = {
    "UTG": (16, 15, 14), "UTG1": (18, 17, 16), "UTG2": (20, 19, 18), "UTG3": (21, 20, 19),
    "LJ": (22, 21, 20), "HJ": (27, 25, 24), "CO": (34, 32, 30), "BTN": (50, 46, 42),
    "SB": (32, 30, 28),          # SB also limps (see SB_LIMP_PCT)
}
SB_LIMP_PCT = (30, 28, 22)
POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN"]


def _bucket(eff):
    return 0 if eff > 40 else 1 if eff > 25 else 2


def _top_by(score: np.ndarray, pct: float) -> np.ndarray:
    """0..1 vector with the top `pct`% of combos by score (last class partial)."""
    out = np.zeros(169)
    target = pct / 100 * TOTAL_COMBOS
    acc = 0.0
    for k in np.argsort(-score):
        if acc >= target:
            break
        take = min(1.0, (target - acc) / W[k])
        out[k] = take
        acc += take * W[k]
    return out


def _hand_meta():
    hi = np.zeros(169, int)
    lo = np.zeros(169, int)
    kind = np.empty(169, dtype=object)
    for k, h in enumerate(HAND_NAMES):
        hi[k], lo[k] = RANK_VALUE[h[0]], RANK_VALUE[h[1]]
        kind[k] = "p" if len(h) == 2 else h[2]
    return hi, lo, kind


HI, LO, KIND = _hand_meta()
SUITED = KIND == "s"
PAIR = KIND == "p"
OFFSUIT = KIND == "o"


class Engine:
    def __init__(self):
        self.solved = None
        self.EQ = equity.load()
        eq_rand = self.EQ @ W / W.sum()
        self.eq_rand = eq_rand
        top = _top_by(eq_rand, 25)
        base = self.EQ @ (W * top) / (W * top).sum()
        gap = HI - LO - 1
        bonus = np.zeros(169)
        bonus += SUITED * 0.025
        bonus += SUITED * np.select([gap == 0, gap == 1, gap == 2], [0.022, 0.012, 0.0], 0) * (LO >= 3)
        bonus -= (SUITED & (HI < 12) & (LO <= 3) & (gap >= 3)) * 0.02  # K2s/Q3s-type
        bonus += (SUITED & (HI == 12) & (LO <= 3)) * 0.015              # A5s-A2s
        bonus -= (OFFSUIT & (LO < 7)) * 0.04                            # offsuit weak kickers
        bonus -= (OFFSUIT & (HI == 12) & (LO < 8)) * 0.01
        bonus += (OFFSUIT & (LO >= 8)) * 0.012                          # offsuit broadways
        bonus -= (OFFSUIT & (HI < 9) & (gap > 0)) * 0.02
        self.open_score = base + bonus

    # ------------------------------------------------------------- equity helpers
    def eq_vs(self, rng: np.ndarray) -> np.ndarray:
        w = W * rng
        s = w.sum()
        if s <= 0:
            return np.full(169, 0.5)
        return self.EQ @ w / s

    # ------------------------------------------------------------- push / fold Nash
    def pushfold(self, pot, hero_total, hero_bet, callers, iters=3000):
        """Hero is first in (or only limpers before). callers = [{'total','bet'}] in order.

        Returns (push_freq[169], call_freqs[list of 169], ev_push[169]) in BB, EV relative to folding.
        """
        p = np.ones(169) * 0.5
        cs = [np.ones(169) * 0.3 for _ in callers]
        Es = [min(hero_total, c["total"]) for c in callers]
        FPs = [pot + (E - hero_bet) + (E - c["bet"]) for E, c in zip(Es, callers)]
        ev = np.zeros(169)
        for it in range(iters):
            lr = 1.0 / (it + 2)
            eqv = self.eq_vs(p)
            new_cs = []
            for c, E, FP in zip(callers, Es, FPs):
                ev_call = eqv * FP - (E - c["bet"])
                new_cs.append((ev_call > 0).astype(float))
            pfold = 1.0
            ev = np.zeros(169)
            for cj, E, FP in zip(cs, Es, FPs):
                w = W * cj
                pc = w.sum() / TOTAL_COMBOS
                if pc > 0:
                    ev += pfold * pc * (self.EQ @ w / w.sum() * FP - (E - hero_bet))
                pfold *= 1 - pc
            ev += pfold * pot
            br = (ev > 0).astype(float)
            p += lr * (br - p)
            cs = [cj + lr * (nc - cj) for cj, nc in zip(cs, new_cs)]
        # clean up the averaged (equilibrium) strategies: pure except where a
        # hand is (nearly) indifferent; EVs are measured against the averages
        band = 0.06
        eqv = self.eq_vs(p)
        cs_out = [np.clip(0.5 + (eqv * FP - (E - c["bet"])) / (2 * band), 0, 1)
                  for c, E, FP in zip(callers, Es, FPs)]
        pfold, ev = 1.0, np.zeros(169)
        for cj, E, FP in zip(cs, Es, FPs):
            w = W * cj
            pc = w.sum() / TOTAL_COMBOS
            if pc > 0:
                ev += pfold * pc * (self.EQ @ w / w.sum() * FP - (E - hero_bet))
            pfold *= 1 - pc
        ev += pfold * pot
        p = np.clip(0.5 + ev / (2 * band), 0, 1)
        return p, cs_out, ev

    def call_allin(self, villain_range, pot, hero_total, hero_bet, villain_bet, band=0.08):
        """Hero facing an all-in: call if EV > 0.  pot = current total pot incl. all bets."""
        E = min(hero_total, villain_bet)
        final_pot = pot - villain_bet + E + (E - hero_bet)
        ev = self.eq_vs(villain_range) * final_pot - (E - hero_bet)
        freq = np.clip(0.5 + ev / (2 * band), 0, 1)
        return freq, ev

    # ------------------------------------------------------------- RFI model
    def open_range(self, pos, eff):
        b = _bucket(eff)
        pct = OPEN_PCT.get(pos, OPEN_PCT["CO"])[b]
        return _top_by(self.open_score, pct)

    def villain_open_range(self, pos, eff):
        """Estimated raise range of someone who opened from `pos` (not all-in)."""
        if pos == "BB":
            pos = "SB"
        if eff <= 12:
            return _top_by(self.open_score, OPEN_PCT.get(pos, OPEN_PCT["CO"])[2] * 0.8)
        return self.open_range(pos, eff)

    # ------------------------------------------------------------- spot solvers
    def solve(self, spot, use_solver=True) -> dict:
        """Solver chart when the spot is in the solved tree, else the approximate model."""
        status = None
        if use_solver:
            if self.solved is None:
                from .solved import SolvedCharts
                self.solved = SolvedCharts()
            try:
                ch, status = self.solved.chart(spot, getattr(self, "game", None))
            except Exception as e:  # never lose the chart because of the solver
                ch, status = None, f"solver error: {type(e).__name__}: {e}"
            if ch is not None:
                return ch
        ch = self._solve_model(spot)
        if status and ch.get("freqs") is not None:
            ch["note"] = f"Approximate model shown: {status}"
        if status and ("solving" in status or "queued" in status):
            ch["pending_solver"] = True
        return ch

    def _solve_model(self, spot) -> dict:
        t = spot.get("type")
        if t == "RFI":
            return self._rfi(spot)
        if t == "VS_SHOVE":
            return self._vs_shove(spot)
        if t == "VS_OPEN":
            return self._vs_open(spot)
        if t == "VS_3BET":
            return self._vs_3bet(spot)
        if t == "LIMPED":
            return self._limped(spot)
        return {"actions": [], "freqs": None, "title": "No decision", "method": "-", "approx": False}

    def _chart(self, title, method, approx, **acts):
        acts = {k: np.clip(v, 0, 1) for k, v in acts.items() if v is not None}
        keys = [a for a in ACTION_ORDER if a in acts and a != "fold"]
        tot = sum(acts[a] for a in keys) if keys else np.zeros(169)
        over = tot > 1
        for a in keys:  # normalise if overlapping
            acts[a] = np.where(over, acts[a] / np.maximum(tot, 1e-9), acts[a])
        tot = sum(acts[a] for a in keys) if keys else np.zeros(169)
        fold_key = "check" if "check" in acts and "fold" not in acts else "fold"
        if fold_key == "fold":
            acts["fold"] = np.clip(1 - tot, 0, 1)
        keys = [a for a in ACTION_ORDER if a in acts]
        freqs = np.stack([acts[a] for a in keys], 1)
        freqs[freqs < 0.02] = 0
        freqs = freqs / np.maximum(freqs.sum(1, keepdims=True), 1e-9)
        pct = {a: round(float((freqs[:, i] * W).sum() / TOTAL_COMBOS * 100), 1) for i, a in enumerate(keys)}
        return {"title": title, "method": method, "approx": approx, "actions": keys,
                "freqs": np.round(freqs, 3).tolist(), "pct": pct}

    def _rfi(self, spot):
        pos, eff = spot["hero_pos"], spot["eff_bb"]
        hero = spot["hero"]
        callers = [{"total": p["total"], "bet": p["bet"]} for p in spot["behind"]]
        pot = spot["pot"]
        push, _, ev = self.pushfold(pot, hero["total"], hero["bet"], callers)
        if eff <= 12 or not callers:
            ch = self._chart(f"{pos} first in · {eff:g}bb · push/fold", "Nash push/fold (computed)", False,
                             allin=push)
            ch["ev"] = np.round(ev, 2).tolist()
            return ch
        if pos == "SB":
            b = _bucket(eff)
            raise_ = _top_by(self.open_score, OPEN_PCT["SB"][b])
            limp = np.clip(_top_by(self.open_score, OPEN_PCT["SB"][b] + SB_LIMP_PCT[b]) - raise_, 0, 1)
            # mix strong hands into the limp range like modern SB strategies
            trap = _top_by(self.open_score, 6) * 0.25
            raise_ = raise_ - trap
            limp = limp + trap
            allin = None
            if eff <= 20:
                allin = push * (1 - _top_by(self.open_score, 4)) * (raise_ > 0.01)
                raise_ = np.clip(raise_ - allin, 0, 1)
            return self._chart(f"SB first in · {eff:g}bb", "Approximate SB raise/limp model", True,
                               allin=allin, **{"raise": raise_, "call": limp})
        rng = self.open_range(pos, eff)
        if eff <= 20:
            # raise/jam split: jam hands that are +EV shoves but play badly vs a 3-bet jam
            jam_type = (PAIR & (HI <= 8)) | (HI == 12) & (LO <= 8) | (OFFSUIT & (HI == 11) & (LO >= 8))
            premium = _top_by(self.open_score, 4)
            jam = np.minimum(push, 1.0) * jam_type * (1 - premium)
            raise_ = np.clip(rng - jam, 0, 1)
            raise_ = np.where(jam > 0.5, 0, raise_)
            return self._chart(f"{pos} first in · {eff:g}bb · raise / jam", "Raise model + Nash jam EV", True,
                               allin=jam, **{"raise": raise_})
        return self._chart(f"{pos} first in · {eff:g}bb", "Approximate open-raise model", True,
                           **{"raise": rng})

    def _villain_shove_range(self, spot):
        v = spot["villain"]
        hero = spot["hero"]
        if spot.get("hero_opened") or spot.get("n_raises", 1) >= 2:
            pct = float(np.clip(42 - v["total"], 8, 30))
            return _top_by(self.eq_rand + 0.02 * SUITED, pct), f"re-jam range ~{pct:.0f}% (approx)", True
        # villain was first in: their Nash push range vs everyone behind them
        order_after = [p for p in spot["behind"] if p["seat"] != v["seat"]]
        callers = [{"total": hero["total"], "bet": hero["bet"]}] + \
                  [{"total": p["total"], "bet": p["bet"]} for p in order_after]
        pot_before = spot["pot"] - v["bet"] + v["posted"]
        push, _, _ = self.pushfold(pot_before, v["total"], v["posted"], callers, iters=300)
        if v["total"] > 20:
            push = np.minimum(push, _top_by(self.eq_rand, 25))
        return push, f"{v['pos']} Nash shove range {range_pct(push):.0f}%", v["total"] > 20

    def _vs_shove(self, spot):
        v, hero = spot["villain"], spot["hero"]
        vr, desc, approx = self._villain_shove_range(spot)
        freq, ev = self.call_allin(vr, spot["pot"], hero["total"], hero["bet"], v["bet"])
        ch = self._chart(f"{spot['hero_pos']} vs {v['pos']} all-in · {spot['eff_bb']:g}bb", f"Call EV vs {desc}",
                         approx, call=freq)
        ch["ev"] = np.round(ev, 2).tolist()
        ch["villain_range"] = np.round(vr, 3).tolist()
        return ch

    def _realization(self, hero_pos, villain_pos):
        ip = POSTFLOP_ORDER.index(hero_pos) > POSTFLOP_ORDER.index(villain_pos) if hero_pos in POSTFLOP_ORDER and villain_pos in POSTFLOP_ORDER else False
        r = np.full(169, 0.92 if ip else 0.74)
        r += SUITED * 0.07 + PAIR * 0.05
        r -= (OFFSUIT & (LO < 6)) * 0.10
        r -= (OFFSUIT & (HI - LO > 4)) * 0.04
        return r, ip

    def _vs_open(self, spot):
        v, hero = spot["villain"], spot["hero"]
        pos, eff = spot["hero_pos"], spot["eff_bb"]
        vr = self.villain_open_range(v["pos"], v["total"])
        eqv = self.eq_vs(vr)
        to_call = spot["to_call"]
        pot = spot["pot"]
        R, ip = self._realization(pos, v["pos"])
        players_behind = len(spot["behind"])
        margin = 0.12 * to_call * players_behind
        ev_call = eqv * R * (pot + to_call) - to_call - margin
        title = f"{pos} vs {v['pos']} open {v['bet']:g}bb · {eff:g}bb"
        if eff <= 25:
            # 3-bet jam: villain calls with the part of their range that beats our jam range
            jam = (eqv > 0.55).astype(float)
            E = min(hero["total"], v["total"])
            fp = pot - v["bet"] - hero["bet"] + 2 * E
            for _ in range(60):
                eq_v = self.eq_vs(jam)  # villain hands vs our jam range
                vcall = vr * (eq_v * fp - (E - v["bet"]) > 0)
                w = W * vcall
                pc = w.sum() / max((W * vr).sum(), 1e-9)
                eq_h = self.EQ @ w / max(w.sum(), 1e-9)
                ev_jam = (1 - pc) * pot + pc * (eq_h * fp - (E - hero["bet"]))
                best = (ev_jam > np.maximum(ev_call, 0)).astype(float)
                jam += 0.2 * (best - jam)
            call = (ev_call > np.maximum(ev_jam, 0)).astype(float) * (1 - jam)
            ch = self._chart(title, "3-bet jam EV (computed) + flat-call heuristic", True, allin=jam, call=call)
            ch["ev"] = np.round(np.maximum(0, np.maximum(ev_jam, ev_call)), 2).tolist()
            return ch
        # deeper: 3-bet for value/bluff, flat by realised equity
        value = (eqv > 0.60) | (PAIR & (HI >= 10))
        bluff = SUITED & (HI == 12) & (LO <= 3) | SUITED & (HI >= 10) & (LO >= 8) & (eqv > 0.45)
        three = value.astype(float) + bluff * (0.5 if (ip or pos == "BB") else 0.35) * (~value)
        call = (ev_call > 0) * (1 - three)
        if pos != "BB" and not ip:
            call *= 0.5   # out of position cold-calls are mostly 3-bet-or-fold
        return self._chart(title, "Approximate 3-bet/flat model (equity vs estimated open range)", True,
                           **{"raise": three, "call": call})

    def _vs_3bet(self, spot):
        v, hero = spot["villain"], spot["hero"]
        eqv = self.eq_vs(_top_by(self.eq_rand + 0.02 * SUITED, 10))
        to_call = spot["to_call"]
        pot = spot["pot"]
        R, _ = self._realization(spot["hero_pos"], v["pos"])
        ev_call = eqv * R * (pot + to_call) - to_call
        jam = (eqv > 0.55).astype(float)
        call = (ev_call > 0) * (1 - jam)
        key = "allin" if spot["eff_bb"] <= 40 else "raise"
        return self._chart(f"{spot['hero_pos']} vs {v['pos']} 3-bet · {spot['eff_bb']:g}bb",
                           "Approximate vs 3-bet model (villain ~10% range)", True, call=call, **{key: jam})

    def _limped(self, spot):
        pos, eff, hero = spot["hero_pos"], spot["eff_bb"], spot["hero"]
        callers = [{"total": p["total"], "bet": p["bet"]} for p in spot["behind"] + spot["limpers"]]
        if eff <= 15:
            push, _, ev = self.pushfold(spot["pot"], hero["total"], hero["bet"], callers)
            extra = {"check": np.ones(169) - push} if pos == "BB" else {}
            ch = self._chart(f"{pos} vs limp · {eff:g}bb · iso-jam", "Nash jam vs limpers (computed)", False,
                             allin=push, **extra)
            ch["ev"] = np.round(ev, 2).tolist()
            return ch
        n_l = len(spot["limpers"])
        pct = OPEN_PCT.get(pos, OPEN_PCT["CO"])[_bucket(eff)] * (0.55 if pos == "BB" else 0.65) / (1 + 0.25 * (n_l - 1))
        iso = _top_by(self.open_score, pct)
        if pos == "BB":
            return self._chart(f"BB vs {n_l} limp(s) · {eff:g}bb", "Approximate iso-raise model", True,
                               **{"raise": iso, "check": 1 - iso})
        over = (SUITED | PAIR) * (1 - iso) * (_top_by(self.open_score, pct * 1.8) > 0) * (0.6 if pos in ("SB", "BTN", "CO") else 0.3)
        return self._chart(f"{pos} vs {n_l} limp(s) · {eff:g}bb", "Approximate iso-raise model", True,
                           **{"raise": iso, "call": over})
