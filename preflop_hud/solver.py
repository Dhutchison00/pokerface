"""Preflop-only CFR solver (chip EV) with a modelled postflop.

Game tree (n-handed, action order UTG ... SB, BB):
  first in      fold / open-raise / open-jam (<= open_jam_max bb) ; SB may also limp
  vs open       fold / call / 3-bet / jam       (first defender to continue ends the
                                                 others' options: no squeezes/multiway)
  vs 3-bet      fold / call / 4-bet jam
  vs 4-bet jam  fold / call
  vs a jam      fold / call
  SB limp       BB: check / iso-raise / jam ; SB vs iso: fold / call / jam
Terminals:
  * someone wins uncontested
  * all-in showdown       -> exact equity (169x169 table)
  * pot goes to the flop  -> pot share from equity scaled by a realisation model
    (position, stack-to-pot ratio, hand playability).  This model is the
    approximation; everything preflop is solved.

Algorithm: Discounted CFR (alpha=1.5, beta=0, gamma=2), simultaneous updates,
vectorised over the 169 hand classes.  Card removal between players ignored.
Convergence is measured by each player's best-response gain (bb per hand).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np

from . import equity
from .cards import COMBO_WEIGHTS as W, TOTAL_COMBOS, RANK_VALUE, HAND_NAMES
from .spot import POSITION_NAMES

POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL_DIR = os.path.join(ROOT, "data", "solutions")
STACK_GRID = [10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60, 80, 100]
MODEL_VERSION = 2

DEFAULT_PARAMS = {
    # open-raise size (bb, raise-to) by effective stack: first threshold that eff <= wins
    "open_size": [[20, 2.0], [40, 2.2], [1000, 2.3]],
    "sb_open_size": [[20, 2.5], [1000, 3.0]],
    "threebet_ip": 3.0,          # x the open, defender in position
    "threebet_oop": 4.0,         # x the open, defender out of position (blinds)
    "max_raise_frac": 0.40,      # a raise that commits > this share of the stack becomes jam-only
    "open_jam_max": 25,          # open-jam option only at or below this effective stack
    "bb_iso": 3.5,               # BB raise-to vs an SB limp
    # postflop realisation model (logit scale):
    #   share = sigmoid(gamma * logit(equity) + position edge + playability difference)
    # gamma > 1: strong hands realise more than their equity, weak hands less.
    # All three terms fade towards raw equity as the stack-to-pot ratio falls.
    "gamma": 1.6,                # at SPR == spr_full; grows with sqrt(SPR) up to gamma_spr_cap
    "gamma_spr_cap": 1.6,
    "pos_edge_base": 0.18, "pos_edge_per_bb": 0.004, "pos_edge_max": 0.25,
    "spr_full": 3.0,
    "hf_suited": 0.12, "hf_pair": 0.0, "hf_connected": 0.06, "hf_broadway": 0.05,
    "hf_offsuit_gapped": -0.08, "hf_offsuit_low": -0.12,
    "init_edge": 0.12,           # preflop aggressor's edge (range + initiative), logit units
    "squeeze_tax": 0.04,         # pot share lost by opener + cold-caller per player left behind
}


def _hand_factor(p):
    hf = np.zeros(169)
    for k, h in enumerate(HAND_NAMES):
        hi, lo = RANK_VALUE[h[0]], RANK_VALUE[h[1]]
        if len(h) == 2:
            hf[k] += p["hf_pair"]
            continue
        gap = hi - lo - 1
        if h[2] == "s":
            hf[k] += p["hf_suited"]
            if gap <= 1 and lo >= 3:
                hf[k] += p["hf_connected"]
        else:
            if gap >= 2:
                hf[k] += p["hf_offsuit_gapped"]
            if lo < 6:
                hf[k] += p["hf_offsuit_low"]
        if lo >= 8:
            hf[k] += p["hf_broadway"]
    return hf


def _size(table, eff):
    for thr, s in table:
        if eff <= thr:
            return s
    return table[-1][1]


class Tree:
    """Flat preorder list of nodes.  Decision nodes: player, actions, children.
    Terminal nodes: kind 'win' | 'showdown' | 'flop' with investments."""

    def __init__(self, n, stacks, dead, params=None, jamfold_only=False, posted=None):
        self.n = n
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.names = POSITION_NAMES[n]
        self.stacks = np.array(stacks, float)       # total chips incl. blinds, action order
        self.dead = float(dead)                     # antes in the pot
        self.ante = self.dead / n
        self.jamfold_only = jamfold_only
        self.sb = n - 2 if n >= 3 else 0
        self.bb = n - 1
        self.EQ = equity.load()
        self.hf = _hand_factor(self.p)
        self._share_cache = {}
        self.nodes = []
        if posted is not None:
            inv = np.array(posted, float)
        else:
            inv = np.zeros(n)
            inv[self.sb] = min(0.5, self.stacks[self.sb])
            inv[self.bb] = min(1.0, self.stacks[self.bb])
        self._first_in(0, inv)
        self._finalise()

    # ---------------------------------------------------------------- helpers
    def _ip(self, a, b):
        """True if player index a acts after b postflop."""
        na, nb = self.names[a], self.names[b]
        if self.n == 2:
            return na == "BTN"
        return POSTFLOP_ORDER.index(na) > POSTFLOP_ORDER.index(nb)

    def _eff(self, a, b):
        return float(min(self.stacks[a], self.stacks[b]))

    def _add(self, node):
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _decision(self, player, key):
        idx = self._add({"kind": "dec", "player": player, "key": key, "actions": [], "children": []})
        return idx

    def _child(self, idx, action, child_idx):
        self.nodes[idx]["actions"].append(action)
        self.nodes[idx]["children"].append(child_idx)

    def _win(self, winner, inv):
        return self._add({"kind": "win", "winner": winner, "inv": inv.copy()})

    def _showdown(self, a, b, inv):
        inv = inv.copy()
        cap = self._eff(a, b)
        inv[a] = inv[b] = cap
        return self._add({"kind": "sd", "a": a, "b": b, "inv": inv, "S": self.EQ})

    def _flop(self, a, b, inv, caller=None, behind=0, aggressor=None):
        """Pot goes to a flop between a and b; caller with `behind` players still to act pays the squeeze tax."""
        pot = self.dead + inv.sum()
        eff_left = self._eff(a, b) - max(inv[a], inv[b])
        spr = eff_left / pot if pot > 0 else 0
        p = self.p
        f = min(1.0, spr / p["spr_full"])
        # gamma keeps growing with depth (implied / reverse-implied odds), sub-linearly
        fg = min(spr / p["spr_full"], p["gamma_spr_cap"]) ** 0.5
        edge = min(p["pos_edge_max"], p["pos_edge_base"] + p["pos_edge_per_bb"] * self._eff(a, b)) * f
        a_ip = self._ip(a, b)
        # squeeze / multiway risk: when an open is cold-called with players still behind,
        # both players give up a share tau of the pot to the big blind (the would-be squeezer)
        tau = p["squeeze_tax"] * behind if (caller is not None and behind > 0) else 0.0
        third = self.bb if tau > 0 else None
        init = 0.0
        if aggressor is not None:
            init = p["init_edge"] * f * (1 if aggressor == a else -1)
        key = (round(f, 3), round(fg, 3), round(edge, 3), a_ip, round(init, 3))
        S = self._share_cache.get(key)
        if S is None:
            eq = np.clip(self.EQ, 1e-4, 1 - 1e-4)
            g = 1 + (p["gamma"] - 1) * fg
            z = g * np.log(eq / (1 - eq)) + (edge if a_ip else -edge) + init + \
                (self.hf[:, None] - self.hf[None, :]) * f
            S = 1 / (1 + np.exp(-z))
            S = np.clip(S, 0, 1)
            self._share_cache[key] = S
        return self._add({"kind": "flop", "a": a, "b": b, "inv": inv.copy(), "S": S,
                          "tau": tau, "third": third})

    # ---------------------------------------------------------------- tree
    def _first_in(self, i, inv):
        n, p = self.n, self.p
        if i == self.bb:
            return self._win(self.bb, inv)
        pos = self.names[i]
        idx = self._decision(i, f"RFI:{pos}")
        self._child(idx, "fold", self._first_in(i + 1, inv))
        eff = float(min(self.stacks[i], max(self.stacks[i + 1:])))
        if not self.jamfold_only:
            if i == self.sb:
                limp = inv.copy()
                limp[i] = 1.0
                self._child(idx, "call", self._bb_vs_limp(limp))
            size = _size(p["sb_open_size"] if (i == self.sb and n > 2) else p["open_size"], eff)
            if size < p["max_raise_frac"] * eff:
                r = inv.copy()
                r[i] = size
                self._child(idx, "raise", self._defend(i, i + 1, r, size))
        if self.jamfold_only or eff <= p["open_jam_max"]:
            j = inv.copy()
            j[i] = self.stacks[i]
            self._child(idx, "allin", self._vs_first_jam(i, i + 1, j))
        return idx

    def _vs_first_jam(self, j, d, inv):
        if d >= self.n:
            return self._win(j, inv)
        idx = self._decision(d, f"VS_JAM:{self.names[d]}:{self.names[j]}")
        self._child(idx, "fold", self._vs_first_jam(j, d + 1, inv))
        self._child(idx, "call", self._showdown(j, d, inv))
        return idx

    def _defend(self, o, d, inv, raise_to):
        n, p = self.n, self.p
        if d >= n:
            return self._win(o, inv)
        idx = self._decision(d, f"VS_OPEN:{self.names[d]}:{self.names[o]}")
        self._child(idx, "fold", self._defend(o, d + 1, inv, raise_to))
        eff = self._eff(o, d)
        c = inv.copy()
        c[d] = min(raise_to, self.stacks[d])
        if c[d] >= self.stacks[d] - 1e-9:
            self._child(idx, "call", self._showdown(o, d, c))
        else:
            self._child(idx, "call", self._flop(o, d, c, caller=d, behind=n - 1 - d, aggressor=o))
        mult = p["threebet_ip"] if self._ip(d, o) else p["threebet_oop"]
        size3 = raise_to * mult
        if size3 < p["max_raise_frac"] * eff:
            t = inv.copy()
            t[d] = size3
            self._child(idx, "raise", self._opener_vs_3bet(o, d, t))
        if eff > raise_to:
            j = inv.copy()
            j[d] = self.stacks[d]
            self._child(idx, "allin", self._opener_vs_jam(o, d, j, f"VS_3BETJAM:{self.names[o]}:{self.names[d]}"))
        return idx

    def _opener_vs_3bet(self, o, d, inv):
        idx = self._decision(o, f"VS_3BET:{self.names[o]}:{self.names[d]}")
        self._child(idx, "fold", self._win(d, inv))
        c = inv.copy()
        c[o] = inv[d]
        self._child(idx, "call", self._flop(o, d, c, aggressor=d))
        j = inv.copy()
        j[o] = self.stacks[o]
        k = self._decision(d, f"VS_4BET:{self.names[d]}:{self.names[o]}")
        self._child(k, "fold", self._win(o, inv))
        self._child(k, "call", self._showdown(o, d, j))
        self._child(idx, "allin", k)
        return idx

    def _opener_vs_jam(self, o, d, inv, key):
        idx = self._decision(o, key)
        self._child(idx, "fold", self._win(d, inv))
        self._child(idx, "call", self._showdown(o, d, inv))
        return idx

    def _bb_vs_limp(self, inv):
        sb, bb, p = self.sb, self.bb, self.p
        idx = self._decision(bb, "BB_VS_LIMP")
        self._child(idx, "check", self._flop(sb, bb, inv))
        eff = self._eff(sb, bb)
        if p["bb_iso"] < p["max_raise_frac"] * eff:
            r = inv.copy()
            r[bb] = p["bb_iso"]
            s = self._decision(sb, "SB_VS_ISO")
            self._child(s, "fold", self._win(bb, r))
            c = r.copy()
            c[sb] = r[bb]
            self._child(s, "call", self._flop(sb, bb, c, aggressor=bb))
            j = r.copy()
            j[sb] = self.stacks[sb]
            self._child(s, "allin", self._opener_vs_jam(bb, sb, j, "BB_VS_SBJAM_ISO"))
            self._child(idx, "raise", s)
        j = inv.copy()
        j[bb] = self.stacks[bb]
        self._child(idx, "allin", self._opener_vs_jam(sb, bb, j, "SB_VS_BBJAM"))
        return idx

    def _finalise(self):
        D, n = self.dead, self.n
        rake_pct, rake_cap = self.p.get("rake_pct", 0.0) / 100.0, self.p.get("rake_cap", 0.0)
        for nd in self.nodes:
            if nd["kind"] == "dec":
                continue
            inv = nd["inv"]
            assert np.all(inv <= self.stacks + 1e-9), ("investment above stack", nd)
            nd["pot"] = D + inv.sum()
            if rake_pct and nd["kind"] != "win":
                # cash rake, "no flop no drop": only pots that see a flop or a showdown are raked
                nd["pot"] -= min(rake_pct * nd["pot"], rake_cap if rake_cap > 0 else np.inf)
            if nd["kind"] != "win":
                S = nd["S"]
                assert S.min() >= -1e-12 and S.max() <= 1 + 1e-12
            # zero-sum check: sum of utilities over all players = 0
            if nd["kind"] == "flop":
                assert 0 <= nd.get("tau", 0.0) < 1
        self.dec = [i for i, nd in enumerate(self.nodes) if nd["kind"] == "dec"]
        self.parent = {}
        for i in self.dec:
            for c in self.nodes[i]["children"]:
                self.parent[c] = i


def _excl_products(m):
    """prod_{k != q} m_k for every q, without division (m may contain zeros)."""
    n = len(m)
    pre = np.ones(n + 1)
    suf = np.ones(n + 1)
    for i in range(n):
        pre[i + 1] = pre[i] * m[i]
        suf[n - 1 - i] = suf[n - i] * m[n - 1 - i]
    return pre[:n] * suf[1:]


def check_zero_sum(tree, trials=3, seed=0):
    """Numerically verify every terminal pays out exactly what was put in: with random
    reach vectors, sum over players of E[utility] must be 0.  Returns the worst error."""
    rng = np.random.default_rng(seed)
    cfr = CFR(tree)
    worst = 0.0
    for _ in range(trials):
        reach = rng.random((tree.n, 169))
        for nd in tree.nodes:
            if nd["kind"] == "dec":
                continue
            vals = cfr._terminal(nd, reach)
            total = sum((reach[q] * cfr.wn) @ vals[q] for q in range(tree.n))
            worst = max(worst, abs(total))
    return worst


class CFR:
    def __init__(self, tree: Tree):
        self.t = tree
        self.n = tree.n
        self.wn = W / TOTAL_COMBOS
        self.regret = {}
        self.ssum = {}
        for i in tree.dec:
            A = len(tree.nodes[i]["actions"])
            self.regret[i] = np.zeros((A, 169))
            self.ssum[i] = np.zeros((A, 169))
        self.iters = 0

    def _sigma(self, i):
        r = np.maximum(self.regret[i], 0)
        s = r.sum(0)
        A = r.shape[0]
        return np.where(s > 0, r / np.where(s > 0, s, 1), 1.0 / A)

    def avg_strategy(self, i):
        s = self.ssum[i]
        tot = s.sum(0)
        A = s.shape[0]
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 1.0 / A)

    # ---------------------------------------------------------------- traversal
    def _terminal(self, nd, reach):
        n = self.n
        m = reach @ self.wn                     # marginal reach of each player
        excl = _excl_products(m)
        inv, pot, ante = nd["inv"], nd["pot"], self.t.ante
        vals = np.empty((n, 169))
        if nd["kind"] == "win":
            for q in range(n):
                vals[q] = ((pot if q == nd["winner"] else 0.0) - inv[q] - ante) * excl[q]
            return vals
        a, b, S = nd["a"], nd["b"], nd["S"]
        others = 1.0
        for k in range(n):
            if k != a and k != b:
                others *= m[k]
        wa = reach[a] * self.wn
        wb = reach[b] * self.wn
        for q in range(n):
            if q != a and q != b:
                vals[q] = (-inv[q] - ante) * excl[q]
        tau = nd.get("tau", 0.0)
        third = nd.get("third")
        if tau:
            vals[third] = (tau * pot - inv[third] - ante) * excl[third]
        sp = pot * (1.0 - tau)
        vals[a] = others * (sp * (S @ wb) - (inv[a] + ante) * wb.sum())
        vals[b] = others * (sp * ((1.0 - S).T @ wa) - (inv[b] + ante) * wa.sum())
        return vals

    def _traverse(self, sig_fn, br_player_mask=None, record=None):
        """Generic pass.  sig_fn(i) -> strategy (A,169) used for reach and values.
        br_player_mask: if given, players with True maximise instead (best response values).
        record: dict to fill with per-node action values."""
        t = self.t
        nodes = t.nodes
        reach = {0: np.ones((self.n, 169))}
        sigmas = {}
        for i in t.dec:  # preorder: parents before children
            sg = sig_fn(i)
            sigmas[i] = sg
            p = nodes[i]["player"]
            r = reach[i]
            for a, c in enumerate(nodes[i]["children"]):
                rc = r.copy()
                rc[p] = r[p] * sg[a]
                reach[c] = rc
        vals = {}
        for i in range(len(nodes) - 1, -1, -1):
            nd = nodes[i]
            if nd["kind"] != "dec":
                vals[i] = self._terminal(nd, reach[i])
                continue
            p = nd["player"]
            ch = np.stack([vals[c] for c in nd["children"]])   # (A, n, 169)
            v = ch.sum(0)
            if br_player_mask is not None and br_player_mask[p]:
                v[p] = ch[:, p, :].max(0)
            else:
                v[p] = (sigmas[i] * ch[:, p, :]).sum(0)
            vals[i] = v
            if record is not None:
                record[i] = (ch[:, p, :], reach[i], sigmas[i])
        return vals, reach, sigmas

    def iterate(self, k=1):
        for _ in range(k):
            self.iters += 1
            T = self.iters
            vals, reach, sig = self._traverse(self._sigma, record=(rec := {}))
            pos_w = T ** 1.5 / (T ** 1.5 + 1)
            avg_w = (T / (T + 1)) ** 2
            for i, (cv, r, sg) in rec.items():
                p = self.t.nodes[i]["player"]
                node_v = (sg * cv).sum(0)
                R = self.regret[i]
                R *= np.where(R > 0, pos_w, 0.5)   # DCFR discounting
                R += cv - node_v
                self.ssum[i] *= avg_w
                self.ssum[i] += r[p] * sg

    def exploitability(self):
        """Best-response gain per player (bb per hand) vs the average strategy."""
        n = self.n
        ev_vals, _, _ = self._traverse(self.avg_strategy)
        br_vals, _, _ = self._traverse(self.avg_strategy, br_player_mask=[True] * n)
        ev = ev_vals[0] @ self.wn
        br = br_vals[0] @ self.wn
        return br - ev, ev

    def solve(self, iters=600, check_every=100, tol=0.005, verbose=False):
        t0 = time.time()
        gains = None
        while self.iters < iters:
            self.iterate(min(check_every, iters - self.iters))
            gains, ev = self.exploitability()
            if verbose:
                print(f"  iter {self.iters:5d}  max BR gain {gains.max():.4f} bb  ({time.time() - t0:.0f}s)", flush=True)
            if gains.max() < tol:
                break
        return gains

    def export(self):
        """Per spot key: actions, average strategy, conditional EV per action, hero reach."""
        rec = {}
        self._traverse(self.avg_strategy, record=rec)
        out = {}
        for i, (cv, r, sg) in rec.items():
            nd = self.t.nodes[i]
            p = nd["player"]
            m = r @ self.wn
            excl = _excl_products(m)[p]
            ev = cv / excl if excl > 1e-12 else np.zeros_like(cv)
            # utility includes -ante for everyone; report EV relative to the ante already paid
            ev = ev + self.t.ante
            key = nd["key"]
            if key in out:  # same public spot reached by different paths shouldn't happen
                continue
            out[key] = {"actions": nd["actions"], "strategy": sg, "ev": ev, "reach": r[p],
                        "reach_others": float(excl)}
        return out


def live_jamfold(stacks, posted, dead, iters=250, params=None):
    """Jam/fold equilibrium for the players still in, using their real stacks.

    stacks/posted are in action order starting with the first player to act
    (everyone earlier has folded).  params: DEFAULT_PARAMS overrides (rake).  Returns (tree, cfr)."""
    n = len(stacks)
    t = Tree(n, stacks, dead, params, jamfold_only=True, posted=posted)
    c = CFR(t)
    c.iterate(iters)
    return t, c


def node_export(cfr, i):
    """Strategy / conditional EV / own reach at one decision node of a solved tree."""
    rec = {}
    cfr._traverse(cfr.avg_strategy, record=rec)
    cv, r, sg = rec[i]
    p = cfr.t.nodes[i]["player"]
    excl = _excl_products(r @ cfr.wn)[p]
    ev = cv / excl + cfr.t.ante if excl > 1e-12 else np.zeros_like(cv)
    return {"actions": cfr.t.nodes[i]["actions"], "strategy": sg, "ev": ev, "reach": r[p]}


# -------------------------------------------------------------------- storage
def sol_path(n, stack, dead, params_tag="default"):
    return os.path.join(SOL_DIR, f"v{MODEL_VERSION}_{params_tag}_n{n}_s{stack}_d{dead:.2f}.npz")


def solve_and_save(n, stack, dead, iters=600, params=None, params_tag="default", verbose=True):
    tree = Tree(n, [stack] * n, dead, params)
    cfr = CFR(tree)
    gains = cfr.solve(iters=iters, tol=0.002, verbose=verbose)
    data = cfr.export()
    os.makedirs(SOL_DIR, exist_ok=True)
    arrays = {}
    meta = {"n": n, "stack": stack, "dead": dead, "iters": cfr.iters, "br_gain": gains.tolist(),
            "params": tree.p, "spots": {}}
    for k, d in data.items():
        safe = k.replace(":", "__")
        arrays[f"{safe}__strategy"] = d["strategy"].astype(np.float32)
        arrays[f"{safe}__ev"] = d["ev"].astype(np.float32)
        arrays[f"{safe}__reach"] = d["reach"].astype(np.float32)
        meta["spots"][k] = {"actions": d["actions"], "reach_others": d["reach_others"]}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path = sol_path(n, stack, dead, params_tag)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path, meta


class Solution:
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(bytes(z["meta"]).decode())
        self.z = z
        self.path = path

    def spot(self, key):
        info = self.meta["spots"].get(key)
        if info is None:
            return None
        safe = key.replace(":", "__")
        return {"actions": info["actions"], "strategy": self.z[f"{safe}__strategy"],
                "ev": self.z[f"{safe}__ev"], "reach": self.z[f"{safe}__reach"],
                "reach_others": info["reach_others"]}


class SolverStore:
    """Finds the solved grid point for (players, effective stack, antes); solves
    missing ones in a background thread (one at a time) and caches them on disk."""

    def __init__(self, iters=600, params=None, tag="default"):
        """params: overrides of DEFAULT_PARAMS (cash sizes, rake); tag names their cache files."""
        self.iters = iters
        self.params = params or None
        self.tag = tag
        self._cache = {}
        self._queue = []
        self._busy = None
        self._lock = threading.Lock()
        self._thread = None
        self.last_error = None

    @staticmethod
    def grid_stack(eff):
        return min(STACK_GRID, key=lambda s: abs(s - eff))

    @staticmethod
    def dead_key(dead):
        return round(dead * 4) / 4

    def get(self, n, eff, dead):
        if n < 2 or n > 9:
            return None, "unsupported table size"
        s = self.grid_stack(eff)
        d = self.dead_key(dead)
        key = (n, s, d)
        if key in self._cache:
            return self._cache[key], None
        path = sol_path(n, s, d, self.tag)
        if os.path.exists(path):
            sol = Solution(path)
            self._cache[key] = sol
            return sol, None
        self._enqueue(key)
        return None, f"solving {n}-handed {s}bb (first time only)…" if self._busy == key else \
            f"queued solve for {n}-handed {s}bb"

    def _enqueue(self, key):
        with self._lock:
            if key not in self._queue and key != self._busy:
                self._queue.append(key)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def _worker(self):
        while True:
            with self._lock:
                if not self._queue:
                    self._busy = None
                    return
                key = self._queue.pop(0)
                self._busy = key
            try:
                n, s, d = key
                solve_and_save(n, s, d, iters=self.iters, params=self.params, params_tag=self.tag,
                               verbose=False)
            except Exception as e:  # keep serving fallback charts
                self.last_error = f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Precompute preflop solutions")
    ap.add_argument("--players", type=int, nargs="+", default=[8])
    ap.add_argument("--stacks", type=int, nargs="+", default=STACK_GRID)
    ap.add_argument("--ante-per-player", type=float, default=0.125, help="ante in bb per player")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--mode", choices=["tournament", "cash"], default="tournament",
                    help="cash: 2.5bb opens and rake (antes default to 0)")
    ap.add_argument("--rake-pct", type=float, default=5.0)
    ap.add_argument("--rake-cap", type=float, default=3.0)
    a = ap.parse_args()
    from preflop_hud import modes
    game = {"mode": a.mode, "rake_pct": a.rake_pct, "rake_cap": a.rake_cap}
    params, tag = modes.solver_params(game), modes.solver_tag(game)
    if a.mode == "cash" and "--ante-per-player" not in sys.argv:
        a.ante_per_player = 0.0
    for n in a.players:
        dead = SolverStore.dead_key(a.ante_per_player * n)
        for s in a.stacks:
            t0 = time.time()
            path, meta = solve_and_save(n, s, dead, iters=a.iters, params=params, params_tag=tag, verbose=False)
            print(f"{n}-handed {s:3d}bb ante {dead}: max BR gain {max(meta['br_gain']):.4f} bb, "
                  f"{meta['iters']} iters, {time.time() - t0:.0f}s -> {os.path.basename(path)}", flush=True)
