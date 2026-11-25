"""Live postflop advice, heads-up: GTOpen's postflop solver on the hand the tracker followed.

Each street is solved on its own, from that street's start, on a gto-server of the HUD's own:

  1. the flop: both players' ranges from the preflop line, each player's own reach at their last
     preflop decision times how often they took the action they took there, from the same preflop
     charts the HUD shows (GTOpen exports first, else the HUD solver's cache).
  2. the turn and river: each player's range is their reach at the end of the previous street's
     solved tree, combo by combo (so it holds the betting so far). A player who took a line the
     solver almost never takes keeps the range they had before it (noted).
  3. each street's spot: the board so far, the pot and stacks the street started with, a bet menu
     (rake in cash mode; chip EV in tournaments, ICM isn't applied postflop yet). A bet or raise
     the menu doesn't have is added to that street's menu and the street re-solved, so the tree
     holds the size that was really used (the old answer shows, marked, while it re-solves).
  4. solving runs in the background, quickly to ~2.5% of the pot first, then on to 0.3% while the
     advice is already shown; the street's logged actions are then followed down the tree to the
     hero's decision, and the hero's exact combo read: how often to take each action, each one's EV.

Anything it can't do yet comes back as a reason instead of advice.
"""
from __future__ import annotations

import atexit
import collections
import os
import threading
import time

import numpy as np

from .cards import HAND_INDEX, HAND_NAMES, hole_cards_to_class
from .tracker import POSTFLOP_ORDER
from . import multiway as mw
from .multiway import players_at

CHIPS = 10                      # chips per big blind sent to GTOpen (its trees are scale-free)
PORT = 3741                     # the preflop batch uses 3739
HERO_FLOOR = 0.02               # the hero's class is kept in the hero's range at least this much
ROUGH_PCT, FINAL_PCT = 2.5, 0.3  # exploitability targets, % of the pot: a first answer, then refined
MAX_ITERS = (600, 2000)
MW_ITERS = (1000, 3000)          # the 3-way solver counts iterations per call, not in total
MW_FINAL_PCT = 0.5
PREFETCH_DEPTH = 4              # after a solve, read the street's nodes this many actions deep
SIZE_NOTE = 0.15                # an observed bet this far (relative) from the menu's nearest is re-solved
STREETS = ("flop", "turn", "river")
LOW_LINE = 0.02                 # a line a player takes less often than this: their earlier range is kept

# bet menus, % of the pot (raises: % of the pot after calling; "a" = all-in), and the raise cap.
# Deep pots with wide ranges (limped pots) make big trees: over BIG_TREE_MB of VRAM, the spot is
# rebuilt with the small menu, which roughly halves the tree and the solve time (and fits beside OCR).
SIZES = {"flop": ("33 75", "60"), "turn": ("75", "60"), "river": ("75", "60")}
SIZES_SMALL = {"flop": ("33", "60"), "turn": ("75", "a"), "river": ("75", "a")}
MAX_RAISES = {"full": 3, "small": 2}
BIG_TREE_MB = 6000
DEEP_SPR = 4.5                  # stack-to-pot ratio from which the small menu is used straight away
SMALL_NOTE = "deep pot: solved with a smaller bet menu (one 33% flop bet, all-in turn/river raises) to answer in seconds"
# trees rooted on the turn or river are tiny (thousands of nodes, ~0.1 s), so their menu is richer,
# and the player out of position may lead into the last street's bettor ("donk"). Flop trees don't
# get donk bets: they'd grow ~25% (a deep limped pot's first answer: 4.7 s -> 6.7 s), and the turn
# and river are solved again from their own start anyway.
LATER_SIZES = {"turn": ("33 75", "60 a"), "river": ("33 75 a", "60 a")}
LATER_MAX_RAISES = 4

RANK_ORDER = "23456789TJQKA"
SUIT_ORDER = "cdhs"


def card_index(c):
    """GTOpen's card numbering: rank * 4 + suit (2c = 0 ... As = 51)."""
    return RANK_ORDER.index(c[0].upper()) * 4 + SUIT_ORDER.index(c[1].lower())


# ---------------------------------------------------------------- 1. ranges from the preflop line
def _sb_name(n):
    return "BTN" if n == 2 else "SB"


LIMP_NOTE = ("{l} limped, which the charts almost never do from there (only the SB limps at these "
             "depths): {l}'s range is assumed to be the hands {l} would open-raise")


def preflop_plan(hand):
    """Which preflop node, and which action there, made each flop player's range.
    Returns ({pos: (node key, action)}, line text, notes) or (None, reason, [])."""
    plan, line = _preflop_plan(hand)
    notes = []
    if plan is not None:
        notes = [v[2] for v in plan.values() if len(v) > 2]          # assumed ranges, said on screen
        plan = {pos: v[:2] for pos, v in plan.items()}
    return plan, line, notes


def _preflop_plan(hand):
    positions = hand.get("positions") or {}
    n = len(positions)
    if not n:
        return None, "positions unknown"
    for note in hand.get("notes") or []:
        if note.startswith("joined"):
            return None, "the preflop action wasn't all seen (tracking started late): ranges unknown"
    seq = [(a["pos"], a["action"]) for a in hand["actions"] if a["street"] == "preflop" and a["action"] != "fold"]
    flop_pos = {positions.get(str(s)) for s in players_at(hand, "flop")}
    sb, bb = _sb_name(n), "BB"
    txt = ", ".join(f"{p} {a}" for p, a in seq) or "no preflop action seen"
    if any(p not in flop_pos for p, _ in seq):
        return None, f"preflop line {txt}: a player put chips in and folded later (not in the charts yet)"
    if len(seq) == 2 and seq[0][1] == "raise" and seq[1][1] == "call":
        o, c = seq[0][0], seq[1][0]
        return {o: (f"RFI:{o}", "raise"), c: (f"VS_OPEN:{c}:{o}", "call")}, f"{o} open, {c} call"
    if seq == [(sb, "call"), (bb, "check")]:
        return {sb: (f"RFI:{sb}", "call"), bb: ("BB_VS_LIMP", "check")}, f"{sb} limp, {bb} check"
    if seq == [(sb, "call"), (bb, "raise"), (sb, "call")]:
        return {sb: ("SB_VS_ISO", "call"), bb: ("BB_VS_LIMP", "raise")}, f"{sb} limp, {bb} raise, {sb} call"
    # a limp from any other seat: an equilibrium line only for the SB, so the solver's nodes after it
    # aren't converged. Assumed ranges instead: the limper plays the hands it would open-raise, an
    # iso-raise is the raiser's own open range (the BB's: its raise against an SB limp)
    if len(seq) in (2, 3) and seq[0][1] == "call" and seq[0][0] not in (sb, bb):
        lmp, other = seq[0][0], seq[1][0]
        note = LIMP_NOTE.format(l=lmp)
        if [a for _, a in seq] == ["call", "check"] and other == bb:
            return ({lmp: (f"RFI:{lmp}", "raise", note), bb: ("BB_VS_LIMP", "check",
                     "the BB's checking range is the one it has against an SB limp")}, f"{lmp} limp, {bb} check")
        if [a for _, a in seq] == ["call", "raise", "call"] and seq[2][0] == lmp:
            iso = (("BB_VS_LIMP", "raise", "the BB's iso-raise range is the one it has against an SB limp")
                   if other == bb else (f"RFI:{other}", "raise", f"{other}'s iso-raise range is assumed to be its open range"))
            return {lmp: (f"RFI:{lmp}", "raise", note), other: iso}, f"{lmp} limp, {other} raise, {lmp} call"
    if len(seq) == 3 and [a for _, a in seq] == ["raise", "raise", "call"] and seq[0][0] == seq[2][0]:
        o, t = seq[0][0], seq[1][0]
        return {o: (f"VS_3BET:{o}:{t}", "call"), t: (f"VS_OPEN:{t}:{o}", "raise")}, f"{o} open, {t} 3-bet, {o} call"
    return None, f"preflop line {txt} isn't in the preflop charts yet (over-limps, limp-reraises, 4-bets, cold calls)"


def _start_stack(hand, seat):
    pre = (hand.get("start") or {}).get("preflop") or {}
    return (pre.get("stacks") or {}).get(seat, (pre.get("stacks") or {}).get(str(seat)))


def _flop_stack(hand, seat):
    fl = (hand.get("start") or {}).get("flop") or {}
    return (fl.get("stacks") or {}).get(seat, (fl.get("stacks") or {}).get(str(seat)))


def build_ranges(hand, lookup, game, multiway=False):
    """lookup(n, eff, dead, game, key) -> (node, source) | (None, reason), e.g. SolvedCharts.node.
    multiway: the flop had three or more players (mw.multiway_plan's line mapping).
    Returns {"weights": {pos: array(169)}, "line", "source", "notes"} or (None, reason)."""
    if multiway:
        plan, line = mw.multiway_plan(hand)
        plan_notes = [] if plan is None else list(dict.fromkeys(v[2] for v in plan.values() if len(v) > 2))
        plan = None if plan is None else {p: v[:2] for p, v in plan.items()}
    else:
        plan, line, plan_notes = preflop_plan(hand)
    if plan is None:
        return None, line
    positions = hand["positions"]
    n = len(positions)
    seat_of = {v: int(k) for k, v in positions.items()}
    stacks = []
    for pos in plan:
        s = _start_stack(hand, seat_of[pos])
        if s is None:                        # preflop stacks unread: the flop stack plus what went in
            to = max((a["to"] or 0.0 for a in hand["actions"] if a["street"] == "preflop" and a["pos"] == pos),
                     default=0.0)
            fs = _flop_stack(hand, seat_of[pos])
            s = None if fs is None else fs + to
        if s is not None:
            stacks.append(s)
    if len(stacks) < 2:
        return None, "stacks not read"
    eff = min(stacks)
    dead = ((hand.get("start") or {}).get("preflop") or {}).get("dead") or 0.0
    weights, sources = {}, set()
    for pos, (key, action) in plan.items():
        node, src = lookup(n, eff, dead, game, key)
        if node is None:
            return None, f"no preflop chart for {key}: {src}"
        if action not in node["actions"]:
            return None, f"{key} has no {action} in the charts"
        strat = np.asarray(node["strategy"], float)[node["actions"].index(action)]
        weights[pos] = np.clip(np.asarray(node["reach"], float) * strat, 0.0, 1.0)
        sources.add(src)
    return {"weights": weights, "line": line, "source": " + ".join(sorted(sources)), "eff_preflop": eff,
            "notes": plan_notes}, None


def range_string(w, eps=0.0005):
    out = []
    for name, f in zip(HAND_NAMES, w):
        if f >= 1 - eps:
            out.append(name)
        elif f > eps:
            out.append(f"{name}:{f:.3f}")
    return ",".join(out)


# ---------------------------------------------------------------- 2. the street's spot
def _street_stack(hand, seat, street):
    st = (hand.get("start") or {}).get(street) or {}
    return (st.get("stacks") or {}).get(seat, (st.get("stacks") or {}).get(str(seat)))


def check_hand(hand):
    """None when the current street can be solved for the hero, else the reason it can't (yet)."""
    if hand is None:
        return "no hand tracked yet"
    if hand.get("finished"):
        return "hand finished"
    street = hand.get("street")
    if street not in STREETS:
        return "waiting for the flop"
    if len(hand.get("board") or []) != 3 + STREETS.index(street):
        return f"waiting for the {street} card to be read"
    if 0 not in hand["in_hand"]:
        return "you're out of this hand"
    if len(hand["in_hand"]) < 2:
        return "hand over: everyone else folded"
    allin = set(hand.get("allin") or []) & set(hand["in_hand"])
    shoved_now = {a["seat"] for a in hand["actions"] if a["street"] == street and a["action"] == "allin"}
    live = [s for s in hand["in_hand"] if s not in allin]
    if 0 in allin or (len(live) < 2 and not allin & shoved_now):
        # the hero is all-in, or nobody with chips is left to bet against: nothing left to decide.
        # (A villain shoving this street leaves the hero a call-or-fold decision, which is advised;
        # in a 3-way pot one player all-in leaves the other two betting into a side pot.)
        return "a player is all-in: no decisions left"
    hero = hand.get("hero") or ""
    if len(hero) != 4 or "?" in hero:
        return "a hole card's suit wasn't read"
    if any("?" in c for c in hand["board"]):
        return "a board card wasn't read"
    if len({hero[0:2], hero[2:4], *hand["board"]}) != 2 + len(hand["board"]):
        return "hole cards and board overlap (misread)"
    return None


def _combo_string(hands, eps=1e-5):
    """GTOpen range text from {combo: weight}, combo by combo ("AhKh:0.12345")."""
    return ",".join(f"{c}:{w:.5f}" for c, w in hands.items() if w >= eps)


def _tokens(txt):
    return [t for t in txt.replace(",", " ").split() if t]


def off_tree_sizes(hand, street, players, req):
    """Bets and raises logged on this street that the request's menu doesn't hold (more than
    SIZE_NOTE from its nearest size): ({(player, "bet"|"raise"): [size tokens]}, notes).
    Follows GTOpen's sizing: a bet is % of the pot, a raise % of the pot after calling, on top
    of the call; anything within the all-in threshold of the stack is the all-in."""
    si = STREETS.index(street)
    pot0, eff = req["starting_pot"], req["effective_stack"]
    snap = req.get("allin_threshold", 85) / 100.0
    put, sb = [pot0 / 2, pot0 / 2], [0.0, 0.0]
    extra, notes = {}, []
    for a in hand["actions"]:
        p = players.get(a["seat"])
        if a["street"] != street or p is None:
            continue
        o = 1 - p
        if a["action"] == "call":
            put[p] += sb[o] - sb[p]
            sb[p] = sb[o]
            continue
        if a["action"] not in ("bet", "raise", "allin") or not a.get("to"):
            continue
        to = a["to"] * CHIPS
        max_to = sb[p] + eff - (put[p] - pot0 / 2)
        facing = sb[o] - sb[p]
        if facing > 1e-9 and to <= sb[o] + 1e-9:            # all-in for less than the bet: a call
            put[p] += to - sb[p]
            sb[p] = to
            continue
        side = "oop" if p == 0 else "ip"
        if facing <= 1e-9:
            field, base, start = "bet", put[0] + put[1], 0.0
        else:
            field, base, start = "raise", put[0] + put[1] + facing, sb[o]
        pct = (to - start) / base * 100
        allin = a["action"] == "allin" or to >= snap * max_to - 1e-9
        target = max_to if allin else to
        have = _tokens(req[side][si][field]) + extra.get((p, field), [])
        cands = []
        for t in have:
            c = max_to if t == "a" else start + float(t) / 100.0 * base
            cands.append(max_to if c >= snap * max_to - 1e-9 else c)
        if not cands or min(abs(c - target) for c in cands) / max(target, 1e-9) > SIZE_NOTE:
            extra.setdefault((p, field), []).append("a" if allin else f"{pct:.1f}")
            what = "all-in" if allin else f"{a['action']} {a['to']:g}bb ({pct:.0f}% of the pot)"
            notes.append(f"{a['pos']} {what}: not in the bet menu, so the {street} was solved with it added")
        put[p] += min(to, max_to) - sb[p]
        sb[p] = min(to, max_to)
    return extra, notes


def build_spot(hand, ranges, game, street="flop", carry=None):
    """The GTOpen /api/spot request for this street plus what the advice needs to read it back.
    ranges: the preflop ranges (build_ranges); carry: on the turn and river, the ranges the previous
    street's solve ended with ({"hands": {seat: {combo: weight}}, "notes"}, see PostflopEngine)."""
    si = STREETS.index(street)
    positions = hand["positions"]
    n = len(positions)
    seats = players_at(hand, street)
    pos = {s: positions[str(s)] for s in seats}
    if n == 2:
        oop = next(s for s in seats if pos[s] == "BB")
    else:
        oop = min(seats, key=lambda s: POSTFLOP_ORDER.index(pos[s]) if pos[s] in POSTFLOP_ORDER else 99)
    ip = next(s for s in seats if s != oop)
    pot = ((hand.get("start") or {}).get(street) or {}).get("pot")
    stacks = [_street_stack(hand, s, street) for s in (oop, ip)]
    if not pot or any(s is None for s in stacks):
        return None, f"the {street}'s pot or stacks weren't read"
    hero = hand["hero"]
    notes = list(ranges.get("notes") or [])
    hero_side = "oop" if oop == 0 else "ip"
    if carry is None:
        w = {p: ranges["weights"][pos[s]].copy() for p, s in (("oop", oop), ("ip", ip))}
        cls = hole_cards_to_class(hero[0], hero[2], hero[1] == hero[3])
        k = HAND_INDEX.get(cls)
        if k is not None and w[hero_side][k] < HERO_FLOOR:
            notes.append(f"{cls} is rarely in your range on this line ({w[hero_side][k] * 100:.1f}%): "
                         "kept in at a small weight to read its advice")
            w[hero_side][k] = HERO_FLOOR
        rng = {p: range_string(w[p]) for p in ("oop", "ip")}
    else:
        notes += carry.get("notes") or []
        rng = {}
        for p, s in (("oop", oop), ("ip", ip)):
            hands = dict(carry["hands"][s])
            if p == hero_side:
                want = {card_index(hero[0:2]), card_index(hero[2:4])}
                top = max(hands.values(), default=1.0) or 1.0
                c = next((c for c in hands if {card_index(c[0:2]), card_index(c[2:4])} == want), hero)
                if hands.get(c, 0.0) < HERO_FLOOR * top:
                    notes.append(f"{hero} is rarely in your range by now ({hands.get(c, 0.0) / top * 100:.1f}% "
                                 "of your likeliest hands): kept in at a small weight to read its advice")
                    hands[c] = HERO_FLOOR * top
            rng[p] = _combo_string(hands)
    from . import modes
    g = modes.normalize(game)
    rake = g["mode"] == "cash"
    if g["icm_stage"]:
        notes.append("ICM isn't applied postflop yet: chip EV")
    eff = min(stacks)

    def menu(sizes, donk=False):
        return [{"bet": sizes[st][0], "raise": sizes[st][1], "donk": sizes[st][0] if donk else ""}
                for st in STREETS]
    req = {"board": "".join(hand["board"][:3 + si]), "range_oop": rng["oop"], "range_ip": rng["ip"],
           "starting_pot": round(pot * CHIPS, 2), "effective_stack": round(eff * CHIPS, 2),
           "rake_pct": g["rake_pct"] if rake else 0.0, "rake_cap": g["rake_cap"] * CHIPS if rake else 0.0,
           "allin_threshold": 85, "add_allin": False}
    if si == 0:
        req.update(max_raises=MAX_RAISES["full"], oop=menu(SIZES), ip=menu(SIZES))
        small = dict(req, max_raises=MAX_RAISES["small"], oop=menu(SIZES_SMALL), ip=menu(SIZES_SMALL))
    else:
        later = dict(LATER_SIZES, flop=LATER_SIZES["turn"])     # the flop's entry isn't read below the root
        req.update(max_raises=LATER_MAX_RAISES, oop=menu(later, donk=True), ip=menu(later))
        small = None
    players = {oop: 0, ip: 1}
    extra, size_notes = off_tree_sizes(hand, street, players, req)
    for r in (req, small):
        if r is None:
            continue
        for (p, field), toks in extra.items():
            side = "oop" if p == 0 else "ip"
            r[side] = [dict(m) for m in r[side]]
            r[side][si][field] = " ".join(_tokens(r[side][si][field]) + toks)
    meta = {"kind": "hu", "players": players, "pos": pos, "hero_player": 0 if oop == 0 else 1, "street": street,
            "pot": pot, "eff": eff, "notes": notes + size_notes, "line": ranges["line"],
            "source": ranges["source"], "resized": bool(extra)}
    return {"request": req, "request_small": small, "meta": meta}, None


def job_key(hand, spot):
    """One solve: the hand, the street and everything sent to GTOpen (ranges, sizes, pot, stacks)."""
    import hashlib
    import json
    digest = hashlib.sha1(json.dumps([spot["request"], spot.get("request_small")], sort_keys=True)
                          .encode()).hexdigest()
    return (hand["id"], spot["meta"]["street"], digest)


# ---------------------------------------------------------------- 4. walking the solved tree
class NeedView(Exception):
    """The walk needs a node view the solver hasn't been asked for yet."""

    def __init__(self, path):
        super().__init__(path)
        self.path = path


def _step(i):
    return {"type": "action", "index": i}


def _tk(path):
    """A path as a hashable key (action steps by index, card steps by card)."""
    return tuple((s["type"], s.get("index", s.get("card"))) for s in path)


def _match(view, act):
    """Index of the tree action matching one logged action, and a note when the size is off-tree."""
    acts = view["actions"]
    kinds = [a["kind"] for a in acts]
    kind = act["action"]
    if kind in ("check", "call", "fold"):
        return (kinds.index(kind), None) if kind in kinds else (None, f"no {kind} in the tree here")
    sized = [i for i, a in enumerate(acts) if a["kind"] in ("bet", "raise")]
    if kind == "allin":
        if not sized:
            return (kinds.index("call"), None) if "call" in kinds else (None, "no all-in in the tree here")
        i = max(sized, key=lambda j: acts[j]["amount"])
    else:
        cand = [j for j in sized if acts[j]["kind"] == kind] or sized
        if not cand:
            return None, f"no {kind} in the tree here"
        i = min(cand, key=lambda j: abs(acts[j]["amount"] - (act.get("to") or 0.0) * CHIPS))
    to = (act.get("to") or 0.0) * CHIPS
    amt = acts[i]["amount"]
    note = None
    if to > 0 and abs(amt - to) / to > SIZE_NOTE:
        pot = view["pot"]
        note = (f"{act['pos']} {kind} {to / CHIPS:g}bb ({to / pot * 100:.0f}% pot): solved as the tree's "
                f"nearest size, {round(amt / CHIPS, 2):g}bb")
    return i, note


def walk(get, actions, players, hero_player, hero_turn, trail=None):
    """Follow a street's logged actions from the root. get(path) -> node view (may raise NeedView).
    trail, if given, collects every view passed (root first). Returns (view, path, notes) or
    (None, path, reason)."""
    path, notes = [], []
    view = get(path)
    if trail is not None:
        trail.append(view)
    for act in actions:
        p = players.get(act["seat"])
        if p is None:
            continue
        if view["node_type"] != "action":
            return None, path, "the logged actions go past the end of the street in the tree"
        for _ in range(2):
            if view["node_type"] != "action" or view["player"] == p:
                break
            # the player to act in the tree checked without it being seen: step past the check
            kinds = [a["kind"] for a in view["actions"]]
            if "check" not in kinds:
                return None, path, "the logged action order doesn't fit the tree"
            path = path + [_step(kinds.index("check"))]
            view = get(path)
            if trail is not None:
                trail.append(view)
        if view["node_type"] != "action" or view["player"] != p:
            return None, path, "the logged action order doesn't fit the tree"
        i, note = _match(view, act)
        if i is None:
            return None, path, note
        if note:
            notes.append(note)
        path = path + [_step(i)]
        view = get(path)
        if trail is not None:
            trail.append(view)
    for _ in range(2):
        if not (view["node_type"] == "action" and view["player"] != hero_player and hero_turn):
            break
        kinds = [a["kind"] for a in view["actions"]]
        if "check" not in kinds:
            break
        path = path + [_step(kinds.index("check"))]         # villain checked, the label wasn't seen yet
        view = get(path)
    return view, path, notes


def _label(a, view):
    amt = round(a["amount"] / CHIPS, 2)
    if a["kind"] in ("check", "fold"):
        return a["kind"].title()
    if a["kind"] == "call":
        return f"Call {amt:g}"
    if a["label"].lower().startswith("all-in"):
        return f"All-in {amt:g}"
    if a["kind"] == "bet":
        return f"Bet {amt:g} ({a['amount'] / view['pot'] * 100:.0f}%)"
    return f"Raise to {amt:g}"


def hero_advice(view, hero_player, hero_cards):
    """The hero's combo at a node where the hero acts: frequencies and EVs per action (bb)."""
    want = {card_index(hero_cards[0:2]), card_index(hero_cards[2:4])}
    hands = view["players"][hero_player]["hands"]
    h = next((x for x in hands if {x["c1"], x["c2"]} == want), None)
    if h is None or h.get("strategy") is None:
        return None
    evs = [None if e is None else e / CHIPS for e in (h.get("evs") or [None] * len(view["actions"]))]
    best = max((e for e in evs if e is not None), default=None)
    # how the whole range plays this node (reach-weighted)
    tot = np.zeros(len(view["actions"]))
    for x in hands:
        if x.get("strategy"):
            tot += np.asarray(x["strategy"]) * x["reach"]
    rng = tot / max(tot.sum(), 1e-12)
    rows = []
    for i, a in enumerate(view["actions"]):
        rows.append({"label": _label(a, view), "kind": a["kind"], "freq": round(h["strategy"][i], 3),
                     "ev": None if evs[i] is None else round(evs[i], 2),
                     "ev_loss": None if evs[i] is None or best is None else round(best - evs[i], 2),
                     "range_freq": round(float(rng[i]), 3)})
    return {"hand": hero_cards, "actions": rows, "equity": None if h.get("eq") is None else round(h["eq"], 3),
            "ev": None if h.get("ev") is None else round(h["ev"] / CHIPS, 2), "weight": h["weight"]}


# ---------------------------------------------------------------- 3. the background solver
class PostflopEngine:
    """Owns a gto-server and one street's solve at a time; advise() never blocks."""

    def __init__(self, lookup, gtopen_dir=None, port=PORT, api=None):
        self.lookup = lookup
        self.gtopen_dir, self.port = gtopen_dir, port
        self._api = api                       # api(path, body) -> dict; tests pass a fake
        self._server = None
        self.version = 0                      # bumped whenever new results land
        self._cv = threading.Condition()
        self._want = None                     # (key, spot) the HUD wants solved
        self._job = None                      # key being / last solved
        self._phase = None                    # starting | building | solving | rough | refining | done | error
        self._error = None
        self._progress = {}
        self._views = {}                      # (key, phase tag, path) -> view
        self._view_tag = 0
        self._wanted_paths = []
        self._prefetch = collections.deque()  # flop nodes read ahead, so the walk rarely waits
        self._tree_note = None                # set when the small bet menu was used
        self._job_si = 0                      # the street the solved tree starts on (0 = flop)
        self._carry = {}                      # earlier street's solve key -> the ranges it ended with
        self._last_ok = {}                    # (hand id, street) -> (key, spot) of its last readable solve
        self._eq_cache = {}                   # (hand id, board, villains, ranges) -> equity
        self._job_kind = "hu"                 # "hu": /api/*; "mw": the 3-way /api/mw/*
        self._job_start_put = 0.0             # 3-way: each player's share of the starting pot (chips)
        self._warm = False
        self._thread = None

    # ------------------------------------------------------------ HUD side
    def warm(self):
        """Start the gto-server ahead of the first flop (it takes a second or two to come up)."""
        with self._cv:
            if self._warm:
                return
            self._warm = True
            self._ensure_thread()
            self._cv.notify_all()

    def advise(self, hand, game):
        """Chart dict for the current postflop spot (advice, or the reason there isn't any)."""
        why = check_hand(hand)
        if why:
            return self._chart(hand, None, status="unsupported", message=why)
        if len(players_at(hand, "flop")) > 2:
            return self._advise_multiway(hand, game)
        ranges, why = build_ranges(hand, self.lookup, game)
        if ranges is None:
            return self._no_ranges(hand, why)
        street = hand["street"]
        carry = None
        for s in STREETS[:STREETS.index(street) + 1]:
            spot, why = build_spot(hand, ranges, game, s, carry)
            if spot is None:
                return self._chart(hand, None, status="unsupported", message=why)
            key = job_key(hand, spot)
            if s == street:
                break
            # an earlier street: the ranges its solve ended with go into the next street's solve
            carry, why = self._carry_of(hand, spot, key, carry)
            if carry is None:
                return self._carry_wait(hand, s, why)
        return self._advise_stage(hand, spot, key)

    def _no_ranges(self, hand, why):
        # the HUD solver computing a missing preflop grid point: retry until it lands
        pending = "solving" in why or "queued" in why
        return self._chart(hand, None, status="solving" if pending else "unsupported",
                           message=f"Waiting for the preflop charts: {why}" if pending else why)

    def _carry_wait(self, hand, s, why):
        if why:
            return self._chart(hand, None, status="unsupported", message=why)
        nxt = STREETS[STREETS.index(s) + 1]
        with self._cv:
            prog = dict(self._progress)
        return self._chart(hand, None, status="solving", progress=prog,
                           message=f"Reading the ranges the {s} ended with, for the {nxt}…")

    def _advise_stage(self, hand, spot, key, equity=None):
        """The current street's solve: request it, follow the street's actions, read the hero's advice."""
        street = hand["street"]
        phase, err, prog, spot = self._request(key, spot)
        if phase == "error":
            return self._chart(hand, spot, status="error", message=err)
        if phase not in ("rough", "refining", "done"):
            old = self._previous_answer(hand, key, prog, equity)
            if old is not None:
                return old
            return self._chart(hand, spot, status="solving", message=f"Solving the {street}…", progress=prog,
                               equity=equity)
        meta = spot["meta"]
        acts = [a for a in hand["actions"] if a["street"] == street]
        try:
            view, path, notes = walk(lambda p: self._get_view(key, p), acts, meta["players"],
                                     meta["hero_player"], hand.get("hero_turn"))
        except NeedView:
            return self._chart(hand, spot, status="solving", message="Reading the solved tree…", progress=prog,
                               equity=equity)
        if view is None:
            return self._chart(hand, spot, status="unsupported", message=notes, progress=prog, equity=equity)
        with self._cv:
            self._last_ok = {k: v for k, v in self._last_ok.items() if k[0] == hand["id"]}
            self._last_ok[(hand["id"], street)] = (key, spot)
        if view["node_type"] != "action":
            return self._chart(hand, spot, status="waiting", message=f"{street.title()} betting is over",
                               notes=notes, progress=prog, equity=equity)
        if view["player"] != meta["hero_player"]:
            return self._chart(hand, spot, status="waiting", message="Waiting for another player to act",
                               notes=notes, progress=prog, equity=equity)
        adv = hero_advice(view, meta["hero_player"], hand["hero"])
        if adv is None:
            return self._chart(hand, spot, status="unsupported", message="your hand isn't in the solved range",
                               notes=notes, progress=prog, equity=equity)
        if adv.get("equity") is None and equity is not None and not equity.get("stale"):
            adv["equity"] = equity["equity"]
        return self._chart(hand, spot, status="ready", message=None, notes=notes, progress=prog, advice=adv,
                           refining=phase != "done", equity=equity)

    # ------------------------------------------------------------ 3-way and bigger pots
    def _advise_multiway(self, hand, game):
        """3+ players saw the flop. A 3-way flop is too big to solve live: it shows your equity. After a
        flop everyone checked, the turn is solved 3-way from the preflop ranges (a checked flop narrows
        them little: heads-up, that beat every reduced flop tree tried), and the river from the ranges
        the turn ended with (3-way, or heads-up if a player folded the turn). With mw.ESTIMATOR_LIVE the
        flop's ranges come from a reduced 3-way flop solve instead, bet flops included. Your equity on
        every street; equity alone where something can't be solved, with the reason."""
        street = hand["street"]
        ranges, why = build_ranges(hand, self.lookup, game, multiway=True)
        if ranges is None:
            return self._no_ranges(hand, why)
        n_flop = len(players_at(hand, "flop"))
        notes = list(ranges.get("notes") or [])

        def equity_only(reason):
            return self._chart(hand, None, status="equity", equity=self._equity(hand, ranges, None),
                               message=f"{reason}: your equity instead", notes=notes, ranges=ranges)
        if n_flop > 3:
            return equity_only(f"{n_flop}-way pots can't be solved")
        est = mw.build_estimator_spot(hand, ranges, game)[0] if mw.ESTIMATOR_LIVE else None
        if street == "flop":
            if est is not None:
                self._request(job_key(hand, est), est)     # its ranges are ready when the turn comes
            return equity_only("3-way flops are too big to solve live")
        if est is not None:
            start, spot, first_note = "flop", est, mw.ESTIMATOR_NOTE
        elif mw.checked_through(hand, "flop") and len(players_at(hand, "turn")) == 3:
            spot, why = mw.build_mw_spot(hand, ranges, game, "turn")
            if spot is None:
                return equity_only(why)
            start, first_note = "turn", mw.CHECKED_NOTE
        else:
            return equity_only("the 3-way flop was bet, and a 3-way flop can't be solved live, so the ranges "
                               "for the turn are unknown")
        key, carry = job_key(hand, spot), None
        for s in STREETS[STREETS.index(start) + 1:STREETS.index(street) + 1]:
            prev = STREETS[STREETS.index(s) - 1]
            carry, why = self._carry_of(hand, spot, key, carry)
            if carry is None:
                if why is not None:
                    return equity_only(why)
                with self._cv:
                    prog = dict(self._progress)
                msg = ("Estimating the ranges the 3-way flop ended with (reduced flop solve)…" if prev == "flop"
                       else f"Reading the ranges the {prev} ended with, for the {s}…")
                return self._chart(hand, None, status="solving", message=msg, progress=prog,
                                   equity=self._equity(hand, ranges, None), notes=notes, ranges=ranges)
            if prev == start:
                carry = dict(carry, notes=list(carry["notes"]) + [first_note])
            n = len(players_at(hand, s))
            spot, why = (mw.build_mw_spot(hand, ranges, game, s, carry) if n == 3
                         else build_spot(hand, ranges, game, s, carry))
            if spot is None:
                return equity_only(why)
            key = job_key(hand, spot)
        if start == street:
            spot["meta"]["notes"].append(first_note)
        return self._advise_stage(hand, spot, key, equity=self._equity(hand, ranges, carry))

    def _equity(self, hand, ranges, carry):
        """Your equity against everyone else still in (their carried ranges if known, else preflop),
        and the pot odds when you face a bet. Cached per spot, so the number holds still."""
        seats = [s for s in hand["in_hand"] if s != 0]
        pos = hand["positions"]
        narrowed = carry is not None and all(s in carry["hands"] for s in seats)
        key = (hand["id"], tuple(hand["board"]), tuple(seats), narrowed and id(carry))
        with self._cv:
            hit = self._eq_cache.get(key)
        if hit is None:
            vill = [carry["hands"][s] if narrowed else ranges["weights"][pos[str(s)]] for s in seats]
            e = mw.equity(hand["hero"], hand["board"], vill)
            hit = {"equity": e, "players": len(seats), "narrowed": narrowed, "hand": hand["hero"]}
            with self._cv:
                self._eq_cache = {k: v for k, v in self._eq_cache.items() if k[0] == hand["id"]}
                self._eq_cache[key] = hit
        out = dict(hit)
        # a bet or raise from someone else on this street: their range is stronger than the one used
        out["stale"] = any(a["street"] == hand["street"] and a["seat"] != 0 and a["action"] in ("bet", "raise", "allin")
                           for a in hand["actions"])
        po = mw.pot_odds(hand)
        if po:
            out.update(to_call=po[0], pot=po[1], need=round(po[0] / (po[1] + po[0]), 3))
        return out

    def _request(self, key, spot):
        """Ask the worker for this solve (if it isn't the one wanted already); its phase now."""
        with self._cv:
            if self._want is None or self._want[0] != key:
                self._want = (key, spot)
                self._wanted_paths = []
                self._ensure_thread()
                self._cv.notify_all()
            phase, err, prog = self._phase if self._job == key else "queued", self._error, dict(self._progress)
            if self._job == key and self._tree_note:
                spot = dict(spot, meta=dict(spot["meta"], notes=spot["meta"]["notes"] + [self._tree_note]))
        return phase, err, prog, spot

    def _carry_of(self, hand, spot, key, prev_carry):
        """The ranges an earlier street's solve ended with, following that street's logged actions to
        the end of its betting: ({"hands": {seat: {combo: reach}}, "notes"}, None), (None, None) while
        it's being solved or read, or (None, reason). A player who took a line the solver almost never
        takes there (under LOW_LINE of their range) keeps the range they had before it: their reach after
        it is a handful of unconverged combos."""
        line = tuple((a["seat"], a["action"], a["to"]) for a in hand["actions"] if a["street"] == spot["meta"]["street"])
        if (key, line) in self._carry:
            return self._carry[(key, line)], None
        phase, err, _, spot = self._request(key, spot)
        if phase == "error":
            return None, err
        if phase not in ("rough", "refining", "done"):
            return None, None
        meta = spot["meta"]
        street = meta["street"]
        acts = [a for a in hand["actions"] if a["street"] == street]
        trail = []
        try:
            view, path, why = walk(lambda p: self._get_view(key, p), acts, meta["players"], meta["hero_player"],
                                   False, trail)
        except NeedView:
            return None, None
        if view is None:
            return None, f"the {street}'s actions don't fit its solved tree ({why}): no ranges for later streets"
        if view["node_type"] != "chance":
            return None, f"the {street}'s betting didn't end in its solved tree: no ranges for later streets"
        seat_of = {p: seat for seat, p in meta["players"].items()}
        notes = list((prev_carry or {}).get("notes") or [])
        if meta.get("kind") == "mw":
            return self._mw_carry(hand, key, meta, path, trail, notes, line)
        hands = {}
        for p in (0, 1):
            src = view
            for a, b in zip(trail, trail[1:]):
                if a.get("node_type") != "action" or a.get("player") != p:
                    continue
                before = sum(x["reach"] for x in a["players"][p]["hands"])
                after = sum(x["reach"] for x in b["players"][p]["hands"])
                if before > 0 and after / before < LOW_LINE:
                    who = "you" if seat_of[p] == 0 else meta["pos"][seat_of[p]]
                    notes.append(f"on the {street} {who} took a line the solver almost never takes "
                                 f"({after / before * 100:.1f}% of the range): the range from before it is kept")
                    src = a
                    break
            hands[seat_of[p]] = {x["combo"]: float(x["reach"]) for x in src["players"][p]["hands"]}
        carry = {"hands": hands, "notes": notes}
        with self._cv:
            self._carry = {k: v for k, v in self._carry.items() if k[0][0] == hand["id"]}
            self._carry[(key, line)] = carry
        return carry, None

    def _mw_carry(self, hand, key, meta, path, trail, notes, line):
        """A 3-way street's end ranges: the 3-way view holds only the actor's hands, so each player's
        reach is read where they first act on the next street (its first node, then after one check,
        then two: checking doesn't change the others' reach). A rare line (under LOW_LINE of a player's
        weighted combos) keeps the range from before it, read at that player's own node."""
        street = meta["street"]
        nxt = STREETS[STREETS.index(street) + 1]
        card = hand["board"][3 + STREETS.index(nxt) - 1] if len(hand["board"]) > 2 + STREETS.index(nxt) else None
        if card is None:
            return None, None
        live = set(players_at(hand, nxt))
        seat_of = {p: seat for seat, p in meta["players"].items()}
        hands = {}
        step = path + [{"type": "card", "card": card}]
        try:
            for _ in range(len(live)):
                v = self._get_view(key, step)
                if v["node_type"] != "action" or v["player"] is None:
                    return None, "a player can't act on the next street (all-in): no ranges for it"
                hands[seat_of[v["player"]]] = {x["combo"]: float(x["reach"]) for x in v["players"][v["player"]]["hands"]}
                kinds = [a["kind"] for a in v["actions"]]
                if "check" not in kinds:
                    break
                step = step + [_step(kinds.index("check"))]
        except NeedView:
            return None, None
        if not live <= set(hands):
            return None, "not every player's range could be read for the next street"
        for a, b in zip(trail, trail[1:]):
            p = a.get("player")
            rc_a, rc_b = a.get("range_combos"), b.get("range_combos")
            if a.get("node_type") != "action" or p is None or not rc_a or not rc_b or rc_a[p] <= 0:
                continue
            if rc_b[p] / rc_a[p] < LOW_LINE and seat_of[p] in live:
                who = "you" if seat_of[p] == 0 else meta["pos"][seat_of[p]]
                notes.append(f"on the {street} {who} took a line the solver almost never takes "
                             f"({rc_b[p] / rc_a[p] * 100:.1f}% of the range): the range from before it is kept")
                hands[seat_of[p]] = {x["combo"]: float(x["reach"]) for x in a["players"][p]["hands"]}
        carry = {"hands": {s: h for s, h in hands.items() if s in live}, "notes": notes}
        with self._cv:
            self._carry = {k: v for k, v in self._carry.items() if k[0][0] == hand["id"]}
            self._carry[(key, line)] = carry
        return carry, None

    def _previous_answer(self, hand, key, prog, equity=None):
        """While a street is re-solved with a bet size its first tree didn't have, the first tree's
        answer (that bet mapped to its nearest size), marked as re-solving. None if there isn't one."""
        with self._cv:
            prev = self._last_ok.get((hand["id"], hand["street"]))
        if prev is None or prev[0] == key:
            return None
        pkey, pspot = prev
        meta = pspot["meta"]
        acts = [a for a in hand["actions"] if a["street"] == hand["street"]]
        try:
            view, _, notes = walk(lambda p: self._cached_view(pkey, p), acts, meta["players"],
                                  meta["hero_player"], hand.get("hero_turn"))
        except NeedView:
            return None
        if view is None or view["node_type"] != "action" or view["player"] != meta["hero_player"]:
            return None
        adv = hero_advice(view, meta["hero_player"], hand["hero"])
        if adv is None:
            return None
        if adv.get("equity") is None and equity is not None and not equity.get("stale"):
            adv["equity"] = equity["equity"]
        return self._chart(hand, pspot, status="ready", message=f"Re-solving the {hand['street']} with the bet "
                           "size really used; meanwhile it's mapped to the nearest size", notes=notes,
                           progress=prog, advice=adv, refining=True, equity=equity)

    def _chart(self, hand, spot, status, message, notes=(), progress=None, advice=None, refining=False,
               equity=None, ranges=None):
        meta = (spot or {}).get("meta") or {}
        street = (hand or {}).get("street") or ""
        title = f"Postflop · {street}"
        pos = meta.get("pos")
        if not pos and hand and ranges is not None:        # equity only: the hand's own players
            pos = {s: hand["positions"][str(s)] for s in players_at(hand, street)}
        if pos and 0 in pos:
            vs = [p for s, p in pos.items() if s != 0]
            st = (hand.get("start") or {}).get(street) or {}
            pot = meta.get("pot", st.get("pot"))
            behind = meta.get("eff")
            if behind is None:
                stk = st.get("stacks") or {}
                vals = [stk.get(s, stk.get(str(s))) for s in pos]
                left = [v for v in vals if v is not None and v > 0.05]      # an all-in player has nothing behind
                behind = min(left) if left and all(v is not None for v in vals) else None
            way = f"{len(pos)}-way · " if len(pos) > 2 else ""
            title = (f"{street.title()} · {way}{pos[0]} vs {' and '.join(vs)}"
                     + (f" · {pot:g}bb pot" if pot else "") + (f", {behind:g}bb behind" if behind is not None else ""))
        src = meta or (ranges or {})
        pf = {"status": status, "message": message,
              "notes": list(dict.fromkeys(list(meta.get("notes") or []) + list(notes))),
              "line": src.get("line"), "ranges": src.get("source"), "progress": progress or {},
              "advice": advice, "refining": refining, "equity": equity}
        if meta.get("street", "flop") != "flop":
            pf["ranges"] = f"{meta.get('source')}, narrowed by each street's solve"
        if equity and (status == "equity" or meta.get("kind") == "mw"):
            if not equity.get("narrowed"):
                pf["notes"].append("equity is against their preflop ranges: the betting since isn't accounted "
                                   "for, so it's likely higher than your real equity")
            elif equity.get("stale"):
                pf["notes"].append(f"equity is against their ranges at the start of the {street}: this street's "
                                   "bets aren't accounted for, so it's likely higher than your real equity")
        if meta.get("kind") == "mw":
            method = "GTOpen 3-way solver (beta) · one bet size and all-in raises per street"
        elif meta:
            method = "GTOpen postflop solver · heads-up · each street solved from its start"
        else:
            method = "Postflop advice: heads-up flop to river; 3-way turn and river; equity otherwise"
        return {"actions": [], "freqs": None, "approx": False, "title": title, "method": method,
                "postflop": pf, "pending_solver": status == "solving"}

    def _cached_view(self, key, path):
        """A view already read for this solve (any read-through), never asking for one."""
        tkey = _tk(path)
        with self._cv:
            hits = [(k[1], v) for k, v in self._views.items() if k[0] == key and k[2] == tkey]
        if not hits:
            raise NeedView(path)
        return max(hits, key=lambda t: t[0])[1]

    def _get_view(self, key, path):
        tkey = _tk(path)
        with self._cv:
            v = self._views.get((key, self._view_tag, tkey))
            if v is not None:
                return v
            if tkey not in [_tk(p) for p in self._wanted_paths]:
                self._wanted_paths.append(path)
                self._cv.notify_all()
        raise NeedView(path)

    # ------------------------------------------------------------ worker side
    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True, name="postflop-solver")
            self._thread.start()

    def api(self, path, body=None):
        if self._api is not None:
            return self._api(path, body)
        if self._server is None:
            from .gtopen import Server, find_gtopen_dir
            try:
                gdir = find_gtopen_dir(self.gtopen_dir)
            except SystemExit as e:
                raise RuntimeError(str(e))
            log = os.path.join(gdir, "saves", "postflop_server.log")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            srv = Server(gdir, log, self.port)
            srv.__enter__()
            self._server = srv
            atexit.register(self.close)
        return self._server.api(path, body, timeout=120)

    def close(self):
        if self._server is not None:
            try:
                self._server.__exit__(None, None, None)
            finally:
                self._server = None

    def _set(self, **kw):
        with self._cv:
            for k, v in kw.items():
                setattr(self, k, v)
            self.version += 1

    def _superseded(self, key):
        with self._cv:
            return self._want is None or self._want[0] != key

    def _run_solve(self, key, target, max_iters, pausable=False, cap=None):
        """One /api/solve to a target; False if a newer spot arrived meanwhile (solve stopped).
        pausable: the walk asking for a node pauses the solve to read it, then the solve continues
        (warm) -- the hero's advice never waits for the refinement."""
        body = {"max_iterations": max_iters, "target_exploit_pct": target, "check_every": 10}
        api = "/api/mw" if self._job_kind == "mw" else "/api"
        self.api(api + "/solve", body)
        t_start = time.time()
        while True:
            st = self.api(api + "/status")
            with self._cv:
                # GTOpen's status drops the GPU flag between runs: once seen this solve, keep it
                gpu = bool(st.get("gpu")) or st.get("engine") == "gpu" or bool(self._progress.get("gpu"))
                self._progress = {"iteration": st.get("iteration"), "exploit_pct": st.get("exploit_pct"),
                                  "gpu": gpu, "secs": round(time.time() - self._t0, 1)}
            if st.get("state") != "running":
                if st.get("error"):
                    raise RuntimeError(st["error"])
                return True
            if self._superseded(key):
                self.api(api + "/stop", {})
                return False
            if cap is not None and time.time() - t_start > cap:
                # out of time: take the strategy wherever it is (the estimator's ranges)
                self.api(api + "/stop", {})
                while self.api(api + "/status").get("state") == "running":
                    time.sleep(0.05)
                return True
            if pausable and self._missing_views(running=True):
                self.api(api + "/stop", {})
                while self.api(api + "/status").get("state") == "running":
                    time.sleep(0.05)
                self._fetch_views(key)
                self.api(api + "/solve", body)
            time.sleep(0.15)

    def _fetch(self, key, tag, path):
        tkey = _tk(path)
        if (key, tag, tkey) in self._views:
            return False
        if self._job_kind == "mw":
            v = mw.mw_view(self.api("/api/mw/node", {"path": mw.mw_path(path), "ev": False}), self._job_start_put)
        else:
            v = self.api("/api/node", {"path": path})
        with self._cv:
            self._views[(key, tag, tkey)] = v
        return True

    def _fetch_views(self, key, tag=None):
        """Read the wanted nodes into the cache under `tag` (default: the current one)."""
        with self._cv:
            paths = list(self._wanted_paths)
            tag = self._view_tag if tag is None else tag
        got = False
        for p in paths:
            if self._superseded(key):
                return
            got |= self._fetch(key, tag, p)
        if got:
            self._set()

    def _prefetch_one(self, key):
        with self._cv:
            if not self._prefetch:
                return
            path, tag = self._prefetch.popleft(), self._view_tag
        try:
            got = self._fetch(key, tag, path)
        except Exception:        # reading ahead is optional: a node the walk needs is read again
            return
        with self._cv:
            v = self._views.get((key, tag, _tk(path))) or {}
            if v.get("node_type") == "action" and v.get("street", 0) == self._job_si and len(path) < PREFETCH_DEPTH:
                self._prefetch.extend(path + [_step(i)] for i in range(len(v["actions"])))
        if got:
            self._set()

    def _worker(self):
        if self._warm and self._api is None:
            try:
                self.api("/api/version")
            except Exception:
                pass                     # reported when a flop needs it
        while True:
            with self._cv:
                while self._want is None or (self._want[0] == self._job and not self._missing_views()
                                             and not (self._prefetch and self._phase == "done")):
                    self._cv.wait()
                key, spot = self._want
            if key != self._job:
                self._solve_job(key, spot)
            else:
                try:
                    if self._phase in ("rough", "done"):
                        self._fetch_views(key)           # nodes the walk asked for come first
                    if self._phase == "done" and not self._missing_views():
                        self._prefetch_one(key)
                except Exception as e:  # keep the worker alive
                    self._set(_phase="error", _error=f"{type(e).__name__}: {e}")

    def _missing_views(self, running=False):
        if self._phase not in ("rough", "done") + (("refining",) if running else ()):
            return False
        return any((self._job, self._view_tag, _tk(p)) not in self._views
                   for p in self._wanted_paths)

    def _solve_job(self, key, spot):
        self._t0 = time.time()
        with self._cv:
            old = self._job
            self._job, self._error, self._progress = key, None, {}
            self._job_si = STREETS.index(spot["meta"]["street"])
            self._job_kind = spot["meta"].get("kind", "hu")
            self._job_start_put = spot["meta"].get("start_put", 0.0)
            self._wanted_paths = [[]]           # the root: read as soon as the first solve lands
            # the previous solve's views stay: a re-solve with a new bet size shows its answer meanwhile
            self._views = {k: v for k, v in self._views.items() if k[0] in (key, old)}
            self._view_tag += 1
            self._prefetch.clear()
        try:
            self._set(_phase="building", _tree_note=None)
            req, small = spot["request"], spot.get("request_small")
            mwj = self._job_kind == "mw"
            if mwj:
                self.api("/api/mw/spot", req)
            elif small and req["effective_stack"] >= DEEP_SPR * req["starting_pot"]:
                self.api("/api/spot", small)        # deep: the full menu's tree would be too big anyway
                self._set(_tree_note=SMALL_NOTE)
            else:
                info = self.api("/api/spot", req)
                if (info or {}).get("vram_mb", 0) > BIG_TREE_MB and small:
                    self.api("/api/spot", small)
                    self._set(_tree_note=SMALL_NOTE)
            self._set(_phase="solving")
            if spot["meta"].get("estimator"):
                # the 3-way flop range estimator: one run to ~1% of the pot, capped in time; no refining
                if not self._run_solve(key, mw.ESTIMATOR_PCT, 400, cap=mw.ESTIMATOR_SECS):
                    return
                self._set(_phase="done", _view_tag=self._view_tag + 1)
                self._fetch_views(key)
                return
            if not self._run_solve(key, ROUGH_PCT, MW_ITERS[0] if mwj else MAX_ITERS[0]):
                return
            self._set(_phase="rough", _view_tag=self._view_tag + 1)
            self._fetch_views(key)
            with self._cv:
                self._prefetch.clear()
            if self._superseded(key):
                return
            self._set(_phase="refining")
            if not self._run_solve(key, MW_FINAL_PCT if mwj else FINAL_PCT, MW_ITERS[1] if mwj else MAX_ITERS[1],
                                   pausable=True):
                return
            # refined strategies: re-read the nodes, then switch over (the rough ones show meanwhile)
            with self._cv:
                self._prefetch.clear()
            self._fetch_views(key, self._view_tag + 1)
            with self._cv:
                self._prefetch.clear()
                self._prefetch.append([])           # then read the street's nodes ahead, from the root
            self._set(_phase="done", _view_tag=self._view_tag + 1)
            self._fetch_views(key)
        except Exception as e:
            self._set(_phase="error", _error=f"postflop solver: {type(e).__name__}: {e}")

