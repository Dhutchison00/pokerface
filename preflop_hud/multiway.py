"""3-way (and bigger) pots: preflop ranges for multiway lines, your equity against every range,
pot odds, and the lean 3-way spots GTOpen's three-player solver answers live.

A 3-way flop is far too big to solve live on one GPU (a 20bb pot with one bet size: 2.7M nodes,
still 4.6% of the pot from equilibrium after ten minutes), so flops show equity only. A 3-way turn
or river with one bet size and all-in raises solves in 1-3 s (turn) or under 1 s (river): those are
solved when the earlier streets' ranges are known well enough, that is when the flop was checked
through (its checking ranges are taken as the preflop ranges, noted), and the river after such a
turn (ranges carried from the turn's solve). Everything else shows equity.
"""
from __future__ import annotations

import numpy as np

from .cards import class_combos
from .evaluator import evaluate7

CHIPS = 10
STREETS = ("flop", "turn", "river")
MW_SIZES = {"turn": ("75", "a"), "river": ("33 75 a", "a")}
MW_LATER_RIVER = ("75", "a")     # the river inside a turn-rooted 3-way tree
MW_MAX_RAISES = 2
SIZE_NOTE = 0.15
EQUITY_SAMPLES = 8000

RANK_ORDER = "23456789TJQKA"
SUIT_ORDER = "cdhs"


def card_index(c):
    return RANK_ORDER.index(c[0].upper()) * 4 + SUIT_ORDER.index(c[1].lower())


def card_name(i):
    return RANK_ORDER[i // 4] + SUIT_ORDER[i % 4]


# ---------------------------------------------------------------- who was in, street by street
def players_at(hand, street):
    """Seats still in when the street began: those in now plus those who folded on it or later."""
    later = STREETS[STREETS.index(street):]
    gone = {a["seat"] for a in hand["actions"] if a["action"] == "fold" and a["street"] in later}
    return sorted(set(hand["in_hand"]) | gone)


def action_order(hand, seats):
    from .tracker import POSTFLOP_ORDER
    pos = hand["positions"]
    return sorted(seats, key=lambda s: POSTFLOP_ORDER.index(pos[str(s)]) if pos[str(s)] in POSTFLOP_ORDER else 99)


def checked_through(hand, street):
    """Every player at the street's start checked and nobody bet."""
    acts = [a for a in hand["actions"] if a["street"] == street]
    return bool(acts) and all(a["action"] == "check" for a in acts) and \
        {a["seat"] for a in acts} >= set(players_at(hand, street))


# ---------------------------------------------------------------- 1. preflop ranges for a multiway line
LIMP_NOTE = ("{l} limped, which the charts almost never do from there (only the SB limps at these depths): "
             "{l}'s range is assumed to be the hands {l} would open-raise")


def multiway_plan(hand):
    """Each flop player's range from their own heads-up node against the first raiser (or as a limper):
    ({pos: (node key, action[, note])}, line text) or (None, reason). The charts have no multiway
    nodes, so an overcall is its call against a lone open, and limps follow the heads-up assumptions."""
    positions = hand.get("positions") or {}
    n = len(positions)
    if not n:
        return None, "positions unknown"
    for note in hand.get("notes") or []:
        if note.startswith("joined"):
            return None, "the preflop action wasn't all seen (tracking started late): ranges unknown"
    sb, bb = ("BTN" if n == 2 else "SB"), "BB"
    seq = [(a["pos"], a["action"]) for a in hand["actions"] if a["street"] == "preflop" and a["action"] != "fold"]
    flop_pos = {positions.get(str(s)) for s in players_at(hand, "flop")}
    txt = ", ".join(f"{p} {a}" for p, a in seq) or "no preflop action seen"
    if any(p not in flop_pos for p, _ in seq):
        return None, f"preflop line {txt}: a player put chips in and folded later (not in the charts yet)"
    raises = [i for i, (_, a) in enumerate(seq) if a in ("raise", "allin")]
    if len(raises) > 1:
        return None, f"preflop line {txt}: 3-bet multiway pots aren't in the charts"
    plan = {}
    if not raises:                                  # limped pot
        for p, a in seq:
            if a == "call":
                plan[p] = (f"RFI:{sb}", "call") if p == sb else (f"RFI:{p}", "raise", LIMP_NOTE.format(l=p))
            elif a == "check" and p == bb:
                plan[p] = ("BB_VS_LIMP", "check", "the BB's checking range is the one it has against an SB limp")
        line = " ".join(["limped pot:", txt])
    else:
        ri = raises[0]
        r = seq[ri][0]
        limpers = [p for p, a in seq[:ri] if a == "call"]
        if not limpers:
            plan[r] = (f"RFI:{r}", "raise")
        elif r == bb:
            plan[r] = ("BB_VS_LIMP", "raise", "the BB's iso-raise range is the one it has against an SB limp")
        else:
            plan[r] = (f"RFI:{r}", "raise", f"{r}'s iso-raise range is assumed to be its open range")
        callers = []
        for p, a in seq[ri + 1:]:
            if a != "call":
                return None, f"preflop line {txt} isn't in the charts yet"
            if p in limpers:
                plan[p] = ((f"RFI:{sb}", "call", "the SB's limp-call range is assumed to be its limping range")
                           if p == sb else (f"RFI:{p}", "raise", LIMP_NOTE.format(l=p)))
            else:
                note = (f"{p} called after {callers[-1]} had called: its range is its call against a lone "
                        f"{r} raise") if callers else None
                plan[p] = (f"VS_OPEN:{p}:{r}", "call") + ((note,) if note else ())
            callers.append(p)
        line = txt
    if set(plan) != flop_pos:
        return None, f"preflop line {txt}: not every flop player's range could be placed"
    return plan, line


# ---------------------------------------------------------------- 2. equity and pot odds
def _combos(weights):
    """(cards[n, 2], weight[n]) from 169 class weights or {combo text: weight}."""
    if isinstance(weights, dict):
        items = [((card_index(c[0:2]), card_index(c[2:4])), w) for c, w in weights.items() if w > 0]
    else:
        items = [(cc, float(weights[k])) for k in range(169) if weights[k] > 0 for cc in class_combos(k)]
    if not items:
        return np.zeros((0, 2), int), np.zeros(0)
    return np.array([c for c, _ in items]), np.array([w for _, w in items], float)


def equity(hero, board, villains, samples=EQUITY_SAMPLES, seed=7):
    """Your share of the pot at showdown against every villain range (Monte Carlo, card removal
    between all hands and the board; ties split). Seeded, so the same spot gives the same number."""
    rng = np.random.default_rng(seed)
    dead = [card_index(hero[0:2]), card_index(hero[2:4])] + [card_index(c) for c in board]
    picks = []
    for w in villains:
        cards, wt = _combos(w)
        ok = ~np.isin(cards, dead).any(axis=1)
        cards, wt = cards[ok], wt[ok]
        if not len(cards) or wt.sum() <= 0:
            return None
        picks.append(cards[rng.choice(len(cards), samples, p=wt / wt.sum())])
    vc = np.concatenate(picks, axis=1)                                   # (N, 2 * villains)
    valid = np.array([len(set(r)) == len(r) for r in vc])                # villains don't share a card
    keys = rng.random((samples, 52))
    keys[:, dead] = 2.0
    np.put_along_axis(keys, vc, 2.0, axis=1)
    k = 5 - len(board)
    extra = np.argsort(keys, axis=1)[:, :k]
    common = np.concatenate([np.tile(np.array([card_index(c) for c in board], int), (samples, 1)), extra], axis=1)
    hs = evaluate7(np.concatenate([np.tile(dead[:2], (samples, 1)), common], axis=1))
    vs = np.stack([evaluate7(np.concatenate([vc[:, 2 * i:2 * i + 2], common], axis=1))
                   for i in range(len(villains))], axis=1)
    best = np.maximum(hs, vs.max(axis=1))
    share = np.where(hs == best, 1.0 / (1 + (vs == best[:, None]).sum(axis=1)), 0.0)
    return round(float(share[valid].mean()), 3) if valid.any() else None


def pot_odds(hand):
    """(to call, pot now) in bb for the hero on this street, or None when there's no bet to call."""
    street = hand["street"]
    start = ((hand.get("start") or {}).get(street) or {})
    pot = start.get("pot")
    if pot is None:
        return None
    put = {}
    for a in hand["actions"]:
        if a["street"] == street and a.get("to"):
            put[a["seat"]] = a["to"]
    hi = max(put.values(), default=0.0)
    to_call = hi - put.get(0, 0.0)
    stack = (start.get("stacks") or {}).get(0, (start.get("stacks") or {}).get("0"))
    if stack is not None:
        to_call = min(to_call, stack - put.get(0, 0.0))
    if to_call <= 1e-9:
        return None
    return round(to_call, 2), round(pot + sum(put.values()), 2)


# ---------------------------------------------------------------- 3. the lean 3-way spot (turn/river)
def _tokens(txt):
    return [t for t in txt.replace(",", " ").split() if t]


def off_tree_sizes(hand, street, order, req):
    """The street's logged bets and raises against the shared menu: {"bet"|"raise": [tokens]} to
    put in its place (one size each keeps the tree small: a second size makes a 3-way turn 3-5x
    slower), and notes. A bet is % of the pot, a raise % of the pot after calling, on top of the bet."""
    si = STREETS.index(street)
    pot0 = req["starting_pot"]
    n = len(order)
    snap = req.get("allin_threshold", 85) / 100.0
    put = {s: pot0 / n for s in order}
    sb = {s: 0.0 for s in order}
    stack = dict(zip(order, req["stacks"]))
    swap, notes = {}, []
    for a in hand["actions"]:
        s = a["seat"]
        if a["street"] != street or s not in sb:
            continue
        hi = max(sb.values())
        if a["action"] == "call":
            add = min(hi - sb[s], stack[s] - (put[s] - pot0 / n))
            put[s] += add
            sb[s] += add
            continue
        if a["action"] not in ("bet", "raise", "allin") or not a.get("to"):
            continue
        to = a["to"] * CHIPS
        own_max = sb[s] + stack[s] - (put[s] - pot0 / n)
        facing = hi - sb[s]
        if facing > 1e-9 and to <= hi + 1e-9:
            put[s] += to - sb[s]
            sb[s] = to
            continue
        field, base, start = ("bet", sum(put.values()), 0.0) if facing <= 1e-9 else \
            ("raise", sum(put.values()) + facing, hi)
        pct = (to - start) / base * 100
        allin = a["action"] == "allin" or to >= snap * own_max - 1e-9
        target = min(to, own_max) if not allin else own_max
        have = swap.get(field) or _tokens(req["sizes"][si][field])
        cands = [own_max if t == "a" else min(start + float(t) / 100.0 * base, own_max) for t in have]
        cands = [own_max if c >= snap * own_max - 1e-9 else c for c in cands]
        if not cands or min(abs(c - target) for c in cands) / max(target, 1e-9) > SIZE_NOTE:
            swap[field] = ["a"] if allin else [f"{pct:.1f}", "a"] if field == "raise" else [f"{pct:.1f}"]
            what = "all-in" if allin else f"{a['action']} {a['to']:g}bb ({pct:.0f}% of the pot)"
            notes.append(f"{a['pos']} {what}: solved with that size in place of the menu's")
        put[s] += min(to, own_max) - sb[s]
        sb[s] = min(to, own_max)
    return swap, notes


def build_mw_spot(hand, ranges, game, street, carry=None, sizes=None):
    """The /api/mw/spot request for a 3-way street, from its start (sizes: the [flop, turn, river]
    menu rows, default the lean turn/river menu)."""
    from .postflop import HERO_FLOOR, hole_cards_to_class, HAND_INDEX, range_string, _combo_string
    from . import modes
    seats = action_order(hand, players_at(hand, street))
    pos = {s: hand["positions"][str(s)] for s in seats}
    st = (hand.get("start") or {}).get(street) or {}
    pot = st.get("pot")
    stk = st.get("stacks") or {}
    stacks = [stk.get(s, stk.get(str(s))) for s in seats]
    if not pot or any(x is None for x in stacks):
        return None, f"the {street}'s pot or stacks weren't read"
    if any(x <= 0.05 for x in stacks):
        return None, "a player is all-in: side pots are shown as equity only"
    hero = hand["hero"]
    notes = list(ranges.get("notes") or [])
    rng = []
    for s in seats:
        if carry is None:
            w = ranges["weights"][pos[s]].copy()
            if s == 0:
                cls = hole_cards_to_class(hero[0], hero[2], hero[1] == hero[3])
                k = HAND_INDEX.get(cls)
                if k is not None and w[k] < HERO_FLOOR:
                    notes.append(f"{cls} is rarely in your range on this line ({w[k] * 100:.1f}%): kept in at a "
                                 "small weight to read its advice")
                    w[k] = HERO_FLOOR
            rng.append(range_string(w))
        else:
            hands = dict(carry["hands"][s])
            if s == 0:
                want = {card_index(hero[0:2]), card_index(hero[2:4])}
                top = max(hands.values(), default=1.0) or 1.0
                c = next((c for c in hands if {card_index(c[0:2]), card_index(c[2:4])} == want), hero)
                if hands.get(c, 0.0) < HERO_FLOOR * top:
                    notes.append(f"{hero} is rarely in your range by now: kept in at a small weight to read its advice")
                    hands[c] = HERO_FLOOR * top
            rng.append(_combo_string(hands))
    if carry is not None:
        notes += carry.get("notes") or []
    g = modes.normalize(game)
    rake = g["mode"] == "cash"
    if g["icm_stage"]:
        notes.append("ICM isn't applied postflop yet: chip EV")
    si = STREETS.index(street)
    # a turn-rooted tree's river is only its continuation: one size keeps the turn's first answer quick
    river = MW_SIZES["river"] if street == "river" else MW_LATER_RIVER
    sizes = [dict(r) for r in sizes] if sizes else [
        {"bet": MW_SIZES["turn"][0], "raise": MW_SIZES["turn"][1]},
        {"bet": MW_SIZES["turn"][0], "raise": MW_SIZES["turn"][1]}, {"bet": river[0], "raise": river[1]}]
    req = {"board": "".join(hand["board"][:3 + si]), "ranges": rng, "starting_pot": round(pot * CHIPS, 2),
           "stacks": [round(x * CHIPS, 2) for x in stacks], "sizes": sizes, "max_raises": MW_MAX_RAISES,
           "allin_threshold": 85, "rake_pct": g["rake_pct"] if rake else 0.0,
           "rake_cap": g["rake_cap"] * CHIPS if rake else 0.0}
    swap, size_notes = off_tree_sizes(hand, street, seats, req)
    for field, toks in swap.items():
        req["sizes"] = [dict(r) for r in req["sizes"]]
        req["sizes"][si][field] = " ".join(toks)
    meta = {"kind": "mw", "players": {s: i for i, s in enumerate(seats)}, "pos": pos, "hero_player": seats.index(0),
            "street": street, "pot": pot, "eff": min(stacks), "notes": notes + size_notes, "line": ranges["line"],
            "source": ranges["source"], "resized": bool(swap), "start_put": round(pot * CHIPS, 2) / len(seats)}
    return {"request": req, "request_small": None, "meta": meta}, None


# ---------------------------------------------------------------- 3b. the 3-way flop range estimator
# Off live: measured heads-up (where the full flop tree solves in seconds) over six flops, the reduced
# tree's end-of-flop ranges were further from the full tree's than no narrowing at all (the OOP range
# 0.59 vs 0.39 total variation; after a checked flop 0.69-0.83 vs 0.06), and the turn played from them
# was worse (OOP betting 0% where the full tree bets 35-48%). With the turn and river checked down, the
# flop is the only street to bet a strong hand, so checking ranges come out capped; one turn bet size
# didn't fix it. validate_multiway checks the same against full 3-way solves.
ESTIMATOR_LIVE = False
CHECKED_NOTE = ("the flop was checked through, and its checking ranges are taken as the preflop ranges: "
                "rough (heads-up, the turn then bet about half as often as from the real ranges)")
ESTIMATOR_BET = "33"            # the flop bet when nobody has bet (yet): a checking range needs one to narrow
ESTIMATOR_SECS = 8.0            # time cap: the estimate is taken wherever the solve is by then
ESTIMATOR_PCT = 1.0
ESTIMATOR_NOTE = ("the 3-way flop's ranges come from a reduced flop solve (the size really bet, no other "
                  "raises, turn and river checked down): approximate, it underrates position and later bets")


def build_estimator_spot(hand, ranges, game):
    """A reduced 3-way flop tree, solved in seconds, whose only job is each player's range at the end of
    the flop: one bet size (the one really bet, else 33%), a raise only at a size really used, the turn
    and river checked down (no bets). A full 3-way flop doesn't solve live; this keeps the multiway
    dynamics of who bets, calls and folds with what."""
    seats = action_order(hand, players_at(hand, "flop"))
    if len(seats) != 3:
        return None, "not a 3-way flop"
    # the flop as it stood: its players (later folds are cut with the later actions), board and actions
    spot, why = build_mw_spot(dict(hand, street="flop", board=hand["board"][:3], in_hand=seats,
                                   actions=[a for a in hand["actions"] if a["street"] in ("preflop", "flop")]),
                              ranges, game, "flop", sizes=[{"bet": ESTIMATOR_BET, "raise": ""},
                                                           {"bet": "", "raise": ""}, {"bet": "", "raise": ""}])
    if spot is None:
        return None, why
    req = spot["request"]
    flop_raise = [t for t in _tokens(req["sizes"][0]["raise"]) if t != "a"] or _tokens(req["sizes"][0]["raise"])
    req["sizes"][0]["raise"] = " ".join(flop_raise)
    raises = sum(1 for a in hand["actions"] if a["street"] == "flop" and a["action"] in ("raise",)) + \
        sum(1 for a in hand["actions"] if a["street"] == "flop" and a["action"] == "allin" and
            any(b["street"] == "flop" and b["action"] in ("bet", "allin") for b in hand["actions"][:hand["actions"].index(a)]))
    req["max_raises"] = max(1, min(raises, 2))
    spot["meta"].update(estimator=True, notes=[n for n in spot["meta"]["notes"] if "in place of" not in n])
    return spot, None


# ---------------------------------------------------------------- 4. reading the 3-way solver's nodes
def mw_path(path):
    """The heads-up path format ({"type": "action", "index": i} / {"type": "card", "card": c}) as
    the 3-way API takes it (an index, or the card text)."""
    return [s["index"] if s["type"] == "action" else s["card"] for s in path]


def mw_view(v, start_put):
    """A /api/mw/node view in the heads-up view's shape, so the same walk and advice code reads it.
    Amounts become the actor's street total (the 3-way view gives the chips an action adds); only the
    acting player's hands come back, with their reach as the weight; per-action EVs aren't given."""
    kind = {"action": "action", "chance": "chance"}.get(v["kind"], "terminal")
    actor = v.get("actor")
    acts = []
    for a in v.get("actions") or []:
        to = 0.0
        if actor is not None and a["kind"] in ("bet", "raise", "call"):
            to = v["put"][actor] - start_put + a["amount"]
        acts.append({"label": a["label"], "kind": a["kind"], "amount": to, "freq": a.get("freq")})
    players = [{"hands": []} for _ in v.get("put") or [0, 0, 0]]
    if actor is not None:
        players[actor]["hands"] = [
            {"combo": h["combo"], "c1": card_index(h["combo"][0:2]), "c2": card_index(h["combo"][2:4]),
             "weight": h["weight"], "reach": h["weight"], "strategy": h.get("strategy"), "evs": None,
             "eq": None, "ev": h.get("ev")} for h in v.get("hands") or []]
    return {"node_type": kind, "player": actor, "pot": v["pot"], "put": v["put"], "street": v["street"],
            "actions": acts, "players": players, "range_combos": v.get("range_combos"),
            "live": v.get("live"), "cards": v.get("cards") or []}
