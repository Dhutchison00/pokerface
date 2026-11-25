"""Work out the hero's preflop decision point from a TableState.

Seats are clockwise from the hero (seat 0); clockwise is the order the action
moves in.  Players still to act in front of the hero are assumed to fold, so
the chart is ready before the action reaches you.
"""
from __future__ import annotations

POSITION_NAMES = {
    2: ["BTN", "BB"],  # heads-up: BTN is also the small blind
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    10: ["UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}

EPS = 0.05


def derive_spot(st) -> dict:
    spot = {"type": "UNKNOWN", "notes": [], "warnings": []}
    if not st.ok:
        spot["warnings"].append(st.error)
        return spot
    seats = st.seats
    n_seats = len(seats)
    hero = seats[0]
    if not hero.present:
        spot["warnings"].append("Hero seat (bottom centre) not found")
        return spot

    def dealt(s):
        return s.present and (s.idx == 0 or s.has_cards or s.folded or s.bet > 0 or s.all_in)

    in_hand = [s for s in seats if dealt(s)]
    n = len(in_hand)
    if n < 2:
        spot["warnings"].append("Fewer than 2 players found")
        return spot
    ring = [s.idx for s in in_hand]  # clockwise order

    def prev_in(idx, k=1):
        i = ring.index(idx)
        return ring[(i - k) % n]

    def next_in(idx, k=1):
        i = ring.index(idx)
        return ring[(i + k) % n]

    # --- locate the blinds / button
    sb_seat = st.sb_seat if st.sb_seat in ring else None
    bb_seat = st.bb_seat if st.bb_seat in ring else None
    source = ""
    if bb_seat is not None:
        source = "SB/BB labels" if sb_seat is not None else "BB label"
    elif sb_seat is not None:
        bb_seat = next_in(sb_seat)
        source = "SB label"
    elif st.button_seat in ring:
        bb_seat = next_in(st.button_seat, 1 if n == 2 else 2)
        source = "dealer button"
    else:
        # last resort: posted blinds (bets of 0.5 / 1 BB before any action)
        halves = [s.idx for s in in_hand if abs(s.bet - st.sb / st.bb) < EPS]
        if halves:
            bb_seat = next_in(halves[0])
            source = "posted bets"
    if bb_seat is None:
        spot["warnings"].append("Couldn't locate the blinds or dealer button")
        return spot
    if n == 2:
        btn_seat = sb_seat = prev_in(bb_seat)
    else:
        sb_seat = prev_in(bb_seat)
        btn_seat = prev_in(bb_seat, 2)
    if st.button_seat is not None and st.button_seat in ring and st.button_seat != btn_seat:
        spot["warnings"].append(f"Dealer button seen at seat {st.button_seat} but blinds imply seat {btn_seat}")

    names = POSITION_NAMES.get(n) or (["UTG"] * (n - 9) + POSITION_NAMES[9])
    order = [next_in(bb_seat, k) for k in range(1, n + 1)]  # UTG ... BB
    pos = {seat: names[k] for k, seat in enumerate(order)}
    posted = {s: 0.0 for s in ring}
    sb_amt = st.sb / st.bb
    posted[sb_seat] = sb_amt if n > 2 or sb_seat != bb_seat else sb_amt
    posted[bb_seat] = 1.0

    by_idx = {s.idx: s for s in seats}
    total_bets = sum(by_idx[s].bet for s in ring)
    pot = st.pot if st.pot is not None else total_bets + n * st.ante / st.bb
    dead = max(0.0, pot - total_bets)

    def player(seat):
        s = by_idx[seat]
        return {"seat": seat, "pos": pos[seat], "name": s.name, "bet": round(s.bet, 3),
                "stack": s.stack, "total": round((s.stack or 0) + s.bet, 3),
                "posted": posted[seat], "all_in": s.all_in}

    hero_i = order.index(0)
    hero_p = player(0)
    spot.update({
        "hero_pos": pos[0], "n_players": n, "positions": {str(k): v for k, v in pos.items()},
        "button_seat": btn_seat, "sb_seat": sb_seat, "bb_seat": bb_seat, "blind_source": source,
        "pot": round(pot, 3), "dead": round(dead, 3), "ante_bb": round(st.ante / st.bb, 4),
        "sb_bb": round(sb_amt, 3), "hero": hero_p, "hero_total": hero_p["total"],
    })
    if not (st.hero_cards and st.hero_cards.get("ranks")):
        spot["notes"].append("No hole cards visible for hero")

    # --- the flop is out: preflop charts don't apply. (Its pot would otherwise read as preflop
    # dead money and send the solver after a nonsense grid point.) A flop is always 3 cards, so a
    # stray 1-2 card detection doesn't switch the charts off.
    board = getattr(st, "board", None) or []
    if len(board) >= 3:
        street = {3: "flop", 4: "turn", 5: "river"}.get(len(board), "postflop")
        spot.update(type="POSTFLOP", street=street, dead=0.0)
        spot["notes"].append(f"Postflop ({street}): preflop charts don't apply")
        return spot

    # --- classify the action so far
    raises, limpers, callers, pending, folded = [], [], [], [], []
    for seat in order:
        if seat == 0:
            continue
        s = by_idx[seat]
        p = player(seat)
        if s.folded or (not s.has_cards and not s.all_in and s.bet <= posted[seat] + EPS):
            folded.append(p)
            continue
        if s.sitting_out:
            folded.append(p)
            spot["notes"].append(f"{p['pos']} is sitting out (counted as a fold)")
            continue
        if s.bet > 1.0 + EPS or (s.all_in and s.bet > posted[seat] + EPS):
            raises.append(p)
        elif abs(s.bet - 1.0) < EPS and seat != bb_seat:
            limpers.append(p)
        else:
            pending.append(p)

    hero_bet = hero_p["bet"]
    hero_vol = hero_bet > posted[0] + EPS  # hero already put voluntary chips in
    raises.sort(key=lambda p: p["bet"])
    # players still able to act behind the hero (not folded, not sitting out)
    before = set(order[:hero_i])
    behind = [p for p in pending if p["seat"] not in before]
    pending_before = [p for p in pending if p["seat"] in before]
    if pending_before:
        spot["notes"].append("Assuming " + ", ".join(p["pos"] for p in pending_before) + " fold")
    if not behind and not raises and not limpers and not any(by_idx[s].has_cards for s in ring if s != 0):
        spot["warnings"].append("No opponents' cards detected: check card_back_box / card_back_min_frac "
                                "(every opponent reads as folded)")
    spot["behind"] = behind
    spot["pending_before"] = [p["pos"] for p in pending_before]
    spot["limpers"] = limpers

    if not raises:
        if hero_vol:
            spot["type"] = "WAITING"      # hero limped/completed, nobody raised: postflop next
            spot["notes"].append("You already acted and nobody raised")
        elif limpers:
            spot["type"] = "LIMPED"
        elif pos[0] == "BB":
            spot["type"] = "WAITING"      # BB has no first-in decision
            spot["notes"].append("BB: waiting for someone to open (folds = walk)")
        else:
            spot["type"] = "RFI"
        opp_totals = [p["total"] for p in behind + limpers]
        spot["eff_bb"] = round(min(hero_p["total"], max(opp_totals)) if opp_totals else hero_p["total"], 2)
        return spot

    villain = raises[-1]
    spot["villain"] = villain
    spot["raises"] = raises
    spot["to_call"] = round(min(villain["bet"], hero_p["total"]) - hero_bet, 3)
    spot["eff_bb"] = round(min(hero_p["total"], villain["total"]), 2)
    shove = villain["all_in"] or villain["bet"] >= hero_p["total"] - EPS
    distinct = sorted({round(p["bet"], 2) for p in raises})
    if hero_vol and hero_bet > 1.0 + EPS:
        spot["type"] = "VS_SHOVE" if shove else "VS_3BET"
        spot["hero_opened"] = True
    elif shove:
        spot["type"] = "VS_SHOVE"
    elif len(distinct) >= 2:
        spot["type"] = "VS_3BET"
    else:
        spot["type"] = "VS_OPEN"
    spot["n_raises"] = len(distinct)
    cold_callers = [p for p in raises if abs(p["bet"] - villain["bet"]) < EPS and p["seat"] != villain["seat"]]
    if cold_callers or limpers:
        spot["notes"].append("Multiway pot: ranges assume heads-up vs the last aggressor")
    return spot
