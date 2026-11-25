"""Hand tracker on synthetic frame sequences: python tests/test_tracker.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preflop_hud.tracker import Tracker  # noqa: E402

# 8-handed; the hero (seat 0) is on the BTN. Seats run clockwise from the hero.
POS = {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "UTG1", 5: "LJ", 6: "HJ", 7: "CO"}


class Table:
    """Builds observations the way the reader reports a frame."""

    def __init__(self, hero="AhKd"):
        self.hero, self.board, self.t = hero, [], 0.0
        self.seats = {i: {"idx": i, "present": True, "stack": 50.0, "bet": 0.0, "labels": [],
                          "has_cards": True, "all_in": False} for i in POS}
        self.seats[1]["bet"], self.seats[2]["bet"] = 0.5, 1.0
        self.seats[1]["labels"], self.seats[2]["labels"] = ["sb"], ["bb"]

    def fold(self, i):
        self.seats[i]["labels"] = ["fold"]
        self.seats[i]["has_cards"] = False

    def bet(self, i, to):
        s = self.seats[i]                  # chips move from the stack to the bet, as on screen
        s["stack"] = round(s["stack"] - (to - s["bet"]), 2) if s["stack"] is not None else None
        s["bet"] = to

    def check(self, i):
        self.seats[i]["labels"] = ["check"]

    def deal(self, *cards):
        """Next street: bets are swept into the pot, labels clear."""
        self.board += list(cards)
        for s in self.seats.values():
            s["bet"] = 0.0
            if "fold" not in s["labels"]:
                s["labels"] = []

    def obs(self):
        self.t += 0.3
        return {"t": self.t, "ok": True, "hero": self.hero, "pot": None, "board": list(self.board),
                "positions": {str(k): v for k, v in POS.items()},
                "seats": [dict(s, labels=list(s["labels"])) for s in self.seats.values()]}


def line(tr, street):
    h = tr.state()
    return [(a["pos"], a["action"], a["to"], a["inferred"]) for a in h["actions"] if a["street"] == street]


# ---------------------------------------------------------------- a normal hand, one action per frame
tb, tr = Table(), Tracker()
tr.update(tb.obs())
for i in (3, 4, 5, 6, 7):
    tb.fold(i)
    tr.update(tb.obs())
tb.bet(0, 2.5)                      # hero opens on the BTN
tr.update(tb.obs())
tb.fold(1)
tr.update(tb.obs())
tb.bet(2, 2.5)                      # BB calls
tr.update(tb.obs())
assert line(tr, "preflop") == [("UTG", "fold", None, False), ("UTG1", "fold", None, False), ("LJ", "fold", None, False),
                               ("HJ", "fold", None, False), ("CO", "fold", None, False), ("BTN", "raise", 2.5, False),
                               ("SB", "fold", None, False), ("BB", "call", 2.5, False)], line(tr, "preflop")
tb.deal("Td", "7c", "2h")
tr.update(tb.obs())
tb.check(2)
tr.update(tb.obs())
tb.bet(0, 1.8)
tr.update(tb.obs())
tb.bet(2, 6.0)                      # check-raise
tr.update(tb.obs())
tb.bet(0, 6.0)
tr.update(tb.obs())
assert line(tr, "flop") == [("BB", "check", None, False), ("BTN", "bet", 1.8, False), ("BB", "raise", 6.0, False),
                            ("BTN", "call", 6.0, False)], line(tr, "flop")
assert tr.state()["street"] == "flop" and tr.state()["board"] == ["Td", "7c", "2h"]
print("normal hand OK:", " | ".join(ln["text"] for ln in tr.state()["lines"]))

# ---------------------------------------------------------------- fast bots: actions merge or get swept
tb, tr = Table(), Tracker()
tr.update(tb.obs())
for i in (3, 4, 5, 6):
    tb.fold(i)
tb.bet(7, 2.2)                      # four folds and the CO open in ONE frame
tr.update(tb.obs())
assert [a[:2] for a in line(tr, "preflop")] == [("UTG", "fold"), ("UTG1", "fold"), ("LJ", "fold"), ("HJ", "fold"),
                                                ("CO", "raise")], line(tr, "preflop")
assert any("one frame" in n for n in tr.state()["notes"])
tb.fold(1)
tb.bet(0, 2.2)                      # hero calls, SB folds, ...
tr.update(tb.obs())
tb.deal("As", "8s", "3d")           # ... and the BB's call is never seen: the flop is out
tr.update(tb.obs())
pre = line(tr, "preflop")
assert ("BB", "call", 2.2, True) in pre, pre
tb.deal("Kc")                       # flop checked through without a single CHECK label seen
tr.update(tb.obs())
assert line(tr, "flop") == [("BB", "check", None, True), ("CO", "check", None, True), ("BTN", "check", None, True)], \
    line(tr, "flop")
tb.bet(2, 3.0)                      # BB bets the turn, then CO folds in the frame the river comes
tr.update(tb.obs())
tb.bet(0, 3.0)
tb.fold(7)
tb.deal("4h")
tr.update(tb.obs())
assert line(tr, "turn") == [("BB", "bet", 3.0, False), ("CO", "fold", None, True), ("BTN", "call", 3.0, True)], \
    line(tr, "turn")
assert tr.state()["in_hand"] == [0, 2] and tr.state()["street"] == "river"
print("fast-bot inference OK:", " | ".join(ln["text"] for ln in tr.state()["lines"]))

# ---------------------------------------------------------------- new hands, and an OCR flicker that isn't one
h1 = tr.state()["id"]
flicker = Table(hero="Ks4h")        # one frame reads different hole cards...
tr.update(dict(tb.obs(), hero="Ks4h"))
assert tr.state()["id"] == h1       # ...not a new hand yet
tr.update(tb.obs())                 # back to the real cards
nxt = Table(hero="9c9d")
tr.update(nxt.obs())
tr.update(nxt.obs())                # two frames of new cards (and the board cleared): new hand
assert tr.state()["id"] == h1 + 1 and tr.state()["hero"] == "9c9d" and tr.state()["street"] == "preflop"
assert len(tr.history) == 1 and tr.history[0]["street"] == "river"
print("new hand detection OK")

# ---------------------------------------------------------------- the big blind's option in a limped pot
tb, tr = Table(hero="7c6c"), Tracker()
tr.update(tb.obs())
for i in (3, 4, 5, 6, 7):
    tb.fold(i)
tb.bet(0, 1.0)                      # hero limps the BTN, SB completes, BB checks (no label seen)
tb.bet(1, 1.0)
tr.update(tb.obs())
tb.deal("Jd", "5c", "2s")
tr.update(tb.obs())
assert line(tr, "preflop")[-3:] == [("BTN", "call", 1.0, False), ("SB", "call", 1.0, False),
                                    ("BB", "check", None, True)], line(tr, "preflop")
print("limped pot OK")

# ---------------------------------------------------------------- recorded hands (real reads, frame by frame)
import json  # noqa: E402
FIX = os.path.join(ROOT, "tests", "fixtures")
REC = {
    # file: (hero cards, final board, {street: [(pos, action, to)]}), straight from the recorded reads
    "rec_ah2h_call_then_fold.json": ("Ah2h", [], {"preflop": [
        ("UTG", "fold", None), ("UTG1", "allin", 2.71), ("LJ", "call", 2.71), ("HJ", "fold", None),
        ("CO", "call", 2.71), ("BTN", "allin", 17.89), ("SB", "fold", None), ("BB", "fold", None),
        ("LJ", "fold", None)]}),
    # cards animate away at the end (not a new hand); the BB's cards vanish at showdown (a muck, not a fold)
    "rec_ksjs_flop_bet_check_down.json": ("KsJs", ["Kd", "Ad", "5c", "Qs", "2h"], {
        "preflop": [("UTG", "fold", None), ("UTG1", "fold", None), ("LJ", "raise", 3.0), ("HJ", "fold", None),
                    ("CO", "fold", None), ("BTN", "fold", None), ("SB", "fold", None), ("BB", "call", 3.0)],
        "flop": [("BB", "check", None), ("LJ", "bet", 1.88), ("BB", "call", 1.88)],
        "turn": [("BB", "check", None), ("LJ", "check", None)],
        "river": [("BB", "check", None), ("LJ", "check", None)]}),
    # recording starts with two players already all-in: their shoves are logged (inferred) in seat
    # order with the folds, no owed calls or checks for them, and no actions while the board runs out
    "rec_ks9s_call_allins_late_start.json": ("Ks9s", ["2d", "4s", "4h", "Td", "Ad"], {
        "preflop": [("UTG", "allin", 7.81), ("UTG1", "fold", None), ("LJ", "fold", None), ("HJ", "fold", None),
                    ("CO", "allin", 10.96), ("BTN", "fold", None), ("SB", "fold", None), ("BB", "call", 10.96)],
        "flop": [], "turn": [], "river": []}),
    # recording starts on the flop (no invented preflop); the river jam shows ALL-IN a frame before its chips
    "rec_qcth_flop_turn_calls_river_jam_fold.json": ("QcTh", ["Jd", "6h", "9h", "4c", "Qd"], {
        "preflop": [], "flop": [("SB", "bet", 2.4), ("BB", "call", 2.4)],
        "turn": [("SB", "bet", 5.49), ("BB", "call", 5.49)],
        "river": [("SB", "allin", 43.38), ("BB", "fold", None)]}),
    # seats already folded when tracking starts are logged as inferred folds; limp, two checked streets, river bet.
    # The K's suit is unreadable at first (K?), then read as a spade: the hand keeps the better read
    "rec_kjh_sb_limp_checks_river_bet_call.json": ("KsJh", ["9h", "4h", "5c", "2s", "3h"], {
        "preflop": [("UTG", "fold", None), ("UTG1", "fold", None), ("LJ", "fold", None), ("HJ", "fold", None),
                    ("CO", "fold", None), ("BTN", "fold", None), ("SB", "call", 1.0), ("BB", "check", None)],
        "flop": [("SB", "check", None), ("BB", "check", None)], "turn": [("SB", "check", None), ("BB", "check", None)],
        "river": [("SB", "bet", 1.5), ("BB", "call", 1.5)]}),
    # tracking starts with UTG's limp already in front of them: logged (inferred) before the hero's raise.
    # After the turn's raise and call the pot shows 13.38 with no bets out, before the river comes
    "rec_9h9c_utg_limp_turn_raise_river_bet.json": ("9h9c", ["Kh", "Kd", "8d", "2s", "6d"], {
        "preflop": [("UTG", "call", 1.0), ("LJ", "raise", 3.0), ("HJ", "fold", None), ("CO", "fold", None),
                    ("BTN", "fold", None), ("SB", "fold", None), ("BB", "fold", None), ("UTG", "call", 3.0)],
        "flop": [("UTG", "check", None), ("LJ", "check", None)],
        "turn": [("UTG", "bet", 1.0), ("LJ", "raise", 2.5), ("UTG", "call", 2.5)],
        "river": [("UTG", "check", None), ("LJ", "bet", 3.35), ("UTG", "call", 3.35)]}),
    # 3-way: two limps, the hero's SB raise, two calls; a flop shove for less, called by both. Two frames
    # misread a bet mid-animation (the hero's 2.99 call as 5.0, 10.45 in front of the all-in LJ who had
    # 3.0): more than the seat had, so ignored (they used to log an SB raise and a phantom 10.45 shove)
    "rec_8s8h_3way_limps_flop_shove_river_bet.json": ("8s8h", ["2h", "Td", "Ks", "Kc", "9c"], {
        "preflop": [("UTG", "call", 1.0), ("UTG1", "fold", None), ("LJ", "call", 1.0), ("HJ", "fold", None),
                    ("CO", "fold", None), ("BTN", "fold", None), ("SB", "raise", 2.5), ("BB", "fold", None),
                    ("UTG", "call", 2.5), ("LJ", "call", 2.5)],
        "flop": [("SB", "bet", 2.5), ("UTG", "call", 2.5), ("LJ", "allin", 2.99), ("SB", "call", 2.99),
                 ("UTG", "call", 2.99)],
        "turn": [("SB", "check", None), ("UTG", "check", None)],
        "river": [("SB", "bet", 2.0), ("UTG", "fold", None)]}),
    # re-read with the glyph-first board reader: the green 6 used to read as a 9 (and a 10 on the river);
    # tracking starts on the flop, which three players check through
    "rec_5d3d_3way_flop_checked_turn_bet.json": ("5d3d", ["6c", "9c", "Jh", "Qs", "3h"], {
        "preflop": [], "flop": [("SB", "check", None), ("HJ", "check", None), ("CO", "check", None)],
        "turn": [("SB", "bet", 2.0), ("HJ", "fold", None), ("CO", "call", 2.0)],
        "river": [("SB", "check", None), ("CO", "check", None)]}),
}
for name, (hero, board, streets) in REC.items():
    with open(os.path.join(FIX, name)) as f:
        rec = json.load(f)
    tr = Tracker()
    for obs in rec["frames"]:
        tr.update(obs)
    hands = tr.history + ([tr.hand.summary()] if tr.hand else [])
    assert len(hands) == 1, (name, [h["id"] for h in hands])       # one hand, no phantom extras
    h = hands[0]
    assert h["hero"] == hero and h["board"] == board, (name, h["hero"], h["board"])
    for street, want in streets.items():
        got = [(a["pos"], a["action"], a["to"]) for a in h["actions"] if a["street"] == street]
        assert got == want, (name, street, got)
    print(f"recording {name} OK: {hero} | " + " | ".join(ln["text"] for ln in h["lines"]))
    if name.startswith("rec_9h9c"):
        # each street's start: read before its bets, never after they're swept into the pot
        got = {k: (v["pot"], v["stacks"][0], v["stacks"][6]) for k, v in h["start"].items() if k != "preflop"}
        assert got == {"flop": (8.38, 9.1, 8.5), "turn": (8.38, 9.1, 8.5), "river": (13.38, 6.6, 6.0)}, got
        assert h["start"]["preflop"]["dead"] == 0.88 and h["start"]["preflop"]["stacks"][0] == 12.1
# ---------------------------------------------------------------- whose turn: pre-action boxes aren't the hero's turn
from types import SimpleNamespace as N  # noqa: E402
from preflop_hud.tracker import hero_turn  # noqa: E402
seats = lambda me: [N(idx=0, acting=me), N(idx=5, acting=not me)]  # noqa: E731
assert not hero_turn(N(seats=seats(False), hero_buttons=["FOLD/CHECK", "CHECK"]))   # the BB still thinking
assert not hero_turn(N(seats=seats(False), hero_buttons=["CALL ANY", "FOLD"]))
assert hero_turn(N(seats=seats(False), hero_buttons=["BET", "FOLD", "CHECK"]))      # recorded: the hero's turn
assert hero_turn(N(seats=seats(True), hero_buttons=[]))
print("hero's turn OK")
# a flop card misread while it animated in (the 4 as a 10), then the settled screen read: corrected at once
tb, tr = Table(), Tracker()
tr.update(tb.obs())
for i in (3, 4, 5, 6, 7):
    tb.fold(i)
    tr.update(tb.obs())
tb.bet(0, 2.5)
tr.update(tb.obs())
tb.fold(1)
tr.update(tb.obs())
tb.bet(2, 2.5)
tr.update(tb.obs())
tb.deal("7d", "Ad", "Tc")
tr.update(tb.obs())                                   # mid-animation read
tb.board[2] = "4c"
assert tr.update(dict(tb.obs(), settled=True))["board"] == ["7d", "Ad", "4c"]
tb.board[2] = "Tc"                                    # an unsettled one-off misread doesn't change it back
assert tr.update(tb.obs())["board"] == ["7d", "Ad", "4c"]
print("settled read corrects a misread board card OK")


# ---------------------------------------------------------------- a board that isn't read isn't a new hand
def flop_hand():
    """The hero opens the BTN, the BB calls, the flop comes Td 7c 2h and the BB checks."""
    tb, tr = Table(), Tracker()
    tr.update(tb.obs())
    for i in (3, 4, 5, 6, 7):
        tb.fold(i)
        tr.update(tb.obs())
    tb.bet(0, 2.5)
    tr.update(tb.obs())
    tb.fold(1)
    tr.update(tb.obs())
    tb.bet(2, 2.5)
    tr.update(tb.obs())
    tb.deal("Td", "7c", "2h")
    tr.update(tb.obs())
    tb.check(2)
    tr.update(tb.obs())
    return tb, tr


FLOP_BET = [("BB", "check", None, False), ("BTN", "bet", 1.8, False)]
# one read without the board (an overlay, a failed board read): the hand carries on untouched
tb, tr = flop_hand()
tr.update(dict(tb.obs(), board=[]))
assert tr.state()["id"] == 1 and not tr.state().get("finished") and not tr.history, tr.state()
tb.bet(0, 1.8)
tr.update(tb.obs())
assert tr.state()["id"] == 1 and line(tr, "flop") == FLOP_BET and len(line(tr, "preflop")) == 8, tr.state()
# two reads without it: taken as cleared and filed, but the same hole cards don't start another hand...
tb, tr = flop_hand()
for _ in range(3):
    tr.update(dict(tb.obs(), board=[]))
assert tr.hand is None and tr.state()["finished"] and len(tr.history) == 1, tr.state()
tb.bet(0, 1.8)
tr.update(tb.obs())                 # ...and when the same board is back, it's the same hand again
assert tr.state()["id"] == 1 and not tr.state().get("finished") and not tr.history, tr.state()
assert line(tr, "flop") == FLOP_BET and len(line(tr, "preflop")) == 8, tr.state()
# a settled read without the board is believed at once
tb, tr = flop_hand()
tr.update(dict(tb.obs(), board=[], settled=True))
assert tr.hand is None and tr.state()["finished"], tr.state()
# the hand really ended: once the cards have gone, the same ranks dealt again are a new hand
tb, tr = flop_hand()
for _ in range(2):
    tr.update(dict(tb.obs(), board=[]))
tr.update(dict(tb.obs(), board=[], hero=None))
tr.update(Table(hero="AsKc").obs())
assert tr.state()["id"] == 2 and tr.state()["hero"] == "AsKc" and tr.state()["street"] == "preflop", tr.state()
assert len(tr.history) == 1 and tr.history[0]["street"] == "flop"
# ...and so are the same cards still on screen once the button has moved
tb, tr = flop_hand()
for _ in range(2):
    tr.update(dict(tb.obs(), board=[]))
moved = Table()
tr.update(dict(moved.obs(), positions={str(k): POS[(k + 1) % 8] for k in POS}))
assert tr.state()["id"] == 2 and tr.state()["street"] == "preflop" and len(tr.history) == 1, tr.state()


def play_on(tr, nxt):
    """The next hand, read many times over: folds to the hero, who opens; the BB calls; the flop."""
    for i in (3, 4, 5, 6, 7):
        nxt.fold(i)
        tr.update(nxt.obs())
    nxt.bet(0, 2.5)
    tr.update(nxt.obs())
    nxt.fold(1)
    tr.update(nxt.obs())
    nxt.bet(2, 2.5)
    tr.update(nxt.obs())
    nxt.deal("9s", "5d", "3c")
    tr.update(nxt.obs())
    return tr.state()


# the hand after one whose board cleared is tracked like any other, from its deal to its flop
tb, tr = flop_hand()
for _ in range(2):
    tr.update(dict(tb.obs(), board=[]))
nxt = Table(hero="QsJs")
tr.update(nxt.obs())
h = play_on(tr, nxt)
assert h["id"] == 2 and h["hero"] == "QsJs" and h["street"] == "flop" and not h.get("finished"), h
assert [a[:2] for a in line(tr, "preflop")][-3:] == [("BTN", "raise"), ("SB", "fold"), ("BB", "call")], h
assert len(tr.history) == 1 and tr.history[0]["id"] == 1
# new hole cards while the last hand's board is still on screen, then that board clears: the new
# hand is tracked from there (it starts again as a preflop hand once the old board has gone)
tb, tr = flop_hand()
nxt = Table(hero="QsJs")
for _ in range(2):
    tr.update(dict(nxt.obs(), board=list(tb.board)))
assert tr.state()["hero"] == "QsJs"
for _ in range(2):
    tr.update(nxt.obs())
h = play_on(tr, nxt)
assert h["hero"] == "QsJs" and h["street"] == "flop" and h["board"] == ["9s", "5d", "3c"], h
assert [a[:2] for a in line(tr, "preflop")][-3:] == [("BTN", "raise"), ("SB", "fold"), ("BB", "call")], h
print("unread board OK")
print("all tests passed")
