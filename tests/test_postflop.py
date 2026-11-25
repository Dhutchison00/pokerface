"""Live flop advice: python tests/test_postflop.py

Ranges, the flop spot and the tree walk run on synthetic hands and a fake solver tree; the
background engine on a fake gto-server; the recorded hands through the tracker. A real GTOpen
solve runs too when HUD_TEST_GTOPEN=1 (needs the GTOpen checkout in config.json and the charts).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preflop_hud import postflop as pf  # noqa: E402
from preflop_hud.cards import HAND_INDEX  # noqa: E402
from preflop_hud.tracker import Tracker  # noqa: E402

POS8 = {"0": "BTN", "1": "SB", "2": "BB", "3": "UTG", "4": "UTG1", "5": "LJ", "6": "HJ", "7": "CO"}


def act(street, seat, action, to=None, inferred=False):
    return {"street": street, "seat": seat, "pos": POS8[str(seat)], "action": action, "to": to, "inferred": inferred}


def hand(pre, flop=(), hero="AhKd", board=("Td", "7c", "2h"), in_hand=(0, 2), **kw):
    folds = [act("preflop", s, "fold", inferred=True) for s in (3, 4, 5, 6, 7)]
    h = {"id": 1, "hero": hero, "street": "flop", "board": list(board), "actions": folds + list(pre) + list(flop),
         "notes": [], "in_hand": list(in_hand), "positions": dict(POS8), "allin": [], "hero_turn": False,
         "start": {"preflop": {"pot": None, "dead": 1.0, "stacks": {0: 40.0, 1: 30.0, 2: 25.0}},
                   "flop": {"pot": 6.5, "dead": None, "stacks": {0: 37.5, 1: 30.0, 2: 22.5}}}}
    h.update(kw)
    return h


def fake_lookup(calls):
    """Every node: reach 1, and the action's frequency 0.5 (AA: 1)."""
    def lookup(n, eff, dead, game, key):
        calls.append((n, eff, dead, key))
        acts = ["fold", "call", "raise", "check", "allin"]
        strat = np.full((len(acts), 169), 0.5)
        strat[:, HAND_INDEX["AA"]] = 1.0
        return {"actions": acts, "strategy": strat, "reach": np.ones(169)}, "fake charts"
    return lookup


# ---------------------------------------------------------------- 1. preflop lines -> chart nodes
open_call = hand([act("preflop", 0, "raise", 2.5), act("preflop", 1, "fold"), act("preflop", 2, "call", 2.5, True)])
plan, line, _ = pf.preflop_plan(open_call)
assert plan == {"BTN": ("RFI:BTN", "raise"), "BB": ("VS_OPEN:BB:BTN", "call")} and line == "BTN open, BB call", plan
three_bet = hand([act("preflop", 0, "raise", 2.5), act("preflop", 1, "fold"), act("preflop", 2, "raise", 9.0),
                  act("preflop", 0, "call", 9.0)])
assert pf.preflop_plan(three_bet)[0] == {"BTN": ("VS_3BET:BTN:BB", "call"), "BB": ("VS_OPEN:BB:BTN", "raise")}
limped = hand([act("preflop", 0, "fold"), act("preflop", 1, "call", 1.0), act("preflop", 2, "check", None, True)],
              in_hand=(1, 2), hero="9s8s")
assert pf.preflop_plan(limped)[0] == {"SB": ("RFI:SB", "call"), "BB": ("BB_VS_LIMP", "check")}
iso = hand([act("preflop", 0, "fold"), act("preflop", 1, "call", 1.0), act("preflop", 2, "raise", 4.0),
            act("preflop", 1, "call", 4.0)], in_hand=(1, 2))
assert pf.preflop_plan(iso)[0] == {"SB": ("SB_VS_ISO", "call"), "BB": ("BB_VS_LIMP", "raise")}
cold = hand([act("preflop", 7, "raise", 2.5), act("preflop", 0, "call", 2.5), act("preflop", 7, "fold")])
assert pf.preflop_plan(cold)[0] is None and "folded later" in pf.preflop_plan(cold)[1]
late = dict(open_call, notes=["joined on the flop: earlier streets weren't seen"])
assert pf.preflop_plan(late)[0] is None and "tracking started late" in pf.preflop_plan(late)[1]
# a limp from a seat that isn't the SB: assumed ranges (the charts' limp nodes there aren't converged)
def utg_limp(*after, in_hand=(0, 3)):
    pre = [act("preflop", 3, "call", 1.0, True)] + [act("preflop", s, "fold") for s in (4, 5, 6, 7)] + list(after)
    st = {"preflop": {"pot": None, "dead": 1.0, "stacks": {0: 40.0, 2: 25.0, 3: 30.0}},
          "flop": {"pot": 7.0, "dead": None, "stacks": {0: 37.0, 2: 22.0, 3: 27.0}}}
    return hand([], actions=pre, in_hand=list(in_hand), start=st)


iso_btn = utg_limp(act("preflop", 0, "raise", 3.0), act("preflop", 1, "fold"), act("preflop", 2, "fold"),
                   act("preflop", 3, "call", 3.0, True))
plan, line, notes = pf.preflop_plan(iso_btn)
assert plan == {"UTG": ("RFI:UTG", "raise"), "BTN": ("RFI:BTN", "raise")} and line == "UTG limp, BTN raise, UTG call"
assert len(notes) == 2 and "UTG limped" in notes[0] and "BTN's iso-raise range is assumed" in notes[1], notes
iso_bb = utg_limp(act("preflop", 0, "fold"), act("preflop", 1, "fold"), act("preflop", 2, "raise", 4.0),
                  act("preflop", 3, "call", 4.0), in_hand=(2, 3))
assert pf.preflop_plan(iso_bb)[0] == {"UTG": ("RFI:UTG", "raise"), "BB": ("BB_VS_LIMP", "raise")}
limp_chk = utg_limp(act("preflop", 0, "fold"), act("preflop", 1, "fold"), act("preflop", 2, "check"), in_hand=(2, 3))
plan, line, notes = pf.preflop_plan(limp_chk)
assert plan == {"UTG": ("RFI:UTG", "raise"), "BB": ("BB_VS_LIMP", "check")} and line == "UTG limp, BB check"
# the SB's own limp lines are real chart nodes: no assumption notes
assert pf.preflop_plan(limped)[2] == [] and pf.preflop_plan(iso)[2] == []
# an over-limp or limp-reraise is still not in the charts
overlimp = utg_limp(act("preflop", 0, "call", 1.0), act("preflop", 1, "fold"), act("preflop", 2, "check"),
                    in_hand=(0, 2, 3))
assert pf.preflop_plan(overlimp)[0] is None
# the assumption reaches the panel's notes, the ranges come from the open-raise nodes
rng, why = pf.build_ranges(iso_btn, fake_lookup(calls := []), None)
assert why is None and [c[3] for c in calls] == ["RFI:UTG", "RFI:BTN"], (why, calls)
assert rng["notes"] == pf.preflop_plan(iso_btn)[2]
sp, why = pf.build_spot(iso_btn, rng, None)
assert why is None and any("UTG limped" in n for n in sp["meta"]["notes"]), sp["meta"]["notes"]
assert sp["meta"]["line"] == "UTG limp, BTN raise, UTG call"
print("preflop lines OK")

# ---------------------------------------------------------------- 2. ranges and the flop spot
calls = []
rng, why = pf.build_ranges(open_call, fake_lookup(calls), None)
assert why is None and set(rng["weights"]) == {"BTN", "BB"}
assert calls == [(8, 25.0, 1.0, "RFI:BTN"), (8, 25.0, 1.0, "VS_OPEN:BB:BTN")], calls   # depth: the smaller start stack
assert rng["weights"]["BTN"][HAND_INDEX["AA"]] == 1.0 and abs(rng["weights"]["BB"][HAND_INDEX["72o"]] - 0.5) < 1e-9
spot, why = pf.build_spot(open_call, rng, None)
req, meta = spot["request"], spot["meta"]
assert meta["players"] == {2: 0, 0: 1} and meta["hero_player"] == 1, meta        # BB out of position, BTN in
assert req["starting_pot"] == 65.0 and req["effective_stack"] == 225.0 and req["board"] == "Td7c2h", req
assert req["rake_pct"] == 0 and req["rake_cap"] == 0
assert "AA," in req["range_ip"] + "," and "72o:0.500" in req["range_oop"]
cash = pf.build_spot(open_call, rng, {"mode": "cash"})[0]["request"]
assert cash["rake_pct"] == 5.0 and cash["rake_cap"] == 30.0, cash                  # 3bb cap in chips (x10)
# a hand the hero's range (almost) never has is kept at a small weight, and said so
rng0 = {"weights": {"BTN": np.zeros(169), "BB": np.ones(169)}, "line": "x", "source": "x"}
sp0 = pf.build_spot(open_call, rng0, None)[0]
assert "AKo:0.020" in sp0["request"]["range_ip"] and any("rarely" in n for n in sp0["meta"]["notes"])
icm = pf.build_spot(open_call, rng, {"mode": "tournament", "icm_stage": "ft"})[0]
assert any("ICM" in n for n in icm["meta"]["notes"])
# what can't be solved yet says why
assert pf.check_hand(dict(open_call, in_hand=[0, 2, 7])) is None             # 3-way: the multiway path
assert "turn card" in pf.check_hand(dict(open_call, street="turn"))              # board still shows 3 cards
assert pf.check_hand(dict(open_call, street="turn", board=["Td", "7c", "2h", "Qs"])) is None
assert pf.check_hand(dict(open_call, street="river", board=["Td", "7c", "2h", "Qs", "3c"])) is None
assert "overlap" in pf.check_hand(dict(open_call, street="turn", board=["Td", "7c", "2h", "Ah"]))
assert "suit" in pf.check_hand(dict(open_call, hero="A?Kd"))
assert "all-in" in pf.check_hand(dict(open_call, allin=[2]))           # all-in before the flop
assert "all-in" in pf.check_hand(dict(open_call, allin=[0], actions=open_call["actions"] + [act("flop", 0, "allin", 22.5)]))
# the villain shoving the flop leaves the hero a call-or-fold decision
assert pf.check_hand(dict(open_call, allin=[2], actions=open_call["actions"] + [act("flop", 2, "allin", 22.5)])) is None
assert "hand over" in pf.check_hand(dict(open_call, in_hand=[0]))
assert "out of" in pf.check_hand(dict(open_call, in_hand=[2, 7]))
assert pf.check_hand(open_call) is None
# bets the menu doesn't have are added to it (GTOpen's units: a raise is % of the pot after calling)
players = {2: 0, 0: 1}
ex, nt = pf.off_tree_sizes(dict(open_call, actions=[act("flop", 2, "bet", 1.6)]), "flop", players, req)
assert ex == {(0, "bet"): ["24.6"]} and "BB bet 1.6bb (25% of the pot)" in nt[0], (ex, nt)   # 16 of 65 chips
assert pf.off_tree_sizes(dict(open_call, actions=[act("flop", 2, "bet", 2.2)]), "flop", players, req)[0] == {}
raised = dict(open_call, actions=[act("flop", 2, "bet", 2.2), act("flop", 0, "raise", 6.0)])
ex, _ = pf.off_tree_sizes(raised, "flop", players, req)
# BB 22 into 65; the BTN's 60 = 22 to call + 38 more, of a 109 pot after calling: 34.9% (the menu's 60% is 87.4)
assert ex == {(1, "raise"): ["34.9"]}, ex
jam = dict(open_call, actions=[act("flop", 2, "allin", 22.5)])
assert pf.off_tree_sizes(jam, "flop", players, req)[0] == {(0, "bet"): ["a"]}
sp_r = pf.build_spot(dict(open_call, actions=open_call["actions"] + [act("flop", 2, "bet", 1.6)]), rng, None)[0]
assert sp_r["request"]["oop"][0]["bet"] == "33 75 24.6" and sp_r["request_small"]["oop"][0]["bet"] == "33 24.6"
assert sp_r["request"]["ip"][0]["bet"] == "33 75" and sp_r["meta"]["resized"]
assert any("not in the bet menu" in n for n in sp_r["meta"]["notes"])
assert pf.job_key(open_call, sp_r) != pf.job_key(open_call, spot)
# the turn: ranges carried combo by combo, the turn's pot and stacks, a richer menu with leads
turn = dict(open_call, street="turn", board=["Td", "7c", "2h", "Qs"],
            start=dict(open_call["start"], turn={"pot": 10.9, "dead": None, "stacks": {0: 35.3, 2: 20.3}}))
carry = {"hands": {2: {"QhQd": 0.8, "7s7d": 0.4}, 0: {"AhKd": 0.0, "AsAd": 0.9}}, "notes": ["kept from the flop"]}
tsp, why = pf.build_spot(turn, rng, None, "turn", carry)
assert why is None and tsp["request"]["board"] == "Td7c2hQs", why
assert tsp["request"]["range_oop"] == "QhQd:0.80000,7s7d:0.40000"
assert "AsAd:0.90000" in tsp["request"]["range_ip"] and "AhKd:0.01800" in tsp["request"]["range_ip"]  # floored
assert tsp["request"]["starting_pot"] == 109.0 and tsp["request"]["effective_stack"] == 203.0
assert tsp["request"]["max_raises"] == 4 and tsp["request_small"] is None
assert tsp["request"]["oop"][2]["donk"] == "33 75 a" and tsp["request"]["ip"][2]["donk"] == ""
assert "kept from the flop" in tsp["meta"]["notes"] and any("rarely" in n for n in tsp["meta"]["notes"])
assert tsp["meta"]["street"] == "turn" and pf.job_key(turn, tsp)[1] == "turn"
print("ranges and flop spot OK")


# ---------------------------------------------------------------- 3. walking a (fake) solved tree
def A(kind, amount=0.0, label=None):
    return {"kind": kind, "amount": amount, "label": label or kind.title()}


def node(player, actions, pot=65.0):
    return {"node_type": "action", "player": player, "pot": pot, "put": [32.5, 32.5], "actions": actions,
            "players": [{"hands": []}, {"hands": []}]}


TREE = {   # chips; BB (player 0) acts first
    (): node(0, [A("check"), A("bet", 21.45), A("bet", 48.75)]),
    (0,): node(1, [A("check"), A("bet", 21.45), A("bet", 48.75)]),
    (1,): node(1, [A("fold"), A("call", 21.45), A("raise", 77.0)], pot=86.45),
    (2,): node(1, [A("fold"), A("call", 48.75), A("raise", 150.0)], pot=113.75),
    (0, 0): {"node_type": "chance", "player": None, "pot": 65.0, "actions": []},
    (0, 1): node(0, [A("fold"), A("call", 21.45), A("raise", 77.0)], pot=86.45),
}


def get(path):
    return TREE[tuple(s["index"] for s in path)]


players = {2: 0, 0: 1}
v, path, notes = pf.walk(get, [], players, 1, hero_turn=False)
assert v["player"] == 0                                      # BB to act: waiting
v, path, notes = pf.walk(get, [], players, 1, hero_turn=True)
assert [s["index"] for s in path] == [0] and v["player"] == 1   # the hero's buttons are up: BB checked unseen
v, path, notes = pf.walk(get, [act("flop", 2, "bet", 1.6)], players, 1, hero_turn=True)
assert [s["index"] for s in path] == [1] and v["player"] == 1
assert "nearest size" in notes[0] and "25% pot" in notes[0], notes   # 1.6bb into 6.5bb: off the 33% size
v, path, notes = pf.walk(get, [act("flop", 2, "bet", 2.2)], players, 1, hero_turn=True)
assert not notes                                             # 2.2bb is the 33% bet (2.145bb) within 15%
v, path, notes = pf.walk(get, [act("flop", 0, "bet", 2.1)], players, 0, hero_turn=False)
assert [s["index"] for s in path] == [0, 1] and v["player"] == 0   # BTN bet: the BB's check was never logged
v, path, notes = pf.walk(get, [act("flop", 2, "allin", 22.5)], players, 1, hero_turn=True)
assert [s["index"] for s in path] == [2] and v["player"] == 1          # the jam: the tree's biggest bet
trail = []
v, path, why = pf.walk(get, [act("flop", 2, "check"), act("flop", 0, "check")], players, 1, hero_turn=False,
                       trail=trail)
assert v["node_type"] == "chance" and [t["node_type"] for t in trail] == ["action", "action", "chance"]
print("tree walk OK")

# ---------------------------------------------------------------- 4. the hero's combo at a node
hv = node(1, [A("check"), A("bet", 21.5, "Bet 21.5 (33%)"), A("bet", 225.0, "All-in 225")])
hv["players"][1]["hands"] = [
    {"combo": "AhKd", "c1": pf.card_index("Ah"), "c2": pf.card_index("Kd"), "weight": 1.0, "reach": 1.0, "eq": 0.61,
     "ev": 40.0, "strategy": [0.25, 0.75, 0.0], "evs": [39.0, 41.0, 30.0]},
    {"combo": "7s7c", "c1": pf.card_index("7c"), "c2": pf.card_index("7s"), "weight": 1.0, "reach": 1.0, "eq": 0.9,
     "ev": 50.0, "strategy": [1.0, 0.0, 0.0], "evs": [50.0, 49.0, 45.0]}]
adv = pf.hero_advice(hv, 1, "KdAh")                          # card order doesn't matter
assert [r["label"] for r in adv["actions"]] == ["Check", "Bet 2.15 (33%)", "All-in 22.5"], adv
assert [r["freq"] for r in adv["actions"]] == [0.25, 0.75, 0.0]
assert [r["ev_loss"] for r in adv["actions"]] == [0.2, 0.0, 1.1] and adv["equity"] == 0.61
assert [r["range_freq"] for r in adv["actions"]] == [0.625, 0.375, 0.0]
assert pf.card_index("2c") == 0 and pf.card_index("As") == 51
print("hero advice OK")


# ---------------------------------------------------------------- 5. the engine, on a fake gto-server
class FakeServer:
    def __init__(self, tree, solve_polls=3):
        self.tree, self.calls, self.polls, self.solve_polls, self.built = tree, [], 0, solve_polls, None

    def __call__(self, path, body=None):
        self.calls.append(path)
        if path == "/api/spot":
            self.built = body
            return {"nodes": 1}
        if path == "/api/solve":
            self.polls = 0
            return {"ok": True}
        if path == "/api/status":
            self.polls += 1
            running = self.polls < self.solve_polls
            return {"state": "running" if running else "done", "iteration": 10 * self.polls,
                    "exploit_pct": 2.0 / self.polls, "gpu": True}
        if path == "/api/node":
            return self.tree.get(tuple(s["index"] for s in body["path"]),
                                 {"node_type": "terminal", "player": None, "pot": 0, "actions": []})
        if path == "/api/stop":
            return {}
        raise AssertionError(path)


def until(fn, secs=10):
    t0 = time.time()
    while time.time() - t0 < secs:
        r = fn()
        if r:
            return r
        time.sleep(0.02)
    raise AssertionError("timed out")


tree = dict(TREE)
tree[(1,)] = dict(TREE[(1,)])
tree[(1,)]["players"] = [{"hands": []}, {"hands": [dict(hv["players"][1]["hands"][0], strategy=[0.1, 0.6, 0.3],
                                                         evs=[1.0, 2.0, 3.0])]}]
fake = FakeServer(tree)
eng = pf.PostflopEngine(fake_lookup([]), api=fake)
h = dict(open_call, hero_turn=False)
first = eng.advise(h, None)["postflop"]
assert first["status"] == "solving", first
until(lambda: eng.advise(h, None)["postflop"]["status"] == "waiting")    # BB to act
assert fake.built["board"] == "Td7c2h" and fake.calls.count("/api/spot") == 1
h2 = dict(open_call, actions=open_call["actions"] + [act("flop", 2, "bet", 2.2)], hero_turn=True)
res = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng.advise(h2, None)))
p = res["postflop"]
assert p["advice"]["hand"] == "AhKd" and [r["freq"] for r in p["advice"]["actions"]] == [0.1, 0.6, 0.3], p
assert res["title"].startswith("Flop · BTN vs BB") and p["line"] == "BTN open, BB call"
assert fake.calls.count("/api/spot") == 1 and fake.calls.count("/api/solve") >= 2   # rough, then refined
v0 = eng.version
# a new hand: the old solve is dropped and the new spot built
h3 = dict(open_call, id=2, board=["As", "8s", "3d"])
until(lambda: eng.advise(h3, None)["postflop"]["status"] in ("waiting", "ready"))
assert fake.calls.count("/api/spot") == 2 and fake.built["board"] == "As8s3d" and eng.version > v0
# after the solve, the flop's nodes are read ahead without the walk asking (so the hero's turn doesn't wait)
fake3 = FakeServer(TREE)
eng3 = pf.PostflopEngine(fake_lookup([]), api=fake3)
eng3.advise(open_call, None)
until(lambda: eng3._phase == "done" and len(eng3._views) >= 6)
assert any(k[2] == (("action", 0), ("action", 1)) for k in eng3._views)     # check, bet: two actions deep
# a big tree is rebuilt with the small bet menu, and said so
class BigFake(FakeServer):
    def __call__(self, path, body=None):
        r = super().__call__(path, body)
        return {"nodes": 900000, "vram_mb": 15000} if path == "/api/spot" else r


fake4 = BigFake(tree)
eng4 = pf.PostflopEngine(fake_lookup([]), api=fake4)
c = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng4.advise(h2, None)))
assert fake4.calls.count("/api/spot") == 2 and fake4.built["max_raises"] == 2, fake4.built
assert fake4.built["oop"][0]["bet"] == "33" and fake4.built["oop"][1]["raise"] == "a"
assert any("smaller bet menu" in n for n in c["postflop"]["notes"])
# preflop charts still being solved by the HUD's solver: pending (re-advised), not a dead end
eng2 = pf.PostflopEngine(lambda *a: (None, "solving 6-handed 25bb (first time only)…"), api=FakeServer(TREE))
c = eng2.advise(open_call, None)
assert c["postflop"]["status"] == "solving" and c["pending_solver"] and "preflop charts" in c["postflop"]["message"]
# unsupported spots never start a solve
before = len(fake.calls)
assert eng.advise(dict(open_call, in_hand=[0, 2, 7]), None)["postflop"]["status"] == "unsupported"
time.sleep(0.1)
assert len(fake.calls) == before

# ---- the turn: the flop's end-of-betting ranges carried into a turn solve
def H(combo, reach, **kw):
    return dict({"combo": combo, "c1": pf.card_index(combo[2:4]), "c2": pf.card_index(combo[0:2]),
                 "weight": 1.0, "reach": reach}, **kw)


ctree = dict(tree)
ctree[()] = dict(TREE[()], players=[{"hands": [H("QhQd", 1.0), H("7s7d", 1.0)]},
                                    {"hands": [H("AhKd", 1.0), H("AsAd", 1.0)]}])
ctree[(0,)] = dict(TREE[(0,)], players=[{"hands": [H("QhQd", 0.5), H("7s7d", 0.9)]},
                                       {"hands": [H("AhKd", 1.0), H("AsAd", 1.0)]}])
ctree[(0, 0)] = {"node_type": "chance", "player": None, "pot": 65.0, "actions": [],
                 "players": [{"hands": [H("QhQd", 0.5), H("7s7d", 0.9)]},
                             {"hands": [H("AhKd", 0.6), H("AsAd", 0.25)]}]}
# the turn tree (any spot gets this fake): the hero (IP) acts after the BB's check
ctree[(0,)]["players"][1]["hands"][0].update(strategy=[0.3, 0.5, 0.2], evs=[1.0, 2.0, 3.0])
fake5 = FakeServer(ctree)
eng5 = pf.PostflopEngine(fake_lookup([]), api=fake5)
flop_cc = open_call["actions"] + [act("flop", 2, "check"), act("flop", 0, "check")]
turn_h = dict(open_call, street="turn", board=["Td", "7c", "2h", "Qs"], actions=flop_cc, hero_turn=True,
              start=dict(open_call["start"], turn={"pot": 6.5, "dead": None, "stacks": {0: 37.5, 2: 22.5}}))
first = eng5.advise(turn_h, None)["postflop"]
assert first["status"] == "solving" and "ended with" in first["message"], first      # the flop is solved first
res = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng5.advise(turn_h, None)))
assert fake5.calls.count("/api/spot") == 2 and fake5.built["board"] == "Td7c2hQs", fake5.built["board"]
assert fake5.built["range_oop"] == "QhQd:0.50000,7s7d:0.90000"                 # the BB's reach after its check
assert fake5.built["range_ip"] == "AhKd:0.60000,AsAd:0.25000", fake5.built["range_ip"]
assert res["title"].startswith("Turn · BTN vs BB") and "narrowed" in res["postflop"]["ranges"]
assert [r["freq"] for r in res["postflop"]["advice"]["actions"]] == [0.3, 0.5, 0.2]
# a line the solver almost never takes (the BB checks 1% of the time): its range from before is kept
ctree2 = dict(ctree)
ctree2[(0,)] = dict(ctree[(0,)], players=[{"hands": [H("QhQd", 0.01), H("7s7d", 0.01)]},
                                         ctree[(0,)]["players"][1]])
ctree2[(0, 0)] = dict(ctree[(0, 0)], players=[{"hands": [H("QhQd", 0.01), H("7s7d", 0.01)]},
                                             ctree[(0, 0)]["players"][1]])
fake6 = FakeServer(ctree2)
eng6 = pf.PostflopEngine(fake_lookup([]), api=fake6)
res = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng6.advise(turn_h, None)))
assert fake6.built["range_oop"] == "QhQd:1.00000,7s7d:1.00000", fake6.built["range_oop"]
assert any("almost never takes" in n and "BB" in n for n in res["postflop"]["notes"]), res["postflop"]["notes"]
# ---- a bet the menu doesn't have: the street is re-solved with it; the old tree's answer shows meanwhile
fake7 = FakeServer(tree)
eng7 = pf.PostflopEngine(fake_lookup([]), api=fake7)
until(lambda: eng7.advise(dict(open_call, hero_turn=False), None)["postflop"]["status"] == "waiting")
until(lambda: eng7._phase == "done" and len(eng7._views) >= 4)                  # the flop read ahead
fake7.solve_polls = 40                                                          # a slow re-solve
small_bet = dict(open_call, actions=open_call["actions"] + [act("flop", 2, "bet", 1.6)], hero_turn=True)
mid = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng7.advise(small_bet, None)))
assert mid["postflop"]["refining"] and "Re-solving" in mid["postflop"]["message"], mid["postflop"]
assert any("nearest size" in n for n in mid["postflop"]["notes"])
until(lambda: fake7.built["oop"][0]["bet"] == "33 75 24.6")                   # rebuilt with the real size
done = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" and not c["postflop"]["message"] else None)(
    eng7.advise(small_bet, None)), secs=20)
assert any("not in the bet menu" in n for n in done["postflop"]["notes"])
print("background engine OK")

# ---------------------------------------------------------------- 5b. 3-way pots
from preflop_hud import multiway as mw  # noqa: E402

pre3 = [act("preflop", 0, "raise", 2.5), act("preflop", 1, "call", 2.5), act("preflop", 2, "call", 2.5)]
st3 = {"preflop": {"pot": None, "dead": 1.0, "stacks": {0: 40.0, 1: 30.0, 2: 25.0}},
       "flop": {"pot": 8.5, "dead": None, "stacks": {0: 37.5, 1: 27.5, 2: 22.5}},
       "turn": {"pot": 8.5, "dead": None, "stacks": {0: 37.5, 1: 27.5, 2: 22.5}},
       "river": {"pot": 8.5, "dead": None, "stacks": {0: 37.5, 1: 27.5, 2: 22.5}}}
three = hand(pre3, in_hand=(0, 1, 2), start=st3)
plan, line = mw.multiway_plan(three)
assert plan == {"BTN": ("RFI:BTN", "raise"), "SB": ("VS_OPEN:SB:BTN", "call"),
                "BB": ("VS_OPEN:BB:BTN", "call", "BB called after SB had called: its range is its call against "
                       "a lone BTN raise")}, plan
limp3 = hand([act("preflop", 3, "call", 1.0), act("preflop", 0, "fold"), act("preflop", 1, "call", 1.0),
              act("preflop", 2, "check")], in_hand=(1, 2, 3), hero="9s8s",
             actions=[act("preflop", s, "fold") for s in (4, 5, 6, 7)] + [
                 act("preflop", 3, "call", 1.0), act("preflop", 0, "fold"), act("preflop", 1, "call", 1.0),
                 act("preflop", 2, "check")])
plan, line = mw.multiway_plan(limp3)
assert plan["UTG"][:2] == ("RFI:UTG", "raise") and plan["SB"] == ("RFI:SB", "call") and plan["BB"][:2] == ("BB_VS_LIMP", "check")
assert mw.multiway_plan(hand(pre3 + [act("preflop", 0, "raise", 9.0)], in_hand=(0, 1, 2)))[0] is None   # 3-bet
assert mw.players_at(dict(three, actions=three["actions"] + [act("flop", 2, "fold")], in_hand=[0, 1]), "flop") == [0, 1, 2]
cc = [act("flop", 1, "check"), act("flop", 2, "check"), act("flop", 0, "check")]
assert mw.checked_through(dict(three, actions=three["actions"] + cc), "flop")
assert not mw.checked_through(dict(three, actions=three["actions"] + cc[:2]), "flop")
rng3, why = pf.build_ranges(three, fake_lookup([]), None, multiway=True)
assert why is None and set(rng3["weights"]) == {"BTN", "SB", "BB"} and any("lone BTN" in n for n in rng3["notes"])
# the lean 3-way turn: action order SB, BB, BTN; per-seat stacks; one shared bet size
turn3 = dict(three, street="turn", board=["Td", "7c", "2h", "Qs"], actions=three["actions"] + cc)
sp3, why = mw.build_mw_spot(turn3, rng3, None, "turn")
r3 = sp3["request"]
assert why is None and r3["board"] == "Td7c2hQs" and r3["stacks"] == [275.0, 225.0, 375.0] and r3["starting_pot"] == 85.0
assert r3["sizes"][1] == {"bet": "75", "raise": "a"} and r3["sizes"][2] == {"bet": "75", "raise": "a"}  # lean river inside
river_sp = mw.build_mw_spot(dict(turn3, street="river", board=turn3["board"] + ["3c"]), rng3, None, "river")[0]
assert river_sp["request"]["sizes"][2] == {"bet": "33 75 a", "raise": "a"}                       # a river root: richer
assert sp3["meta"]["players"] == {1: 0, 2: 1, 0: 2} and sp3["meta"]["hero_player"] == 2 and sp3["meta"]["kind"] == "mw"
# a 25% turn bet: swapped in for the 75% (a second size would make the tree 3-5x slower); a raise keeps all-in
bet25 = dict(turn3, actions=turn3["actions"] + [act("turn", 1, "bet", 2.12)])
sp_b = mw.build_mw_spot(bet25, rng3, None, "turn")[0]
assert sp_b["request"]["sizes"][1]["bet"] == "24.9" and any("in place of" in n for n in sp_b["meta"]["notes"])
assert pf.job_key(bet25, sp_b) != pf.job_key(turn3, sp3)
# the 3-way view in the heads-up shape: the street total, the actor's hands with reach
raw = {"kind": "action", "street": 1, "pot": 120.0, "put": [40.0, 40.0, 40.0 - 11.67], "actor": 2,
       "actions": [{"label": "Fold", "kind": "fold", "amount": 0, "freq": 0.1},
                   {"label": "Call 35", "kind": "call", "amount": 35.0, "freq": 0.9}],
       "hands": [{"combo": "AhKd", "weight": 0.4, "strategy": [0.25, 0.75], "ev": 12.0}], "range_combos": [1, 2, 3]}
v = mw.mw_view(raw, 28.33)
assert v["node_type"] == "action" and v["player"] == 2 and v["players"][2]["hands"][0]["reach"] == 0.4
assert abs(v["actions"][1]["amount"] - 35.0) < 0.01          # 28.33 in before the street + 35 = the call to 35
assert mw.mw_path([{"type": "action", "index": 1}, {"type": "card", "card": "3c"}]) == [1, "3c"]
print("3-way ranges, spots and views OK")


def M(actor, acts, hands=(), street=1, pot=85.0, rc=(100.0, 100.0, 100.0)):
    return {"kind": "action", "street": street, "pot": pot, "put": [pot / 3] * 3, "actor": actor,
            "actions": [{"label": k.title(), "kind": k, "amount": amt, "freq": 0.5} for k, amt in acts],
            "hands": [{"combo": c, "weight": w, "strategy": st, "ev": 1.0} for c, w, st in hands],
            "range_combos": list(rc)}


CHK = [("check", 0.0), ("bet", 63.75)]
CH = {"kind": "chance", "street": 1, "pot": 85.0, "put": [28.3] * 3, "actor": None, "actions": [], "hands": [],
      "range_combos": [100.0, 100.0, 100.0]}
MWTREE = {   # any 3-way spot gets this tree: the flop estimator, a turn, a river
    (): M(0, CHK), (0,): M(1, CHK),
    (0, 0): M(2, CHK + [("bet", 375.0)], [("AhKd", 1.0, [0.2, 0.7, 0.1])]),
    (0, 0, 0): CH,
    # the estimator (checked flop): each player's reach where they first act on the turn
    (0, 0, 0, "Qs"): M(0, [("check", 0)], [("QhQd", 0.7, [1.0])]),
    (0, 0, 0, "Qs", 0): M(1, [("check", 0)], [("7s7d", 0.6, [1.0])]),
    (0, 0, 0, "Qs", 0, 0): M(2, [("check", 0)], [("AhKd", 0.95, [1.0]), ("AsAd", 0.2, [1.0])]),
    # the turn tree, checked through: the river
    (0, 0, 0, "3c"): M(0, CHK, [("QhQd", 0.5, [1.0, 0.0])], street=2),
    (0, 0, 0, "3c", 0): M(1, CHK, [("7s7d", 0.4, [1.0, 0.0])], street=2),
    (0, 0, 0, "3c", 0, 0): M(2, CHK, [("AhKd", 0.9, [1.0, 0.0]), ("AsAd", 0.3, [1.0, 0.0])], street=2),
    # the SB bets, the BB folds, the BTN calls: heads-up next street (SB, then the BTN)
    (1,): M(1, [("fold", 0), ("call", 63.75)]), (1, 0): M(2, [("fold", 0), ("call", 63.75)]),
    (1, 0, 1): dict(CH, pot=212.5, put=[92, 28.3, 92], range_combos=[100.0, 0.0, 100.0]),
    (1, 0, 1, "Qs"): M(0, [("check", 0)], [("QhQd", 0.65, [1.0])]),
    (1, 0, 1, "Qs", 0): M(2, [("check", 0)], [("AhKd", 0.85, [1.0])]),
    (1, 0, 1, "3c"): M(0, CHK, [("QhQd", 0.6, [1.0, 0.0])], street=2),
    (1, 0, 1, "3c", 0): M(2, CHK, [("AhKd", 0.8, [1.0, 0.0])], street=2),
}


class MwFake(FakeServer):
    def __init__(self):
        super().__init__(tree)
        self.mw_built, self.mw_builds = None, []

    def __call__(self, path, body=None):
        if path.startswith("/api/mw/"):
            self.calls.append(path)
            sub = path[len("/api/mw"):]
            if sub == "/spot":
                self.mw_built = body
                self.mw_builds.append(body)
                return {"nodes": 1}
            if sub == "/node":
                assert body.get("ev") is False                       # EVs off: they're computed on the CPU
                return MWTREE.get(tuple(body["path"]), {"kind": "showdown", "street": 2, "pot": 0, "put": [0, 0, 0],
                                                        "actor": None, "actions": [], "hands": []})
            if sub == "/status":
                self.polls += 1
                return {"state": "running" if self.polls < self.solve_polls else "done", "iteration": 10 * self.polls,
                        "exploit_pct": 2.0 / self.polls, "engine": "gpu", "error": ""}
            return FakeServer.__call__(self, "/api" + sub, body)                 # /solve and /stop
        return FakeServer.__call__(self, path, body)


# ---- the default (the estimator off: heads-up tests showed it worse than no narrowing)
assert not mw.ESTIMATOR_LIVE
fk0 = MwFake()
eng0 = pf.PostflopEngine(fake_lookup([]), api=fk0)
assert eng0.advise(three, None)["postflop"]["status"] == "equity"
time.sleep(0.2)
assert not fk0.calls                                                   # nothing solved on a 3-way flop
bet_turn = dict(turn3, actions=three["actions"] + [act("flop", 1, "bet", 4.0), act("flop", 2, "call", 4.0),
                                                   act("flop", 0, "call", 4.0)])
p = eng0.advise(bet_turn, None)["postflop"]
assert p["status"] == "equity" and "was bet" in p["message"] and p["equity"]["equity"] is not None, p
turn_ready = dict(turn3, actions=turn3["actions"] + [act("turn", 1, "check"), act("turn", 2, "check")], hero_turn=True)
res = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng0.advise(turn_ready, None)))
assert fk0.mw_builds[0]["board"] == "Td7c2hQs" and len(fk0.mw_builds) == 1             # the turn, nothing before it
assert all(len(t.split(":")[0]) <= 3 for r in fk0.mw_built["ranges"] for t in r.split(",")), fk0.mw_built["ranges"]  # classes: preflop
assert any("checked through" in n for n in res["postflop"]["notes"])
river0 = dict(turn3, street="river", board=turn3["board"] + ["3c"], hero_turn=False,
              actions=turn3["actions"] + [act("turn", 1, "check"), act("turn", 2, "check"), act("turn", 0, "check")])
until(lambda: eng0.advise(river0, None)["postflop"]["status"] == "waiting")
assert fk0.mw_built["ranges"] == ["QhQd:0.50000", "7s7d:0.40000", "AhKd:0.90000,AsAd:0.30000"], fk0.mw_built["ranges"]

# ---- the estimator (mw.ESTIMATOR_LIVE): kept for validate_multiway and a later decision
mw.ESTIMATOR_LIVE = True
fk = MwFake()
eng8 = pf.PostflopEngine(fake_lookup([]), api=fk)
# the flop: your equity; meanwhile the reduced 3-way flop tree (33% bet, turn and river checked down) is solved
c = eng8.advise(three, None)
p = c["postflop"]
assert p["status"] == "equity" and "too big" in p["message"] and 0 < p["equity"]["equity"] < 1, p
assert p["equity"]["players"] == 2 and c["title"].startswith("Flop · 3-way · BTN vs SB and BB"), c["title"]
assert any("preflop ranges" in n for n in p["notes"])
until(lambda: fk.mw_builds)
est = fk.mw_builds[0]
assert est["board"] == "Td7c2h" and est["sizes"][0] == {"bet": "33", "raise": ""}, est["sizes"]
assert est["sizes"][1] == {"bet": "", "raise": ""} and est["sizes"][2] == {"bet": "", "raise": ""}
# a 4bb bet (47% of the pot): the estimator takes that size instead; you face it: the pot odds
bet = dict(three, actions=three["actions"] + [act("flop", 1, "bet", 4.0)], hero_turn=True)
e = eng8.advise(bet, None)["postflop"]["equity"]
assert e["to_call"] == 4.0 and e["pot"] == 12.5 and e["need"] == round(4.0 / 16.5, 3), e
assert eng8.advise(bet, None)["postflop"]["equity"]["equity"] == e["equity"]       # cached: holds still
until(lambda: fk.mw_built["sizes"][0]["bet"] == "47.1")
# a checked flop: the turn is solved 3-way from the estimator's ranges; the hero acts after two checks
turn_ready = dict(turn3, actions=turn3["actions"] + [act("turn", 1, "check"), act("turn", 2, "check")], hero_turn=True)
res = until(lambda: (lambda c: c if c["postflop"]["status"] == "ready" else None)(eng8.advise(turn_ready, None)))
p = res["postflop"]
assert fk.mw_built["board"] == "Td7c2hQs" and "/api/spot" not in fk.calls
assert fk.mw_built["ranges"] == ["QhQd:0.70000", "7s7d:0.60000", "AhKd:0.95000,AsAd:0.20000"], fk.mw_built["ranges"]
assert [r["freq"] for r in p["advice"]["actions"]] == [0.2, 0.7, 0.1] and p["advice"]["equity"] == p["equity"]["equity"]
assert res["title"].startswith("Turn · 3-way") and "3-way solver" in res["method"]
assert any("reduced flop solve" in n for n in p["notes"]) and p["equity"]["narrowed"]
# the turn checked through too: the river is solved 3-way from the ranges the turn ended with
river3 = dict(turn3, street="river", board=turn3["board"] + ["3c"], hero_turn=False,
              actions=turn3["actions"] + [act("turn", 1, "check"), act("turn", 2, "check"), act("turn", 0, "check")])
until(lambda: eng8.advise(river3, None)["postflop"]["status"] == "waiting")
assert fk.mw_built["board"] == "Td7c2hQs3c", fk.mw_built["board"]
assert fk.mw_built["ranges"] == ["QhQd:0.50000", "7s7d:0.40000", "AhKd:0.90000,AsAd:0.30000"], fk.mw_built["ranges"]
# the BB folds the turn: the river is heads-up, from the 3-way turn's ranges
fold_r = dict(river3, in_hand=[0, 1], start=dict(st3, river={"pot": 21.25, "dead": None, "stacks": {0: 31.1, 1: 21.1}}),
              actions=turn3["actions"] + [act("turn", 1, "bet", 6.38), act("turn", 2, "fold"), act("turn", 0, "call", 6.38)])
until(lambda: eng8.advise(fold_r, None)["postflop"]["status"] in ("waiting", "ready"))
assert fk.built["board"] == "Td7c2hQs3c" and fk.built["range_oop"] == "QhQd:0.60000", fk.built
assert fk.built["range_ip"] == "AhKd:0.80000"
# a bet flop the BB folds: the turn is heads-up, from the estimator's ranges
fold_f = dict(turn3, in_hand=[0, 1], hero_turn=False,
              actions=three["actions"] + [act("flop", 1, "bet", 4.0), act("flop", 2, "fold"), act("flop", 0, "call", 4.0)])
until(lambda: eng8.advise(fold_f, None)["postflop"]["status"] in ("waiting", "ready"))
assert fk.built["board"] == "Td7c2hQs" and fk.built["range_oop"] == "QhQd:0.65000", fk.built
assert fk.built["range_ip"] == "AhKd:0.85000"
assert any(b["sizes"][0]["bet"] == "47.1" and b["board"] == "Td7c2h" for b in fk.mw_builds)   # the estimator, with the bet
mw.ESTIMATOR_LIVE = False
print("3-way engine OK")

# ---------------------------------------------------------------- 6. recorded hands
FIX = os.path.join(ROOT, "tests", "fixtures")


def flop_states(name):
    with open(os.path.join(FIX, name)) as f:
        rec = json.load(f)
    tr, out = Tracker(), []
    for obs in rec["frames"]:
        st = tr.update(obs)
        if st and st.get("street") == "flop" and not st.get("finished"):
            out.append(st)
    return out


calls = []
# K♠J♠: LJ open, BB call; the flop's pot and stacks come from the reads; LJ is in position
states = flop_states("rec_ksjs_flop_bet_check_down.json")
h0 = states[0]
spot, why = pf.build_spot(h0, pf.build_ranges(h0, fake_lookup(calls), None)[0], None)
assert why is None and spot["request"]["starting_pot"] == 75.0 and spot["request"]["effective_stack"] == 179.0, spot
assert spot["meta"]["hero_player"] == 1 and calls[0][1] == 20.9, calls      # BB's 20.9bb start stack is the depth
assert spot["request"]["board"] == "KdAd5c" and pf.check_hand(h0) is None
# the LJ's 25% bet after the BB's check, then the BB's call
last = states[-1]
fl = [(a["pos"], a["action"], a["to"]) for a in last["actions"] if a["street"] == "flop"]
assert fl == [("BB", "check", None), ("LJ", "bet", 1.88), ("BB", "call", 1.88)], fl
# K♠J♥ SB limp, BB check: the hero (SB) is out of position
states = flop_states("rec_kjh_sb_limp_checks_river_bet_call.json")
plan, line, _ = pf.preflop_plan(states[0])
assert line == "SB limp, BB check" and pf.check_hand(states[0]) is None
sp = pf.build_spot(states[0], pf.build_ranges(states[0], fake_lookup([]), None)[0], None)[0]
assert sp["meta"]["hero_player"] == 0 and sp["request"]["starting_pot"] == 30.0
# tracking started on the flop: no ranges, said so
st = flop_states("rec_qcth_flop_turn_calls_river_jam_fold.json")[0]
assert pf.build_ranges(st, fake_lookup([]), None)[0] is None
assert "late" in pf.build_ranges(st, fake_lookup([]), None)[1]
# three all-ins preflop: no decisions on the flop
st = flop_states("rec_ks9s_call_allins_late_start.json")
assert all(pf.check_hand(s) for s in st)
# UTG limp, LJ (hero) iso-raise, UTG call: assumed open ranges, said so (not read as open + call)
st = flop_states("rec_9h9c_utg_limp_turn_raise_river_bet.json")[0]
plan, line, notes = pf.preflop_plan(st)
assert plan == {"UTG": ("RFI:UTG", "raise"), "LJ": ("RFI:LJ", "raise")} and line == "UTG limp, LJ raise, UTG call"
assert len(notes) == 2, notes
print("recorded hands OK")

# ---------------------------------------------------------------- 7. a real GTOpen solve (opt-in)
if os.environ.get("HUD_TEST_GTOPEN") == "1":
    from preflop_hud.solved import SolvedCharts
    eng = pf.PostflopEngine(SolvedCharts().node, gtopen_dir=json.load(open(os.path.join(ROOT, "config.json")))
                            .get("gtopen_dir"))
    states = flop_states("rec_ksjs_flop_bet_check_down.json")
    h = next(s for s in states if [a for a in s["actions"] if a["street"] == "flop"])   # after the BB's check
    res = until(lambda: (lambda c: c if c["postflop"]["status"] in ("ready", "error") else None)(eng.advise(h, None)),
                secs=120)
    p = res["postflop"]
    assert p["status"] == "ready", p
    print("real GTOpen solve OK:", ", ".join(f"{r['label']} {r['freq']:.0%}" for r in p["advice"]["actions"]),
          "|", p["progress"])
    # the whole chain on the 9h9c hand: flop solve -> its end ranges -> the turn (re-solved with the 1bb lead
    # and the 2.5 raise) -> its end ranges -> the river decision after UTG's check
    with open(os.path.join(FIX, "rec_9h9c_utg_limp_turn_raise_river_bet.json")) as f:
        rec = json.load(f)
    tr = Tracker()
    river = None
    for obs in rec["frames"]:
        st = tr.update(obs)
        if st and st["street"] == "river" and st["hero_turn"] and not st.get("finished"):
            river = st
            break
    res = until(lambda: (lambda c: c if c["postflop"]["status"] in ("ready", "error", "unsupported") else None)(
        eng.advise(river, None)), secs=120)
    p = res["postflop"]
    assert p["status"] == "ready" and res["title"].startswith("River"), (res["title"], p)
    assert any("1bb (12% of the pot)" in n for n in p["notes"]) is False      # the turn's notes stay on the turn
    print("real GTOpen river OK:", ", ".join(f"{r['label']} {r['freq']:.0%}" for r in p["advice"]["actions"]))
    eng.close()
print("all tests passed")
