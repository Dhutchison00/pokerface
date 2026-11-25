"""Hand tracker: rebuilds the action of a hand from successive screen reads.

A single frame shows bets in front of seats, stacks, labels (FOLD, CHECK, ALL-IN...), the board
and the pot, but not what happened earlier. The tracker compares frames and turns the changes
into actions, street by street:

  * a new hand starts when the hero's hole cards change, or the board clears
  * a seat's bet growing is a call (up to the current bet), a bet (first chips on a street
    postflop) or a raise (above it); all-in when the seat is all-in
  * FOLD / CHECK labels appearing, or a seat's cards disappearing, are folds and checks
  * the board growing closes the street: bets are swept into the pot, so anyone still in who
    hadn't matched the bet must have called, and on an unbet street everyone checked

Bots can act faster than the capture rate. Actions seen together in one frame are put in seat
order, and actions only deduced when a street closes are marked "inferred". Lines that can't be
told apart from the frames are flagged instead of guessed.

    python -m preflop_hud.tracker replay recordings/<folder>          # replay recorded reads
    python -m preflop_hud.tracker replay recordings/<folder> --reread # re-read the saved frames first
"""
from __future__ import annotations

import json
import os
import sys
import time

EPS = 0.05
READS_KEPT = 15                  # street-start pot/stack reads kept for the median
STREETS = ["preflop", "flop", "turn", "river"]
POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG1", "UTG2", "UTG3", "LJ", "HJ", "CO", "BTN"]


# ---------------------------------------------------------------- observations
REAL_BUTTONS = ("BET", "RAISE", "CALL", "ALL-IN", "ALLIN")


def hero_turn(st) -> bool:
    """The hero is to act: the hero's seat is highlighted, or real action buttons are up (BET / CALL /
    RAISE). Pre-action checkboxes shown while others act ("FOLD/CHECK", "CHECK", "CALL ANY") don't count."""
    if any(s.idx == 0 and s.acting for s in st.seats):
        return True
    btns = [str(b).upper() for b in (getattr(st, "hero_buttons", None) or [])]
    if any("/" in b or "ANY" in b for b in btns):
        return False
    return any(b.startswith(REAL_BUTTONS) for b in btns)


def observe(st, spot=None, t=None) -> dict:
    """The parts of one screen read the tracker uses (JSON-friendly, so recordings can replay)."""
    hc = st.hero_cards or {}
    hero = None
    if hc.get("complete") and hc.get("ranks"):
        suits = hc.get("suits") or [None, None]
        hero = "".join(f"{r}{s or '?'}" for r, s in zip(hc["ranks"], suits))
    return {
        "t": time.time() if t is None else t, "ok": bool(st.ok),
        "hero": hero, "pot": st.pot,
        "hero_turn": hero_turn(st),
        "board": [f"{c.get('rank') or '?'}{c.get('suit') or '?'}" for c in (st.board or [])],
        "positions": (spot or {}).get("positions") or {},
        "seats": [{"idx": s.idx, "present": s.present, "stack": s.stack, "bet": round(s.bet or 0.0, 3),
                   "labels": list(s.labels), "has_cards": bool(s.has_cards), "all_in": bool(s.all_in)}
                  for s in st.seats],
    }


# ---------------------------------------------------------------- the tracker
class Hand:
    def __init__(self, obs, hand_id):
        self.id = hand_id
        self.started = obs["t"]
        self.hero = obs["hero"]
        self.positions = {int(k): v for k, v in (obs.get("positions") or {}).items()}
        self.street = 0
        self.board = []
        self.actions = []            # {street, seat, pos, action, to, inferred}
        self.notes = []
        self.folded, self.allin = set(), set()
        # who was dealt in: the hero plus seats showing cards (or with chips in) at the start
        self.players = {s["idx"] for s in obs["seats"]
                        if s["present"] and (s["idx"] == 0 or s["has_cards"] or s["bet"] > 0)}
        self.contrib = {s["idx"]: s["bet"] for s in obs["seats"] if s["idx"] in self.players}  # blinds
        self.max_bet = max(self.contrib.values(), default=0.0)
        self.acted = set()           # acted voluntarily on this street
        self.last_actor = None
        self.no_cards_frames = {}    # seat -> consecutive frames without cards (fold debounce)
        self.players_folded_before = False   # seats already folded when we joined: logged once
        self._last_board_read = None
        self.hero_turn = False
        # what each street started with: the pot (read while no bets are out) and every player's
        # stack plus chips in front, which stays constant until the street's bets are swept.
        # Medians over several frames, so one misread doesn't matter. Postflop solving needs them.
        self.reads = {}                    # street -> {"pot": [...], "stacks": {seat: [...]}, "dead": [...]}
        # all-in before this street (or before we joined): those seats have nothing left to do
        self.allin_before = {s["idx"] for s in obs["seats"] if s["idx"] in self.players and s["all_in"]}
        self.allin |= self.allin_before
        # joined after the flop: start on the street showing; earlier streets weren't seen
        board = [c for c in obs["board"] if c[0] != "?"]
        street = {3: 1, 4: 2, 5: 3}.get(len(board), 0)
        if street:
            self.street, self.board = street, board
            self.contrib = {s["idx"]: s["bet"] for s in obs["seats"] if s["idx"] in self.players}
            self.max_bet = max(self.contrib.values(), default=0.0)
            self.notes.append(f"joined on the {STREETS[street]}: earlier streets weren't seen")
        self._note_reads(obs)

    # ------------------------------------------------------------ helpers
    def pos(self, seat):
        return self.positions.get(seat, f"seat {seat}")

    def active(self):
        return [p for p in self.players if p not in self.folded]

    def order(self):
        """Seats in this street's action order."""
        if self.street == 0 and self.positions:
            # preflop: UTG ... BB, the order positions are named in (spot.POSITION_NAMES)
            from .spot import POSITION_NAMES
            names = POSITION_NAMES.get(len(self.positions), [])
            by_pos = {v: k for k, v in self.positions.items()}
            seq = [by_pos[n] for n in names if n in by_pos]
            if seq:
                return seq
        if self.positions:
            n = len(self.positions)
            rank = (lambda p: 0 if p == "BB" else 1) if n == 2 else (
                lambda p: POSTFLOP_ORDER.index(p) if p in POSTFLOP_ORDER else 99)
            return sorted(self.positions, key=lambda s: rank(self.positions[s]))
        return sorted(self.players)

    def after_last(self, seats):
        """Sort seats in action order, starting after the last seat that acted."""
        order = [s for s in self.order() if s in self.players] or sorted(self.players)
        start = order.index(self.last_actor) + 1 if self.last_actor in order else 0
        rot = order[start:] + order[:start]
        return sorted(seats, key=lambda s: rot.index(s) if s in rot else 99)

    def add(self, seat, action, to=None, inferred=False):
        self.actions.append({"street": STREETS[self.street], "seat": seat, "pos": self.pos(seat),
                             "action": action, "to": None if to is None else round(to, 2), "inferred": inferred})
        if action == "fold":
            self.folded.add(seat)
        if action not in ("fold",):
            self.acted.add(seat)
        self.last_actor = seat

    # ------------------------------------------------------------ one frame
    def betting_done(self):
        """This street's betting is complete: everyone still in (and not all-in) has acted and
        matched the bet, or at most one player can still act."""
        live = [i for i in self.active() if i not in self.allin]
        if len(self.active()) <= 1:
            return True
        # a seat that went all-in on this street but whose chips haven't shown yet still has to act
        if any(i not in self.allin_before and i not in self.acted for i in self.allin if i in self.active()):
            return False
        if len(live) <= 1 and all(self.contrib.get(i, 0.0) >= self.max_bet - EPS for i in live):
            return True
        return all(i in self.acted and self.contrib.get(i, 0.0) >= self.max_bet - EPS for i in live)

    @property
    def over(self):
        """No more decisions: one player left; the river's betting is complete (showdown: cards
        disappearing after that are mucks, not folds); or after preflop, at most one player who
        isn't all-in remains, so the board just runs out."""
        live = [i for i in self.active() if i not in self.allin]
        return (len(self.active()) <= 1 or (self.street == 3 and self.betting_done())
                or (self.street > 0 and len(live) <= 1 and self.betting_done()))

    def _note_reads(self, obs):
        r = self.reads.setdefault(self.street, {"pot": [], "stacks": {}, "dead": []})
        seats = [s for s in obs["seats"] if s["present"]]
        out = sum(s["bet"] or 0.0 for s in seats)
        # once this street's bets are swept (the betting closed but the next card isn't out yet)
        # the pot already holds them and stacks are lower: those reads aren't the street's start
        if self.max_bet > EPS and out <= EPS:
            return
        pot = obs.get("pot")
        if pot is not None and pot > 0:
            if self.street > 0 and out <= EPS and len(r["pot"]) < READS_KEPT:
                r["pot"].append(pot)
            if self.street == 0 and len(r["dead"]) < READS_KEPT and pot - out > -EPS:
                r["dead"].append(max(0.0, pot - out))      # antes: the pot minus the blinds in front
        for s in seats:
            if s["idx"] in self.players and s["stack"] is not None:
                lst = r["stacks"].setdefault(s["idx"], [])
                if len(lst) < READS_KEPT:
                    lst.append(s["stack"] + (s["bet"] or 0.0))

    def _bet_possible(self, seat, s):
        """A bet can't be more than the seat had this street (its stack plus chips in front, a
        constant until the bets are swept) minus what it still has behind. Frames that break this
        are misreads while chips animate (a 5.0 bet over a 2.99 call; another seat's chips
        flying past an all-in player's spot)."""
        import statistics
        vals = (self.reads.get(self.street) or {}).get("stacks", {}).get(seat)
        if not vals or s["stack"] is None:
            return True
        total = statistics.median(vals)
        return s["bet"] <= total - s["stack"] + max(0.1, 0.03 * total)

    def street_start(self, street=None):
        """{"pot", "stacks": {seat: bb}, "dead"} at the start of a street (medians; None if unread)."""
        import statistics
        r = self.reads.get(self.street if street is None else street) or {}
        med = (lambda v: round(statistics.median(v), 2) if v else None)
        return {"pot": med(r.get("pot")), "dead": med(r.get("dead")),
                "stacks": {i: med(v) for i, v in (r.get("stacks") or {}).items() if v}}

    def update(self, obs):
        self.hero_turn = bool(obs.get("hero_turn"))
        if obs.get("hero") and self.hero and "?" in self.hero and obs["hero"].count("?") < self.hero.count("?")                 and obs["hero"][0::2] == self.hero[0::2]:
            self.hero = obs["hero"]          # a suit that was unreadable at first has been read
        for s in obs["seats"]:                # all-in is a state, not only an action we saw
            if s["idx"] in self.players and s["all_in"] and s["idx"] not in self.folded:
                self.allin.add(s["idx"])
        if self.over:
            # no more actions, but keep showing the runout (an all-in's turn and river)
            board = [c for c in obs["board"] if c[0] != "?"]
            if len(board) > len(self.board) and len(board) in (3, 4, 5):
                self.board = board
                self.street = max(self.street, {3: 1, 4: 2, 5: 3}[len(board)])
            return
        if not self.positions and obs.get("positions"):
            self.positions = {int(k): v for k, v in obs["positions"].items()}
        if self.street == 0 and not self.actions and not self.players_folded_before:
            # seats already showing FOLD when we start folded earlier in this hand: log them (in
            # seat order, inferred) so the preflop line is complete for building ranges
            early = [s["idx"] for s in obs["seats"]
                     if s["present"] and "fold" in s["labels"] and s["idx"] not in self.players and s["idx"] != 0]
            self.players_folded_before = True
            # chips already in front of a seat other than the blinds: a limp or raise made before
            # we started (shown as a bet with a CALL/RAISE label). Logged with the folds, in order.
            blinds = {k for k, v in self.positions.items() if v in ("SB", "BB")}
            put = {s["idx"]: s["bet"] for s in obs["seats"]
                   if s["idx"] in self.players and s["idx"] not in blinds and s["idx"] != 0 and s["bet"] > EPS}
            if (early or put) and self.positions:
                self.players |= set(early)
                bb = max((self.contrib.get(k, 0.0) for k in blinds), default=0.0)
                level = bb
                for i in self.after_last(early + list(put)):
                    if i not in put:
                        self.add(i, "fold", inferred=True)
                        continue
                    allin = any(s["idx"] == i and s["all_in"] for s in obs["seats"])
                    kind = "allin" if allin else "raise" if put[i] > level + EPS else "call"
                    self.add(i, kind, put[i], inferred=True)
                    level = max(level, put[i])
        if self.street == 0 and not any(not a["inferred"] for a in self.actions):
            # the deal may still be animating on the first frames: keep adding seats that show cards
            for s in obs["seats"]:
                if s["present"] and s["idx"] not in self.players and (s["has_cards"] or s["bet"] > 0):
                    self.players.add(s["idx"])
                    self.contrib[s["idx"]] = s["bet"]
                    self.max_bet = max(self.max_bet, s["bet"])
        seats = {s["idx"]: s for s in obs["seats"] if s["idx"] in self.players}
        # seats out of the hand in this frame (for closing a street swept in the same frame)
        gone_now = {i for i, s in seats.items()
                    if i != 0 and ("fold" in s["labels"] or (not s["has_cards"] and not s["all_in"]))}
        # 1. the board grew: close the street(s) before reading this frame's bets (they were swept)
        board = [c for c in obs["board"] if c[0] != "?"]
        target = {0: 0, 3: 1, 4: 2, 5: 3}.get(len(board))
        if target is not None and target > self.street:
            while self.street < target:
                self.close_street(gone_now)
            self.board = board
        elif len(board) > len(self.board):
            self.board = board
        elif board and len(board) == len(self.board) and board != self.board and                 (board == self._last_board_read or obs.get("settled")):
            # a card misread while it animated in: two agreeing frames correct it, or one read of the
            # settled screen (taken once it stopped changing, so nothing was mid-animation)
            self.board = board
        self._last_board_read = board
        self._note_reads(obs)

        # 2. what each seat did since the last frame: chips added, a FOLD label (or its cards gone
        #    for two frames), a CHECK label. Several can land in one frame when bots act quickly,
        #    so they are applied together in seat order, starting after the last actor.
        events = {}
        for i, s in seats.items():
            if i in self.folded:
                continue
            gone = not s["has_cards"] and not s["all_in"] and i != 0
            self.no_cards_frames[i] = self.no_cards_frames.get(i, 0) + 1 if gone else 0
            if s["bet"] > self.contrib.get(i, 0.0) + EPS:
                if not self._bet_possible(i, s):
                    continue                    # a misread mid-animation: more than the seat has
                events[i] = "chips"
            elif "fold" in s["labels"] or self.no_cards_frames[i] >= 2:
                events[i] = "fold"
            elif "check" in s["labels"] and i not in self.acted:
                events[i] = "check"
        if len(events) > 1:
            self.flag(f"several actions in one frame on the {STREETS[self.street]}: put in seat order")
        for i in self.after_last(list(events)):
            s, ev = seats[i], events[i]
            if ev == "fold":
                self.add(i, "fold")
            elif ev == "check":
                if self.contrib.get(i, 0.0) >= self.max_bet - EPS:
                    self.add(i, "check")
            else:
                new = s["bet"]
                if s["all_in"]:
                    kind = "allin"
                    self.allin.add(i)
                elif new <= self.max_bet + EPS:
                    kind = "call"
                elif self.max_bet <= EPS:
                    kind = "bet"
                else:
                    kind = "raise"
                self.add(i, kind, new)
                self.contrib[i] = new
                self.max_bet = max(self.max_bet, new)

    def close_street(self, gone=()):
        """Bets were swept: whoever hadn't matched the bet called it (or folded, if their cards
        are gone in the same frame); on an unbet street, everyone left checked."""
        betting_possible = len([i for i in self.active() if i not in self.allin]) > 1
        for i in self.after_last(self.active()):
            if i in self.allin:
                continue
            if i in gone:
                self.add(i, "fold", inferred=True)
                continue
            if self.max_bet > EPS and self.contrib.get(i, 0.0) < self.max_bet - EPS:
                self.add(i, "call", self.max_bet, inferred=True)
            elif not betting_possible:
                continue                                     # nobody left to bet against: no checks
            elif self.max_bet <= EPS and i not in self.acted:
                self.add(i, "check", inferred=True)
            elif self.street == 0 and i not in self.acted and self.contrib.get(i, 0.0) >= self.max_bet - EPS:
                self.add(i, "check", inferred=True)          # the big blind's option
        self.street += 1
        self.allin_before = set(self.allin)
        self.contrib = {i: 0.0 for i in self.players}
        self.max_bet = 0.0
        self.acted = set()
        self.last_actor = None

    def flag(self, note):
        if note not in self.notes:
            self.notes.append(note)

    # ------------------------------------------------------------ output
    def summary(self):
        lines = []
        for st in STREETS[:self.street + 1]:
            acts = [a for a in self.actions if a["street"] == st]
            txt = ", ".join(f"{a['pos']} {a['action']}" + (f" {a['to']:g}" if a["to"] is not None else "")
                            + ("*" if a["inferred"] else "") for a in acts)
            lines.append({"street": st, "board": self.board[:{0: 0, 1: 3, 2: 4, 3: 5}[STREETS.index(st)]],
                          "text": txt})
        # copies: a summary is a snapshot of this frame, later frames mustn't change it
        return {"id": self.id, "hero": self.hero, "street": STREETS[self.street], "board": list(self.board),
                "actions": [dict(a) for a in self.actions], "lines": lines, "notes": list(self.notes),
                "in_hand": sorted(self.active()), "inferred_marker": "*",
                # for the postflop solver
                "positions": {str(k): v for k, v in self.positions.items()},
                "allin": sorted(self.allin), "hero_turn": self.hero_turn,
                "start": {STREETS[k]: self.street_start(k) for k in sorted(self.reads)}}


class Tracker:
    """Feeds frames to the current Hand and starts a new one when a new hand is dealt."""

    def __init__(self, keep=20):
        self.hand = None
        self.history = []
        self.keep = keep
        self._n = 0
        self._last_board = 0
        self._other_hero = (None, 0)          # different hole cards, and in how many frames running
        self._cleared = 0                     # reads in a row without a board, after one that had it
        self._filed = None                    # the Hand last filed because its board cleared
        self._cards_gone = False              # ...and whether the hole cards have left the screen since
        self._old_board = False               # the hand was dealt with the last hand's board still up

    @staticmethod
    def _ranks(hero):
        return "".join(sorted(hero[0::2])) if hero else None   # "Qs6s" -> "6Q": suits can flicker

    def _same_deal(self, obs, board):
        """After a hand was filed for its cleared board, what the same hole cards on screen mean:
        "resume" when its board is back (it was only unread for a while: the hand goes on), "stale"
        while they're the old cards still lying there (no new hand), None for a new deal: other
        cards, or the same ranks dealt again (the cards left the screen in between, or the button
        has moved)."""
        f = self._filed
        if f is None or self._ranks(obs.get("hero")) != self._ranks(f.hero):
            return None
        if len(f.board) >= 3 and board[:len(f.board)] == f.board:
            return "resume"
        pos = {int(k): v for k, v in (obs.get("positions") or {}).items()}
        if self._cards_gone or (pos and f.positions and pos != f.positions):
            return None
        return "stale"

    def update(self, obs):
        if not obs.get("ok"):
            return self.state()
        board = [c for c in obs["board"] if c[0] != "?"]
        board_n = len(board)
        changed = False
        if self.hand is not None and obs.get("hero") and self.hand.hero:
            r = self._ranks(obs["hero"])
            if r != self._ranks(self.hand.hero):
                # new hole cards: believe it once two frames agree (one misread isn't a new hand)
                n = self._other_hero[1] + 1 if self._other_hero[0] == r else 1
                self._other_hero = (r, n)
                changed = n >= 2
            else:
                self._other_hero = (None, 0)
        # the board cleared: the hand is over (the next starts when hole cards show). Only a fully
        # cleared board counts: at the end of a hand the cards animate away a few at a time. And only
        # once a second read agrees, or on a read of the settled screen: one read without a board is
        # as likely an overlay or a failed board read in the middle of the hand
        if self.hand is None or board_n > 0:
            self._cleared = 0                 # counted per hand: a new hand starts without a board
        elif self._last_board > 0 or self._cleared:
            self._cleared += 1
        self._last_board = board_n
        if self.hand is not None and self._cleared:
            if self._cleared >= 2 or obs.get("settled"):
                # a hand dealt while the last one's board was still up loses that board, not its
                # own: it isn't over, and starts again below as the preflop hand it is
                self._filed, self._cards_gone = (None if self._old_board else self.hand), False
                self._file()
                self._cleared = 0
            elif not changed:
                return self.state()           # unconfirmed: the hand isn't shown this read
        if self.hand is None and not obs.get("hero"):
            self._cards_gone = True
        same = self._same_deal(obs, board) if self.hand is None and obs.get("hero") else None
        if same == "resume":
            # the board was unread for a few reads, not cleared: take the hand up where it was
            self.hand, self._filed = self._filed, None
            if self.history and self.history[-1]["id"] == self.hand.id:
                self.history.pop()
        new = (self.hand is None and obs.get("hero") and same is None) or (self.hand is not None and changed)
        if new:
            # new hole cards with a board still on screen: that board is the last hand's
            self._old_board = self.hand is not None and board_n > 0
            if self.hand is not None:
                self._file()
            self._n += 1
            self.hand = Hand(obs, self._n)
            self._other_hero = (None, 0)
            self._filed, self._cleared = None, 0
            if self.hand.street == 0 and self.hand.max_bet > 1.0 + EPS:
                self.hand.flag("joined after the first raise: earlier preflop actions are missing")
        if self.hand is not None:
            if not self.hand.hero and obs.get("hero"):
                self.hand.hero = obs["hero"]
            self.hand.update(obs)
        return self.state()

    def _file(self):
        """Move the current hand to the history."""
        self.history.append(self.hand.summary())
        self.history = self.history[-self.keep:]
        self.hand = None

    @property
    def in_hand(self):
        return self.hand is not None and self.hand.hero is not None and 0 not in self.hand.folded

    def state(self):
        """The current hand; between hands, the one that just finished (marked finished)."""
        if self.hand:
            return self.hand.summary()
        return dict(self.history[-1], finished=True) if self.history else None


# ---------------------------------------------------------------- replay
def replay(folder, reread=False):
    """Run a fresh tracker over a recording: its saved reads, or the frames re-read from scratch."""
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    tr = Tracker(keep=1000)
    for f in files:
        with open(os.path.join(folder, f)) as fh:
            rec = json.load(fh)
        obs = rec["obs"]
        if reread:
            import cv2
            from .reader import read_table
            from .spot import derive_spot
            img = cv2.imread(os.path.join(folder, f.replace(".json", ".png")))
            st = read_table(img)
            obs = observe(st, derive_spot(st), t=obs["t"])
        tr.update(obs)
    hands = tr.history + ([tr.hand.summary()] if tr.hand else [])
    for h in hands:
        print(f"hand {h['id']}: {h['hero'] or '??'}  board {' '.join(h['board']) or '-'}")
        for ln in h["lines"]:
            print(f"   {ln['street']:8s} {' '.join(ln['board']):15s} {ln['text']}")
        for n in h["notes"]:
            print(f"   note: {n}")
    return hands


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.tracker")
    ap.add_argument("command", choices=["replay"])
    ap.add_argument("folder")
    ap.add_argument("--reread", action="store_true", help="re-read the saved frames instead of the saved reads")
    a = ap.parse_args()
    replay(a.folder, a.reread)
