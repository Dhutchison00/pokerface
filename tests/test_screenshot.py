"""Regression test: python tests/test_screenshot.py

Parses the reference screenshot, checks every field, then mutates the parsed
table to simulate other spots (facing a shove / an open / limpers / deep).
"""
import copy
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tempfile  # noqa: E402
os.environ["HUD_GTOPEN_CHARTS"] = tempfile.mkdtemp()   # local GTOpen exports must not change these results

from preflop_hud.reader import read_table          # noqa: E402
from preflop_hud.spot import derive_spot           # noqa: E402
from preflop_hud.strategy import Engine            # noqa: E402
from preflop_hud.cards import HAND_INDEX, range_pct  # noqa: E402

img = cv2.imread(os.path.join(ROOT, "samples", "clubwpt_cutoff.webp"))
st = read_table(img)
assert st.ok, st.error
assert (st.sb, st.bb, st.ante) == (200, 400, 50)
assert st.pot == 2.5 and st.unit == "BB"
stacks = [s.stack for s in st.seats]
assert stacks == [19.7, 9.5, 19.4, 14.9, 21.8, 11.4, 4.5, 8.8], stacks
assert [s.bet for s in st.seats] == [0, 0, 0.5, 1.0, 0, 0, 0, 0]
assert st.button_seat == 1 and st.sb_seat == 2 and st.bb_seat == 3
assert st.seats[4].folded and st.seats[5].folded and st.seats[6].sitting_out
assert st.hero_cards["ranks"] == ["J", "T"] and st.hero_cards["suited"] is False

spot = derive_spot(st)
assert spot["type"] == "RFI" and spot["hero_pos"] == "CO" and spot["n_players"] == 8, spot
assert spot["eff_bb"] == 19.7 and abs(spot["ante_bb"] - 0.125) < 1e-9
assert [p["pos"] for p in spot["behind"]] == ["BTN", "SB", "BB"]
print("screenshot parse OK:", spot["hero_pos"], spot["eff_bb"], "bb", spot["type"], spot["notes"])

eng = Engine()


def run(name, mutate, expect_type, check=None, solver=True):
    s = copy.deepcopy(st)
    mutate(s)
    sp = derive_spot(s)
    assert sp["type"] == expect_type, (name, sp["type"], sp.get("warnings"), sp.get("notes"))
    ch = eng.solve(sp)
    assert bool(ch.get("solver")) == solver, (name, ch.get("method"), ch.get("note"))
    F = np.array(ch.get("freqs_raw") or ch["freqs"])
    if check:
        check(sp, ch, F)
    print(f"{name:26s} -> {ch['title']:38s} {ch['pct']}  [{ch['method']}]")


def hand(ch, F, h, act):
    return F[HAND_INDEX[h], ch["actions"].index(act)] if act in ch["actions"] else 0.0


def seat(s, i, **kw):
    for k, v in kw.items():
        setattr(s.seats[i], k, v)


# HJ (8.8bb) shoves, hero CO must call/fold
def hj_shove(s):
    seat(s, 7, bet=8.8, stack=0.0)
    s.pot = 2.5 + 8.8
run("HJ shoves 8.8bb", hj_shove, "VS_SHOVE",
    lambda sp, ch, F: (hand(ch, F, "AA", "call") == 1 and hand(ch, F, "72o", "call") == 0) or sys.exit("bad call range"))


# HJ opens to 2.2bb (20bb eff): jam/call/fold
def hj_open(s):
    seat(s, 7, bet=2.2, stack=6.6)
    s.pot = 2.5 + 2.2
run("HJ opens 2.2bb", hj_open, "VS_OPEN",
    lambda sp, ch, F: hand(ch, F, "AA", "allin") + hand(ch, F, "AA", "call") > 0.99 or sys.exit("AA must continue"))


# deep: everyone 60bb, HJ opens 2.3bb
def deep_open(s):
    for i in range(8):
        if s.seats[i].stack is not None:
            s.seats[i].stack = 60.0
    seat(s, 7, bet=2.3, stack=57.7)
    s.pot = 2.5 + 2.3
run("deep: HJ opens, hero CO", deep_open, "VS_OPEN",
    lambda sp, ch, F: (hand(ch, F, "AA", "fold") < 0.01 and hand(ch, F, "AA", "raise") + hand(ch, F, "AA", "allin") >= 0.5)
    or sys.exit("AA must never fold and should mostly re-raise deep"))


def deep_rfi(s):
    for i in range(8):
        if s.seats[i].stack is not None:
            s.seats[i].stack = 60.0
run("deep: first in", deep_rfi, "RFI",
    lambda sp, ch, F: 28 < ch["pct"]["raise"] < 40 or sys.exit("CO deep open % off"))


# HJ limps
def hj_limp(s):
    seat(s, 7, bet=1.0, stack=7.8)
    s.pot = 3.5
run("HJ limps", hj_limp, "LIMPED", solver=False)   # not in the solver tree: approximate model


# short: hero 8bb first in -> pure push/fold
def short(s):
    seat(s, 0, stack=8.0)
run("hero 8bb first in", short, "RFI",
    lambda sp, ch, F: ch["actions"][0] == "allin" or sys.exit("expected push/fold"))


# hero on the SB: move blinds so hero (seat 0) is SB, seat 1 BB, button seat 7
def hero_sb(s):
    for x in s.seats:
        x.labels = [l for l in x.labels if l not in ("sb", "bb")]
        x.bet = 0.0
    s.seats[0].labels.append("sb"); s.seats[0].bet = 0.5; s.seats[0].stack = 19.2
    s.seats[1].labels.append("bb"); s.seats[1].bet = 1.0; s.seats[1].stack = 8.5
    for i in (2, 3, 7):
        s.seats[i].labels.append("fold"); s.seats[i].has_cards = False
    s.sb_seat, s.bb_seat = 0, 1
    s.pot = 2.5
run("hero SB, folded to", hero_sb, "RFI")

# ------------------------------------------------------------ spot -> solver node mapping
from preflop_hud.solved import SolvedCharts  # noqa: E402


def deep(s, stack=40.0):
    for x in s.seats:
        if x.stack is not None:
            x.stack = stack


def only(s, keep):
    """Everyone not in `keep` has folded (no cards, FOLD label)."""
    for x in s.seats:
        if x.idx not in keep and x.present:
            x.has_cards = False
            x.labels = [l for l in x.labels if l not in ("sb", "bb", "sitting_out")] + ["fold"]
            x.bet = 0.0


def set_blinds(s, sb, bb):
    for x in s.seats:
        x.labels = [l for l in x.labels if l not in ("sb", "bb")]
    s.seats[sb].labels.append("sb")
    s.seats[bb].labels.append("bb")
    s.sb_seat, s.bb_seat = sb, bb


def m_3bet(s):        # CO (hero) opens 2.2, BTN 3-bets to 6.6
    deep(s); seat(s, 0, bet=2.2, stack=37.8); seat(s, 1, bet=6.6, stack=33.4); s.pot = 2.5 + 2.2 + 6.6


def m_3betjam(s):     # CO opens, BB jams
    deep(s); seat(s, 0, bet=2.2, stack=37.8); seat(s, 3, bet=40.0, stack=0.0); s.pot = 1.0 + 0.5 + 2.2 + 40


def m_4bet(s):        # UTG opens, CO 3-bets to 6.6, UTG jams
    deep(s); seat(s, 4, bet=40.0, stack=0.0, labels=[], has_cards=True); seat(s, 0, bet=6.6, stack=33.4)
    s.pot = 1.0 + 1.5 + 40 + 6.6


def m_bb_vs_limp(s):  # hero is BB, HJ (seat 7) is SB and limps, everyone else folded
    deep(s); set_blinds(s, 7, 0); only(s, {0, 7}); seat(s, 7, bet=1.0, stack=39.0); seat(s, 0, bet=1.0, stack=39.0)
    s.pot = 3.0


def m_sb_vs_iso(s):   # hero SB limped, BB (seat 1) raises to 3.5
    deep(s); set_blinds(s, 0, 1); only(s, {0, 1}); seat(s, 0, bet=1.0, stack=39.0); seat(s, 1, bet=3.5, stack=36.5)
    s.pot = 1.0 + 4.5


def m_hu(s):          # heads-up: hero is BTN/SB, first in
    deep(s, 30.0)
    for x in s.seats[2:]:
        x.present = False; x.has_cards = False; x.labels = []; x.bet = 0.0; x.name = ""; x.stack = None
    set_blinds(s, 0, 1); seat(s, 0, bet=0.5, stack=29.5); seat(s, 1, bet=1.0, stack=29.0); s.pot = 1.5 + 0.25


for name, mut, typ, key in [
    ("opened, faces 3-bet", m_3bet, "VS_3BET", "VS_3BET:CO:BTN"),
    ("opened, BB jams", m_3betjam, "VS_SHOVE", "VS_3BETJAM:CO:BB"),
    ("3-bet, opener jams", m_4bet, "VS_SHOVE", "VS_4BET:CO:UTG"),
    ("BB vs SB limp", m_bb_vs_limp, "LIMPED", "BB_VS_LIMP"),
    ("SB limped, BB raises", m_sb_vs_iso, "VS_OPEN", "SB_VS_ISO"),
    ("heads-up first in", m_hu, "RFI", "RFI:BTN"),
]:
    s2 = copy.deepcopy(st)
    mut(s2)
    sp = derive_spot(s2)
    assert sp["type"] == typ, (name, sp["type"], sp.get("warnings"), sp.get("notes"))
    k = SolvedCharts._key(sp)
    assert k == key, (name, k, key)
    ch = eng.solve(sp)
    assert ch.get("solver"), (name, ch.get("method"), ch.get("note"))
    print(f"{name:24s} -> {k:18s} {ch['title']:34s} {ch['pct']}")

print("spot tests passed")

# ------------------------------------------------------------ same-colour suit detection
red = img[813, 848].copy()          # inside the heart pip of the 10
white = img[830, 815].copy()        # card face


def red_jack(im):
    im = im.copy()
    patch = im[748:800, 800:822].astype(float)
    # recolour black ink -> red, keeping the anti-aliasing (t = ink coverage 0..1)
    t = np.clip((patch.mean(axis=2, keepdims=True) - white.mean()) / (30 - white.mean()), 0, 1)
    im[748:800, 800:822] = (white * (1 - t) + red * t).astype(np.uint8)
    return im


hh = red_jack(img)
hh[804:825, 802:826] = white
hh[804:824, 803:825] = img[803:823, 838:860]   # paste the heart pip over the spade
hc = read_table(hh).hero_cards
assert hc["suited"] is True, hc
hc2 = read_table(red_jack(img)).hero_cards      # red J with a spade-shaped pip: different suits
assert hc2["suited"] is False, hc2
print("suited detection OK:", hc["suits"], hc2["suits"])

# ------------------------------------------------------------ four-colour deck: colour alone decides
from preflop_hud import reader as reader_mod      # noqa: E402

diamond = np.zeros((24, 24), bool)
diamond[4:20, 4:20] = True
spade = np.zeros((24, 24), bool)
spade[8:22, 2:22] = True


def pip(color, shape):
    return {"color": color, "pip": shape, "fill": 0.6, "suit": None}


# same blue/green colour is suited even when one pip is missing or the pips compare badly
assert reader_mod.decide_suited([pip("blue", diamond), pip("blue", None)], False) is True
assert reader_mod.decide_suited([pip("blue", diamond), pip("blue", spade)], False) is True
assert reader_mod.decide_suited([pip("green", None), pip("green", None)], False) is True
assert reader_mod.decide_suited([pip("blue", diamond), pip("green", diamond)], False) is False
# red/red: pips decide in a two-colour deck, colour decides in a four-colour one
assert reader_mod.decide_suited([pip("red", diamond), pip("red", spade)], False) is False
assert reader_mod.decide_suited([pip("red", diamond), pip("red", None)], False) is None
assert reader_mod.decide_suited([pip("red", diamond), pip("red", spade)], True) is True
assert reader_mod.decide_suited([pip("red", diamond), None], True) is None

# 4d3d on a four-colour deck (samples/hero_43s_river.webp). Hero has folded there, so the
# app dims the cards: undo the dimming on the hero-card area before reading it.
reader_mod._seen_four_color = False
img43 = cv2.imread(os.path.join(ROOT, "samples", "hero_43s_river.webp"))
fx, fy, fw, fh = reader_mod.find_felt(img43, reader_mod.DEFAULT_LAYOUT)
hu0, hv0, hu1, hv1 = reader_mod.DEFAULT_LAYOUT["hero_cards"]
ys = slice(int(fy + hv0 * fh), int(fy + hv1 * fh))
xs = slice(int(fx + hu0 * fw), int(fx + hu1 * fw))
img43[ys, xs] = np.clip(img43[ys, xs].astype(float) * 255 / 135, 0, 255).astype(np.uint8)
hc43 = read_table(img43).hero_cards
assert hc43["ranks"] == ["4", "3"] and hc43["suited"] is True and hc43["suits"] == ["d", "d"], hc43
assert hc43["four_color"] is True
# same hand with the 3 recoloured green (4d3c): different colours -> offsuit
img43c = img43.copy()
right = cv2.cvtColor(img43c[ys, 955:1075], cv2.COLOR_BGR2HSV)
blue_ink = (right[..., 1] > 90) & (abs(right[..., 0].astype(int) - 103) < 12)
right[..., 0][blue_ink] = 77
img43c[ys, 955:1075] = cv2.cvtColor(right, cv2.COLOR_HSV2BGR)
hc43c = read_table(img43c).hero_cards
assert hc43c["colors"] == ["blue", "green"] and hc43c["suited"] is False, hc43c
reader_mod._seen_four_color = False                # later tests use the two-colour reference
print("four-colour suits OK:", hc43["suits"], hc43c["suits"])

# an OCR box that starts below the top of the glyph (seen on the GPU build) must still find the pip
fx, fy, fw, fh = reader_mod.find_felt(hh, reader_mod.DEFAULT_LAYOUT)
crop = hh[int(fy + hv0 * fh):int(fy + hv1 * fh), int(fx + hu0 * fw):int(fx + hu1 * fw)]
up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
low_box = reader_mod._suit_info(up, ((125, 122), (197, 243)), 3)
assert low_box and low_box["pip"] is not None and low_box["color"] == "red", low_box
print("low OCR box OK")

# ------------------------------------------------------------ community cards
board = read_table(cv2.imread(os.path.join(ROOT, "samples", "hero_43s_river.webp"))).board
assert [(c["rank"], c["suit"]) for c in board] == [("T", "d"), ("T", "c"), ("9", "c"), ("5", "d")], board
assert st.board == [], st.board                       # the reference screenshot is preflop
print("board OK:", " ".join(c["rank"] + c["suit"] for c in board))

# a board out means postflop: no preflop chart, no solver work, no pot read as preflop dead money
st_turn = read_table(cv2.imread(os.path.join(ROOT, "samples", "hero_43s_river.webp")))
sp_turn = derive_spot(st_turn)
assert sp_turn["type"] == "POSTFLOP" and sp_turn["street"] == "turn" and sp_turn["dead"] == 0.0, sp_turn
ch_turn = eng.solve(sp_turn)
assert ch_turn["freqs"] is None and ch_turn["title"] == "Postflop · turn", ch_turn
st_flop = copy.deepcopy(st_turn)
st_flop.board = st_flop.board[:2]                      # a stray 1-2 card read doesn't switch charts off
assert derive_spot(st_flop)["type"] != "POSTFLOP"
print("postflop detection OK")

# ------------------------------------------------------------ hole-card ranks: this app's tall Q, 2 vs 7
q6 = cv2.imread(os.path.join(ROOT, "samples", "hero_q6s_utg.webp"))
hq = read_table(q6).hero_cards
assert hq["ranks"] == ["Q", "6"] and hq["suited"] is True and hq["suits"] == ["s", "s"], hq   # OCR alone: "n"/"?"
# the same hand with a red Q and a blue 6: each card's colour comes from its own glyph -> offsuit
fx, fy, fw, fh = reader_mod.find_felt(q6, reader_mod.DEFAULT_LAYOUT)
ys = slice(int(fy + hv0 * fh), int(fy + hv1 * fh))
xs = slice(int(fx + hu0 * fw), int(fx + hu1 * fw))
crop = q6[ys, xs].copy()
hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
ink = (hsv_c[..., 1] < 90) & (hsv_c[..., 2] < 200)
t = np.clip((200 - hsv_c[..., 2].astype(float)) / 160, 0, 1)[..., None]
cols = np.arange(crop.shape[1])[None, :].repeat(crop.shape[0], 0)
for m, colour in ((ink & (cols < 0.29 * crop.shape[1]), (45, 45, 205)), (ink & (cols >= 0.29 * crop.shape[1]), (200, 110, 30))):
    crop[m] = (np.array([245.0] * 3) * (1 - t[m]) + np.array(colour, float) * t[m]).astype(np.uint8)
rb = q6.copy()
rb[ys, xs] = crop
hrb = read_table(rb).hero_cards
assert hrb["ranks"] == ["Q", "6"] and hrb["colors"] == ["red", "blue"] and hrb["suited"] is False, hrb
# 2 vs 7: a 2 has a flat base, a 7 ends in a narrow stroke
two = np.zeros((30, 20), bool)
two[0:4, 2:18] = True
two[26:30, 1:19] = True
seven = np.zeros((30, 20), bool)
seven[0:4, 1:19] = True
for r in range(4, 30):
    seven[r, max(0, 16 - r // 2):18 - r // 2] = True
assert reader_mod._fix_2_7("7", two) == "2" and reader_mod._fix_2_7("2", seven) == "7"
assert reader_mod._fix_2_7("K", two) == "K"
# a real one: OCR reads the 2 as "7" (93%) and doesn't detect the red Q at all
hq2 = read_table(cv2.imread(os.path.join(ROOT, "samples", "hero_qh2s_2read7.png"))).hero_cards
assert hq2["ranks"] == ["Q", "2"] and hq2["suits"] == ["h", "s"] and hq2["suited"] is False, hq2
# red A and 2 that OCR text detection doesn't see (it read "2" for the A, then "T"): found from the card faces
reader_mod._seen_four_color = False
ha2 = read_table(cv2.imread(os.path.join(ROOT, "samples", "hero_ah2h_red_rec.png"))).hero_cards
assert ha2["ranks"] == ["A", "2"] and ha2["colors"] == ["red", "red"], ha2
print("hole-card ranks OK: Q6s read, red/blue overlap offsuit, 2/7 by shape")

# red ink straddles OpenCV's hue wrap (0..180): split red pixels plus a few anti-aliased in-between
# hues used to give a median of ~81 (green) or ~95 (blue): a 2 of hearts read as clubs
hsv_red = np.zeros((10, 10, 3), np.uint8)
hsv_red[..., 1], hsv_red[..., 2] = 200, 180
hsv_red[:5, :, 0] = 2
hsv_red[5:9, :, 0] = 177
hsv_red[9, :, 0] = [20, 30, 81, 95, 110, 120, 140, 150, 160, 170]
assert reader_mod._ink_color(hsv_red, np.ones((10, 10), bool)) == "red"
for hue, name in ((60, "green"), (110, "blue")):
    hsv_c = hsv_red.copy()
    hsv_c[..., 0] = hue
    assert reader_mod._ink_color(hsv_c, np.ones((10, 10), bool)) == name
print("red across the hue wrap OK")

# ------------------------------------------------------------ red table theme
red = img.copy()
hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
lay = reader_mod.DEFAULT_LAYOUT
green = cv2.inRange(hsv, tuple(lay["felt_hsv_lo"]), tuple(lay["felt_hsv_hi"])) > 0
hsv[..., 0][green] = 2                                 # repaint the felt red
hsv[..., 1][green] = np.maximum(hsv[..., 1][green], 150)
red = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
assert reader_mod.find_felt(red, lay) is not None
st_red = read_table(red)
assert [s.stack for s in st_red.seats] == stacks, [s.stack for s in st_red.seats]
assert st_red.hero_cards["ranks"] == ["J", "T"] and (st_red.sb, st_red.bb, st_red.ante) == (200, 400, 50)
print("red felt OK")

# ------------------------------------------------------------ live loop (change detection + confirmation)
from preflop_hud import server  # noqa: E402

no_btn_cards = img.copy()
no_btn_cards[660:752, 285:415] = (38, 30, 28)  # Brock777 (BTN) folds: card backs disappear
frames = [img, img, no_btn_cards, no_btn_cards, no_btn_cards]
app = server.App()
app.cfg = {}
it = iter(frames)
app._frame = lambda: next(it)
statuses = []
for _ in frames:
    app.tick()
    statuses.append((app.state["status"], [p["pos"] for p in app.state.get("spot", {}).get("behind", [])]))
print("live ticks:", statuses)
assert statuses[1] == ("live", ["BTN", "SB", "BB"]), statuses
assert statuses[3] == ("live", ["SB", "BB"]), statuses
assert statuses[4] == ("live", ["SB", "BB"]), statuses
# a FOLD-label-sized change must register as a change
small_change = img.copy()
small_change[355:378, 1335:1395] = 200
a = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (320, 220), interpolation=cv2.INTER_AREA)
b = cv2.resize(cv2.cvtColor(small_change, cv2.COLOR_BGR2GRAY), (320, 220), interpolation=cv2.INTER_AREA)
assert int((cv2.absdiff(a, b) > 12).sum()) > 8
# ------------------------------------------------------------ uncached grid point -> fallback -> solver
import tempfile, time  # noqa: E401,E402
from preflop_hud import solver as solver_mod  # noqa: E402
saved_dir = solver_mod.SOL_DIR
solver_mod.SOL_DIR = tempfile.mkdtemp()
try:
    app2 = server.App()
    app2.cfg = {}
    app2._frame = lambda: img
    app2.tick(); app2.tick()
    first = app2.state["chart"]
    assert not first.get("solver") and first.get("pending_solver"), first.get("method")
    print("uncached:", first.get("note"))
    sig0 = app2._accepted
    t0 = time.time()
    while not app2.state["chart"].get("solver") and time.time() - t0 < 120:
        time.sleep(0.5)
        app2.tick()
    assert app2.state["chart"].get("solver"), "chart never upgraded to the solver"
    assert app2._accepted == sig0
    print(f"upgraded to solver chart after {time.time() - t0:.1f}s without a table change")
finally:
    solver_mod.SOL_DIR = saved_dir
print("all tests passed")
