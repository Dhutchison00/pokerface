"""Tournament / cash modes: python tests/test_modes.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp()
os.environ["HUD_GTOPEN_CHARTS"] = TMP

from preflop_hud import gtopen, modes, solver as solver_mod   # noqa: E402
from preflop_hud.cards import COMBO_WEIGHTS as W, HAND_INDEX  # noqa: E402
from preflop_hud.solved import SolvedCharts                  # noqa: E402

solver_mod.SOL_DIR = tempfile.mkdtemp()      # never touch the real cache

# ---------------------------------------------------------------- settings
cash = {"mode": "cash"}
bubble = {"mode": "tournament", "icm_stage": "bubble"}
assert modes.normalize(None) == modes.DEFAULT_GAME
assert modes.normalize({"mode": "cash", "icm_stage": "bubble"})["icm_stage"] is None   # no ICM in cash
for bad in ({"mode": "sng"}, {"icm_stage": "nope"}, {"mode": "cash", "rake_pct": -1}):
    try:
        modes.normalize(bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
assert modes.solver_tag(None) == "default" and modes.solver_tag(bubble) == "default"   # HUD solver: chip EV
assert modes.solver_tag(cash) == "cash_r5c3" and modes.gtopen_tag(cash) == "_cash_r5c3"
assert modes.gtopen_tag(None) == "" and modes.gtopen_tag(bubble) == "_icm_bubble"
assert modes.solver_params(cash)["open_size"] == [[1000, 2.5]] and modes.solver_params(None) == {}
assert modes.describe(cash) == "Cash · 5% rake, 3bb cap"
pay = modes.standard_payouts()
assert len(pay) == 150 and abs(sum(pay) - 100) < 0.01 and pay[0] > pay[1] > pay[-1]
icm = modes.icm_config("bubble", 8, 25)
assert len(icm["payouts"]) == 150 and icm["field_groups"] == [{"stack": 25.0, "count": 144}]
assert "field_groups" not in modes.icm_config("ft", 8, 25)
print("settings OK")

# ---------------------------------------------------------------- GTOpen configs per mode
c = gtopen.game_config(8, 100, 0.0, False, cash)
assert c["rake_pct"] == 5 and c["rake_cap"] == 3 and c["open_raises"] == [2.5] and c["open_raises_by_seat"][6] == [3.0]
c = gtopen.game_config(8, 25, 1.0, False, bubble)
assert c["icm"]["payouts"][:2] == pay[:2] and c["realization"] == "balanced" and not c["card_removal"]
assert "leaf_net" not in c and c["rake_pct"] == 0
assert gtopen.game_name(8, 100, 0.0, False, cash) == "hud_v1_n8_s100_d0p00_cash_r5c3"
print("GTOpen configs OK")

# ---------------------------------------------------------------- rake in the HUD solver
def bb_defend(params):
    t = solver_mod.Tree(3, [100] * 3, 0.0, params)
    c = solver_mod.CFR(t)
    c.solve(iters=400, tol=0.002)
    d = c.export()["VS_OPEN:BB:BTN"]
    cont = 1 - d["strategy"][d["actions"].index("fold")]
    return float((cont * W * d["reach"]).sum() / (W * d["reach"]).sum())


t = solver_mod.Tree(3, [100] * 3, 0.0, modes.solver_params(cash))
flop = [nd for nd in t.nodes if nd["kind"] == "flop"]
assert all(abs(nd["pot"] - (nd["inv"].sum() - min(0.05 * nd["inv"].sum(), 3.0))) < 1e-9 for nd in flop)
wins = [nd for nd in t.nodes if nd["kind"] == "win"]
assert all(nd["pot"] == nd["inv"].sum() for nd in wins)          # no flop, no drop
no_rake, raked = bb_defend({"open_size": [[1000, 2.5]], "sb_open_size": [[1000, 3.0]]}), bb_defend(modes.solver_params(cash))
assert raked < no_rake, (raked, no_rake)
print(f"rake OK: BB defends vs BTN {no_rake * 100:.1f}% without rake, {raked * 100:.1f}% with 5%/3bb")

# ---------------------------------------------------------------- exports per mode
def write_export(n, stack, dead, game, raise_hands):
    strat = np.zeros((2, 169), np.float32)
    strat[0] = 1.0
    for h in raise_hands:
        strat[:, HAND_INDEX[h]] = [0.0, 1.0]
    meta = {"source": "gtopen", "export_version": gtopen.EXPORT_VERSION, "n": n, "stack": stack, "dead": dead,
            "limp": False, "tag": modes.gtopen_tag(game), "iterations": 1, "gap_total": 0.0, "params": {},
            "spots": {"RFI:CO": {"actions": ["fold", "raise"], "low_reach": False}}}
    np.savez_compressed(os.path.join(TMP, gtopen.game_name(n, stack, dead, False, game) + ".npz"),
                        RFI__CO__strategy=strat, RFI__CO__reach=np.ones(169, np.float32),
                        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


write_export(8, 25, 1.0, None, ["AA"])
write_export(8, 25, 1.0, bubble, ["KK"])
write_export(8, 100, 0.0, cash, ["QQ"])
store = gtopen.GtopenStore(TMP)
assert store.get(8, 25, 1.0, "").meta["tag"] == ""
assert store.get(8, 25, 1.0, "_icm_bubble").meta["tag"] == "_icm_bubble"
assert store.get(8, 25, 1.0, "_icm_ft") is None and store.get(8, 100, 0.0, "") is None

charts = SolvedCharts(gtopen_dir=TMP)
rfi = {"type": "RFI", "eff_bb": 25, "n_players": 8, "hero_pos": "CO", "hero": {"bet": 0.0},
       "raises": [], "limpers": [], "dead": 1.0}


def raises(ch, hand):
    return ch["freqs"][HAND_INDEX[hand]][ch["actions"].index("raise")]


ch, _ = charts.chart(rfi, None)
assert ch["source"] == "gtopen" and raises(ch, "AA") == 1 and raises(ch, "KK") == 0 and "chip EV" in ch["method"]
ch, _ = charts.chart(rfi, bubble)
assert ch["source"] == "gtopen" and raises(ch, "KK") == 1 and "ICM bubble" in ch["method"], ch["method"]
ch, msg = charts.chart(dict(rfi, hero_pos="BTN"), bubble)          # no ICM export for BTN: chip EV + note
assert ch is None or ("ICM" in (ch.get("note") or "") and ch["source"] == "hud"), ch
ch, _ = charts.chart(dict(rfi, eff_bb=100, dead=0.0), cash)
assert ch["source"] == "gtopen" and raises(ch, "QQ") == 1 and ch["method"].startswith("GTOpen solver · Cash")
assert charts._store(cash).tag == "cash_r5c3" and charts._store(None).tag == "default"
print("charts per mode OK")

# ---------------------------------------------------------------- ICM at short stacks, low reach, rescans
def write_jam_export(stack, game, call_hands, low_reach=False, name_suffix=""):
    strat = np.zeros((2, 169), np.float32)
    strat[0] = 1.0
    for h in call_hands:
        strat[:, HAND_INDEX[h]] = [0.0, 1.0]
    meta = {"source": "gtopen", "export_version": gtopen.EXPORT_VERSION, "n": 8, "stack": stack, "dead": 1.0,
            "limp": False, "tag": modes.gtopen_tag(game), "iterations": 1, "gap_total": 0.0, "params": {},
            "spots": {"VS_JAM:BB:BTN": {"actions": ["fold", "call"], "low_reach": low_reach}}}
    np.savez_compressed(os.path.join(TMP, gtopen.game_name(8, stack, 1.0, False, game) + name_suffix + ".npz"),
                        VS_JAM__BB__BTN__strategy=strat, VS_JAM__BB__BTN__reach=np.ones(169, np.float32),
                        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))


vs_jam = {"type": "VS_SHOVE", "eff_bb": 15, "n_players": 8, "hero_pos": "BB", "hero": {"bet": 1.0, "total": 15.0},
          "villain": {"pos": "BTN", "bet": 15.0}, "raises": [{"pos": "BTN"}], "limpers": [], "hero_opened": False,
          "dead": 1.0}
assert charts._key(vs_jam) == "VS_JAM:BB:BTN"
write_jam_export(15, bubble, ["AA"])                   # appears while the HUD runs: picked up by the rescan
ch, _ = charts.chart(vs_jam, bubble)
assert ch and ch["source"] == "gtopen" and ch["freqs"][HAND_INDEX["AA"]][ch["actions"].index("call")] == 1, ch
write_jam_export(15, None, ["AA"], low_reach=True)     # unconverged line: not used
assert store.get(8, 15, 1.0, "") is not None and store.get(8, 15, 1.0, "").spot("VS_JAM:BB:BTN") is None
print("ICM short-stack export, low-reach fallback and rescans OK")
print("all tests passed")
