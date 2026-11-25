"""GTOpen chart export/lookup: python tests/test_gtopen.py

No GTOpen needed: builds a synthetic export file and checks that the HUD prefers it, falls back
to its own solver for nodes the export lacks, and matches grid points the way the store intends.
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp()
os.environ["HUD_GTOPEN_CHARTS"] = TMP

from preflop_hud import gtopen                              # noqa: E402
from preflop_hud.cards import HAND_INDEX, HAND_NAMES        # noqa: E402
from preflop_hud.solved import SolvedCharts                  # noqa: E402

# ---------------------------------------------------------------- class order
order = gtopen._gtopen_to_hud_order()
assert sorted(order) == list(range(169))
ranks = "23456789TJQKA"
assert HAND_NAMES[order[12 * 13 + 12]] == "AA"
assert HAND_NAMES[order[12 * 13 + 11]] == "AKs"     # row > col: suited
assert HAND_NAMES[order[11 * 13 + 12]] == "AKo"     # row < col: offsuit
assert HAND_NAMES[order[5 * 13 + 0]] == "72s" and HAND_NAMES[order[0 * 13 + 5]] == "72o"

# ---------------------------------------------------------------- lines
lines = gtopen.hud_lines(8, limp=False)
assert "RFI:UTG" in lines and "RFI:SB" not in lines and "BB_VS_LIMP" not in lines
assert not any(k.endswith(":SB") and k.startswith("VS_OPEN") for k in lines)   # no SB-opener lines
assert lines["VS_4BET:BB:CO"] == ([("CO", "raise"), ("BB", "raise"), ("CO", "jam")], "BB")
lines_l = gtopen.hud_lines(8, limp=True)
assert "RFI:SB" in lines_l and lines_l["SB_VS_ISO"] == ([("SB", "call"), ("BB", "raise")], "SB")
assert "RFI:BTN" in gtopen.hud_lines(2, limp=True) and "RFI:BTN" not in gtopen.hud_lines(2, limp=False)
cfg = gtopen.game_config(8, 60, 1.0, limp=True)
assert cfg["open_raises"] == [2.3] and cfg["open_raises_by_seat"][6] == [3.0] and cfg["ante"] == 0.125
assert cfg["raise_mults_oop"] == [4.0] and abs(cfg["limper_extra"] - 1.2) < 1e-9
assert gtopen.game_name(8, 60, 1.0, True) == "hud_v1_n8_s60_d1p00_limp"
print("mapping OK")


# ---------------------------------------------------------------- synthetic export
def write_export(n, stack, dead, limp, keys):
    arrays, spots = {}, {}
    for key, raise_hands in keys.items():
        strat = np.zeros((2, 169), np.float32)
        strat[0] = 1.0
        for h in raise_hands:
            strat[:, HAND_INDEX[h]] = [0.0, 1.0]
        safe = key.replace(":", "__")
        arrays[f"{safe}__strategy"] = strat
        arrays[f"{safe}__reach"] = np.ones(169, np.float32)
        spots[key] = {"actions": ["fold", "raise"], "low_reach": False}
    meta = {"source": "gtopen", "export_version": gtopen.EXPORT_VERSION, "n": n, "stack": stack, "dead": dead,
            "limp": limp, "iterations": 100, "gap_total": 0.01, "params": {}, "spots": spots}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(os.path.join(TMP, gtopen.game_name(n, stack, dead, limp) + ".npz"), **arrays)


write_export(8, 21, 1.0, False, {"RFI:CO": ["AA", "KK"]})
write_export(8, 21, 1.0, True, {"RFI:CO": ["AA"]})          # same grid point with limps: preferred
write_export(8, 21, 2.5, False, {"RFI:CO": ["72o"]})        # far-off antes: never chosen for 1.0

store = gtopen.GtopenStore(TMP)
assert store.get(8, 21, 1.0).meta["limp"] is True
assert store.get(8, 21, 0.75).meta["dead"] == 1.0           # 0.12/player x 7 rounds to 0.75: still matches
assert store.get(8, 21, 2.4).meta["dead"] == 2.5
assert store.get(8, 25, 1.0) is None and store.get(7, 21, 1.0) is None
print("store matching OK")

charts = SolvedCharts(gtopen_dir=TMP)
rfi = {"type": "RFI", "eff_bb": 20.5, "n_players": 8, "hero_pos": "CO", "hero": {"bet": 0.0},
       "raises": [], "limpers": [], "dead": 1.0}
ch, msg = charts.chart(rfi)
assert ch and ch["source"] == "gtopen", (ch, msg)
assert ch["actions"] == ["raise", "fold"]
# the limp export (AA only) wins over the no-limp one (AA, KK)
assert ch["freqs"][HAND_INDEX["AA"]] == [1.0, 0.0] and ch["freqs"][HAND_INDEX["KK"]] == [0.0, 1.0]
assert ch["ev"] is None and "ev_actions" not in ch and "GTOpen" in ch["method"] and "limps" in ch["method"]
btn = dict(rfi, hero_pos="BTN")                              # not in the export: the HUD's own solver
ch2, msg2 = charts.chart(btn)
assert ch2 is None or ch2["source"] == "hud", ch2
print("HUD prefers GTOpen, falls back for missing nodes OK:", ch["method"], "|", (ch2 or {}).get("method", msg2))

# ---------------------------------------------------------------- re-exporting a chart the HUD has loaded
# export() writes next to the chart and replaces it. Both stores above have this chart loaded, and on
# Windows a file that is still open can't be replaced.
path = os.path.join(TMP, gtopen.game_name(8, 21, 1.0, True) + ".npz")
assert store.get(8, 21, 1.0).spot("RFI:CO")["strategy"][1, HAND_INDEX["KK"]] == 0.0
with np.load(path) as z:
    arrays = {k: z[k] for k in z.files}
arrays["RFI__CO__strategy"] = arrays["RFI__CO__strategy"].copy()
arrays["RFI__CO__strategy"][:, HAND_INDEX["KK"]] = [0.0, 1.0]
tmp = path + ".tmp.npz"
np.savez_compressed(tmp, **arrays)
os.replace(tmp, path)
assert gtopen.GtopenChart(path).spot("RFI:CO")["strategy"][1, HAND_INDEX["KK"]] == 1.0
print("re-export over a loaded chart OK")

# ---------------------------------------------------------------- resuming a checkpointed solve
# A loaded save already has its iterations, and GTOpen reports the GPU only once the solve is set up:
# the first status of a resumed chunk shows neither progress nor the GPU, which isn't a CPU fallback.
import time    # noqa: E402
import types   # noqa: E402


class FakeServer:
    calls = []

    def __init__(self, *a, **k):
        self.running, self.polls, self.iteration = False, 0, 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def api(self, path, body=None, timeout=None):
        FakeServer.calls.append(path)
        if path == "/api/preflop/solve":
            self.running, self.polls = True, 0
        if path != "/api/preflop/status":
            return {}
        if not self.running:
            done = {"stop_reason": "target_reached", "gap_total": 0.01, "gaps": [0.01]} if self.iteration > 200 else {}
            return dict(done, state="idle", iteration=self.iteration)
        self.polls += 1
        if self.polls == 1:
            return {"state": "running", "iteration": 200}                 # still setting up
        if self.polls == 2:
            return {"state": "running", "iteration": 200, "gpu": True}    # on the GPU
        self.running, self.iteration = False, 300
        return {"state": "idle", "iteration": 300, "gpu": True, "gap_total": 0.01, "stop_reason": "target_reached"}


gdir = tempfile.mkdtemp()
os.makedirs(os.path.join(gdir, "saves", "preflop"))
open(os.path.join(gdir, "saves", "preflop", gtopen.game_name(8, 60, 1.0, True) + ".gtop"), "w").close()
real = gtopen.Server, gtopen.time
gtopen.Server = FakeServer
gtopen.time = types.SimpleNamespace(sleep=lambda s: None, time=time.time, strftime=time.strftime)
try:
    ok = gtopen.solve(8, 60, 1.0, True, gdir)
finally:
    gtopen.Server, gtopen.time = real
assert ok and "/api/preflop/stop" not in FakeServer.calls, FakeServer.calls
assert "/api/preflop/load" in FakeServer.calls and "/api/preflop/save" in FakeServer.calls
print("resumed solve waits for the GPU report OK")
print("all tests passed")
