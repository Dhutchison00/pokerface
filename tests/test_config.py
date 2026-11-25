"""config.json saves: python tests/test_config.py"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preflop_hud import server  # noqa: E402

# A HUD that started before "gtopen_dir" was added by hand must not drop it when it
# saves the region or the game mode (that's how the flop solver lost its GTOpen path).
with tempfile.TemporaryDirectory() as d:
    server.CONFIG = os.path.join(d, "config.json")
    stale = {"region": {"left": 1}, "game": {"mode": "tournament"}}
    with open(server.CONFIG, "w") as f:
        json.dump(dict(stale, gtopen_dir="C:/gtopen"), f)
    server.save_config({"region": {"left": 2}})
    server.save_config({"game": {"mode": "cash"}})
    with open(server.CONFIG) as f:
        cfg = json.load(f)
    assert cfg == {"region": {"left": 2}, "game": {"mode": "cash"}, "gtopen_dir": "C:/gtopen"}, cfg

    os.remove(server.CONFIG)                      # no file yet: the save creates it
    server.save_config({"region": {"left": 3}})
    assert server.load_config() == {"region": {"left": 3}}
print("config saves keep other keys OK")
print("all tests passed")
