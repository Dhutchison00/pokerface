"""User-supplied range files that replace the built-in charts.

Drop *.json files in the ranges/ folder (re-read automatically when changed):

{
  "spot": "RFI",              // RFI | VS_OPEN | VS_SHOVE | VS_3BET | LIMPED
  "hero": "CO",               // hero position
  "villain": "UTG",           // optional: aggressor position (VS_* spots)
  "stack_bb": 20,             // effective stack the chart was solved for
  "stack_tolerance": 3,       // optional, default max(2, 15% of stack_bb)
  "players": 8,               // optional: only use at this table size
  "actions": {                // range strings; whatever is left is fold
    "raise": "AA,KK,AKs,AQs:0.5,A5s-A2s,77+",
    "allin": "66-22:0.5"
  }
}

A file may also hold a list of such objects.  The closest stack wins.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .cards import parse_range, COMBO_WEIGHTS as W, TOTAL_COMBOS
from .strategy import ACTION_ORDER


class Overrides:
    def __init__(self, folder):
        self.folder = folder
        self._sig = None
        self.entries = []
        self.errors = []

    def _scan(self):
        files = sorted(glob.glob(os.path.join(self.folder, "**", "*.json"), recursive=True))
        sig = tuple((f, os.path.getmtime(f)) for f in files)
        if sig == self._sig:
            return
        self._sig = sig
        self.entries, self.errors = [], []
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, d in enumerate(data if isinstance(data, list) else [data]):
                    if "spot" not in d or "actions" not in d:
                        continue
                    acts = {a.lower(): parse_range(r) for a, r in d["actions"].items()}
                    for a in acts:
                        if a not in ACTION_ORDER:
                            raise ValueError(f"unknown action {a!r} (use {ACTION_ORDER})")
                    d = dict(d, _acts=acts, _file=os.path.relpath(f, self.folder) + (f"#{i}" if isinstance(data, list) else ""))
                    self.entries.append(d)
            except Exception as e:  # keep going; show the error on the page
                self.errors.append(f"{os.path.basename(f)}: {e}")

    def find(self, spot):
        self._scan()
        eff = spot.get("eff_bb")
        if eff is None:
            return None
        villain = (spot.get("villain") or {}).get("pos")
        best = None
        for d in self.entries:
            if d["spot"].upper() != spot.get("type"):
                continue
            if d.get("hero", "").upper() != spot.get("hero_pos"):
                continue
            if d.get("villain") and d["villain"].upper() != villain:
                continue
            if d.get("players") and int(d["players"]) != spot.get("n_players"):
                continue
            sbb = float(d.get("stack_bb", eff))
            tol = float(d.get("stack_tolerance", max(2.0, 0.15 * sbb)))
            gap = abs(sbb - eff)
            if gap <= tol and (best is None or gap < best[0]):
                best = (gap, d)
        if not best:
            return None
        d = best[1]
        acts = dict(d["_acts"])
        tot = sum(acts.values())
        if "fold" not in acts and "check" not in acts:
            acts["fold"] = np.clip(1 - tot, 0, 1)
        keys = [a for a in ACTION_ORDER if a in acts]
        freqs = np.stack([acts[a] for a in keys], 1)
        freqs = freqs / np.maximum(freqs.sum(1, keepdims=True), 1e-9)
        pct = {a: round(float((freqs[:, i] * W).sum() / TOTAL_COMBOS * 100), 1) for i, a in enumerate(keys)}
        title = f"{spot.get('hero_pos')} {spot.get('type')} · {d.get('stack_bb', eff)}bb (your file)"
        return {"title": title, "method": f"Override: {d['_file']}", "approx": False, "override": True,
                "actions": keys, "freqs": np.round(freqs, 3).tolist(), "pct": pct}
