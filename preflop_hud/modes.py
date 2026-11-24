"""Game modes: tournament (chip EV, or ICM at a stage preset) and cash (rake, cash sizes).

The mode is kept in config.json under "game" and switched from the page. It decides
  * the HUD solver's tree sizes and rake, and which cache it reads (data/solutions/v2_<tag>_...),
  * which GTOpen exports are used (each export records the mode it was solved for).
The HUD's own solver has no ICM: in an ICM stage only GTOpen exports solved for that stage
are ICM-aware, everything else is shown as chip EV with a note.
"""
from __future__ import annotations

DEFAULT_GAME = {"mode": "tournament", "icm_stage": None, "rake_pct": 5.0, "rake_cap": 3.0}

# stage presets of a 1,000-player tournament (15% paid), as in GTOpen's Preflop Lab
ICM_STAGES = {
    "ft": ("Final table", 8),
    "t2": ("Two tables left", 16),
    "pct10": ("10% left, in the money", 100),
    "bubble": ("Bubble (152 left, 150 paid)", 152),
    "pct50": ("Half the field left", 500),
}

# cash tree sizes; everything else (3-bet multipliers, jam rule) is the HUD's default
CASH_SIZES = {"open_size": [[1000, 2.5]], "sb_open_size": [[1000, 3.0]]}


def normalize(game) -> dict:
    g = dict(DEFAULT_GAME)
    g.update({k: v for k, v in (game or {}).items() if k in DEFAULT_GAME})
    if g["mode"] not in ("tournament", "cash"):
        raise ValueError(f"unknown mode {g['mode']!r} (tournament or cash)")
    if g["mode"] == "cash" or g["icm_stage"] in ("", "chip", "none"):
        g["icm_stage"] = None
    if g["icm_stage"] is not None and g["icm_stage"] not in ICM_STAGES:
        raise ValueError(f"unknown ICM stage {g['icm_stage']!r} ({', '.join(ICM_STAGES)})")
    g["rake_pct"], g["rake_cap"] = float(g["rake_pct"]), float(g["rake_cap"])
    if not (0 <= g["rake_pct"] < 100 and g["rake_cap"] >= 0):
        raise ValueError("rake_pct must be 0..100 and rake_cap >= 0")
    return g


def _num(x):
    return f"{x:g}".replace(".", "p")


def solver_params(game) -> dict:
    """Overrides of solver.DEFAULT_PARAMS for the HUD's own solver."""
    g = normalize(game)
    if g["mode"] == "cash":
        return dict(CASH_SIZES, rake_pct=g["rake_pct"], rake_cap=g["rake_cap"])
    return {}


def solver_tag(game) -> str:
    """Cache tag of the HUD solver's solutions; "default" keeps the existing tournament cache."""
    g = normalize(game)
    return f"cash_r{_num(g['rake_pct'])}c{_num(g['rake_cap'])}" if g["mode"] == "cash" else "default"


def gtopen_tag(game) -> str:
    """Suffix of GTOpen export names; empty for tournament chip EV (the original exports)."""
    g = normalize(game)
    if g["mode"] == "cash":
        return f"_cash_r{_num(g['rake_pct'])}c{_num(g['rake_cap'])}"
    return f"_icm_{g['icm_stage']}" if g["icm_stage"] else ""


def describe(game) -> str:
    g = normalize(game)
    if g["mode"] == "cash":
        return f"Cash · {g['rake_pct']:g}% rake, {g['rake_cap']:g}bb cap"
    return f"Tournament · ICM {ICM_STAGES[g['icm_stage']][0].lower()}" if g["icm_stage"] else "Tournament · chip EV"


def standard_payouts(entrants=1000, paid_frac=0.15):
    """Approximate MTT payouts in % of the pool: 1/place, min cash 1.5 buy-ins (GTOpen's preset)."""
    paid = max(1, round(entrants * paid_frac))
    min_cash = 1.5 / entrants
    raw = [1 / (k + 1) for k in range(paid)]
    s = sum(raw)
    raw = [max(x / s, min_cash) for x in raw]
    t = sum(raw)
    return [round(100 * x / t, 4) for x in raw]


def icm_config(stage, n, stack) -> dict:
    """GTOpen PreflopConfig.icm for a stage: the payouts still to be won, and the players at
    other tables (assumed to have the table's stack)."""
    left = ICM_STAGES[stage][1]
    payouts = standard_payouts()[:left]
    cfg = {"payouts": payouts}
    if left > n:
        cfg["field_groups"] = [{"stack": float(stack), "count": left - n}]
    return cfg
