"""GTOpen preflop charts for the HUD.

GTOpen (https://github.com/MatthewPDingle/GTOpen) solves the full multiway preflop tree on the
GPU, with flop values learned from exact flop solves. A solve takes minutes (hours with limps),
far too slow to run live, so this module precomputes grid points offline and exports the nodes
the HUD shows into small files under data/gtopen/. The HUD reads those files with no GTOpen
dependency at run time and falls back to its own solver for anything missing.

    python -m preflop_hud.gtopen solve  --players 8 --stack 60 [--limp] [--mode cash | --icm-stage bubble]
    python -m preflop_hud.gtopen export --players 8 --stack 60 [--limp] [...same options]
    python -m preflop_hud.gtopen compare --players 8 --stack 60 [--limp] [...same options]

`solve` runs the game in chunks and saves a GTOpen checkpoint after each one, so an
interrupted run resumes where it stopped. `export` loads the finished save and writes the
HUD chart file. The GTOpen checkout is found from --gtopen-dir, HUD_GTOPEN_DIR or
"gtopen_dir" in config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import modes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# exported charts the HUD reads; HUD_GTOPEN_CHARTS points it somewhere else (tests, experiments)
OUT_DIR = os.environ.get("HUD_GTOPEN_CHARTS") or os.path.join(ROOT, "data", "gtopen")
EXPORT_VERSION = 1
PORT = 3739


# ---------------------------------------------------------------- game definition
def game_name(n, stack, dead, limp, game=None):
    # no dots: GTOpen strips them from save names ("d1.00" would be saved as "d100")
    d = f"{dead:.2f}".replace(".", "p")
    return (f"hud_v{EXPORT_VERSION}_n{n}_s{stack}_d{d}" + ("_limp" if limp else "")
            + modes.gtopen_tag(game))


def game_config(n, stack, dead, limp, game=None, leaf_net="v3"):
    """The HUD's own tree settings (solver.DEFAULT_PARAMS plus the mode's) as a GTOpen PreflopConfig.

    Stacks are chips behind with the ante posted on top, the same in both. Seats keep the
    HUD's names; GTOpen orders seats postflop from the blinds, not the names. Cash adds rake;
    an ICM stage adds its payouts (GTOpen then uses its "balanced" flop values without blockers,
    as its learned values and blocker model don't support ICM).
    """
    from .solver import DEFAULT_PARAMS, _size
    from .spot import POSITION_NAMES
    g = modes.normalize(game)
    P = dict(DEFAULT_PARAMS, **modes.solver_params(g))
    names = POSITION_NAMES[n]
    posts = [0.0] * n
    posts[-1] = 1.0
    posts[-2] = 0.5                      # heads-up: the BTN posts the small blind
    sb = n - 2 if n >= 3 else None
    by_seat = [[] for _ in range(n)]     # [] = the global open size
    if sb is not None:
        by_seat[sb] = [_size(P["sb_open_size"], stack)]
    cfg = {
        "positions": names, "stack": float(stack), "ante": dead / n, "posts": posts,
        "limp": bool(limp),
        "open_raises": [_size(P["open_size"], stack)],
        "open_raises_by_seat": by_seat,
        "raise_mults": [P["threebet_ip"]], "raise_mults_oop": [P["threebet_oop"]],
        # opening over limpers: the HUD's BB iso-raise size against one limper
        "limper_extra": round(P["bb_iso"] - _size(P["open_size"], stack), 3) if limp else 0.0,
        "max_raises": 3, "add_allin": True, "allin_threshold": P["max_raise_frac"] + 1e-9,
        "rake_pct": P.get("rake_pct", 0.0), "rake_cap": P.get("rake_cap", 0.0), "no_flop_no_drop": True,
        "realization": "leafnet", "leaf_net": leaf_net, "card_removal": True,
    }
    if g["icm_stage"]:
        cfg.update(icm=modes.icm_config(g["icm_stage"], n, stack), realization="balanced", card_removal=False)
        cfg.pop("leaf_net")
    return cfg


# ---------------------------------------------------------------- GTOpen server
def find_gtopen_dir(arg=None):
    cand = arg or os.environ.get("HUD_GTOPEN_DIR")
    if not cand:
        try:
            with open(os.path.join(ROOT, "config.json")) as f:
                cand = json.load(f).get("gtopen_dir")
        except (OSError, ValueError):
            cand = None
    if not cand or not os.path.isfile(os.path.join(cand, "target", "release", "gto-server.exe")):
        sys.exit("GTOpen not found: pass --gtopen-dir, set HUD_GTOPEN_DIR, or add \"gtopen_dir\" to "
                 "config.json (a checkout built with: cargo build --release -p server --features gpu)")
    return cand


class Server:
    def __init__(self, gtopen_dir, log_path, port=PORT):
        self.dir, self.port, self.log_path, self.proc = gtopen_dir, port, log_path, None

    def api(self, path, body=None, timeout=3600):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GTOpen {path}: HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")

    def __enter__(self):
        env = dict(os.environ, PORT=str(self.port))
        nvrtc = os.path.join(self.dir, ".cuda-nvrtc", "nvidia", "cuda_nvrtc", "bin")
        if os.path.isdir(nvrtc):
            env["PATH"] = nvrtc + os.pathsep + env.get("PATH", "")
        exe = os.path.join(self.dir, "target", "release", "gto-server.exe")
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen([exe], cwd=self.dir, env=env, stdout=self._log, stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                self.api("/api/version", timeout=2)
                return self
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"gto-server exited; see {self.log_path}")
                time.sleep(0.5)
        raise RuntimeError("gto-server did not start")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._log.close()


# ---------------------------------------------------------------- solving
def solve(n, stack, dead, limp, gtopen_dir, target_gap=0.02, max_iters=3000, chunk=100, port=PORT,
          allow_cpu=False, game=None):
    """Solve one grid point to target_gap (bb, summed over seats), checkpointing every chunk."""
    name = game_name(n, stack, dead, limp, game)
    save_dir = os.path.join(gtopen_dir, "saves", "preflop")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, name + ".log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    with Server(gtopen_dir, os.path.join(save_dir, name + ".server.log"), port) as srv:
        if os.path.exists(os.path.join(save_dir, name + ".gtop")):
            srv.api("/api/preflop/load", {"name": name})
            st = srv.api("/api/preflop/status")
            log(f"resumed {name} at iteration {st['iteration']}, gap {st.get('gap_total')}")
        else:
            built = srv.api("/api/preflop/spot", game_config(n, stack, dead, limp, game))
            log(f"built {name}: {built['nodes']:,} nodes, {built['arena_mb']:.0f} MB")
        t_start = time.time()
        while True:
            st = srv.api("/api/preflop/status")
            if st["iteration"] >= max_iters:
                log(f"stopping at the iteration cap ({max_iters})")
                break
            if st.get("stop_reason") == "target_reached":
                break
            t0 = time.time()
            srv.api("/api/preflop/solve", {"iterations": min(chunk, max_iters - st["iteration"]),
                                           "check_every": 25, "target_gap": target_gap})
            # GTOpen reports GPU vs CPU only once it has set the solve up (~25 s for a big tree)
            t_wait = time.time()
            while True:
                time.sleep(2)
                st = srv.api("/api/preflop/status")
                if (st["state"] != "running" or st.get("gpu") or st.get("gpu_note")
                        or st.get("iteration", 0) > 0 or time.time() - t_wait > 300):
                    break
            if st["state"] == "running" and not st.get("gpu") and not allow_cpu:
                srv.api("/api/preflop/stop", {})
                log(f"aborted: GTOpen fell back to the CPU ({st.get('gpu_note')}); a CPU solve this size won't finish")
                return False
            while st["state"] == "running":
                time.sleep(10)
                st = srv.api("/api/preflop/status")
            if st.get("error"):
                log(f"solver error: {st['error']}")
                return False
            srv.api("/api/preflop/save", {"name": name})
            log(f"iteration {st['iteration']}: gap {st.get('gap_total', 0):.4f} bb "
                f"(target {target_gap}), chunk {time.time() - t0:.0f}s, total {(time.time() - t_start) / 3600:.2f} h, "
                f"{st.get('gpu_note') or ''} -> checkpoint saved")
            if st.get("stop_reason") == "target_reached":
                break
        st = srv.api("/api/preflop/status")
        log(f"done: iteration {st['iteration']}, gap {st.get('gap_total', 0):.4f} bb, "
            f"per seat {[round(g, 4) for g in st.get('gaps') or []]}")
        # a loaded save doesn't report its gap, so keep the final numbers next to it for `export`
        with open(os.path.join(save_dir, name + ".result.json"), "w") as f:
            json.dump({"iteration": st["iteration"], "gap_total": st.get("gap_total"), "gaps": st.get("gaps"),
                       "target_gap": target_gap, "hours": round((time.time() - t_start) / 3600, 3)}, f)
        return True


# ---------------------------------------------------------------- export to HUD charts
KIND_TO_HUD = {"fold": "fold", "check": "check", "call": "call", "raise": "raise", "jam": "allin"}


def _gtopen_to_hud_order():
    """GTOpen class index -> HUD class index. GTOpen: ranks 2..A = 0..12, idx = row*13 + col,
    pairs on the diagonal, suited when row > col (row = high rank), offsuit when row < col."""
    import numpy as np
    from .cards import HAND_INDEX
    ranks = "23456789TJQKA"
    out = []
    for idx in range(169):
        r, c = divmod(idx, 13)
        if r == c:
            name = ranks[r] * 2
        elif r > c:
            name = ranks[r] + ranks[c] + "s"
        else:
            name = ranks[c] + ranks[r] + "o"
        out.append(HAND_INDEX[name])
    return np.array(out)


def hud_lines(n, limp):
    """HUD node key -> (planned non-fold actions as (seat name, GTOpen kind), seat to stop at).

    Every seat not in the plan folds (or checks). Lines where the small-blind seat opens or limps
    are only exported from games with limps: the HUD's own tree lets that seat limp, so a no-limp
    game would give it a different range there.
    """
    from .spot import POSITION_NAMES
    names = POSITION_NAMES[n]
    sb_role = names[-2]                  # SB, or the BTN heads-up
    bb = names[-1]
    lines = {}
    for i, o in enumerate(names[:-1]):
        if o == sb_role and not limp:
            continue
        lines[f"RFI:{o}"] = ([], o)
        for d in names[i + 1:]:
            lines[f"VS_OPEN:{d}:{o}"] = ([(o, "raise")], d)
            lines[f"VS_JAM:{d}:{o}"] = ([(o, "jam")], d)
            lines[f"VS_3BET:{o}:{d}"] = ([(o, "raise"), (d, "raise")], o)
            lines[f"VS_3BETJAM:{o}:{d}"] = ([(o, "raise"), (d, "jam")], o)
            lines[f"VS_4BET:{d}:{o}"] = ([(o, "raise"), (d, "raise"), (o, "jam")], d)
    if limp:
        lines["BB_VS_LIMP"] = ([(sb_role, "call")], bb)
        lines["SB_VS_ISO"] = ([(sb_role, "call"), (bb, "raise")], sb_role)
        lines["SB_VS_BBJAM"] = ([(sb_role, "call"), (bb, "jam")], sb_role)
        lines["BB_VS_SBJAM_ISO"] = ([(sb_role, "call"), (bb, "raise"), (sb_role, "jam")], bb)
    return lines


def _walk(srv, plan, target):
    """Follow a line from the root; returns the target's node view, or (None, reason)."""
    path, plan = [], list(plan)
    for _ in range(64):
        v = srv.api("/api/preflop/node", {"path": path})
        if v.get("kind") != "action":
            return None, f"line ends at a {v.get('kind')} node"
        actor = v["actor_pos"]
        kinds = [a["kind"] for a in v["actions"]]
        unknown = set(kinds) - set(KIND_TO_HUD)
        if unknown:
            raise RuntimeError(f"unknown GTOpen action kind(s) {unknown} at {path}")
        if plan and plan[0][0] == actor:
            want = plan.pop(0)[1]
        elif not plan and actor == target:
            return v, None
        else:
            want = "fold" if "fold" in kinds else "check"
        if want not in kinds:
            return None, f"{actor} has no {want} action ({kinds})"
        path.append(kinds.index(want))
    return None, "line too long"


def export(n, stack, dead, limp, gtopen_dir, port=PORT, srv=None, game=None):
    """Load the finished GTOpen save and write data/gtopen/<name>.npz for the HUD."""
    import numpy as np
    from .cards import HAND_INDEX
    from .solver import DEFAULT_PARAMS
    name = game_name(n, stack, dead, limp, game)
    if srv is None:
        save_dir = os.path.join(gtopen_dir, "saves", "preflop")
        if not os.path.exists(os.path.join(save_dir, name + ".gtop")):
            sys.exit(f"no GTOpen save {name}.gtop in {save_dir}: run `solve` first")
        with Server(gtopen_dir, os.path.join(save_dir, name + ".server.log"), port) as s:
            s.api("/api/preflop/load", {"name": name})
            return export(n, stack, dead, limp, gtopen_dir, port, s, game)
    order = _gtopen_to_hud_order()
    st = srv.api("/api/preflop/status")
    result = os.path.join(gtopen_dir, "saves", "preflop", name + ".result.json")
    log_file = os.path.join(gtopen_dir, "saves", "preflop", name + ".log")
    if os.path.exists(result):
        with open(result) as f:
            st.update(json.load(f))          # the solve's own final iteration and gap
    elif os.path.exists(log_file):           # older runs: the last checkpoint line of the solve log
        import re
        with open(log_file) as f:
            hits = re.findall(r"iteration (\d+): gap ([\d.]+) bb", f.read())
        if hits:
            st["iteration"], st["gap_total"] = int(hits[-1][0]), float(hits[-1][1])
    arrays, spots, skipped = {}, {}, {}
    for key, (plan, target) in hud_lines(n, limp).items():
        v, why = _walk(srv, plan, target)
        if v is None or not v.get("strategy") or not v.get("reach"):
            skipped[key] = why or v.get("strategy_note") or "no strategy"
            continue
        acts = [KIND_TO_HUD[a["kind"]] for a in v["actions"]]
        g = np.array(v["strategy"], dtype=np.float64).reshape(len(acts), 169)
        names = list(dict.fromkeys(acts))            # merge same-kind actions (several raise sizes)
        strat = np.zeros((len(names), 169))
        for i, a in enumerate(acts):
            strat[names.index(a), order] += g[i]
        reach = np.zeros(169)
        reach[order] = np.array(v["reach"], dtype=np.float64)
        safe = key.replace(":", "__")
        arrays[f"{safe}__strategy"] = strat.astype(np.float32)
        arrays[f"{safe}__reach"] = reach.astype(np.float32)
        spots[key] = {"actions": names, "low_reach": bool(v.get("low_reach")),
                      "labels": [a["label"] for a in v["actions"]]}
    # sanity: the first seat opens AA and folds 72o (catches a class-order or path mistake)
    from .spot import POSITION_NAMES
    first = f"RFI:{POSITION_NAMES[n][0]}"
    if first in spots:
        s = arrays[first.replace(":", "__") + "__strategy"]
        fold = spots[first]["actions"].index("fold")
        aa, junk = s[fold, HAND_INDEX["AA"]], s[fold, HAND_INDEX["72o"]]
        # 4+ handed the first seat opens AA and folds 72o; short-handed (the BTN heads-up) 72o can play
        ok = (aa < 0.1 and junk > 0.9) if n >= 4 else (aa < 0.1 and junk > aa)
        assert ok, f"{first}: AA fold {aa:.2f}, 72o fold {junk:.2f}: class order or line mapping is wrong"
    try:
        version = srv.api("/api/version")
    except Exception:
        version = {}
    meta = {"source": "gtopen", "export_version": EXPORT_VERSION, "n": n, "stack": stack, "dead": dead,
            "limp": bool(limp), "iterations": st.get("iteration"), "gap_total": st.get("gap_total"),
            "gaps": st.get("gaps"), "gtopen_version": version.get("version"),
            "gtopen_commit": version.get("commit"), "config": game_config(n, stack, dead, limp, game),
            "tag": modes.gtopen_tag(game), "game": modes.normalize(game),
            "params": DEFAULT_PARAMS, "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spots": spots, "skipped": skipped}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name + ".npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"exported {len(spots)} spots, skipped {len(skipped)}: "
          + ", ".join(f"{k} ({w})" for k, w in list(skipped.items())[:8]))
    return path


def compare(n, stack, dead, limp, game=None):
    """Print each exported node next to the HUD solver's same node (cached HUD solutions only)."""
    import numpy as np
    from .cards import COMBO_WEIGHTS as W
    from .solver import Solution, sol_path
    gt = GtopenChart(os.path.join(OUT_DIR, game_name(n, stack, dead, limp, game) + ".npz"))
    hp = sol_path(n, stack, dead, modes.solver_tag(game))
    hud = Solution(hp) if os.path.exists(hp) else None
    m = gt.meta
    print(f"GTOpen {n}-handed {stack}bb, {dead}bb antes{' (limps)' if limp else ''}: {m['iterations']} iterations, "
          f"gap {m['gap_total']:.4f} bb" + ("" if hud else "  (no cached HUD solution to compare with)"))
    print(f"{'node':22s} {'GTOpen':38s} {'HUD':38s} differ")

    def summary(d):
        reach = np.asarray(d["reach"], float)
        w = W * reach
        tot = max(w.sum(), 1e-9)
        s = np.asarray(d["strategy"], float)
        return {a: (s[i] * w).sum() / tot * 100 for i, a in enumerate(d["actions"])}, s, reach

    low = [k for k, info in m["spots"].items() if info.get("low_reach")]
    for key in m["spots"]:
        if key in low:
            continue
        g, gs, greach = summary(gt.spot(key))
        line = f"{key:22s} " + " ".join(f"{a} {p:4.1f}" for a, p in g.items())
        h = hud.spot(key) if hud else None
        if h is None:
            print(line)
            continue
        hsum, hs, _ = summary(h)
        acts = sorted(set(gt.spot(key)["actions"]) | set(h["actions"]))
        ga = np.array([gs[gt.spot(key)["actions"].index(a)] if a in gt.spot(key)["actions"] else np.zeros(169)
                       for a in acts])
        ha = np.array([hs[h["actions"].index(a)] if a in h["actions"] else np.zeros(169) for a in acts])
        w = W * greach
        tv = (0.5 * np.abs(ga - ha).sum(0) * w).sum() / max(w.sum(), 1e-9) * 100
        print(f"{line:61s} " + " ".join(f"{a} {p:4.1f}" for a, p in hsum.items()).ljust(38) + f" {tv:5.1f}%")
    if low:
        print(f"\n{len(low)} low-reach nodes (not converged; the HUD uses its own solver there): {', '.join(low)}")


# ---------------------------------------------------------------- HUD side
class GtopenChart:
    """One exported grid point; same .spot(key) interface as solver.Solution (no EVs)."""

    def __init__(self, path):
        import numpy as np
        self.z = np.load(path)
        self.meta = json.loads(bytes(self.z["meta"]).decode())
        self.path = path

    def spot(self, key):
        info = self.meta["spots"].get(key)
        if info is None or info.get("low_reach"):
            # GTOpen: a line taken under ~0.1% of the time isn't converged, its ranges mean
            # nothing (e.g. facing an open-jam at deep stacks) -> the HUD's own solver instead
            return None
        safe = key.replace(":", "__")
        return {"actions": info["actions"], "strategy": self.z[f"{safe}__strategy"], "ev": None,
                "reach": self.z[f"{safe}__reach"]}


class GtopenStore:
    """Finds an exported GTOpen chart for (players, grid stack, dead money)."""

    def __init__(self, directory=OUT_DIR, dead_tolerance=0.5):
        self.dir, self.tol = directory, dead_tolerance
        self._index = None
        self._mtime = None
        self._cache = {}

    def _refresh(self):
        """Rescan when the folder changes, so a running HUD picks up new exports."""
        try:
            mtime = os.stat(self.dir).st_mtime
        except OSError:
            mtime = None
        if self._index is None or mtime != self._mtime:
            self._index, self._mtime = self._scan(), mtime
            self._cache = {}

    def _scan(self):
        import numpy as np
        index = []
        if os.path.isdir(self.dir):
            for f in sorted(os.listdir(self.dir)):
                if not f.endswith(".npz") or f.endswith(".tmp.npz"):
                    continue
                try:
                    with np.load(os.path.join(self.dir, f)) as z:
                        m = json.loads(bytes(z["meta"]).decode())
                    if m.get("export_version") == EXPORT_VERSION:
                        index.append((m["n"], m["stack"], m["dead"], m["limp"], m.get("tag", ""), os.path.join(self.dir, f)))
                except Exception:
                    continue
        return index

    def get(self, n, grid_stack, dead, tag=""):
        """tag: modes.gtopen_tag of the current mode; exports of other modes are never used."""
        self._refresh()
        best = None
        for (fn, fs, fd, fl, ft, path) in self._index:
            if fn != n or fs != grid_stack or ft != tag or abs(fd - dead) > self.tol:
                continue
            rank = (abs(fd - dead), not fl)          # nearest antes first, then the game with limps
            if best is None or rank < best[0]:
                best = (rank, path)
        if best is None:
            return None
        path = best[1]
        if path not in self._cache:
            self._cache[path] = GtopenChart(path)
        return self._cache[path]


def batch(n, stacks, dead, limp, gtopen_dir, game=None, target_gap=0.02, max_iters=3000, port=PORT,
          keep_saves=False):
    """Solve and export several grid stacks in turn. With limp=True a stack whose limp tree GTOpen
    refuses (too big for memory) or can't keep on the GPU is solved without limps instead.
    Stacks already exported are skipped, so a stopped batch picks up where it left off."""
    import ctypes
    if os.name == "nt":       # keep the PC from idle-sleeping while this process runs (no settings change)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    save_dir = os.path.join(gtopen_dir, "saves", "preflop")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"batch_n{n}_{time.strftime('%Y%m%d-%H%M%S')}.log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def exported(lp):
        return os.path.exists(os.path.join(OUT_DIR, game_name(n, s, dead, lp, game) + ".npz"))

    log(f"batch: {n}-handed, stacks {stacks}, antes {dead}bb, limps {'on' if limp else 'off'}, "
        f"{modes.describe(game)}")
    for s in stacks:
        if exported(limp) or (limp and exported(False)):
            log(f"{s}bb: already exported, skipped")
            continue
        t0 = time.time()
        tries = [True, False] if limp else [False]
        for lp in tries:
            try:
                ok = solve(n, s, dead, lp, gtopen_dir, target_gap, max_iters, port=port, game=game)
            except RuntimeError as e:
                ok = False
                log(f"{s}bb{' limps' if lp else ''}: {e}")
            if ok:
                path = export(n, s, dead, lp, gtopen_dir, port, game=game)
                log(f"{s}bb{' limps' if lp else ''}: done in {(time.time() - t0) / 60:.0f} min -> {os.path.basename(path)}")
                if not keep_saves:   # 2-4 GB each; the export has what the HUD uses (logs are kept)
                    gtop = os.path.join(save_dir, game_name(n, s, dead, lp, game) + ".gtop")
                    if os.path.exists(gtop):
                        os.remove(gtop)
                break
            if lp:
                log(f"{s}bb: limp tree refused or off the GPU: solving without limps")
        else:
            log(f"{s}bb: FAILED, see the solve log")
    log("batch finished")
    return log_path


def main(argv=None):
    from .solver import SolverStore
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.gtopen", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["solve", "export", "compare", "batch"])
    ap.add_argument("--players", type=int, required=True)
    ap.add_argument("--stack", type=int, nargs="+", required=True,
                    help="stack in bb (a HUD grid depth); batch takes several, solved in the order given")
    ap.add_argument("--ante-per-player", type=float, default=None,
                    help="ante in bb per player (default 0.125 in tournaments, 0 in cash)")
    ap.add_argument("--limp", action="store_true", help="allow open-limps and limping behind (much larger tree)")
    ap.add_argument("--mode", choices=["tournament", "cash"], default="tournament")
    ap.add_argument("--icm-stage", choices=sorted(modes.ICM_STAGES), help="tournament ICM stage (default chip EV)")
    ap.add_argument("--rake-pct", type=float, default=modes.DEFAULT_GAME["rake_pct"], help="cash rake %%")
    ap.add_argument("--rake-cap", type=float, default=modes.DEFAULT_GAME["rake_cap"], help="cash rake cap in bb")
    ap.add_argument("--target-gap", type=float, default=0.02, help="summed best-response gap in bb")
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--gtopen-dir")
    ap.add_argument("--port", type=int, default=PORT, help="port for the GTOpen server this command starts")
    ap.add_argument("--cpu", action="store_true", help="solve on the CPU (small games only)")
    ap.add_argument("--keep-saves", action="store_true",
                    help="batch: keep each GTOpen checkpoint (2-4 GB) after exporting it")
    a = ap.parse_args(argv)
    game = modes.normalize({"mode": a.mode, "icm_stage": a.icm_stage, "rake_pct": a.rake_pct,
                            "rake_cap": a.rake_cap})
    ante = a.ante_per_player if a.ante_per_player is not None else (0.0 if a.mode == "cash" else 0.125)
    dead = SolverStore.dead_key(ante * a.players)
    if a.command != "batch":
        if len(a.stack) != 1:
            ap.error(f"{a.command} takes one --stack (batch takes several)")
        a.stack = a.stack[0]
    if a.command == "compare":
        compare(a.players, a.stack, dead, a.limp, game)
        return 0
    gdir = find_gtopen_dir(a.gtopen_dir)
    if a.cpu:
        os.environ["SOLVER_GPU"] = "0"
    if a.command == "batch":
        batch(a.players, a.stack, dead, a.limp, gdir, game, a.target_gap, a.max_iters, a.port, a.keep_saves)
        return 0
    if a.command == "solve":
        ok = solve(a.players, a.stack, dead, a.limp, gdir, a.target_gap, a.max_iters, port=a.port,
                   allow_cpu=a.cpu, game=game)
        if ok:
            print("wrote", export(a.players, a.stack, dead, a.limp, gdir, a.port, game=game))
        return 0 if ok else 1
    path = export(a.players, a.stack, dead, a.limp, gdir, a.port, game=game)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
