"""Can a small net fit 3-way turn values from an affordable amount of data? The pilot's readout.

    A:\\poker-pilot\\venv\\Scripts\\python.exe -m preflop_hud.pilot_train            (the learning curve)
    A:\\poker-pilot\\venv\\Scripts\\python.exe -m preflop_hud.pilot_train --check    (pipeline checks only)

Reads the solved turns pilot_data wrote (data/pilot/train_*.npz, test_*.npz) and fits the first
player's value for every hand, in pots. The test set is the fixed one (its own seed); examples are
never split by hand. Error is the same measure validate_targets used, so the numbers sit side by
side: the reach-weighted mean difference between predicted and solved value, in % of the pot.

Two nets, both predicting a correction to a showdown baseline (a hand's chance to beat both ranges
by the river, "eq3", which costs nothing to compute):

  per-hand   one small net shared by every hand: the hand's own features (its chance against each
             range, its place in its own range) plus a summary of the spot (stacks, range shapes,
             board texture). Few weights, so it should need the least data.
  global     the DeepStack shape: all three ranges, the board and the stacks in, 1326 values out.

Baselines: every hand worth a third of the pot, and eq3 alone. The readout is a learning curve:
held-out error at growing training sizes, so it shows whether more data would still help.

The rule, fixed before any full run:
  at most TARGET_PCT of the pot   as good as the solver's own drift between a loose and a tight solve
  at most USABLE_PCT              inside what the cheap bet menu already costs against a richer one
  more                            not there
and "still improving" if the last doubling of the data cut the error by more than 5%.

What this cannot show: the other two players' values, ranges after real flop betting (these are
reshaped preflop charts), or whether a flop search built on the net plays well.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .evaluator import evaluate7
from .pilot_data import OUT, PAIRS

TARGET_PCT, USABLE_PCT = 0.5, 1.5
EXPLOIT_MAX = 0.16              # examples solved worse than this (% of the pot) are left out
N_CTX = 38


# ---------------------------------------------------------------- features
def hand_features(board, ranges):
    """Per combo: its chance to beat each opponent's range by the river (ties half), and both at once
    ("eq3"), averaged over river cards. Card removal between the hands is ignored: it is a feature."""
    board = [int(c) for c in board]
    live = np.array([c for c in range(52) if c not in board])
    ok = ~np.isin(PAIRS, board).any(axis=1)
    idx = np.flatnonzero(ok)
    rows = np.empty((len(idx), len(live), 7), np.int64)
    rows[:, :, 0:2] = PAIRS[idx][:, None, :]
    rows[:, :, 2:6] = board
    rows[:, :, 6] = live[None, :]
    score = evaluate7(rows.reshape(-1, 7)).reshape(len(idx), len(live)).astype(np.int64)
    blocked = (PAIRS[idx][:, :, None] == live[None, None, :]).any(axis=1)
    p = np.zeros((2, len(idx), len(live)))
    for o in (1, 2):
        w = ranges[o][idx].astype(np.float64)
        for j in range(len(live)):
            good = ~blocked[:, j]
            sc, ww = score[good, j], w[good]
            order = np.argsort(sc, kind="stable")
            ss, cw = sc[order], np.concatenate([[0.0], np.cumsum(ww[order])])
            lo, hi = np.searchsorted(ss, sc, "left"), np.searchsorted(ss, sc, "right")
            p[o - 1, good, j] = (cw[lo] + 0.5 * (cw[hi] - cw[lo])) / max(cw[-1], 1e-12)
    n = np.maximum((~blocked).sum(axis=1), 1)
    out = np.zeros((3, len(PAIRS)), np.float32)
    out[0, idx] = (p[0] * ~blocked).sum(axis=1) / n
    out[1, idx] = (p[1] * ~blocked).sum(axis=1) / n
    out[2, idx] = (p[0] * p[1] * ~blocked).sum(axis=1) / n
    return out


def context(board, ranges, stacks, feats):
    """A summary of the spot shared by every hand of the example (N_CTX numbers)."""
    board = np.asarray(board, int)
    r, s = board // 4, board % 4
    sc = np.bincount(s, minlength=4)
    rs = np.sort(np.unique(r))
    run = max(int(((rs >= a) & (rs <= a + 4)).sum()) for a in range(-1, 9))
    ctx = [np.log(stacks[0]), np.log(stacks[1]), np.log(stacks[2]), np.log(stacks.min()),
           sc.max() / 4.0, float(len(np.unique(r)) < 4), run / 4.0, r.max() / 12.0, r.mean() / 12.0]
    w0 = ranges[0] / max(ranges[0].sum(), 1e-9)
    for k in range(3):                                   # how the first player's range does against each
        ctx += [float((w0 * feats[k]).sum()), float(np.sqrt((w0 * (feats[k] - (w0 * feats[k]).sum()) ** 2).sum()))]
    ctx += list(np.histogram(feats[2], bins=5, range=(0, 1), weights=w0)[0])
    for o in (0, 1, 2):
        ctx.append(np.log(max(float(ranges[o].sum()), 1.0)) / 7.0)
    for o in (1, 2):                                     # each opponent's range by raw hand strength
        wo = ranges[o] / max(ranges[o].sum(), 1e-9)
        ctx += list(np.histogram(_raw_strength(feats, o), bins=5, range=(0, 1), weights=wo)[0])
    ctx += [0.0] * (N_CTX - len(ctx))
    return np.array(ctx[:N_CTX], np.float32)


def _raw_strength(feats, o):
    """A stand-in for an opponent's hand strength: how its hands fare against the other opponent's
    range (feats[0] is against opponent 1, feats[1] against opponent 2)."""
    return feats[2 - o]


def shard_features(path):
    """Features for one shard, cached beside it (feat_<name>)."""
    cache = os.path.join(os.path.dirname(path), "feat_" + os.path.basename(path))
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return cache
    z = np.load(path)
    feats = np.stack([hand_features(z["board"][i], z["ranges"][i]) for i in range(len(z["board"]))])
    ctx = np.stack([context(z["board"][i], z["ranges"][i], z["stacks"][i], feats[i]) for i in range(len(feats))])
    tmp = cache + ".tmp.npz"
    np.savez_compressed(tmp, feats=feats.astype(np.float16), ctx=ctx)
    os.replace(tmp, cache)
    return cache


def load(split, workers=4, limit=None):
    names = sorted(f for f in os.listdir(OUT) if f.startswith(split + "_") and f.endswith(".npz") and ".tmp" not in f)
    names = [n for n in names if n[len(split) + 1:len(split) + 6].isdigit()]
    if limit:
        names = names[:limit]
    paths = [os.path.join(OUT, n) for n in names]
    if workers > 1 and len(paths) > 2:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            caches = pool.map(shard_features, paths)
    else:
        caches = [shard_features(p) for p in paths]
    cols = {k: [] for k in ("board", "ranges", "stacks", "ev", "exploit", "feats", "ctx", "fam")}
    for p, c in zip(paths, caches):
        z, f = np.load(p), np.load(c)
        meta = json.loads(bytes(z["meta"]).decode())
        keep = np.isfinite(z["exploit"]) & (z["exploit"] <= EXPLOIT_MAX) & (z["nodes"] > 0)
        for k in ("board", "ranges", "stacks", "ev", "exploit"):
            cols[k].append(z[k][keep])
        cols["feats"].append(f["feats"][keep].astype(np.float32))
        cols["ctx"].append(f["ctx"][keep])
        cols["fam"].append(np.array(["+".join(sorted(set(m["families"]))) for m in meta])[keep])
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["mask"] = np.isfinite(d["ev"]) & (d["ranges"][:, 0] > 0)
    d["w"] = np.where(d["mask"], d["ranges"][:, 0], 0.0).astype(np.float32)
    d["w"] /= np.maximum(d["w"].sum(axis=1, keepdims=True), 1e-9)
    d["ev"] = np.where(d["mask"], d["ev"], 0.0).astype(np.float32)
    return d


# ---------------------------------------------------------------- error
def error(pred, d):
    """(mean over examples of the reach-weighted |difference|, the weighted 90th percentile over all
    hands), both in % of the pot, and the per-example errors."""
    diff = np.abs(pred - d["ev"]) * d["mask"] * 100
    per = (diff * d["w"]).sum(axis=1)
    x, w = diff[d["mask"]], d["w"][d["mask"]]
    order = np.argsort(x)
    p90 = float(x[order][np.searchsorted(np.cumsum(w[order]) / w.sum(), 0.9)])
    return float(per.mean()), p90, per


# ---------------------------------------------------------------- nets
def hand_rows(d):
    """One row per hand in the first player's range: its own features, then the spot's summary."""
    ex, h = np.nonzero(d["mask"])
    own = d["ranges"][ex, 0, h]
    top = d["ranges"][:, 0].max(axis=1)[ex]
    pair = (PAIRS[h, 0] // 4 == PAIRS[h, 1] // 4).astype(np.float32)
    suited = (PAIRS[h, 0] % 4 == PAIRS[h, 1] % 4).astype(np.float32)
    x = np.column_stack([d["feats"][ex, 0, h], d["feats"][ex, 1, h], d["feats"][ex, 2, h], own / np.maximum(top, 1e-9),
                         pair, suited, PAIRS[h, 1] // 4 / 12.0, PAIRS[h, 0] // 4 / 12.0, d["ctx"][ex]])
    return x.astype(np.float32), ex, h


def fit_per_hand(tr, te, epochs=30, seed=0, shuffle=False, quiet=True):
    import torch
    torch.manual_seed(seed)
    x, ex, h = hand_rows(tr)
    y = (tr["ev"][ex, h] - tr["feats"][ex, 2, h]).astype(np.float32)
    w = tr["w"][ex, h] * len(tr["ev"])
    if shuffle:
        y = y[np.random.default_rng(seed).permutation(len(y))]
    mu, sd = x.mean(0), x.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-5)
    xt, yt, wt = torch.from_numpy((x - mu) / sd), torch.from_numpy(y), torch.from_numpy(w.astype(np.float32))
    steps = max(1, len(xt) // 2048)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * steps)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(steps):
            b = perm[i * 2048:(i + 1) * 2048]
            loss = (wt[b] * torch.nn.functional.smooth_l1_loss(net(xt[b]).squeeze(1), yt[b], beta=0.05,
                                                              reduction="none")).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    def predict(d):
        xx, e2, h2 = hand_rows(d)
        with torch.no_grad():
            out = net(torch.from_numpy((xx - mu) / sd)).squeeze(1).numpy()
        pred = np.zeros_like(d["ev"])
        pred[e2, h2] = out + d["feats"][e2, 2, h2]
        return pred
    return predict


def fit_global(tr, te, epochs=60, seed=0):
    import torch
    torch.manual_seed(seed)

    def inputs(d):
        rn = d["ranges"] / np.maximum(d["ranges"].sum(axis=2, keepdims=True), 1e-9) * 100
        b = np.zeros((len(d["board"]), 52), np.float32)
        b[np.arange(len(b))[:, None], d["board"].astype(int)] = 1.0
        return np.concatenate([rn.reshape(len(b), -1), d["feats"].reshape(len(b), -1), b, np.log(d["stacks"])],
                              axis=1).astype(np.float32)
    x = torch.from_numpy(inputs(tr))
    y = torch.from_numpy(tr["ev"] - tr["feats"][:, 2])
    w = torch.from_numpy(tr["w"])
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 768), torch.nn.ReLU(), torch.nn.Linear(768, 768),
                              torch.nn.ReLU(), torch.nn.Linear(768, 1326))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = max(1, len(x) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * steps)
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(steps):
            b = perm[i * 64:(i + 1) * 64]
            loss = (w[b] * torch.nn.functional.smooth_l1_loss(net(x[b]), y[b], beta=0.05, reduction="none")).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    def predict(d):
        with torch.no_grad():
            return net(torch.from_numpy(inputs(d))).numpy() + d["feats"][:, 2]
    return predict


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def verdict(e):
    return "target-grade" if e <= TARGET_PCT else "usable" if e <= USABLE_PCT else "not there"


# ---------------------------------------------------------------- runs
def check(tr, te):
    """Pipeline checks on whatever data exists: not a result."""
    print(f"pipeline check on {len(tr['ev'])} training and {len(te['ev'])} test examples (not a result)")
    print(f"  hands per example in the first player's range: median {int(np.median(tr['mask'].sum(1)))}")
    print(f"  baseline, a third of the pot: {error(np.full_like(te['ev'], 1 / 3), te)[0]:.2f}%")
    print(f"  baseline, eq3 alone:          {error(te['feats'][:, 2], te)[0]:.2f}%")
    small = subset(tr, np.arange(min(64, len(tr["ev"]))))
    g = fit_global(small, small, epochs=300)
    print(f"  global net memorising {len(small['ev'])} examples: {error(g(small), small)[0]:.2f}% on them (should be near 0)")
    p = fit_per_hand(tr, te)
    print(f"  per-hand net:                 {error(p(te), te)[0]:.2f}% held out, {error(p(tr), tr)[0]:.2f}% on its training set")
    s = fit_per_hand(tr, te, shuffle=True)
    print(f"  per-hand net, labels shuffled: {error(s(te), te)[0]:.2f}% held out (should be no better than eq3 alone)")


def curve(tr, te, out_path):
    n = len(tr["ev"])
    sizes = [s for s in (250, 500, 1000, 2000, 4000, 8000, 16000, 32000) if s < n] + [n]
    rep = {"train_examples": n, "test_examples": len(te["ev"]), "rule": {"target": TARGET_PCT, "usable": USABLE_PCT},
           "baselines": {"third_of_pot": error(np.full_like(te["ev"], 1 / 3), te)[0],
                         "eq3": error(te["feats"][:, 2], te)[0]}, "curve": []}
    print(f"{n} training examples, {len(te['ev'])} test examples; error = reach-weighted mean difference, % of the pot")
    print(f"  baselines: a third of the pot {rep['baselines']['third_of_pot']:.2f}%, eq3 alone {rep['baselines']['eq3']:.2f}%")
    print(f"  {'examples':>9s} {'per-hand':>9s} {'p90':>6s} {'global':>8s} {'p90':>6s} {'secs':>6s}")
    best = None
    for s in sizes:
        t0 = time.time()
        sub = subset(tr, np.arange(s))
        ph = fit_per_hand(sub, te)
        e1, p1, per1 = error(ph(te), te)
        e2, p2 = float("nan"), float("nan")
        if s <= 16000:
            gl = fit_global(sub, te)
            e2, p2, _ = error(gl(te), te)
        rep["curve"].append({"examples": s, "per_hand": e1, "per_hand_p90": p1, "global": e2, "global_p90": p2})
        print(f"  {s:9d} {e1:8.2f}% {p1:5.2f}% {e2:7.2f}% {p2:5.2f}% {time.time() - t0:6.0f}", flush=True)
        best = (ph, per1)
        with open(out_path, "w") as f:
            json.dump(rep, f, indent=1)
    ph, per = best
    last = rep["curve"][-1]
    e = min(last["per_hand"], last["global"] if np.isfinite(last["global"]) else 9e9)
    rep["verdict"] = verdict(e)
    if len(rep["curve"]) >= 2:
        a, b = rep["curve"][-2], rep["curve"][-1]
        gain = 1 - b["per_hand"] / a["per_hand"]
        rep["still_improving"] = bool(gain > 0.05 * np.log2(b["examples"] / a["examples"]))
    print(f"\nbest at the full set: {e:.2f}% of the pot -> {rep['verdict']}"
          f"{'; still improving with more data' if rep.get('still_improving') else '; flat with more data'}")
    rep["by_family"], rep["by_stack"] = {}, {}
    for fam in sorted(set(te["fam"])):
        m = te["fam"] == fam
        if m.sum() >= 10:
            rep["by_family"][fam] = [int(m.sum()), float(per[m].mean())]
    print("  per-hand net by range family (test examples, error):",
          ", ".join(f"{k} {v[0]}: {v[1]:.2f}%" for k, v in rep["by_family"].items()))
    spr = te["stacks"].min(axis=1)
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 99)):
        m = (spr >= lo) & (spr < hi)
        if m.sum() >= 10:
            rep["by_stack"][f"{lo}-{hi} pots"] = [int(m.sum()), float(per[m].mean())]
    print("  by shortest stack:", ", ".join(f"{k} {v[0]}: {v[1]:.2f}%" for k, v in rep["by_stack"].items()))
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"-> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m preflop_hud.pilot_train", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="pipeline checks on the data so far, not a result")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="use only this many training shards")
    a = ap.parse_args(argv)
    te = load("test", a.workers)
    tr = load("train", a.workers, a.limit)
    if a.check:
        check(tr, te)
    else:
        curve(tr, te, os.path.join(OUT, time.strftime("report_%Y%m%d-%H%M%S.json")))


if __name__ == "__main__":
    main()
