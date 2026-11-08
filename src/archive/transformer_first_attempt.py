"""Small paper-style transformer on raw joint positions: serving arm vs full body.

Input is the smoothed, hip-relative trajectory resampled to 32 frames. Same splits
and targets as experiments.py. Tuning is light: a handful of configs on one
held-out validation split of players (gender, per input), then the chosen config
is reused for every target. Early stopping uses a validation slice of each
training fold. One repeat only.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for serve_data.py etc.
import serve_data as sd
from handedness import hand
from smoothing import hip_relative, smooth

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
RES = sd.ROOT / "results"
T, BOOT = 32, 400
ARM = {"L": (5, 7, 9), "R": (6, 8, 10)}
CONFIGS = [
    dict(d=32, layers=1, drop=0.1, lr=1e-3, wd=1e-2),
    dict(d=64, layers=2, drop=0.2, lr=1e-3, wd=1e-2),
    dict(d=64, layers=2, drop=0.3, lr=5e-4, wd=1e-2),
    dict(d=128, layers=2, drop=0.3, lr=5e-4, wd=5e-2),
    dict(d=32, layers=2, drop=0.2, lr=2e-3, wd=1e-2),
]
SCALE = "--scale" in sys.argv
rng = np.random.RandomState(0)

meta = sd.load_metadata().reset_index(drop=True)
meta["height_cm"] = meta.height_cm.fillna(meta.height_cm.median())
cache = RES / "cache" / "resampled_keypoints_32frames.npy"
if cache.exists():
    raw = np.load(cache)
else:
    kps = sd.load_keypoints(meta)
    raw = np.zeros((len(kps), T, 17, 3), np.float32)
    for i, k in enumerate(kps):
        r = smooth(hip_relative(k))
        t = np.linspace(0, len(r) - 1, T)
        for j in range(17):
            for a in range(3):
                raw[i, :, j, a] = np.interp(t, np.arange(len(r)), r[:, j, a])
    np.save(cache, raw)
if SCALE:  # divide by torso length so size is removed and only proportions/motion remain
    torso = np.linalg.norm(raw[:, :, 5] - raw[:, :, 11], axis=2).mean(1)
    raw = raw / torso[:, None, None, None]
hands = meta.server.map(hand).values
arm = np.stack([raw[i][:, list(ARM[h])] for i, h in enumerate(hands)]).reshape(len(raw), T, 9)
body = raw.reshape(len(raw), T, 51)
INPUT = {"arm": arm.astype(np.float32), "body": body.astype(np.float32)}


class Net(nn.Module):
    def __init__(self, d_in, n_out, d, layers, drop, **_):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        enc = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        dec = nn.TransformerDecoderLayer(d, 4, 2 * d, drop, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, 1)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(self.dec(self.query.expand(len(x), -1, -1), h)[:, 0])


def fit_predict(Xtr, ytr, gtr, Xte, kind, cfg, n_out, epochs=30, seed=0):
    torch.manual_seed(seed)
    ups = np.unique(gtr)
    val_g = np.random.RandomState(seed).choice(ups, max(1, int(0.15 * len(ups))), replace=False)
    va = np.isin(gtr, val_g); tr = ~va
    mu, sd_ = Xtr[tr].reshape(-1, Xtr.shape[-1]).mean(0), Xtr[tr].reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    f = lambda a: torch.tensor((a - mu) / sd_, dtype=torch.float32)
    Xt, Xv, Xe = f(Xtr[tr]), f(Xtr[va]), f(Xte)
    yt, yv = torch.tensor(ytr[tr]), torch.tensor(ytr[va])
    ytype = torch.float32 if kind != "class" else torch.long
    yt, yv = yt.to(ytype), yv.to(ytype)
    net = Net(Xtr.shape[-1], n_out, **cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = {"prob": nn.BCEWithLogitsLoss(), "class": nn.CrossEntropyLoss(), "resid": nn.MSELoss()}[kind]
    out = lambda x: net(x).squeeze(-1) if kind != "class" else net(x)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        for ix in torch.randperm(len(Xt)).split(64):
            opt.zero_grad(); loss_fn(out(Xt[ix]), yt[ix]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(out(Xv), yv).item()
        if vl < best:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        o = out(Xe)
    return (torch.sigmoid(o).numpy() if kind == "prob" else o.argmax(1).numpy() if kind == "class" else o.numpy())


def player_folds(groups, seed=0):
    ups = np.unique(groups)
    perm = dict(zip(ups, np.random.RandomState(seed).permutation(len(ups))))
    return list(GroupKFold(5).split(groups, groups=np.array([perm[a] for a in groups])))


def cv(X, y, groups, kind, cfg, n_out, base=None):
    pred = np.zeros(len(y)) if kind != "class" else np.zeros(len(y), int)
    valid = np.ones(len(y), bool)
    for tr, te in player_folds(groups):
        pred[te] = fit_predict(X[tr], y[tr], groups[tr], X[te], kind, cfg, n_out)
        if kind == "class":
            valid[te] = np.isin(y[te], np.unique(y[tr]))
    return pred, valid


def tune(inp):
    """Pick a config on one held-out set of players (gender, early-stopped on a slice of the rest)."""
    g = meta.server.values; y = (meta.gender == "F").astype(int).values
    ups = np.unique(g); hold = np.random.RandomState(1).choice(ups, int(0.2 * len(ups)), replace=False)
    te = np.isin(g, hold)
    scores = []
    for c in CONFIGS:
        p = fit_predict(INPUT[inp][~te], y[~te], g[~te], INPUT[inp][te], "prob", c, 1)
        scores.append(roc_auc_score(y[te], p))
        print(f"  tune {inp} {c} -> val AUC {scores[-1]:.3f}", flush=True)
    return CONFIGS[int(np.argmax(scores))]


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return f"{np.mean(v):7.3f}  [{lo:6.3f}, {hi:6.3f}]"


def boot_draws(players):
    ups = np.unique(players); idx = {u: np.where(players == u)[0] for u in ups}
    return [np.concatenate([idx[u] for u in rng.choice(ups, len(ups))]) for _ in range(BOOT)]


def run():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["gender", "serve", "identity", "speed"]
    print("scaled by torso length" if SCALE else "raw hip-relative positions", flush=True)
    cfg = {i: tune(i) for i in ("arm", "body")}
    print("chosen:", cfg, flush=True)
    g = meta.server.values
    if "gender" in which:
        y = (meta.gender == "F").astype(int).values
        res = {i: cv(INPUT[i], y, g, "prob", cfg[i], 1)[0] for i in INPUT}
        dr = boot_draws(g)
        s = {i: np.array([roc_auc_score(y[ix], res[i][ix]) for ix in dr]) for i in res}
        print("\n=== GENDER transformer (AUC) ==="); [print(f"  {i:6s} {ci(s[i])}") for i in s]
        print(f"  arm minus body {ci(s['arm'] - s['body'])}", flush=True)
    if "serve" in which:
        y = (meta.serve_number == 2).astype(int).values
        res = {i: cv(INPUT[i], y, g, "prob", cfg[i], 1)[0] for i in INPUT}
        dr = boot_draws(g)
        s = {i: np.array([roc_auc_score(y[ix], res[i][ix]) for ix in dr]) for i in res}
        print("\n=== SERVE NUMBER transformer (AUC) ==="); [print(f"  {i:6s} {ci(s[i])}") for i in s]
        print(f"  arm minus body {ci(s['arm'] - s['body'])}", flush=True)
    if "identity" in which:
        mm = meta.groupby("server").match_id.nunique()
        keep = meta.server.isin(mm[mm >= 2].index).values
        sub = meta[keep].reset_index(drop=True)
        classes = {p: i for i, p in enumerate(sorted(sub.server.unique()))}
        y = sub.server.map(classes).values
        res = {}
        for i in INPUT:
            Xi = INPUT[i][keep]
            pred = np.zeros(len(y), int); valid = np.ones(len(y), bool)
            for tr, te in player_folds(sub.match_id.values):
                pred[te] = fit_predict(Xi[tr], y[tr], sub.match_id.values[tr], Xi[te], "class", cfg[i], len(classes))
                valid[te] = np.isin(y[te], np.unique(y[tr]))
            res[i] = (pred, valid)
        dr = boot_draws(sub.server.values)
        ok = res["arm"][1] & res["body"][1]
        s = {i: np.array([np.mean(res[i][0][ix[ok[ix]]] == y[ix[ok[ix]]]) for ix in dr]) for i in res}
        print(f"\n=== IDENTITY transformer (top-1, {len(classes)} players, split by match) ==="); [print(f"  {i:6s} {ci(s[i])}") for i in s]
        print(f"  arm minus body {ci(s['arm'] - s['body'])}", flush=True)
    if "speed" in which:
        y = meta.speed_kmh.values.astype(float)
        M = pd.get_dummies(meta[["gender", "serve_number", "height_cm"]].astype({"serve_number": str}), drop_first=True).astype(float).values
        base = np.zeros(len(y)); res = {i: np.zeros(len(y)) for i in INPUT}
        for tr, te in player_folds(g):
            lin = LinearRegression().fit(M[tr], y[tr]); rtr = y[tr] - lin.predict(M[tr])
            base[te] = lin.predict(M[te])
            for i in INPUT:
                sdv = rtr.std()
                res[i][te] = base[te] + sdv * fit_predict(INPUT[i][tr], rtr / sdv, g[tr], INPUT[i][te], "resid", cfg[i], 1)
        dr = boot_draws(g)
        sb = np.array([mean_absolute_error(y[ix], base[ix]) for ix in dr])
        s = {i: np.array([mean_absolute_error(y[ix], res[i][ix]) for ix in dr]) - sb for i in res}
        print("\n=== SPEED transformer (MAE change vs metadata baseline, km/h; negative = pose helps) ===")
        [print(f"  {i:6s} {ci(s[i])}") for i in s]
        print(f"  arm minus body {ci(s['arm'] - s['body'])}", flush=True)


run()
