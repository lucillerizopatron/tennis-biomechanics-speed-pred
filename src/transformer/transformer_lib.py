"""Data, model and training loop for the small paper-style transformer.

Input per serve: smoothed, hip-relative joint positions resampled to 32 frames.
  size  arm  : the serving arm (shoulder, elbow, wrist; the side with the faster wrist), 9 numbers per frame
        body : all 17 joints, 51 numbers per frame
  mode  raw    : positions as they are (size, proportions and motion mixed)
        scaled : positions divided by limb size (arm length for arm, torso for body),
                 the same scaling idea as the Random Forest's scaled motion
        motion : scaled positions minus each joint's mean over the serve, so only
                 movement is left
Folds come from splits/splits.py, shared with the Random Forest.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
import serve_data as sd
import joints

torch.set_num_threads(os.cpu_count())
T = 32
RES = sd.ROOT / "results"
CACHE = RES / "cache"
OUT_DIR = RES / "transformer"
CONFIGS = [
    dict(d=32, layers=1, drop=0.1, lr=1e-3, wd=1e-2),
    dict(d=64, layers=2, drop=0.2, lr=1e-3, wd=1e-2),
    dict(d=64, layers=2, drop=0.3, lr=5e-4, wd=1e-2),
    dict(d=128, layers=2, drop=0.3, lr=5e-4, wd=5e-2),
    dict(d=32, layers=2, drop=0.2, lr=2e-3, wd=1e-2),
]


def load_data():
    """Metadata (same row order as splits/splits.py), per-serve side and scales, and the
    32-frame hip-relative positions, (N,32,17,3)."""
    meta = sd.load_metadata().reset_index(drop=True)
    meta["height_cm"] = meta.height_cm.fillna(meta.height_cm.median())
    cache = CACHE / "h36m_inputs_32frames.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        raw, side, sa, st = z["raw"], z["side"], z["scale_arm"], z["scale_torso"]
    else:
        raw = np.zeros((len(meta), T, 17, 3), np.float32)
        side, sa, st = [], [], []
        for i, k in enumerate(sd.load_keypoints(meta)):
            r, s, a, t, _ = joints.serve_info(k)
            side.append(s); sa.append(a); st.append(t)
            tt = np.linspace(0, len(r) - 1, T)
            for j in range(17):
                for a_ in range(3):
                    raw[i, :, j, a_] = np.interp(tt, np.arange(len(r)), r[:, j, a_])
        side, sa, st = np.array(side), np.array(sa), np.array(st)
        np.savez(cache, raw=raw, side=side, scale_arm=sa, scale_torso=st)
    feats = pd.DataFrame({"side": side, "scale_arm": sa, "scale_torso": st})
    return meta, feats, raw


def make_input(meta, feats, raw, size, mode):
    if size == "arm":
        x = np.stack([raw[i][:, list(joints.ARM[s])] for i, s in enumerate(feats.side.values)])  # (N,32,3,3)
        scale = feats.scale_arm.values
    else:
        x, scale = raw, feats.scale_torso.values
    x = x.astype(np.float32)
    if mode in ("scaled", "motion"):
        x = x / scale[:, None, None, None].astype(np.float32)
    if mode == "motion":
        x = x - x.mean(axis=1, keepdims=True)
    return x.reshape(len(x), T, -1)


class Net(nn.Module):
    """Projection + learnable positions + encoder, then a decoder with one learned query."""

    def __init__(self, d_in, n_out, d, layers, drop, **_):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True), layers)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 2 * d, drop, batch_first=True), 1)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.head = nn.Linear(d, n_out)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(self.dec(self.query.expand(len(x), -1, -1), h)[:, 0])


def fit_predict(X, y, fit_idx, val_idx, test_idx, kind, cfg, n_out=1, epochs=25, patience=5, seed=0, off=None):
    """Train on fit_idx, stop early on val_idx, predict test_idx.
    kind: prob (binary, returns P(1)) | class (returns class id) | reg (returns original units).
    off: optional per-serve logit offset (prob only), e.g. from a metadata model; the network learns a correction."""
    o_ = lambda i: torch.zeros(len(i)) if off is None else torch.tensor(off[i], dtype=torch.float32)
    torch.manual_seed(seed)
    mu = X[fit_idx].reshape(-1, X.shape[-1]).mean(0)
    sd_ = X[fit_idx].reshape(-1, X.shape[-1]).std(0) + 1e-6
    f = lambda i: torch.tensor((X[i] - mu) / sd_, dtype=torch.float32)
    Xf, Xv, Xt = f(fit_idx), f(val_idx), f(test_idx)
    ym, ys = 0.0, 1.0
    if kind == "reg":
        ym, ys = float(y[fit_idx].mean()), float(y[fit_idx].std())
    yt = (y - ym) / ys
    yf = torch.tensor(yt[fit_idx], dtype=torch.long if kind == "class" else torch.float32)
    yv = torch.tensor(yt[val_idx], dtype=torch.long if kind == "class" else torch.float32)
    of, ov, ot = o_(fit_idx), o_(val_idx), o_(test_idx)
    net = Net(X.shape[-1], n_out, **cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = {"prob": nn.BCEWithLogitsLoss(), "class": nn.CrossEntropyLoss(), "reg": nn.MSELoss()}[kind]
    out = (lambda x: net(x)) if kind == "class" else (lambda x: net(x).squeeze(-1))
    best, state, bad = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        for ix in torch.randperm(len(Xf)).split(64):
            opt.zero_grad(); loss_fn(out(Xf[ix]) + (of[ix] if kind == "prob" else 0), yf[ix]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(out(Xv) + (ov if kind == "prob" else 0), yv).item()
        if vl < best:
            best, state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        o = out(Xt)
    if kind == "prob":
        return torch.sigmoid(o + ot).numpy()
    if kind == "class":
        return o.argmax(1).numpy()
    return o.numpy() * ys + ym
