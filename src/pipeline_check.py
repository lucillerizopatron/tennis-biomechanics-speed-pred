"""Is the serve speed result a property of the data or of our pipeline?

Checks, in order: size and units of the motion, timing of the peak, within-player
correlation of speed with wrist speed, smoothing cutoff, scaling, and not relying
on the handedness table.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

import serve_data as sd
from pose_features import columns

warnings.filterwarnings("ignore")
RES = sd.ROOT / "results"
df = pd.read_parquet(RES / "features_8hz.parquet")
df["height_cm"] = df.height_cm.fillna(df.height_cm.median())

print("1. SIZE AND UNITS")
print("  torso length (coordinate units): median %.3f" % df.scale_torso.median())
print("  arm length: median %.3f" % df.scale_arm.median())
w = df.wrist_max_peak_speed
print("  peak wrist speed (units/s): median %.2f, 90th pct %.2f" % (w.median(), w.quantile(.9)))
print("  peak wrist speed in torso lengths/s: median %.2f" % (w / df.scale_torso).median())
print("  (a real serve racket-arm wrist moves roughly 10-20 m/s, i.e. 20-40 torso lengths/s)")

print("\n2. TIMING OF THE WRIST SPEED PEAK (fraction of clip)")
t = df.wrist_peak_time_norm
print("  mean %.2f, std %.2f, quartiles %s" % (t.mean(), t.std(), np.round(t.quantile([.25, .5, .75]).values, 2)))
print("  histogram (10 bins):", np.histogram(t, bins=10, range=(0, 1))[0])
print("  n_frames: median %d, std %.1f" % (df.n_frames.median(), df.n_frames.std()))

print("\n3. WITHIN-PLAYER CORRELATION OF SPEED WITH POSE")
feats = ["wrist_max_peak_speed", "arm_wrist_peak_speed", "arm_wrist_mean_speed", "arm_elbow_peak_speed",
         "arm_shoulder_peak_speed", "body_r_ankle_peak_speed", "body_nose_peak_speed"]
for f in feats:
    zs, zf = [], []
    for (_, _), g in df.groupby(["server", "serve_number"]):
        if len(g) >= 8 and g.speed_kmh.std() > 0:
            zs.append(((g.speed_kmh - g.speed_kmh.mean()) / g.speed_kmh.std()).values)
            zf.append(((g[f] - g[f].mean()) / (g[f].std() + 1e-9)).values)
    zs, zf = np.concatenate(zs), np.concatenate(zf)
    r, p = spearmanr(zs, zf)
    print(f"  {f:28s} rho {r:+.3f} (n={len(zs)})")
print("  (same player, same serve number: faster serves should have faster arms)")

print("\n4/5/6. DOES POSE ADD TO THE METADATA BASELINE FOR SPEED?  (grouped 5-fold, out-of-fold R2)")
meta = pd.get_dummies(df[["gender", "serve_number", "height_cm"]].astype({"serve_number": str}), drop_first=True).astype(float)


def gain(frame, cols, seed=0):
    y = frame.speed_kmh.values
    g = frame.server.values
    ups = np.unique(g)
    perm = dict(zip(ups, np.random.RandomState(seed).permutation(len(ups))))
    gid = np.array([perm[a] for a in g])
    X = frame[cols].values
    M = meta.loc[frame.index].values
    base = np.zeros(len(y)); full = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gid):
        lin = LinearRegression().fit(M[tr], y[tr])
        res = y[tr] - lin.predict(M[tr])
        rf = RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=seed).fit(X[tr], res)
        base[te] = lin.predict(M[te]); full[te] = base[te] + rf.predict(X[te])
    return r2_score(y, base), r2_score(y, full)


rows = []
for hz in (6, 8, 12):
    d = pd.read_parquet(RES / f"features_{hz}hz.parquet")
    d["height_cm"] = d.height_cm.fillna(d.height_cm.median())
    for kind in ("motion", "scaled"):
        for size in ("arm", "body"):
            b, f = gain(d, columns(d, kind, size))
            rows.append((f"{hz} Hz", kind, size, b, f, f - b))
b, f = gain(df, ["wrist_max_peak_speed", "wrist_peak_time_norm"])
rows.append(("8 Hz", "both wrists, no handedness", "2 features", b, f, f - b))
out = pd.DataFrame(rows, columns=["cutoff", "features", "set", "R2 baseline", "R2 with pose", "gain"])
print(out.round(4).to_string(index=False))
out.to_csv(RES / "speed_pipeline_check.csv", index=False)
