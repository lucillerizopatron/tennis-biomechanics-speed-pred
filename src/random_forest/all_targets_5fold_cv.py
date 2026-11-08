"""Serving arm vs full body on four targets, with player-level bootstrap CIs.

Targets: gender, serve number (split by player), identity (split by match, only
players seen in 2+ matches), serve speed (gain over the metadata baseline).
Inputs: shape (bone lengths), raw motion, motion scaled by limb size, shape+scaled.
Out-of-fold predictions from every repeat are stacked, then players are resampled.
Folds come from splits/splits.py (shared with the transformer).
Run from src/random_forest/:  python all_targets_5fold_cv.py > ../../results/random_forest/all_targets_5fold_cv/all_targets_experiments.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for serve_data.py etc.
import serve_data as sd
from pose_features import columns

sys.path.insert(0, str(sd.ROOT / "splits"))
import splits

warnings.filterwarnings("ignore")
RES = sd.ROOT / "results"
REPS, TREES, BOOT = 2, 150, 400
df = pd.read_parquet(RES / "cache" / "pose_features_8hz.parquet")
df["height_cm"] = df.height_cm.fillna(df.height_cm.median())
df["female"] = (df.gender == "F").astype(int)
df["second"] = (df.serve_number == 2).astype(int)
rng = np.random.RandomState(0)


def pose_sets(d):
    s = {}
    for size in ("arm", "body"):
        sh, mo, sc = columns(d, "shape", size), columns(d, "motion", size), columns(d, "scaled", size)
        s[f"shape/{size}"], s[f"motion/{size}"], s[f"scaled/{size}"] = sh, mo, sc
        s[f"shape+scaled/{size}"] = sh + sc
    return s


def folds(d, group_col, rep):
    """Shared folds (splits/splits.py). d is the full table (group_col "server") or the identity subset
    in metadata order (group_col "match_id")."""
    task = "player" if group_col == "server" else "match"
    fid = splits.fold_ids(task, rep)
    if task == "match":
        fid = fid[fid >= 0]
    assert len(fid) == len(d)
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(splits.N_FOLDS)]


def oof(d, cols, y, group_col, kind):
    """Stacked out-of-fold predictions over REPS repeats (kind: prob | class | resid)."""
    X = d[cols].values
    out, valid = [], []
    for rep in range(REPS):
        p = np.zeros(len(d)) if kind != "class" else np.empty(len(d), dtype=object)
        v = np.ones(len(d), bool)
        for tr, te in folds(d, group_col, rep):
            if kind == "prob":
                m = RandomForestClassifier(TREES, min_samples_leaf=3, n_jobs=-1, random_state=rep)
                m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
            elif kind == "class":
                m = RandomForestClassifier(TREES, min_samples_leaf=1, n_jobs=-1, random_state=rep)
                m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
                v[te] = np.isin(y[te], np.unique(y[tr]))
            else:
                m = RandomForestRegressor(TREES, min_samples_leaf=5, n_jobs=-1, random_state=rep)
                m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
        out.append(p); valid.append(v)
    return np.concatenate(out), np.concatenate(valid)


def bootstrap(players, metric_fns, n_rows_per_rep):
    """Player-level resampling indices into the stacked arrays."""
    ups = np.unique(players)
    base = {u: np.where(players == u)[0] for u in ups}
    draws = []
    for _ in range(BOOT):
        pick = rng.choice(ups, len(ups))
        draws.append(np.concatenate([base[u] for u in pick]))
    return draws


def summarize(name, vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"{np.mean(vals):7.3f}  [{lo:6.3f}, {hi:6.3f}]"


def report(title, preds, players, metric, valid=None, higher=True):
    print(f"\n=== {title} ===", flush=True)
    stacked_players = np.tile(players, REPS)
    draws = bootstrap(stacked_players, None, len(players))
    scores = {}
    for name, p in preds.items():
        s = []
        for ix in draws:
            ok = ix if valid is None else ix[valid[ix]]
            s.append(metric(ok, p))
        scores[name] = np.array(s)
        print(f"  {name:32s} {summarize(name, scores[name])}", flush=True)
    return scores


def compare(scores, pairs, higher=True):
    print("  -- paired differences (CI excludes 0 = real) --")
    for a, b in pairs:
        if a in scores and b in scores:
            d = scores[a] - scores[b]
            print(f"  {a} minus {b:26s} {summarize('', d)}  P(>0)={np.mean(d > 0):.2f}", flush=True)


ARM_BODY = [(f"{k}/arm", f"{k}/body") for k in ("shape", "motion", "scaled", "shape+scaled")]
which = sys.argv[1:] or ["gender", "serve", "identity", "speed"]

# ---- gender (split by player)
if "gender" in which:
    y = df.female.values
    stk = np.tile(y, REPS)
    preds = {}
    meta_cols = {"majority": None, "height only": ["height_cm"],
                 "meta (height+speed+serve#)": ["height_cm", "speed_kmh", "second"]}
    for k, c in meta_cols.items():
        preds[k] = np.full(len(stk), 0.5) if c is None else oof(df, c, y, "server", "prob")[0]
    for name, c in pose_sets(df).items():
        preds[name] = oof(df, c, y, "server", "prob")[0]
    for kind in ("motion", "scaled"):
        for size in ("arm", "body"):
            preds[f"meta+{kind}/{size}"] = oof(df, ["height_cm", "speed_kmh", "second"] + columns(df, kind, size),
                                                y, "server", "prob")[0]
    sc = report("GENDER (AUC, split by player)", preds, df.server.values,
                lambda ix, p: roc_auc_score(stk[ix], p[ix]))
    compare(sc, ARM_BODY + [("meta+motion/arm", "meta (height+speed+serve#)"),
                            ("meta+scaled/arm", "meta (height+speed+serve#)"),
                            ("meta+scaled/body", "meta (height+speed+serve#)"),
                            ("meta+scaled/body", "meta+scaled/arm")])

# ---- serve number (split by player); speed is a label proxy, kept only as a reference
if "serve" in which:
    y = df.second.values
    stk = np.tile(y, REPS)
    preds = {"speed_kmh only (reference)": oof(df, ["speed_kmh"], y, "server", "prob")[0]}
    for name, c in pose_sets(df).items():
        preds[name] = oof(df, c, y, "server", "prob")[0]
    sc = report("SERVE NUMBER (AUC, split by player; 0.5 = chance)", preds, df.server.values,
                lambda ix, p: roc_auc_score(stk[ix], p[ix]))
    compare(sc, ARM_BODY)

# ---- identity (split by match, players with 2+ matches)
if "identity" in which:
    multi = df.groupby("server").match_id.nunique()
    sub = df[df.server.isin(multi[multi >= 2].index)].reset_index(drop=True)
    y = sub.server.values
    stk = np.tile(y, REPS)
    print(f"\nidentity subset: {sub.server.nunique()} players, {len(sub)} serves; chance {1/sub.server.nunique():.3f}, "
          f"majority {sub.server.value_counts(normalize=True).iloc[0]:.3f}")
    preds, vals = {}, None
    for name, c in pose_sets(sub).items():
        preds[name], v = oof(sub, c, y, "match_id", "class")
        vals = v if vals is None else vals & v
    sc = report("IDENTITY (top-1 accuracy, split by match; scored on players seen in training)", preds,
                sub.server.values, lambda ix, p: np.mean(p[ix] == stk[ix]), valid=vals)
    compare(sc, ARM_BODY)

# ---- serve speed: does pose add to the metadata baseline?
if "speed" in which:
    y = df.speed_kmh.values
    stk = np.tile(y, REPS)
    M = pd.get_dummies(df[["gender", "second", "height_cm"]].astype({"second": str}), drop_first=True).astype(float).values
    base = np.zeros(len(df) * REPS)
    resid_preds = {}
    sets = pose_sets(df)
    for name in sets:
        resid_preds[name] = np.zeros(len(df) * REPS)
    for rep in range(REPS):
        off = rep * len(df)
        for tr, te in folds(df, "server", rep):
            lin = LinearRegression().fit(M[tr], y[tr])
            res = y[tr] - lin.predict(M[tr])
            base[off + te] = lin.predict(M[te])
            for name, c in sets.items():
                X = df[c].values
                m = RandomForestRegressor(TREES, min_samples_leaf=5, n_jobs=-1, random_state=rep).fit(X[tr], res)
                resid_preds[name][off + te] = base[off + te] + m.predict(X[te])
    preds = {"metadata baseline": base, **resid_preds}
    sc = report("SERVE SPEED (MAE in km/h, lower is better; split by player)", preds, df.server.values,
                lambda ix, p: mean_absolute_error(stk[ix], p[ix]))
    d = {n: sc[n] - sc["metadata baseline"] for n in sets}
    print("  -- MAE change vs baseline (negative = pose helps; CI excluding 0 = real) --")
    for n, v in d.items():
        print(f"  {n:32s} {summarize(n, v)}", flush=True)
    compare(sc, [(f"{k}/arm", f"{k}/body") for k in ("motion", "scaled", "shape", "shape+scaled")])
