"""Controls for the gender and identity results (Random Forest).

A. Gender: metadata + SHUFFLED pose vs metadata alone vs metadata + real pose.
   If shuffled pose scores like metadata alone, the gain from real pose is signal.
B. Gender with player-level label shuffle: the pipeline should fall to chance.
C. Identity with match-level label shuffle: accuracy should fall to chance.
Run from src/random_forest/:  python shuffled_controls.py > ../../results/random_forest/all_targets_5fold_cv/shuffled_controls.txt
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for serve_data.py etc.
import serve_data as sd
from pose_features import columns

sys.path.insert(0, str(sd.ROOT / "splits"))
import splits

warnings.filterwarnings("ignore")
REPS, SHUFFLES, TREES, BOOT = 2, 2, 150, 400
RES = sd.ROOT / "results"
df = pd.read_parquet(RES / "cache" / "pose_features_8hz.parquet")
df["height_cm"] = df.height_cm.fillna(df.height_cm.median())
df["female"] = (df.gender == "F").astype(int)
df["second"] = (df.serve_number == 2).astype(int)
META = ["height_cm", "speed_kmh", "second"]
rng = np.random.RandomState(0)


def folds(task, rep):
    """Shared folds (splits/splits.py); for "match" the rows are the identity subset in metadata order."""
    fid = splits.fold_ids(task, rep)
    if task == "match":
        fid = fid[fid >= 0]
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(splits.N_FOLDS)]


def oof(X, y, task, kind):
    outs, valids = [], []
    for rep in range(REPS):
        p = np.zeros(len(y)) if kind == "prob" else np.zeros(len(y), int)
        v = np.ones(len(y), bool)
        for tr, te in folds(task, rep):
            m = RandomForestClassifier(TREES, min_samples_leaf=3 if kind == "prob" else 1,
                                       n_jobs=-1, random_state=rep).fit(X[tr], y[tr])
            if kind == "prob":
                p[te] = m.predict_proba(X[te])[:, 1]
            else:
                p[te] = m.predict(X[te]); v[te] = np.isin(y[te], np.unique(y[tr]))
        outs.append(p); valids.append(v)
    return np.concatenate(outs), np.concatenate(valids)


def draws(players):
    ups = np.unique(players); idx = {u: np.where(players == u)[0] for u in ups}
    return [np.concatenate([idx[u] for u in rng.choice(ups, len(ups))]) for _ in range(BOOT)]


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return f"{np.mean(v):6.3f}  [{lo:6.3f}, {hi:6.3f}]"


def auc_boot(y, p, players, reps):
    ys, ps, pl = np.tile(y, reps), p, np.tile(players, reps)
    return np.array([roc_auc_score(ys[ix], ps[ix]) for ix in draws(pl)])


players = df.server.values
y = df.female.values

# ---- A. metadata + shuffled pose
print("=== A. GENDER (AUC, split by player): does real pose add beyond metadata? ===", flush=True)
scores = {}
p, _ = oof(df[META].values, y, "player", "prob")
scores["metadata alone"] = auc_boot(y, p, players, REPS)
for size in ("arm", "body"):
    cols = columns(df, "scaled", size)
    p, _ = oof(df[META + cols].values, y, "player", "prob")
    scores[f"metadata + real pose ({size})"] = auc_boot(y, p, players, REPS)
    sh = []
    for s in range(SHUFFLES):
        X = df[cols].values[np.random.RandomState(100 + s).permutation(len(df))]
        p, _ = oof(np.hstack([df[META].values, X]), y, "player", "prob")
        sh.append(auc_boot(y, p, players, REPS))
    scores[f"metadata + SHUFFLED pose ({size})"] = np.mean(sh, axis=0)
for k, v in scores.items():
    print(f"  {k:36s} {ci(v)}", flush=True)
for size in ("arm", "body"):
    d = scores[f"metadata + real pose ({size})"] - scores[f"metadata + SHUFFLED pose ({size})"]
    print(f"  real minus shuffled ({size}):         {ci(d)}", flush=True)

# ---- B. gender label shuffle (whole players get another player's label)
print("\n=== B. GENDER with shuffled player labels (should be about 0.5) ===", flush=True)
ups = np.unique(players)
for size in ("arm", "body"):
    cols = columns(df, "motion", size)
    aucs = []
    for s in range(SHUFFLES + 1):
        lab = dict(zip(ups, np.random.RandomState(200 + s).permutation([y[players == u][0] for u in ups])))
        ysh = np.array([lab[u] for u in players])
        p, _ = oof(df[cols].values, ysh, "player", "prob")
        aucs.append(auc_boot(ysh, p, players, REPS))
    print(f"  motion/{size:5s} {ci(np.mean(aucs, axis=0))}", flush=True)

# ---- C. identity label shuffle (whole matches get another match's player label)
print("\n=== C. IDENTITY with shuffled match labels (chance about 0.02 to 0.05) ===", flush=True)
mm = df.groupby("server").match_id.nunique()
sub = df[df.server.isin(mm[mm >= 2].index)].reset_index(drop=True)
matches = sub.match_id.unique()
owner = sub.groupby("match_id").server.first().reindex(matches)
for size in ("arm", "body"):
    cols = columns(sub, "motion", size)
    accs = []
    for s in range(SHUFFLES + 1):
        new = dict(zip(matches, np.random.RandomState(300 + s).permutation(owner.values)))
        ysh = sub.match_id.map(new).values
        p, v = oof(sub[cols].values, pd.factorize(ysh)[0], "match", "class")
        yf = np.tile(pd.factorize(ysh)[0], REPS)
        pl = np.tile(sub.server.values, REPS)
        accs.append(np.array([np.mean(p[ix[v[ix]]] == yf[ix[v[ix]]]) for ix in draws(pl)]))
    print(f"  motion/{size:5s} {ci(np.mean(accs, axis=0))}", flush=True)
