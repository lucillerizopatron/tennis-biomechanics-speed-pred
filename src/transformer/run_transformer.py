"""Transformer experiments: serving arm vs full body on the four targets.

Same design as the Random Forest work: folds from splits/splits.py (player folds for gender,
serve number and speed; match folds for identity), player-level bootstrap intervals,
pose-only results first, metadata baselines (with and without height) as extra rows.

Usage (from src/):  python transformer/run_transformer.py [tune] [gender] [identity] [serve] [speed]
                    QUICK=1 python transformer/run_transformer.py ...   (smoke test, 1 epoch, 1 fold)
No argument runs everything.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score

import transformer_lib as L

sys.path.insert(0, str(L.sd.ROOT / "splits"))
import splits

warnings.filterwarnings("ignore")
QUICK = bool(os.environ.get("QUICK"))
EPOCHS, BOOT = (1, 20) if QUICK else (25, 300)
L.OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG_FILE = L.OUT_DIR / ("tuned_configs_quick.json" if QUICK else "tuned_configs.json")
OUT_CSV = L.OUT_DIR / ("results_quick.csv" if QUICK else f"results{os.environ.get('TAG', '')}.csv")
which = [a for a in sys.argv[1:]] or ["tune", "gender", "identity", "serve", "speed"]

meta, feats, raw = L.load_data()
N = len(meta)
players = meta.server.values
_inputs = {}
rows_out = []


def inp(size, mode):
    if (size, mode) not in _inputs:
        _inputs[(size, mode)] = L.make_input(meta, feats, raw, size, mode)
    return _inputs[(size, mode)]


REPS = 1 if QUICK else int(os.environ.get("REPS", 3))
TAG = os.environ.get("TAG", "")


def fold_list(task, rep=0):
    f = splits.folds(task, rep, meta)
    return f[:1] if QUICK else f


def run_cv(X, y, task, kind, cfg, n_out=1, off=None):
    """Out-of-fold predictions, one array per repeat (NaN where a serve was not tested). Seed = repeat."""
    preds = []
    for rep in range(REPS):
        pred = np.full(N, np.nan)
        for tr, te in fold_list(task, rep):
            fit, val = splits.inner_split(tr, task, meta, seed=rep)
            pred[te] = L.fit_predict(X, y, fit, val, te, kind, cfg, n_out, epochs=EPOCHS, seed=rep, off=off(tr) if off else None)
        preds.append(pred)
    return preds


def rf_cv(cols, y, task, kind):
    """Random Forest reference on the same folds (metadata columns, no pose), one array per repeat."""
    A = meta[cols].values
    preds = []
    for rep in range(REPS):
        pred = np.full(N, np.nan)
        for tr, te in fold_list(task, rep):
            m = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=-1, random_state=rep).fit(A[tr], y[tr])
            pred[te] = m.predict_proba(A[te])[:, 1] if kind == "prob" else m.predict(A[te])
        preds.append(pred)
    return preds


def ok_of(preds, extra=None):
    ok = np.all([~np.isnan(p) for p in preds], axis=0)
    return ok if extra is None else ok & extra


def score(preds, dr, fn, y):
    """Metric averaged over the repeats, for each player-level bootstrap draw."""
    return np.array([np.mean([fn(y[i], p[i]) for p in preds]) for i in dr])


def draws_for(ok):
    idx = np.where(ok)[0]
    ups = np.unique(players[idx])
    by = {u: idx[players[idx] == u] for u in ups}
    rng = np.random.RandomState(0)
    return [np.concatenate([by[u] for u in rng.choice(ups, len(ups))]) for _ in range(BOOT)]


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"{np.mean(v):7.3f}  [{lo:6.3f}, {hi:6.3f}]"


def record(target, name, metric, vals):
    rows_out.append(dict(target=target, model=name, metric=metric, mean=np.mean(vals),
                         lo=np.percentile(vals, 2.5), hi=np.percentile(vals, 97.5)))
    pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
    print(f"  {name:44s} {ci(vals)}", flush=True)


def paired(target, label, a, b, metric):
    d = a - b
    rows_out.append(dict(target=target, model=label, metric=metric + " difference", mean=np.mean(d),
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    print(f"  {label:44s} {ci(d)}", flush=True)


def load_cfg():
    if CFG_FILE.exists():
        return json.loads(CFG_FILE.read_text())
    return {s: L.CONFIGS[1] for s in ("arm", "body")}


# ---------------------------------------------------------------- light tuning
if "tune" in which:
    print("=== LIGHT TUNING (gender, scaled input; validation players taken from fold 0's TRAINING set) ===", flush=True)
    y = (meta.gender == "F").astype(int).values
    tr0, _ = splits.folds("player", 0, meta)[0]
    fit_t, val_t = splits.inner_split(tr0, "player", meta, frac=0.2, seed=1)
    fit2, es = splits.inner_split(fit_t, "player", meta, frac=0.15, seed=0)
    chosen = {}
    for size in ("arm", "body"):
        X = inp(size, "scaled")
        scores = []
        for c in (L.CONFIGS[:2] if QUICK else L.CONFIGS):
            p = L.fit_predict(X, y, fit2, es, val_t, "prob", c, epochs=EPOCHS)
            scores.append(roc_auc_score(y[val_t], p))
            print(f"  {size:5s} {c} -> validation AUC {scores[-1]:.3f}", flush=True)
        chosen[size] = (L.CONFIGS[:2] if QUICK else L.CONFIGS)[int(np.argmax(scores))]
    CFG_FILE.write_text(json.dumps(chosen))
    print("  chosen:", chosen, flush=True)
cfg = load_cfg()
print("configs in use:", cfg, "\n", flush=True)

# ---------------------------------------------------------------- gender
if "gender" in which:
    from sklearn.linear_model import LogisticRegression
    y = (meta.gender == "F").astype(int).values
    auc = lambda yy, pp: roc_auc_score(yy, pp)
    print(f"=== GENDER (AUC, folds split by player, mean of {REPS} repeats) ===", flush=True)
    refs = {"height only (RF)": ["height_cm"], "speed + serve# (RF, no height)": ["speed_kmh", "serve_number"],
            "speed + serve# + height (RF)": ["speed_kmh", "serve_number", "height_cm"]}
    for name, cols in refs.items():
        p = rf_cv(cols, y, "player", "prob"); dr = draws_for(ok_of(p))
        record("gender", "reference: " + name, "AUC", score(p, dr, auc, y))
    for mode in ("raw", "scaled", "motion"):
        sc = {}
        for size in ("arm", "body"):
            p = run_cv(inp(size, mode), y, "player", "prob", cfg[size]); dr = draws_for(ok_of(p))
            sc[size] = score(p, dr, auc, y)
            record("gender", f"transformer {size} [{mode}]", "AUC", sc[size])
        paired("gender", f"  arm minus body [{mode}]", sc["arm"], sc["body"], "AUC")
    # pose beyond metadata: the network learns a correction on top of a logistic model on metadata
    print("  -- pose on top of metadata (scaled input; the transformer corrects a metadata logistic model) --", flush=True)
    for mname, cols in (("speed + serve#", ["speed_kmh", "serve_number"]),
                        ("speed + serve# + height", ["speed_kmh", "serve_number", "height_cm"])):
        A = meta[cols].values.astype(float)
        A = (A - A.mean(0)) / A.std(0)
        lr_all = {}
        def offset_for(tr, A=A):
            m = LogisticRegression(max_iter=1000).fit(A[tr], y[tr])
            o = np.zeros(N); o[:] = m.decision_function(A)
            return o
        base = []
        for rep in range(REPS):
            pred = np.full(N, np.nan)
            for tr, te in fold_list("player", rep):
                pred[te] = LogisticRegression(max_iter=1000).fit(A[tr], y[tr]).predict_proba(A[te])[:, 1]
            base.append(pred)
        dr = draws_for(ok_of(base)); sb = score(base, dr, auc, y)
        record("gender", f"baseline: metadata {mname} (logistic)", "AUC", sb)
        sc = {}
        for size in ("arm", "body"):
            p = run_cv(inp(size, "scaled"), y, "player", "prob", cfg[size], off=offset_for); dr = draws_for(ok_of(p + base))
            sc[size] = score(p, dr, auc, y); sbb = score(base, dr, auc, y)
            record("gender", f"transformer {size} + {mname}", "AUC", sc[size])
            paired("gender", f"  {size} + {mname} minus metadata only", sc[size], sbb, "AUC")
        paired("gender", f"  arm minus body (+ {mname})", sc["arm"], sc["body"], "AUC")

# ---------------------------------------------------------------- identity
if "identity" in which:
    rows = splits.fold_ids("match", 0, meta) >= 0
    codes, names = pd.factorize(meta.server[rows])
    y = np.zeros(N, int); y[rows] = codes
    valid = np.ones(N, bool)
    for rep in range(REPS):
        v = np.zeros(N, bool)
        for tr, te in fold_list("match", rep):
            v[te[splits.seen_in_train(meta, tr, te)]] = True
        valid &= v
    acc = lambda yy, pp: np.mean(yy == pp)
    print(f"\n=== IDENTITY (top-1 accuracy; {len(names)} players seen in 2+ matches; folds split by match; "
          f"chance {1/len(names):.3f}; mean of {REPS} repeats) ===", flush=True)
    p = rf_cv(["height_cm"], y, "match", "class"); dr = draws_for(ok_of(p, valid))
    record("identity", "reference: height only (RF)", "accuracy", score(p, dr, acc, y))
    for mode in ("raw", "scaled", "motion"):
        sc = {}
        for size in ("arm", "body"):
            p = run_cv(inp(size, mode), y, "match", "class", cfg[size], n_out=len(names))
            dr = draws_for(ok_of(p, valid))
            sc[size] = score(p, dr, acc, y)
            record("identity", f"transformer {size} [{mode}]", "accuracy", sc[size])
        paired("identity", f"  arm minus body [{mode}]", sc["arm"], sc["body"], "accuracy")

# ---------------------------------------------------------------- serve number
if "serve" in which:
    y = (meta.serve_number == 2).astype(int).values
    auc = lambda yy, pp: roc_auc_score(yy, pp)
    print(f"\n=== SERVE NUMBER (AUC, 0.5 = chance; folds split by player; mean of {REPS} repeats) ===", flush=True)
    p = rf_cv(["speed_kmh"], y, "player", "prob"); dr = draws_for(ok_of(p))
    record("serve number", "reference: speed only (RF, label sanity)", "AUC", score(p, dr, auc, y))
    for mode in ("raw", "scaled"):
        sc = {}
        for size in ("arm", "body"):
            p = run_cv(inp(size, mode), y, "player", "prob", cfg[size]); dr = draws_for(ok_of(p))
            sc[size] = score(p, dr, auc, y)
            record("serve number", f"transformer {size} [{mode}]", "AUC", sc[size])
        paired("serve number", f"  arm minus body [{mode}]", sc["arm"], sc["body"], "AUC")

# ---------------------------------------------------------------- speed
if "speed" in which:
    y = meta.speed_kmh.values.astype(float)
    female = (meta.gender == "F").astype(float).values
    second = (meta.serve_number == 2).astype(float).values
    M1 = np.c_[female, second]
    M2 = np.c_[female, second, meta.height_cm.values]
    print("\n=== SERVE SPEED (MAE in km/h, lower is better; folds split by player; scaled input) ===", flush=True)
    base = {k: np.full(N, np.nan) for k in ("mean", "B1", "B2")}
    resid = {(s, v): np.full(N, np.nan) for s in ("arm", "body") for v in ("pose only", "pose + B1", "pose + B2")}
    for tr, te in fold_list("player"):
        fit, val = splits.inner_split(tr, "player", meta)
        base["mean"][te] = y[tr].mean()
        lr = {}
        for k, M in (("B1", M1), ("B2", M2)):
            lr[k] = LinearRegression().fit(M[tr], y[tr])
            base[k][te] = lr[k].predict(M[te])
        for size in ("arm", "body"):
            X = inp(size, "scaled")
            resid[(size, "pose only")][te] = L.fit_predict(X, y, fit, val, te, "reg", cfg[size], epochs=EPOCHS)
            for k, M, v in (("B1", M1, "pose + B1"), ("B2", M2, "pose + B2")):
                yres = np.zeros(N); yres[tr] = y[tr] - lr[k].predict(M[tr])
                resid[(size, v)][te] = base[k][te] + L.fit_predict(X, yres, fit, val, te, "reg", cfg[size], epochs=EPOCHS)
    ok = ~np.isnan(base["mean"]); dr = draws_for(ok)
    mae = lambda p: np.array([mean_absolute_error(y[i], p[i]) for i in dr])
    S = {"mean baseline": mae(base["mean"]), "baseline B1: gender + serve#": mae(base["B1"]),
         "baseline B2: gender + serve# + height": mae(base["B2"])}
    for k, v in S.items():
        record("speed", k, "MAE", v)
    for (size, v), p in resid.items():
        S[f"transformer {size}: {v}"] = mae(p)
        record("speed", f"transformer {size}: {v}", "MAE", S[f"transformer {size}: {v}"])
    print("  -- change vs the matching baseline (negative = pose helps; CI excluding 0 = real) --", flush=True)
    for size in ("arm", "body"):
        for v, ref in (("pose only", "mean baseline"), ("pose + B1", "baseline B1: gender + serve#"),
                       ("pose + B2", "baseline B2: gender + serve# + height")):
            paired("speed", f"{size} {v} minus {ref.split(':')[0]}", S[f"transformer {size}: {v}"], S[ref], "MAE")
    for v in ("pose only", "pose + B1", "pose + B2"):
        paired("speed", f"arm minus body ({v})", S[f"transformer arm: {v}"], S[f"transformer body: {v}"], "MAE")
