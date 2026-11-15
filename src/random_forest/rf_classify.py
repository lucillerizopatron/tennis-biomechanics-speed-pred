#!/usr/bin/env python3
"""
Random Forest classification: gender, player identity, serve number.

Serving arm (3 joints) vs. full body (17 joints), with shape, motion and scaled-motion
features (see rf_features.py), on the shared 5-fold splits (3 repeats):
  gender       : whole players held out; balanced accuracy and AUC (chance 50% / 0.5)
  identity     : whole matches held out, 46 players seen in 2+ matches; a test serve is
                 scored only if its player is in the training folds; balanced accuracy
                 and top-1 accuracy (chance 1/46)
  serve_number : whole players held out; class-weighted RF (70% first serves);
                 balanced accuracy and AUC; also metadata-only baselines (gender + height)
                 and pose + metadata
Confidence intervals: bootstrap over players (2,000 resamples); model comparisons use
paired differences on the same resamples.

Run from the repository root:
    python src/random_forest/rf_classify.py --target all
Output: results/random_forest/<target>/results.csv, bootstrap_differences.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import rf_common as rc

META = ["gender_M", "height_cm"]


def model_specs(df, target):
    """(name, model factory, feature columns)."""
    weight = "balanced" if target == "serve_number" else None
    specs = []
    if target == "serve_number":
        specs += [("Metadata: gender + height (logistic)", lambda: rc.logistic(weight), META),
                  ("Metadata: gender + height", lambda: rc.rf_classifier(weight), META)]
    for name, size, kinds in rc.REPRESENTATIONS:
        specs.append((name, lambda: rc.rf_classifier(weight), rc.representation_columns(df, size, kinds)))
    if target == "serve_number":
        for size in ("Arm", "Body"):
            specs.append((f"{size} Shape + Motion + metadata", lambda: rc.rf_classifier(weight),
                          META + rc.representation_columns(df, size.lower(), ("shape", "motion"))))
    return specs


def evaluate(target, y, pcode, n_players, out, mask, n_classes=None):
    """Point metrics, bootstrap CIs and paired differences from out-of-fold predictions."""
    binary = target != "identity"
    # metrics and player-level bootstrap
    rows, boots = [], {}
    if binary:
        W = rc.draw_weights(n_players)
        Ws = W[:, pcode]
        for name, P in out.items():
            pred = (P >= 0.5).astype(int)
            count = lambda c: np.bincount(pcode, weights=c.sum(0).astype(float), minlength=n_players)
            tp1, n1 = count((pred == 1) & (y == 1)), count(np.broadcast_to(y == 1, P.shape))
            tp0, n0 = count((pred == 0) & (y == 0)), count(np.broadcast_to(y == 0, P.shape))
            avg = P.mean(0)
            boots[name] = {"balanced_accuracy": ((W @ tp1) / (W @ n1) + (W @ tp0) / (W @ n0)) / 2,
                           "auc": np.array([roc_auc_score(y, avg, sample_weight=w) for w in Ws])}
            rows.append(dict(model=name, balanced_accuracy=(tp1.sum() / n1.sum() + tp0.sum() / n0.sum()) / 2,
                             accuracy=float((pred == y).mean()),
                             auc=np.mean([roc_auc_score(y, p) for p in P]),
                             share_pred_positive=float(pred.mean())))
    else:
        W = rc.draw_weights(n_classes)       # players are the classes
        for name, P in out.items():
            P = P.astype(int)
            corr = np.bincount(y, weights=((P == y) & mask).sum(0).astype(float), minlength=n_classes)
            sup = np.bincount(y, weights=mask.sum(0).astype(float), minlength=n_classes)
            ok = sup > 0
            rec = np.where(ok, corr / np.maximum(sup, 1), np.nan)
            boots[name] = {"balanced_accuracy": np.array([np.nansum(w * rec) / (w * ok).sum() for w in W]),
                           "top1_accuracy": (W @ corr) / (W @ sup)}
            rows.append(dict(model=name, balanced_accuracy=np.nanmean(rec), top1_accuracy=corr.sum() / sup.sum()))
    res = pd.DataFrame(rows)
    for met in boots[res.model[0]]:
        res[f"{met}_lo95"] = [rc.ci(boots[n][met])[0] for n in res.model]
        res[f"{met}_hi95"] = [rc.ci(boots[n][met])[1] for n in res.model]

    pairs = [(f"Body {k}", f"Arm {k}") for k in rc.KIND_LABELS]
    for s in ("Arm", "Body"):
        pairs += [(f"{s} Shape + Motion", f"{s} Shape"), (f"{s} Motion", f"{s} Shape")]
    if target == "serve_number":
        pairs += [(f"{s} Shape + Motion + metadata", "Metadata: gender + height (logistic)") for s in ("Arm", "Body")]
    point = res.set_index("model")
    diffs = []
    for a, b in pairs:
        for met in boots[a]:
            lo, hi = rc.ci(boots[a][met] - boots[b][met])
            diffs.append(dict(new=a, reference=b, metric=met, diff=point.loc[a, met] - point.loc[b, met],
                              lo95=lo, hi95=hi, ci_excludes_0=bool(lo > 0 or hi < 0)))
    diffs = pd.DataFrame(diffs)

    return res, diffs


def run(target, n_reps=None):
    t0 = time.time()
    data = rc.load_data()
    folds = rc.shared_folds(data, "match" if target == "identity" else "player", n_reps)
    reps = sorted(folds)
    keep = folds[reps[0]] >= 0
    df = data.loc[keep].reset_index(drop=True)
    fold = {r: folds[r][keep] for r in reps}
    binary = target != "identity"
    if target == "gender":
        y = (df["gender"] == "M").astype(int).to_numpy()
    elif target == "serve_number":
        y = df["second"].to_numpy()
    else:
        labels, y = np.unique(df["server"].to_numpy(), return_inverse=True)
    players, pcode = np.unique(df["server"].to_numpy(), return_inverse=True)
    print(f"{target}: {len(df)} serves, {len(players)} players, {len(reps)} repeats x 5 folds", flush=True)

    # identity: a test serve counts only if its player appears in the training folds
    mask = np.ones((len(reps), len(df)), bool)
    if not binary:
        for ri, r in enumerate(reps):
            for k in range(5):
                te = fold[r] == k
                mask[ri, te] = np.isin(y[te], np.unique(y[fold[r] != k]))

    # out-of-fold predictions: probabilities (binary) or labels (identity)
    out = {}
    for name, make, cols in model_specs(df, target):
        X = df[cols].to_numpy(dtype=float)
        P = np.zeros((len(reps), len(df)))
        for ri, r in enumerate(reps):
            for k in range(5):
                te, tr = np.flatnonzero(fold[r] == k), np.flatnonzero(fold[r] != k)
                m = make().fit(X[tr], y[tr])
                P[ri, te] = m.predict_proba(X[te])[:, 1] if binary else m.predict(X[te])
        out[name] = P
        print(f"  {name} ({time.time() - t0:.0f}s)", flush=True)

    res, diffs = evaluate(target, y, pcode, len(players), out, mask, None if binary else len(labels))

    outdir = rc.RESULTS / target
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "results.csv", index=False)
    diffs.to_csv(outdir / "bootstrap_differences.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"-> {outdir} ({time.time() - t0:.0f}s)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all", choices=["gender", "identity", "serve_number", "all"])
    ap.add_argument("--reps", type=int, default=None, help="number of repeats (default: all 3)")
    a = ap.parse_args()
    for t in (["gender", "identity", "serve_number"] if a.target == "all" else [a.target]):
        run(t, a.reps)

