#!/usr/bin/env python3
"""
Random Forest serve-speed prediction, in two stages, on the shared player-level
5-fold splits (3 repeats).

Stage 1 (all serves): how much does the metadata explain, and does pose add anything?
    Mean | LR gender | LR gender + serve number | LR gender + serve number + height
    RF gender + serve number | RF arm only | RF body only
    RF gender + serve number + arm | RF gender + serve number + body
Stage 2 (four subgroups: female/male x first/second serve, so gender and serve number
are fixed): does anything beat the subgroup mean?
    Mean | RF height | RF arm | RF body
    Within each fold, subgroup models are trained on that subgroup's training players
    only. Folds are by player, so a held-out player is held out for both serve types.

Pose = motion features (rf_features.py): serving arm 18, full body 102.
Metric: MAE in km/h. Confidence intervals: bootstrap over players (2,000 resamples),
on MAE and on paired MAE differences.

Run from the repository root:
    python src/random_forest/rf_speed.py
Output: results/random_forest/speed/ results_stage1.csv, bootstrap_stage1.csv,
        results_subgroups.csv, bootstrap_subgroups.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

import rf_common as rc

GROUPS = [("F first", "F", 1), ("F second", "F", 2), ("M first", "M", 1), ("M second", "M", 2)]


def oof(make, X, y, rows, fold, reps):
    """Out-of-fold predictions (repeats x serves) for the serves in `rows`.
    make=None predicts the training mean."""
    P = np.zeros((len(reps), len(rows)))
    for ri, r in enumerate(reps):
        f = fold[r][rows]
        for k in range(5):
            te, tr = np.flatnonzero(f == k), np.flatnonzero(f != k)
            if make is None:
                P[ri, te] = y[rows][tr].mean()
            else:
                P[ri, te] = make().fit(X[rows][tr], y[rows][tr]).predict(X[rows][te])
    return P


def metrics(y, P):
    mae = [np.abs(y - p).mean() for p in P]
    rmse = [np.sqrt(((y - p) ** 2).mean()) for p in P]
    r2 = [1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum() for p in P]
    return dict(MAE=np.mean(mae), MAE_sd_reps=np.std(mae, ddof=1) if len(mae) > 1 else np.nan,
                RMSE=np.mean(rmse), R2=np.mean(r2))


def bootstrap(y, preds, players, pairs, rng):
    """Player bootstrap of MAE and of paired MAE differences (rep-averaged errors)."""
    ups, code = np.unique(players, return_inverse=True)
    n = np.bincount(code, minlength=len(ups)).astype(float)
    W = rc.draw_weights(len(ups), rng)
    err = {k: np.bincount(code, weights=np.abs(y - P).mean(axis=0), minlength=len(ups)) for k, P in preds.items()}
    b = {k: (W @ e) / (W @ n) for k, e in err.items()}
    mae_ci = {k: rc.ci(v) for k, v in b.items()}
    rows = []
    for a, ref in pairs:
        lo, hi = rc.ci(b[a] - b[ref])
        d = np.abs(y - preds[a]).mean(axis=0).mean() - np.abs(y - preds[ref]).mean(axis=0).mean()
        rows.append(dict(new=a, reference=ref, dMAE=d, lo95=lo, hi95=hi,
                         share_new_better=float(((b[a] - b[ref]) < 0).mean()), ci_excludes_0=bool(lo > 0 or hi < 0)))
    return mae_ci, pd.DataFrame(rows)


def main(n_reps=None):
    t0 = time.time()
    df = rc.load_data()
    folds = rc.shared_folds(df, "player", n_reps)
    reps = sorted(folds)
    y = df["speed_kmh"].to_numpy(dtype=float)
    players = df["server"].to_numpy()
    arm = rc.representation_columns(df, "arm", ("motion",))
    body = rc.representation_columns(df, "body", ("motion",))
    meta2 = ["gender_M", "second"]
    allrows = np.arange(len(df))

    stage1 = [("Mean", None, []),
              ("LR gender", rc.linear, ["gender_M"]),
              ("LR gender + serve number", rc.linear, meta2),
              ("LR gender + serve number + height", rc.linear, meta2 + ["height_cm"]),
              ("RF gender + serve number", rc.rf_regressor, meta2),
              ("RF arm only", rc.rf_regressor, arm),
              ("RF body only", rc.rf_regressor, body),
              ("RF gender + serve number + arm", rc.rf_regressor, meta2 + arm),
              ("RF gender + serve number + body", rc.rf_regressor, meta2 + body)]
    stage2 = [("Mean", None, []), ("RF height", rc.rf_regressor, ["height_cm"]),
              ("RF arm", rc.rf_regressor, arm), ("RF body", rc.rf_regressor, body)]

    pred1 = {}
    for name, make, cols in stage1:
        X = df[cols].to_numpy(dtype=float) if cols else np.zeros((len(df), 1))
        pred1[name] = oof(make, X, y, allrows, folds, reps)
        print(f"  stage 1: {name} ({time.time() - t0:.0f}s)", flush=True)
    pred2 = {}
    for g, sex, sn in GROUPS:
        rows = np.flatnonzero((df["gender"] == sex).to_numpy() & (df["serve_number"] == sn).to_numpy())
        for name, make, cols in stage2:
            X = df[cols].to_numpy(dtype=float) if cols else np.zeros((len(df), 1))
            pred2[(g, name)] = (rows, oof(make, X, y, rows, folds, reps))
        print(f"  stage 2: {g} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(rc.BOOT_SEED)     # one stream for all bootstraps
    pairs1 = [("LR gender", "Mean"), ("LR gender + serve number", "LR gender"),
              ("LR gender + serve number + height", "LR gender + serve number"),
              ("RF arm only", "Mean"), ("RF body only", "Mean"), ("RF body only", "RF arm only"),
              ("RF gender + serve number", "LR gender + serve number"),
              ("RF gender + serve number + arm", "RF gender + serve number"),
              ("RF gender + serve number + body", "RF gender + serve number"),
              ("RF gender + serve number + body", "RF gender + serve number + arm")]
    ci1, d1 = bootstrap(y, pred1, players, pairs1, rng)
    r1 = pd.DataFrame([dict(model=k, **metrics(y, P), MAE_lo95=ci1[k][0], MAE_hi95=ci1[k][1])
                       for k, P in pred1.items()])
    r2, d2 = [], []
    for g, _, _ in GROUPS:
        rows = pred2[(g, "Mean")][0]
        preds = {n: pred2[(g, n)][1] for n, _, _ in stage2}
        ci2, d = bootstrap(y[rows], preds, players[rows], [("RF arm", "Mean"), ("RF body", "Mean"),
                           ("RF height", "Mean"), ("RF arm", "RF body")], rng)
        for n, P in preds.items():
            r2.append(dict(group=g, model=n, n_serves=len(rows), n_players=len(np.unique(players[rows])),
                           **metrics(y[rows], P), MAE_lo95=ci2[n][0], MAE_hi95=ci2[n][1]))
        d2.append(d.assign(group=g))
    r2, d2 = pd.DataFrame(r2), pd.concat(d2)

    out = rc.RESULTS / "speed"
    out.mkdir(parents=True, exist_ok=True)
    r1.to_csv(out / "results_stage1.csv", index=False)
    d1.to_csv(out / "bootstrap_stage1.csv", index=False)
    r2.to_csv(out / "results_subgroups.csv", index=False)
    d2.to_csv(out / "bootstrap_subgroups.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nSTAGE 1\n", r1.round(2).to_string(index=False))
    print("\nSUBGROUPS\n", r2.round(2).to_string(index=False))
    print(f"-> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None, help="number of repeats (default: all 3)")
    main(ap.parse_args().reps)
