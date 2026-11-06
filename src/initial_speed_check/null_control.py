"""
Can a different model class extract value from pose features beyond the linear
metadata baseline?  Reads features.csv written by rf_baseline.py.

Models (all use the SAME serves; every fit uses only training players):
  baseline            : linear regression on gender, serve_number, height_cm
  ridge + pose        : ridge on metadata + pose (alpha chosen by inner player-grouped CV)
  residual ridge      : baseline, then ridge on pose features predicting its residuals
  residual RF         : baseline, then a shallow, heavily regularised RF on residuals
  residual GBM        : baseline, then a small gradient-boosting model on residuals
  NULL control        : residual RF with pose rows randomly shuffled across serves
                        (calibrates how much "improvement" pure noise produces)

Each is run for the 3-joint arm and the 17-joint full-body feature sets.

Evaluation:
  A) the original 87/22 player split (the 'split' column)
  B) repeated player-grouped K-fold CV over ALL players (out-of-fold predictions)

Usage:  python null_control.py        (requirements: numpy, pandas, scikit-learn)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================ CONFIGURATION ================================
OUT_DIR = Path(__file__).resolve().parent / "pose_rf_outputs"
FEATURES_CSV = OUT_DIR / "features.csv"
N_SPLITS = 5            # folds per CV repeat
N_REPEATS = 5           # different random player-to-fold assignments
N_BOOT = 2000           # player bootstrap resamples for out-of-fold comparison
SEED = 42
INNER_FOLDS = 5         # inner player-grouped folds for choosing ridge alpha
RIDGE_ALPHAS = np.logspace(-1, 4, 11)
RESID_RIDGE_ALPHAS = np.logspace(0, 5, 11)
META_CAT, META_NUM, TARGET = ["gender", "serve_number"], ["height_cm"], "speed_kmh"
BASELINE = "Linear: metadata (baseline)"
# ===========================================================================


def load() -> tuple[pd.DataFrame, list, list]:
    if not FEATURES_CSV.exists():
        sys.exit(f"ERROR: {FEATURES_CSV} not found. Run rf_baseline.py first.")
    df = pd.read_csv(FEATURES_CSV)
    need = ["server", "split", "pose_ok", TARGET] + META_CAT + META_NUM
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"ERROR: features.csv missing columns {miss}")
    df = df[df["pose_ok"].astype(bool)].reset_index(drop=True)
    arm = [c for c in df.columns if c.startswith("arm_")]
    body = [c for c in df.columns if c.startswith("body_")]
    if not arm or not body:
        sys.exit("ERROR: no arm_/body_ feature columns in features.csv")
    if df[arm + body].isna().any().any():
        sys.exit("ERROR: NaNs in pose features among pose_ok rows")
    print(f"Loaded {len(df)} serves, {df['server'].nunique()} players; "
          f"{len(arm)} arm features, {len(body)} body features")
    return df, arm, body


def prep(num_cols, scale=False, with_meta=True) -> ColumnTransformer:
    num = [("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if scale else [])
    parts = []
    if with_meta:
        parts.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), META_CAT))
    parts.append(("num", Pipeline(num), num_cols))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def fit_baseline(tr):
    pipe = Pipeline([("pre", prep(META_NUM)), ("reg", LinearRegression())])
    pipe.fit(tr[META_CAT + META_NUM], tr[TARGET])
    return pipe


def grouped_ridge(pipe_alpha_key, alphas, X, y, groups):
    n_groups = groups.nunique()
    gs = GridSearchCV(pipe_alpha_key, {"reg__alpha": alphas},
                      cv=GroupKFold(min(INNER_FOLDS, n_groups)),
                      scoring="neg_mean_absolute_error")
    gs.fit(X, y, groups=groups)
    return gs


# ---- model definitions: each returns predictions for `te` given `tr` -------
def m_baseline(tr, te, pose):
    return fit_baseline(tr).predict(te[META_CAT + META_NUM])


def m_ridge(tr, te, pose):
    cols = META_CAT + META_NUM + pose
    pipe = Pipeline([("pre", prep(META_NUM + pose, scale=True)), ("reg", Ridge())])
    gs = grouped_ridge(pipe, RIDGE_ALPHAS, tr[cols], tr[TARGET], tr["server"])
    return gs.predict(te[cols])


def _residual(tr, te, pose, stage2):
    base = fit_baseline(tr)
    r_tr = tr[TARGET].to_numpy() - base.predict(tr[META_CAT + META_NUM])
    p_base = base.predict(te[META_CAT + META_NUM])
    return p_base + stage2(tr, r_tr, te, pose)


def m_resid_ridge(tr, te, pose):
    def s2(tr, r, te, pose):
        pipe = Pipeline([("pre", prep(pose, scale=True, with_meta=False)), ("reg", Ridge())])
        gs = grouped_ridge(pipe, RESID_RIDGE_ALPHAS, tr[pose], r, tr["server"])
        return gs.predict(te[pose])
    return _residual(tr, te, pose, s2)


def _tree_stage2(make):
    def s2(tr, r, te, pose):
        cols = META_CAT + META_NUM + pose
        pipe = Pipeline([("pre", prep(META_NUM + pose)), ("reg", make())])
        pipe.fit(tr[cols], r)
        return pipe.predict(te[cols])
    return s2


def m_resid_rf(tr, te, pose):
    return _residual(tr, te, pose, _tree_stage2(lambda: RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=20, max_features=0.33,
        random_state=SEED, n_jobs=-1)))


def m_resid_gbm(tr, te, pose):
    return _residual(tr, te, pose, _tree_stage2(lambda: GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=2, min_samples_leaf=20,
        subsample=0.7, random_state=SEED)))


def build_models(arm, body):
    models = {BASELINE: (m_baseline, [], False)}
    for tag, cols in (("arm", arm), ("body", body)):
        models[f"Ridge: meta + {tag}"] = (m_ridge, cols, False)
        models[f"Resid ridge: {tag}"] = (m_resid_ridge, cols, False)
        models[f"Resid RF: {tag}"] = (m_resid_rf, cols, False)
        models[f"Resid GBM: {tag}"] = (m_resid_gbm, cols, False)
    models["NULL: resid RF, shuffled arm"] = (m_resid_rf, arm, True)
    models["NULL: resid RF, shuffled body"] = (m_resid_rf, body, True)
    return models


def shuffled(df, cols, seed):
    out = df.copy()
    perm = np.random.default_rng(seed).permutation(len(df))
    out[cols] = df[cols].to_numpy()[perm]
    return out


def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))


def player_bootstrap_diff(players, y, p_new, p_ref, n_boot, seed=0):
    codes, uniq = pd.factorize(players)
    P = len(uniq)
    cnt = np.bincount(codes, minlength=P).astype(float)
    def s(p):
        e = y - p
        return np.bincount(codes, weights=np.abs(e), minlength=P), np.bincount(codes, weights=e ** 2, minlength=P)
    (na, nq), (ra, rq) = s(p_new), s(p_ref)
    d = np.random.default_rng(seed).integers(0, P, size=(n_boot, P))
    c = cnt[d].sum(1)
    return (na[d].sum(1) - ra[d].sum(1)) / c, np.sqrt(nq[d].sum(1) / c) - np.sqrt(rq[d].sum(1) / c)


def run_model(name, spec, df, tr_idx, te_idx, seed):
    fn, pose, null = spec
    data = shuffled(df, pose, seed) if null else df
    return fn(data.iloc[tr_idx], data.iloc[te_idx], pose)


def main():
    pd.set_option("display.width", 200)
    df, arm, body = load()
    models = build_models(arm, body)
    y = df[TARGET].to_numpy()
    t0 = time.time()

    # ---------------- A) original split ----------------
    tr_idx = np.flatnonzero(df["split"] == "train")
    te_idx = np.flatnonzero(df["split"] == "test")
    assert set(df.iloc[tr_idx]["server"]).isdisjoint(set(df.iloc[te_idx]["server"]))
    print(f"\n[A] ORIGINAL SPLIT: {len(tr_idx)} train / {len(te_idx)} test serves, "
          f"{df.iloc[te_idx]['server'].nunique()} test players")
    rows = []
    for name, spec in models.items():
        p = run_model(name, spec, df, tr_idx, te_idx, SEED)
        rows.append({"Model": name, **metrics(y[te_idx], p)})
        print(f"  done {name} ({time.time()-t0:.0f}s)")
    A = pd.DataFrame(rows)
    b = A.set_index("Model").loc[BASELINE]
    for m in ("MAE", "RMSE", "R2"):
        A[f"d{m}"] = A[m] - b[m]
    print("\nORIGINAL SPLIT (dX = change vs baseline; MAE/RMSE negative = better, R2 positive = better)")
    print(A.round(3).to_string(index=False))
    A.to_csv(OUT_DIR / "compare_original_split.csv", index=False)

    # ---------------- B) repeated player-grouped CV ----------------
    players = df["server"].unique()
    print(f"\n[B] REPEATED GROUPED CV: {N_REPEATS} x {N_SPLITS}-fold over {len(players)} players")
    per_repeat, oof0 = [], {}
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(SEED + rep)
        fold_of = dict(zip(rng.permutation(players), np.arange(len(players)) % N_SPLITS))
        folds = df["server"].map(fold_of).to_numpy()
        oof = {n: np.zeros(len(df)) for n in models}
        for f in range(N_SPLITS):
            tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            assert set(df.iloc[tr]["server"]).isdisjoint(set(df.iloc[te]["server"]))
            for name, spec in models.items():
                oof[name][te] = run_model(name, spec, df, tr, te, SEED + 100 * rep + f)
        for name in models:
            per_repeat.append({"repeat": rep, "Model": name, **metrics(y, oof[name])})
        if rep == 0:
            oof0 = oof
        print(f"  repeat {rep+1}/{N_REPEATS} done ({time.time()-t0:.0f}s)")

    R = pd.DataFrame(per_repeat)
    piv = {m: R.pivot(index="repeat", columns="Model", values=m) for m in ("MAE", "RMSE", "R2")}
    summ = []
    for name in models:
        row = {"Model": name}
        for m in ("MAE", "RMSE", "R2"):
            row[m] = piv[m][name].mean()
            row[f"{m}_sd"] = piv[m][name].std(ddof=1) if N_REPEATS > 1 else np.nan
            row[f"d{m}"] = (piv[m][name] - piv[m][BASELINE]).mean()
        row["repeats_MAE_better"] = int((piv["MAE"][name] < piv["MAE"][BASELINE]).sum())
        summ.append(row)
    B = pd.DataFrame(summ)
    print(f"\nGROUPED CV, mean over {N_REPEATS} repeats (all {len(df)} serves, out-of-fold)")
    print(B[["Model", "MAE", "MAE_sd", "RMSE", "R2", "dMAE", "dRMSE", "dR2", "repeats_MAE_better"]]
          .round(3).to_string(index=False))
    B.to_csv(OUT_DIR / "compare_grouped_cv.csv", index=False)

    if N_BOOT > 0:
        brow = []
        for name in models:
            if name == BASELINE:
                continue
            dm, dr = player_bootstrap_diff(df["server"].to_numpy(), y, oof0[name], oof0[BASELINE], N_BOOT)
            brow.append({"Model": name, "dMAE": dm.mean(), "lo95": np.percentile(dm, 2.5),
                         "hi95": np.percentile(dm, 97.5), "share_better": float((dm < 0).mean()),
                         "dRMSE": dr.mean(), "RMSE_lo95": np.percentile(dr, 2.5),
                         "RMSE_hi95": np.percentile(dr, 97.5)})
        C = pd.DataFrame(brow)
        print(f"\nPLAYER BOOTSTRAP on repeat-0 out-of-fold predictions ({df['server'].nunique()} players); "
              f"model minus baseline, negative = better")
        print(C.round(3).to_string(index=False))
        C.to_csv(OUT_DIR / "compare_bootstrap.csv", index=False)

    print("\nHOW TO READ: a pose model is only convincing if (1) it beats the baseline in most "
          "repeats, (2) the bootstrap interval excludes 0, and (3) it clearly beats the NULL "
          "(shuffled-pose) rows. If NULL rows improve about as much, the gain is not pose signal.")
    print(f"Saved CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
