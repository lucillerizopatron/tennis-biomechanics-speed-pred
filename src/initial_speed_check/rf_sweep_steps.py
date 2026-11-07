"""
Frame-skipping sweep: how does serve-speed prediction change as frames are dropped?
(2024 US Open serve data; Random Forest on pose-derived features)

Per-serve pipeline for every FRAME_STEP in STEPS:
    hip-relative -> smoothing (at the FULL frame rate) -> x[::step] -> features
Speeds / accelerations use the EFFECTIVE frame rate FPS / step.

Why this script is more robust than a single train/test split
-------------------------------------------------------------
* Grouped K-fold cross-validation BY PLAYER: every serve is predicted exactly once
  by a model that never saw that player. All 109 players act as test players
  (instead of only 22).
* Repeated N_REPEATS times with different player-to-fold assignments and different
  Random Forest seeds, so every metric comes with a mean and a standard deviation.
* The SAME serves are used for every step (serves that fail at any step are removed
  from all steps), and the SAME folds are used for every step within a repetition,
  so differences between steps are paired.
* Player-level bootstrap of the paired MAE/RMSE differences (step s vs step 1, and
  pose models vs the metadata-only RF), using out-of-fold predictions averaged over
  repetitions.

Models
------
  Linear regression: metadata only                 (does not depend on the step)
  RF: metadata only                                (does not depend on the step)
  RF: metadata + arm (existing features)           (3 dominant-arm joints)
  RF: metadata + body (existing features)          (17 joints)
  RF: metadata + arm + acceleration
  RF: metadata + body + acceleration

Run (from the folder containing this script):
    py rf_sweep_steps.py
Quick test:
    py rf_sweep_steps.py --steps 1,4,8 --repeats 1 --folds 3 --bootstrap 200 --estimators 50

Outputs (in pose_rf_outputs/):
    sweep_summary.csv            mean/std of MAE, RMSE, R2 per model and step
    sweep_paired_bootstrap.csv   paired differences with 95% player-bootstrap CIs
    sweep_mae_vs_fps.png         MAE vs effective FPS, and change vs full frame rate

Requirements: numpy, pandas, scipy, scikit-learn, pyarrow, matplotlib (for the plot).
Feature caches written by rf_baseline_skip2.py (joint_features_cache_step{N}.*) are reused.
All joint-motion features are in COORDINATE UNITS (per second), not metres.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

# ============================================================================
# CONFIGURATION  (edit here, or override from the command line)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR.parent                    # folder containing serve_data.py etc.
DATA_DIR_OVERRIDE = SCRIPT_DIR                    # metadata.parquet and keypoints/ sit next to this script
OUT_DIR = SCRIPT_DIR / "pose_rf_outputs"

STEPS = [1, 2, 3, 4, 6, 8, 10, 12]                # 1 = all frames; must contain 1 (reference)
N_FOLDS = 5                                       # grouped (by player) cross-validation folds
N_REPEATS = 3                                     # repetitions with different folds / RF seeds
N_BOOTSTRAP = 2000                                # player-level bootstrap resamples (0 disables)

SMOOTHING_MODULE = "smoothing"
HANDEDNESS_MODULE = "serve_handedness"
SERVE_DATA_MODULE = "serve_data"

USE_FEATURE_CACHE = True
SMOOTH_CUTOFF_HZ = None                           # None -> smoothing module's DEFAULT_CUTOFF_HZ (8 Hz)
SMOOTH_ORDER = 2

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=5,
                 random_state=42, n_jobs=-1)

# Fallback used ONLY if the handedness module cannot be imported. NOT FULLY VERIFIED.
FALLBACK_LEFT_HANDED = {"Denis Shapovalov", "Beatriz Haddad Maia"}

META_CAT = ["gender", "serve_number"]
META_NUM = ["height_cm"]
TARGET = "speed_kmh"

M_LIN = "Linear regression: metadata"
M_RFMETA = "RF: metadata only"
M_ARM = "RF: metadata + arm (existing features)"
M_BODY = "RF: metadata + body (existing features)"
M_ARM_ACC = "RF: metadata + arm + acceleration"
M_BODY_ACC = "RF: metadata + body + acceleration"
POSE_MODELS = [M_ARM, M_BODY, M_ARM_ACC, M_BODY_ACC]


def parse_args():
    ap = argparse.ArgumentParser(description="Frame-skipping sweep with grouped CV.")
    ap.add_argument("--steps", type=str, default=None,
                    help="comma-separated FRAME_STEP values, e.g. 1,2,4,8 (must include 1)")
    ap.add_argument("--folds", type=int, default=None, help="number of grouped CV folds")
    ap.add_argument("--repeats", type=int, default=None, help="number of repetitions")
    ap.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples (0 disables)")
    ap.add_argument("--estimators", type=int, default=None, help="override RF n_estimators")
    return ap.parse_args()


_args = parse_args()
if _args.steps:
    STEPS = [int(s) for s in _args.steps.split(",") if s.strip()]
if _args.folds:
    N_FOLDS = _args.folds
if _args.repeats:
    N_REPEATS = _args.repeats
if _args.bootstrap is not None:
    N_BOOTSTRAP = _args.bootstrap
if _args.estimators:
    RF_PARAMS["n_estimators"] = _args.estimators

STEPS = sorted(set(STEPS))
if 1 not in STEPS or any((not isinstance(s, int)) or s < 1 for s in STEPS):
    sys.exit("ERROR: STEPS must be integers >= 1 and must include 1 (the full-frame-rate reference).")
if N_FOLDS < 2 or N_REPEATS < 1:
    sys.exit("ERROR: need at least 2 folds and 1 repeat.")

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================
def _require(module: str, pip_name: str | None = None):
    try:
        return importlib.import_module(module)
    except ImportError:
        sys.exit(f"ERROR: missing dependency '{module}'. Install with: pip install {pip_name or module}")

np = _require("numpy")
pd = _require("pandas")
_require("scipy")
_require("sklearn", "scikit-learn")
try:
    importlib.import_module("pyarrow")
except ImportError:
    try:
        importlib.import_module("fastparquet")
    except ImportError:
        sys.exit("ERROR: reading metadata.parquet needs 'pyarrow' (or 'fastparquet'). pip install pyarrow")

from scipy.signal import butter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================================
# IMPORT PROJECT HELPERS
# ============================================================================
sys.path.insert(0, str(HELPER_DIR))
try:
    sd = importlib.import_module(SERVE_DATA_MODULE)
except ImportError as e:
    sys.exit(f"ERROR: cannot import '{SERVE_DATA_MODULE}' from {HELPER_DIR} ({e}). "
             f"Adjust HELPER_DIR / SERVE_DATA_MODULE at the top of this script.")
try:
    sm = importlib.import_module(SMOOTHING_MODULE)
except ImportError as e:
    sys.exit(f"ERROR: cannot import smoothing module '{SMOOTHING_MODULE}' from {HELPER_DIR} ({e}).")
for _fn in ("hip_relative", "smooth"):
    if not hasattr(sm, _fn):
        sys.exit(f"ERROR: smoothing module '{SMOOTHING_MODULE}' has no function '{_fn}'.")

try:
    _hm = importlib.import_module(HANDEDNESS_MODULE)
    LEFT_HANDED = set(getattr(_hm, "LEFT_HANDED"))
    hand = _hm.hand
    HAND_SOURCE = HANDEDNESS_MODULE
except (ImportError, AttributeError):
    warnings.warn(f"Could not import '{HANDEDNESS_MODULE}'; using embedded fallback left-hander list "
                  f"{sorted(FALLBACK_LEFT_HANDED)}. Handedness is NOT FULLY VERIFIED.")
    LEFT_HANDED = set(FALLBACK_LEFT_HANDED)
    hand = lambda s: "L" if s in LEFT_HANDED else "R"          # noqa: E731
    HAND_SOURCE = "embedded fallback"

DATA_DIR = Path(DATA_DIR_OVERRIDE) if DATA_DIR_OVERRIDE else Path(sd.DATA)
FPS = float(sd.FPS)

# ============================================================================
# JOINTS / FEATURE DEFINITIONS
# ============================================================================
JOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
               "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
               "l_wrist", "r_wrist", "l_hip", "r_hip",
               "l_knee", "r_knee", "l_ankle", "r_ankle"]
assert JOINT_NAMES[sd.L_SHOULDER] == "l_shoulder" and JOINT_NAMES[sd.R_SHOULDER] == "r_shoulder"
assert JOINT_NAMES[sd.L_ELBOW] == "l_elbow" and JOINT_NAMES[sd.R_ELBOW] == "r_elbow"
assert JOINT_NAMES[sd.L_WRIST] == "l_wrist" and JOINT_NAMES[sd.R_WRIST] == "r_wrist"
assert JOINT_NAMES[sd.L_HIP] == "l_hip" and JOINT_NAMES[sd.R_HIP] == "r_hip"

BASE_FEATURE_SUFFIXES = ["path_length", "peak_speed", "peak_time_norm", "mean_speed"]
ACCEL_FEATURE_SUFFIXES = ["peak_acceleration", "mean_acceleration", "peak_acceleration_time_norm"]
FEATURE_SUFFIXES = BASE_FEATURE_SUFFIXES + ACCEL_FEATURE_SUFFIXES
ARM_ROLES = ["shoulder", "elbow", "wrist"]
JOINT_FEATURE_COLS = [f"{j}_{s}" for j in JOINT_NAMES for s in FEATURE_SUFFIXES]

_b, _a = butter(SMOOTH_ORDER, (SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ) / (FPS / 2))
MIN_FRAMES = 3 * max(len(_a), len(_b)) + 1        # filtfilt needs len > padlen; smoothing runs at full rate
CUTOFF = SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ
if not (0 < CUTOFF < FPS / 2):
    sys.exit(f"ERROR: smoothing cutoff {CUTOFF} Hz must be in (0, {FPS/2}) Hz at {FPS} FPS.")
MIN_FRAMES_AFTER_STEP = 4                          # 3 velocities -> 2 accelerations


def _status_tag(msg: str) -> str:
    return ("bad_shape" if msg.startswith("unexpected shape") else
            "too_short" if msg.startswith("too_short") else
            "non_finite" if msg.startswith("non_finite") else "value_error")


def validate_keypoints(k: np.ndarray) -> None:
    if k.ndim != 3 or k.shape[1:] != (17, 3):
        raise ValueError(f"unexpected shape {k.shape}, expected (T, 17, 3)")
    if k.shape[0] < MIN_FRAMES:
        raise ValueError(f"too_short: {k.shape[0]} frames < {MIN_FRAMES} needed by filtfilt")
    if not np.isfinite(k).all():
        raise ValueError("non_finite values in keypoints")


def features_from_smoothed(x_full: np.ndarray, step: int) -> dict:
    """Features from a hip-relative, smoothed (full frame rate) sequence, keeping every step-th frame."""
    fps_eff = FPS / step
    x = x_full[::step]
    if x.shape[0] < MIN_FRAMES_AFTER_STEP:
        raise ValueError(f"too_short: only {x.shape[0]} frames after step={step}")
    out = {}
    for j, name in enumerate(JOINT_NAMES):
        velocity = np.diff(x[:, j, :], axis=0) * fps_eff
        speed = np.linalg.norm(velocity, axis=1)
        out[f"{name}_path_length"] = float(speed.sum() / fps_eff)
        out[f"{name}_peak_speed"] = float(speed.max())
        out[f"{name}_peak_time_norm"] = float(np.argmax(speed) / (len(speed) - 1))
        out[f"{name}_mean_speed"] = float(speed.mean())

        acceleration = np.diff(velocity, axis=0) * fps_eff
        acc_mag = np.linalg.norm(acceleration, axis=1)
        out[f"{name}_peak_acceleration"] = float(acc_mag.max())
        out[f"{name}_mean_acceleration"] = float(acc_mag.mean())
        out[f"{name}_peak_acceleration_time_norm"] = float(np.argmax(acc_mag) / (len(acc_mag) - 1))
    if not np.isfinite(list(out.values())).all():
        raise ValueError("non_finite feature values")
    return out


# ----------------------------------------------------------------------------
# Feature extraction for all steps (one pass over the keypoint files)
# ----------------------------------------------------------------------------
def _cache_paths(step: int):
    return (OUT_DIR / f"joint_features_cache_step{step}.csv",
            OUT_DIR / f"joint_features_cache_step{step}.json")


def _cache_key(step: int, rows: pd.DataFrame) -> dict:
    # Same key layout as rf_baseline_skip2.py, so its caches are reused.
    return dict(fps=FPS, frame_step=step, cutoff_hz=CUTOFF, order=SMOOTH_ORDER, n=len(rows),
                files_md5=hashlib.md5("\n".join(rows["filename"]).encode()).hexdigest(),
                feature_suffixes=FEATURE_SUFFIXES)


def load_or_compute_all_steps(rows: pd.DataFrame) -> dict:
    """Return {step: DataFrame(joint feature columns + 'pose_status')} indexed like rows."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats_by_step, missing = {}, []
    for step in STEPS:
        csv, meta = _cache_paths(step)
        if USE_FEATURE_CACHE and csv.exists() and meta.exists():
            try:
                if json.loads(meta.read_text()) == _cache_key(step, rows):
                    cached = pd.read_csv(csv, index_col=0)
                    if cached.index.equals(rows.index):
                        print(f"  step {step:>2}: loaded cache {csv.name}")
                        feats_by_step[step] = cached
                        continue
            except Exception:
                pass
            print(f"  step {step:>2}: cache present but stale; recomputing")
        missing.append(step)

    if missing:
        print(f"  computing features for steps {missing} (single pass over {len(rows)} files)...")
        kp_dir = DATA_DIR / "keypoints"
        records = {s: [] for s in missing}
        statuses = {s: [] for s in missing}
        t0 = time.time()
        for n, fname in enumerate(rows["filename"], 1):
            try:
                k = np.load(kp_dir / fname)
                validate_keypoints(k)
                x_full = sm.smooth(sm.hip_relative(k.astype(np.float64)),
                                   cutoff_hz=CUTOFF, order=SMOOTH_ORDER)
            except FileNotFoundError:
                for s in missing:
                    records[s].append({}); statuses[s].append("missing_file")
                continue
            except ValueError as e:
                tag = _status_tag(str(e))
                if tag in ("bad_shape", "value_error"):
                    print(f"  WARNING {fname}: {e}")
                for s in missing:
                    records[s].append({}); statuses[s].append(tag)
                continue
            except Exception as e:                                       # corrupt .npy etc.
                print(f"  WARNING {fname}: {type(e).__name__}: {e}")
                for s in missing:
                    records[s].append({}); statuses[s].append(f"load_error_{type(e).__name__}")
                continue
            for s in missing:
                try:
                    records[s].append(features_from_smoothed(x_full, s))
                    statuses[s].append("ok")
                except ValueError as e:
                    records[s].append({}); statuses[s].append(_status_tag(str(e)))
            if n % 1000 == 0:
                print(f"    processed {n}/{len(rows)} serves ({time.time() - t0:.0f}s)")
        for s in missing:
            feats = pd.DataFrame(records[s], index=rows.index).reindex(columns=JOINT_FEATURE_COLS)
            feats["pose_status"] = statuses[s]
            feats_by_step[s] = feats
            if USE_FEATURE_CACHE:
                csv, meta = _cache_paths(s)
                feats.to_csv(csv)
                meta.write_text(json.dumps(_cache_key(s, rows)))
    return feats_by_step


def build_representation_columns(df: pd.DataFrame):
    """Create arm_* (dominant side, 3 joints) and body_* (17 joints) feature columns."""
    new = {}
    is_left = (df["hand"] == "L").to_numpy()
    for role in ARM_ROLES:
        for s in FEATURE_SUFFIXES:
            new[f"arm_{role}_{s}"] = np.where(is_left, df[f"l_{role}_{s}"], df[f"r_{role}_{s}"])
    for j in JOINT_NAMES:
        for s in FEATURE_SUFFIXES:
            new[f"body_{j}_{s}"] = df[f"{j}_{s}"].to_numpy()
    new = pd.DataFrame(new, index=df.index)
    arm_cols = [c for c in new if c.startswith("arm_")]
    body_cols = [c for c in new if c.startswith("body_")]
    is_accel = lambda c: any(c.endswith("_" + s) for s in ACCEL_FEATURE_SUFFIXES)   # noqa: E731
    arm_base = [c for c in arm_cols if not is_accel(c)]
    body_base = [c for c in body_cols if not is_accel(c)]
    return new, arm_cols, body_cols, arm_base, body_base


# ============================================================================
# MODELS / METRICS
# ============================================================================
def make_pipeline(regressor, pose_cols: list) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), META_CAT),
         ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), META_NUM + pose_cols)],
        verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("reg", regressor)])


def metrics(y, p) -> dict:
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)))


def player_bootstrap_diff(players, y, p_new, p_ref, n_boot, seed=0):
    """Paired bootstrap over PLAYERS. Returns arrays of (new - ref) MAE and RMSE."""
    codes, uniq = pd.factorize(players)
    P = len(uniq)
    cnt = np.bincount(codes, minlength=P).astype(float)

    def sums(p):
        e = y - p
        return (np.bincount(codes, weights=np.abs(e), minlength=P),
                np.bincount(codes, weights=e ** 2, minlength=P))

    (a_abs, a_sq), (r_abs, r_sq) = sums(p_new), sums(p_ref)
    draws = np.random.default_rng(seed).integers(0, P, size=(n_boot, P))
    c = cnt[draws].sum(1)
    mae = a_abs[draws].sum(1) / c - r_abs[draws].sum(1) / c
    rmse = np.sqrt(a_sq[draws].sum(1) / c) - np.sqrt(r_sq[draws].sum(1) / c)
    return mae, rmse


def make_group_folds(players: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each serve to a fold so that all serves of a player share a fold (players shuffled)."""
    uniq = np.unique(players)
    if len(uniq) < n_folds:
        sys.exit(f"ERROR: only {len(uniq)} players for {n_folds} folds.")
    perm = np.random.default_rng(seed).permutation(uniq)
    fold_of = {p: i % n_folds for i, p in enumerate(perm)}
    return np.array([fold_of[p] for p in players])


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


# ============================================================================
# PLOT
# ============================================================================
def make_plot(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: skipping the plot. Install with: py -m pip install matplotlib)")
        return

    steps = STEPS
    xpos = np.arange(len(steps))
    xlabels = [f"{FPS / s:g}\n(step {s})" for s in steps]
    colors = {M_ARM: "#1f77b4", M_BODY: "#d62728", M_ARM_ACC: "#17becf", M_BODY_ACC: "#ff7f0e"}
    offsets = np.linspace(-0.18, 0.18, len(POSE_MODELS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # (a) MAE vs effective FPS
    ax = axes[0]
    for off, model in zip(offsets, POSE_MODELS):
        d = summary[summary["Model"] == model].set_index("step").loc[steps]
        ax.errorbar(xpos + off, d["MAE_mean"], yerr=d["MAE_std"].fillna(0.0), marker="o", ms=4,
                    capsize=3, lw=1.3, color=colors[model], label=model.replace("RF: metadata + ", "RF + "))
    for model, ls, col in ((M_LIN, "--", "k"), (M_RFMETA, ":", "gray")):
        d = summary[summary["Model"] == model].iloc[0]
        ax.axhline(d["MAE_mean"], ls=ls, color=col, lw=1.2, label=model)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Effective frame rate (FPS)")
    ax.set_ylabel("MAE (km/h), grouped CV, mean ± SD over repeats")
    ax.set_title("Serve-speed MAE vs effective frame rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) paired change vs full frame rate, with 95% player-bootstrap CI
    ax = axes[1]
    if N_BOOTSTRAP > 0 and len(paired):
        for off, model in zip(offsets, POSE_MODELS):
            d = paired[(paired["model"] == model) & (paired["ref"] == "step1")].set_index("step")
            xs, ys, lo, hi = [], [], [], []
            for i, s in enumerate(steps):
                xs.append(i + off)
                if s == 1:
                    ys.append(0.0); lo.append(0.0); hi.append(0.0)
                else:
                    ys.append(d.loc[s, "dMAE_mean"])
                    lo.append(d.loc[s, "dMAE_mean"] - d.loc[s, "dMAE_lo95"])
                    hi.append(d.loc[s, "dMAE_hi95"] - d.loc[s, "dMAE_mean"])
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=4, capsize=3, lw=1.3, color=colors[model])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Change in MAE vs step 1 (km/h), 95% player-bootstrap CI")
        ax.set_title("Paired change vs full frame rate (positive = worse)")
    else:
        ax.text(0.5, 0.5, "bootstrap disabled", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel("Effective frame rate (FPS)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "sweep_mae_vs_fps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR}")
    print(f"Steps: {STEPS}  ->  effective FPS: {[round(FPS / s, 2) for s in STEPS]}")
    print(f"CV: {N_FOLDS} grouped folds x {N_REPEATS} repeats; RF n_estimators={RF_PARAMS['n_estimators']}")
    lowest = FPS / max(STEPS)
    if lowest / 2 < CUTOFF:
        print(f"NOTE: at {lowest:g} FPS the Nyquist frequency ({lowest / 2:g} Hz) is below the "
              f"{CUTOFF} Hz smoothing cutoff. Smoothing is done BEFORE frame skipping here, so this "
              f"tests robustness of the features, not a real low-FPS camera.")

    # ---- 1. metadata + filtering ------------------------------------------
    try:
        meta = sd.load_metadata(DATA_DIR)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nCheck DATA_DIR_OVERRIDE / that keypoints/*.npy exist.")
    except Exception as e:
        sys.exit(f"ERROR reading metadata from {DATA_DIR}: {type(e).__name__}: {e}")
    needed = ["filename", "server"] + META_CAT + META_NUM + [TARGET]
    absent = [c for c in needed if c not in meta.columns]
    if absent:
        sys.exit(f"ERROR: metadata missing required columns: {absent}")

    df = meta[needed].copy()
    for col in ["serve_number", "height_cm", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=[TARGET, "server"])
    df = df[df["serve_number"].isin([1, 2]) & df["gender"].isin(["M", "F"])].copy()
    print(f"\nMetadata rows: {n0} -> {len(df)} after filtering; players: {df['server'].nunique()}")
    df["hand"] = df["server"].map(hand)
    print(f"Handedness source: {HAND_SOURCE} (left-handed serves: {int((df['hand'] == 'L').sum())})")

    # ---- 2. pose features for every step ----------------------------------
    print("\nPose features:")
    feats_by_step = load_or_compute_all_steps(df)
    ok_all = pd.Series(True, index=df.index)
    for s in STEPS:
        ok_s = feats_by_step[s]["pose_status"].eq("ok")
        n_bad = int((~ok_s).sum())
        if n_bad:
            print(f"  step {s:>2}: {n_bad} serves unusable "
                  f"({feats_by_step[s].loc[~ok_s, 'pose_status'].value_counts().to_dict()})")
        ok_all &= ok_s
    n_drop = int((~ok_all).sum())
    print(f"\nCOMMON SAMPLE: {int(ok_all.sum())} serves usable at ALL steps ({n_drop} removed from every step "
          f"so that all steps are compared on identical serves)")
    if n_drop / len(df) > 0.02:
        print("  WARNING: >2% of serves removed; investigate before trusting results.")
    if ok_all.sum() == 0:
        sys.exit("ERROR: no serves usable at all steps.")

    dfc = df[ok_all].copy()
    players = dfc["server"].to_numpy()
    y = dfc[TARGET].to_numpy()
    n_players = len(np.unique(players))
    print(f"  {len(dfc)} serves / {n_players} players")

    # Per-step design matrices (metadata + arm/body features)
    X_by_step = {}
    for s in STEPS:
        joined = dfc.join(feats_by_step[s].loc[dfc.index])
        rep, arm_cols, body_cols, arm_base, body_base = build_representation_columns(joined)
        assert not rep.isna().any().any(), f"NaN in pose features at step {s}"
        X_by_step[s] = pd.concat([dfc[META_CAT + META_NUM], rep], axis=1)
    pose_cols_of = {M_ARM: arm_base, M_BODY: body_base, M_ARM_ACC: arm_cols, M_BODY_ACC: body_cols}
    print(f"  features: arm = {len(arm_cols)} ({len(arm_base)} without acceleration), "
          f"body = {len(body_cols)} ({len(body_base)} without acceleration)")

    # ---- 3. grouped, repeated cross-validation ----------------------------
    n = len(dfc)
    oof = {}                                   # (model, step or None, repeat) -> out-of-fold predictions
    t0 = time.time()
    X_meta = X_by_step[STEPS[0]]
    for rep in range(N_REPEATS):
        fold_id = make_group_folds(players, N_FOLDS, seed=1000 + rep)
        rf_seed = RF_PARAMS["random_state"] + rep
        for model, step in [(M_LIN, None), (M_RFMETA, None)] + [(m, s) for s in STEPS for m in POSE_MODELS]:
            oof[(model, step, rep)] = np.full(n, np.nan)
        for fold in range(N_FOLDS):
            tr = np.where(fold_id != fold)[0]
            te = np.where(fold_id == fold)[0]
            assert set(players[tr]).isdisjoint(set(players[te]))

            def fit_predict(reg, X, pose_cols):
                cols = META_CAT + META_NUM + pose_cols
                pipe = make_pipeline(reg, pose_cols)
                pipe.fit(X.iloc[tr][cols], y[tr])
                return pipe.predict(X.iloc[te][cols])

            oof[(M_LIN, None, rep)][te] = fit_predict(LinearRegression(), X_meta, [])
            rf_params = {**RF_PARAMS, "random_state": rf_seed}
            oof[(M_RFMETA, None, rep)][te] = fit_predict(RandomForestRegressor(**rf_params), X_meta, [])
            for s in STEPS:
                for m in POSE_MODELS:
                    oof[(m, s, rep)][te] = fit_predict(RandomForestRegressor(**rf_params),
                                                       X_by_step[s], pose_cols_of[m])
            print(f"  repeat {rep + 1}/{N_REPEATS}, fold {fold + 1}/{N_FOLDS} done "
                  f"({time.time() - t0:.0f}s elapsed)")
        assert not any(np.isnan(oof[k]).any() for k in oof if k[2] == rep), "some serves were never predicted"

    # ---- 4. summary metrics (mean/std over repeats) ------------------------
    per_rep = {}                               # (model, step) -> list of metric dicts (one per repeat)
    for (model, step, rep), p in oof.items():
        per_rep.setdefault((model, step), []).append(metrics(y, p))

    rows = []
    for (model, step), lst in per_rep.items():
        row = {"Model": model, "step": step if step is not None else np.nan,
               "effective_fps": FPS / step if step is not None else np.nan,
               "n_serves": n, "n_players": n_players, "n_repeats": N_REPEATS}
        for m in ("MAE", "RMSE", "R2"):
            row[f"{m}_mean"], row[f"{m}_std"] = mean_std([d[m] for d in lst])
        if model not in (M_LIN, M_RFMETA):
            ref = per_rep[(M_RFMETA, None)]
            row["dMAE_vs_rf_meta_mean"], row["dMAE_vs_rf_meta_std"] = mean_std(
                [a["MAE"] - b["MAE"] for a, b in zip(lst, ref)])
            ref1 = per_rep[(model, 1)]
            row["dMAE_vs_step1_mean"], row["dMAE_vs_step1_std"] = mean_std(
                [a["MAE"] - b["MAE"] for a, b in zip(lst, ref1)])
        rows.append(row)
    summary = pd.DataFrame(rows)
    order = {M_LIN: 0, M_RFMETA: 1, M_ARM: 2, M_BODY: 3, M_ARM_ACC: 4, M_BODY_ACC: 5}
    summary["_o"] = summary["Model"].map(order)
    summary = summary.sort_values(["_o", "step"]).drop(columns="_o").reset_index(drop=True)
    summary.to_csv(OUT_DIR / "sweep_summary.csv", index=False)

    fmt = lambda m, s: f"{m:.3f}±{s:.3f}" if np.isfinite(s) else f"{m:.3f}"       # noqa: E731
    print("\n" + "=" * 100)
    print(f"MAE (km/h), grouped CV by player, mean ± SD over {N_REPEATS} repeats  "
          f"(columns: step / effective FPS)")
    print("=" * 100)
    table = {}
    for m in (M_LIN, M_RFMETA):
        r = summary[summary["Model"] == m].iloc[0]
        table[m] = {f"{s} / {FPS / s:g}": fmt(r["MAE_mean"], r["MAE_std"]) for s in STEPS}
    for m in POSE_MODELS:
        d = summary[summary["Model"] == m].set_index("step")
        table[m] = {f"{s} / {FPS / s:g}": fmt(d.loc[s, "MAE_mean"], d.loc[s, "MAE_std"]) for s in STEPS}
    print(pd.DataFrame(table).T.to_string())

    print("\nR2, same layout")
    table = {}
    for m in (M_LIN, M_RFMETA):
        r = summary[summary["Model"] == m].iloc[0]
        table[m] = {f"{s} / {FPS / s:g}": fmt(r["R2_mean"], r["R2_std"]) for s in STEPS}
    for m in POSE_MODELS:
        d = summary[summary["Model"] == m].set_index("step")
        table[m] = {f"{s} / {FPS / s:g}": fmt(d.loc[s, "R2_mean"], d.loc[s, "R2_std"]) for s in STEPS}
    print(pd.DataFrame(table).T.to_string())

    print("\nPose gain: MAE(model) - MAE(RF: metadata only), mean ± SD over repeats (negative = pose helps)")
    table = {}
    for m in POSE_MODELS:
        d = summary[summary["Model"] == m].set_index("step")
        table[m] = {f"{s} / {FPS / s:g}": fmt(d.loc[s, "dMAE_vs_rf_meta_mean"], d.loc[s, "dMAE_vs_rf_meta_std"])
                    for s in STEPS}
    print(pd.DataFrame(table).T.to_string())

    print("\nChange vs step 1: MAE(step s) - MAE(step 1), mean ± SD over repeats (positive = worse)")
    table = {}
    for m in POSE_MODELS:
        d = summary[summary["Model"] == m].set_index("step")
        table[m] = {f"{s} / {FPS / s:g}": fmt(d.loc[s, "dMAE_vs_step1_mean"], d.loc[s, "dMAE_vs_step1_std"])
                    for s in STEPS if s != 1}
    print(pd.DataFrame(table).T.to_string())

    # ---- 5. paired player-level bootstrap ---------------------------------
    paired = pd.DataFrame()
    if N_BOOTSTRAP > 0:
        print(f"\nPLAYER-LEVEL BOOTSTRAP ({N_BOOTSTRAP} resamples of the {n_players} players; "
              f"uses out-of-fold predictions averaged over repeats; negative = first model better)")
        avg = {}
        for (model, step, rep), p in oof.items():
            avg.setdefault((model, step), []).append(p)
        avg = {k: np.mean(v, axis=0) for k, v in avg.items()}

        brows = []

        def add(model, step, ref_name, p_new, p_ref):
            dmae, drmse = player_bootstrap_diff(players, y, p_new, p_ref, N_BOOTSTRAP)
            brows.append({"model": model, "step": step, "ref": ref_name,
                          "dMAE_mean": dmae.mean(), "dMAE_lo95": np.percentile(dmae, 2.5),
                          "dMAE_hi95": np.percentile(dmae, 97.5), "share_MAE_better": float((dmae < 0).mean()),
                          "dRMSE_mean": drmse.mean(), "dRMSE_lo95": np.percentile(drmse, 2.5),
                          "dRMSE_hi95": np.percentile(drmse, 97.5)})

        for m in POSE_MODELS:
            for s in STEPS:
                add(m, s, M_RFMETA, avg[(m, s)], avg[(M_RFMETA, None)])
                add(m, s, M_LIN, avg[(m, s)], avg[(M_LIN, None)])
                if s != 1:
                    add(m, s, "step1", avg[(m, s)], avg[(m, 1)])
        paired = pd.DataFrame(brows)
        paired.to_csv(OUT_DIR / "sweep_paired_bootstrap.csv", index=False)
        for ref_name, title in (("step1", "step s vs step 1 (same model)"),
                                (M_RFMETA, "pose model vs RF: metadata only (same step)")):
            sub = paired[paired["ref"] == ref_name]
            print(f"\n  {title}")
            print(sub[["model", "step", "dMAE_mean", "dMAE_lo95", "dMAE_hi95", "share_MAE_better",
                       "dRMSE_mean"]].round(3).to_string(index=False))

    # ---- 6. plot + save ----------------------------------------------------
    print()
    make_plot(summary, paired)
    print(f"\nSaved to {OUT_DIR}: sweep_summary.csv"
          + (", sweep_paired_bootstrap.csv" if N_BOOTSTRAP > 0 else "") + ", sweep_mae_vs_fps.png")
    print("Reading guide: differences between steps smaller than the SD over repeats (or whose bootstrap CI "
          "contains 0) should not be interpreted.")


if __name__ == "__main__":
    main()
