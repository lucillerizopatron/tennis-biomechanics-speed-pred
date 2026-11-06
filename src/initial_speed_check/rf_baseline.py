"""
Random Forest pipeline: does a 3-joint serving-arm representation predict serve
speed as well as a 17-joint full-body representation? (2024 US Open serve data)

Models (identical player-level train/test split, identical serves for all):
  1. Linear regression, metadata only (gender, serve_number, height_cm)  [original baseline]
  2. Random Forest, metadata only                                         [fair RF control]
  3. Random Forest, metadata + dominant-arm features (shoulder/elbow/wrist)
  4. Random Forest, metadata + full-skeleton features (all 17 joints)
  (+ a train-mean predictor for reference)

Save next to serve_data.py and the smoothing / handedness modules, then run:
    python rf_baseline.py

Requirements: numpy, pandas, scipy, scikit-learn, pyarrow (or fastparquet).

All joint-motion features are in COORDINATE UNITS (per second), not metres.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import warnings
from pathlib import Path

# ============================================================================
# CONFIGURATION  (edit here)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR.parent                    # folder containing serve_data.py etc.
DATA_DIR_OVERRIDE = None                     # e.g. Path("/path/to/tennis_serve_data"); None -> serve_data.DATA
OUT_DIR = SCRIPT_DIR / "pose_rf_outputs"     # CSVs and cache are written here

SMOOTHING_MODULE = "smoothing"         # change if your smoothing file is named differently
HANDEDNESS_MODULE = "serve_handedness"       # change if your handedness file is named differently
SERVE_DATA_MODULE = "serve_data"

USE_FEATURE_CACHE = True
SMOOTH_CUTOFF_HZ = None                      # None -> smoothing module's DEFAULT_CUTOFF_HZ (8 Hz)
SMOOTH_ORDER = 2

SPLIT_SEED = 42                              # must stay 42 to reproduce the existing split
TEST_SIZE = 0.2
N_BOOTSTRAP = 2000                           # player-level bootstrap resamples (0 disables)
TOP_N_IMPORTANCES = 15

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=5,
                 random_state=42, n_jobs=-1)

# Reference results from the existing notebook (for sanity checking only)
REF_COUNTS = dict(train_players=87, test_players=22, train_serves=4729, test_serves=1237)
REF_LINEAR = dict(MAE=10.597, RMSE=13.291, R2=0.649)

# Fallback used ONLY if the handedness module cannot be imported. NOT FULLY VERIFIED.
FALLBACK_LEFT_HANDED = {"Denis Shapovalov", "Beatriz Haddad Maia"}

META_CAT = ["gender", "serve_number"]
META_NUM = ["height_cm"]
TARGET = "speed_kmh"

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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    sys.exit(f"ERROR: cannot import smoothing module '{SMOOTHING_MODULE}' from {HELPER_DIR} ({e}). "
             f"Set SMOOTHING_MODULE at the top of this script to the correct filename (without .py).")
for _fn in ("hip_relative", "smooth", "speed"):
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
# Guard against indexing mistakes: names must agree with serve_data's constants.
assert JOINT_NAMES[sd.L_SHOULDER] == "l_shoulder" and JOINT_NAMES[sd.R_SHOULDER] == "r_shoulder"
assert JOINT_NAMES[sd.L_ELBOW] == "l_elbow" and JOINT_NAMES[sd.R_ELBOW] == "r_elbow"
assert JOINT_NAMES[sd.L_WRIST] == "l_wrist" and JOINT_NAMES[sd.R_WRIST] == "r_wrist"
assert JOINT_NAMES[sd.L_HIP] == "l_hip" and JOINT_NAMES[sd.R_HIP] == "r_hip"
assert (sd.R_SHOULDER, sd.R_ELBOW, sd.R_WRIST) == (6, 8, 10)
assert (sd.L_SHOULDER, sd.L_ELBOW, sd.L_WRIST) == (5, 7, 9)

BASE_FEATURE_SUFFIXES = [
    "path_length", "peak_speed", "peak_time_norm", "mean_speed"
]

ACCEL_FEATURE_SUFFIXES = [
    "peak_acceleration", "mean_acceleration", "peak_acceleration_time_norm"
]

FEATURE_SUFFIXES = BASE_FEATURE_SUFFIXES + ACCEL_FEATURE_SUFFIXES
ARM_ROLES = ["shoulder", "elbow", "wrist"]

_b, _a = butter(SMOOTH_ORDER, (SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ) / (FPS / 2))
# filtfilt needs len > padlen (default 3 * max(len(a), len(b)))
MIN_FRAMES = 3 * max(len(_a), len(_b)) + 1
CUTOFF = SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ
if not (0 < CUTOFF < FPS / 2):
    sys.exit(f"ERROR: smoothing cutoff {CUTOFF} Hz must be in (0, {FPS/2}) Hz at {FPS} FPS.")


def joint_features(k: np.ndarray) -> dict:
    """Extract position-derived speed and acceleration features for all joints.

    Coordinates are hip-relative and smoothed first.
    Speed is measured in coordinate units/s.
    Acceleration is measured in coordinate units/s^2.
    """
    if k.ndim != 3 or k.shape[1:] != (17, 3):
        raise ValueError(f"unexpected shape {k.shape}, expected (T, 17, 3)")
    if k.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"too_short: {k.shape[0]} frames < {MIN_FRAMES} needed by filtfilt"
        )
    if not np.isfinite(k).all():
        raise ValueError("non_finite values in keypoints")

    x = sm.smooth(
        sm.hip_relative(k.astype(np.float64)),
        cutoff_hz=CUTOFF,
        order=SMOOTH_ORDER,
    )

    out = {}

    for j, name in enumerate(JOINT_NAMES):
        # Frame-to-frame velocity vectors, in coordinate units/s.
        velocity = np.diff(x[:, j, :], axis=0) * FPS
        speed = np.linalg.norm(velocity, axis=1)

        out[f"{name}_path_length"] = float(speed.sum() / FPS)
        out[f"{name}_peak_speed"] = float(speed.max())
        out[f"{name}_peak_time_norm"] = float(
            np.argmax(speed) / (len(speed) - 1)
        )
        out[f"{name}_mean_speed"] = float(speed.mean())

        # Frame-to-frame acceleration vectors, in coordinate units/s^2.
        acceleration = np.diff(velocity, axis=0) * FPS
        acceleration_magnitude = np.linalg.norm(acceleration, axis=1)

        out[f"{name}_peak_acceleration"] = float(
            acceleration_magnitude.max()
        )
        out[f"{name}_mean_acceleration"] = float(
            acceleration_magnitude.mean()
        )
        out[f"{name}_peak_acceleration_time_norm"] = float(
            np.argmax(acceleration_magnitude)
            / (len(acceleration_magnitude) - 1)
        )

    if not np.isfinite(list(out.values())).all():
        raise ValueError("non_finite feature values")

    return out


JOINT_FEATURE_COLS = [f"{j}_{s}" for j in JOINT_NAMES for s in FEATURE_SUFFIXES]


def compute_or_load_joint_features(rows: pd.DataFrame) -> pd.DataFrame:
    """All-17-joint features for each row (index preserved). Adds 'pose_status' ('ok' or reason)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_csv = OUT_DIR / "joint_features_cache.csv"
    cache_meta = OUT_DIR / "joint_features_cache.json"
    key = dict(fps=FPS, cutoff_hz=CUTOFF, order=SMOOTH_ORDER, n=len(rows),
               files_md5=hashlib.md5("\n".join(rows["filename"]).encode()).hexdigest(),
               feature_suffixes=FEATURE_SUFFIXES)
    if USE_FEATURE_CACHE and cache_csv.exists() and cache_meta.exists():
        if json.loads(cache_meta.read_text()) == key:
            print(f"Loading cached joint features from {cache_csv}")
            cached = pd.read_csv(cache_csv, index_col=0)
            if cached.index.equals(rows.index):
                return cached
        print("Cache present but stale; recomputing.")

    kp_dir = DATA_DIR / "keypoints"
    records, statuses = [], []
    for n, (idx, fname) in enumerate(zip(rows.index, rows["filename"]), 1):
        try:
            k = np.load(kp_dir / fname)
            records.append(joint_features(k))
            statuses.append("ok")
        except FileNotFoundError:
            records.append({}); statuses.append("missing_file")
        except ValueError as e:
            msg = str(e)
            tag = ("bad_shape" if msg.startswith("unexpected shape") else
                   "too_short" if msg.startswith("too_short") else
                   "non_finite" if msg.startswith("non_finite") else "value_error")
            records.append({}); statuses.append(tag)
            if tag in ("bad_shape", "value_error"):
                print(f"  WARNING {fname}: {msg}")
        except Exception as e:                                           # corrupt .npy etc.
            records.append({}); statuses.append(f"load_error_{type(e).__name__}")
            print(f"  WARNING {fname}: {type(e).__name__}: {e}")
        if n % 1000 == 0:
            print(f"  processed {n}/{len(rows)} serves")
    feats = pd.DataFrame(records, index=rows.index).reindex(columns=JOINT_FEATURE_COLS)
    feats["pose_status"] = statuses
    if USE_FEATURE_CACHE:
        feats.to_csv(cache_csv)
        cache_meta.write_text(json.dumps(key))
    return feats


def build_representation_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
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

    arm_base_cols = [
    c for c in arm_cols
    if not any(c.endswith("_" + s) for s in ACCEL_FEATURE_SUFFIXES)
    ]
    body_base_cols = [
        c for c in body_cols
        if not any(c.endswith("_" + s) for s in ACCEL_FEATURE_SUFFIXES)
    ]

    return (
        new,
        arm_cols,
        body_cols,
        arm_base_cols,
        body_base_cols,
    )


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
    return dict(MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=r2_score(y, p))


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


# ============================================================================
# MAIN
# ============================================================================
def main():
    pd.set_option("display.width", 200)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. metadata -------------------------------------------------------
    print(f"Data directory: {DATA_DIR}")
    try:
        meta = sd.load_metadata(DATA_DIR)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}\nCheck DATA_DIR_OVERRIDE / that tennis_serve_data/keypoints/*.npy exist.")
    except Exception as e:
        sys.exit(f"ERROR reading metadata from {DATA_DIR}: {type(e).__name__}: {e}")
    needed = ["filename", "server"] + META_CAT + META_NUM + [TARGET]
    absent = [c for c in needed if c not in meta.columns]
    if absent:
        sys.exit(f"ERROR: metadata missing required columns: {absent}")
    print(f"Metadata rows: {len(meta)}")

    # ---- 2. same filtering as the baseline notebook -----------------------
    df = meta[needed].copy()
    for col in ["serve_number", "height_cm", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=[TARGET, "server"])
    n1 = len(df)
    df = df[df["serve_number"].isin([1, 2]) & df["gender"].isin(["M", "F"])].copy()
    print(f"\nFiltering (same as baseline notebook): {n0} -> {n1} (valid target/server) -> {len(df)} "
          f"(serve_number in {{1,2}}, gender in {{M,F}})")
    print(f"Serves: {len(df)}, players: {df['server'].nunique()}, "
          f"serves with missing height: {int(df['height_cm'].isna().sum())}")

    # ---- 3. handedness -----------------------------------------------------
    df["hand"] = df["server"].map(hand)
    present_left = sorted(LEFT_HANDED & set(df["server"]))
    missing_left = sorted(LEFT_HANDED - set(df["server"]))
    print(f"\nHandedness source: {HAND_SOURCE}  (NOT FULLY VERIFIED; unlisted players default to 'R')")
    print(f"  Left-handed players in data: {present_left} "
          f"({int((df['hand'] == 'L').sum())} serves)")
    if missing_left:
        print(f"  WARNING: listed left-handers NOT found among servers (name mismatch?): {missing_left}")

    # ---- 4. split (computed BEFORE pose exclusions, to match the existing split)
    players = df.groupby("server", as_index=False).agg(height_cm=("height_cm", "median"))
    players["height_for_split"] = players["height_cm"].fillna(players["height_cm"].median())
    players["height_group"] = pd.qcut(players["height_for_split"], q=4, labels=False, duplicates="drop")
    train_players, test_players = train_test_split(
        players["server"], test_size=TEST_SIZE, random_state=SPLIT_SEED,
        stratify=players["height_group"])
    train_set, test_set = set(train_players), set(test_players)
    assert train_set.isdisjoint(test_set), "Train and test players overlap!"
    assert train_set | test_set == set(df["server"]), "Some players are in neither split!"
    df["split"] = np.where(df["server"].isin(test_set), "test", "train")

    got = dict(train_players=len(train_set), test_players=len(test_set),
               train_serves=int((df["split"] == "train").sum()),
               test_serves=int((df["split"] == "test").sum()))
    print("\nPLAYER-LEVEL SPLIT (before pose exclusions)")
    print(f"  {got}")
    print(f"  Reference from existing notebook: {REF_COUNTS}")
    if got != REF_COUNTS:
        print("  WARNING: split does not match the reference counts; comparisons with the "
              "reference baseline metrics are NOT like-for-like.")
    print("  Train/test player sets are disjoint: True")

    # ---- 5. pose features --------------------------------------------------
    print("\nExtracting pose features (hip-relative -> smooth -> speed)...")
    print(f"  FPS={FPS}, cutoff={CUTOFF} Hz, order={SMOOTH_ORDER}, min frames={MIN_FRAMES}")
    jf = compute_or_load_joint_features(df)
    df = df.join(jf)
    df["pose_ok"] = df["pose_status"].eq("ok")
    n_bad = int((~df["pose_ok"]).sum())
    print(f"\nPOSE EXTRACTION: {int(df['pose_ok'].sum())} usable, {n_bad} excluded")
    if n_bad:
        print(df.loc[~df["pose_ok"], "pose_status"].value_counts().to_string())
        print(df.loc[~df["pose_ok"]].groupby("split").size().rename("excluded_by_split").to_string())
        if n_bad / len(df) > 0.02:
            print("  WARNING: >2% of serves excluded; investigate before trusting results.")
    if df["pose_ok"].sum() == 0:
        sys.exit("ERROR: no usable pose sequences. Check array shapes / file contents.")

    (
        rep,
        arm_cols,
        body_cols,
        arm_base_cols,
        body_base_cols,
    ) = build_representation_columns(df)    
    df = pd.concat([df, rep], axis=1)
    pose_df = df[df["pose_ok"]]                         # SAME serves for every model
    train = pose_df[pose_df["split"] == "train"]
    test = pose_df[pose_df["split"] == "test"]
    assert set(train["server"]).isdisjoint(set(test["server"]))
    assert not train[arm_cols + body_cols].isna().any().any() and not test[arm_cols + body_cols].isna().any().any()
    print(f"\nFINAL EVALUATION SAMPLE (used by ALL models)")
    print(f"  train: {len(train)} serves / {train['server'].nunique()} players")
    print(f"  test : {len(test)} serves / {test['server'].nunique()} players")
    print(f"  features: arm = {len(arm_cols)}, full body = {len(body_cols)}")
    print("  NOTE: after hip-centring, l_hip and r_hip are mirror images (identical speeds), "
          "so the full-body set contains some redundant features; mean_speed ~ path_length "
          "up to a factor 1/(T-1).")

    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    # ---- 6. models ---------------------------------------------------------
    specs = {
    "Mean predictor (train mean)": (
        DummyRegressor(strategy="mean"), []
    ),
    "Linear regression: metadata": (
        LinearRegression(), []
    ),
    "RF: metadata only": (
        RandomForestRegressor(**RF_PARAMS), []
    ),

    # Original pose feature sets
    "RF: metadata + arm (existing features)": (
        RandomForestRegressor(**RF_PARAMS), arm_base_cols
    ),
    "RF: metadata + body (existing features)": (
        RandomForestRegressor(**RF_PARAMS), body_base_cols
    ),

    # Expanded feature sets, including acceleration
    "RF: metadata + arm + acceleration": (
        RandomForestRegressor(**RF_PARAMS), arm_cols
    ),
    "RF: metadata + body + acceleration": (
        RandomForestRegressor(**RF_PARAMS), body_cols
    ),
}
    fitted, preds, rows = {}, {}, []
    for name, (reg, pcols) in specs.items():
        pipe = make_pipeline(reg, pcols)
        cols = META_CAT + META_NUM + pcols
        pipe.fit(train[cols], y_train)                  # preprocessing fitted on train only
        p = pipe.predict(test[cols])
        fitted[name], preds[name] = pipe, p
        rows.append({"Model": name, "n_features_pose": len(pcols), **metrics(y_test, p)})
    res = pd.DataFrame(rows)

    lin, rfm = "Linear regression: metadata", "RF: metadata only"
    for ref, tag in ((lin, "vs_linear"), (rfm, "vs_rf_meta")):
        r = res.set_index("Model").loc[ref]
        for m in ("MAE", "RMSE", "R2"):
            res[f"d{m}_{tag}"] = res[m] - r[m]
    res["n_train"], res["n_test"] = len(train), len(test)

    print("\n" + "=" * 78 + "\nRESULTS (test players only; MAE/RMSE in km/h)\n" + "=" * 78)
    print(res[["Model", "n_features_pose", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))
    print("\nChange vs ORIGINAL LINEAR BASELINE (MAE/RMSE: negative = better; R2: positive = better)")
    print(res[["Model", "dMAE_vs_linear", "dRMSE_vs_linear", "dR2_vs_linear"]].round(3).to_string(index=False))
    print("\nChange vs RF METADATA-ONLY CONTROL (isolates the contribution of pose features)")
    print(res[["Model", "dMAE_vs_rf_meta", "dRMSE_vs_rf_meta", "dR2_vs_rf_meta"]].round(3).to_string(index=False))

    # sanity check: original baseline on ALL filtered serves with the same split
    full_tr, full_te = df[df["split"] == "train"], df[df["split"] == "test"]
    chk = make_pipeline(LinearRegression(), [])
    chk.fit(full_tr[META_CAT + META_NUM], full_tr[TARGET])
    cm = metrics(full_te[TARGET].to_numpy(), chk.predict(full_te[META_CAT + META_NUM]))
    print("\nSANITY CHECK: linear baseline on ALL filtered serves (no pose exclusions):")
    print("  this run : " + ", ".join(f"{k}={v:.3f}" for k, v in cm.items()))
    print("  reference: " + ", ".join(f"{k}={v:.3f}" for k, v in REF_LINEAR.items()))
    ok = all(abs(cm[k] - REF_LINEAR[k]) < 0.01 for k in REF_LINEAR)
    print("  -> reproduces existing baseline" if ok else
          "  -> DOES NOT match the reference; check filtering/split before trusting comparisons")

    # ---- 7. uncertainty: bootstrap over test players -----------------------
    if N_BOOTSTRAP > 0:
        print(f"\nPLAYER-LEVEL BOOTSTRAP ({N_BOOTSTRAP} resamples of the {test['server'].nunique()} "
              f"test players; new minus reference, negative = new model better)")
        print("  This reflects test-player sampling noise only, for ONE fixed split and ONE RF seed.")
        boot_rows = []
        for new in specs:
            if not new.startswith("RF: metadata +"):
                continue
            for ref in (rfm, lin):
                dmae, drmse = player_bootstrap_diff(test["server"].to_numpy(), y_test,
                                                    preds[new], preds[ref], N_BOOTSTRAP)
                boot_rows.append({"new": new, "ref": ref,
                                  "dMAE_mean": dmae.mean(), "dMAE_lo95": np.percentile(dmae, 2.5),
                                  "dMAE_hi95": np.percentile(dmae, 97.5),
                                  "share_MAE_better": float((dmae < 0).mean()),
                                  "dRMSE_mean": drmse.mean(), "dRMSE_lo95": np.percentile(drmse, 2.5),
                                  "dRMSE_hi95": np.percentile(drmse, 97.5)})
        boot = pd.DataFrame(boot_rows)
        print(boot.round(3).to_string(index=False))
        boot.to_csv(OUT_DIR / "bootstrap_differences.csv", index=False)

    # ---- 8. feature importances -------------------------------------------
    imp_all = []
    for name in (
    "RF: metadata + arm (existing features)",
    "RF: metadata + arm + acceleration",
    "RF: metadata + body (existing features)",
    "RF: metadata + body + acceleration",
    ):
        pipe = fitted[name]
        names = pipe.named_steps["pre"].get_feature_names_out()
        imp = pd.DataFrame({"model": name, "feature": names,
                            "importance": pipe.named_steps["reg"].feature_importances_})
        imp["group"] = np.where(imp["feature"].str.startswith(("arm_", "body_")), "pose", "metadata")
        imp = imp.sort_values("importance", ascending=False)
        imp_all.append(imp)
        print(f"\nFEATURE IMPORTANCES (impurity-based): {name}")
        print("  total importance by group: " +
              ", ".join(f"{g}={v:.3f}" for g, v in imp.groupby("group")["importance"].sum().items()))
        print(imp.head(TOP_N_IMPORTANCES)[["group", "feature", "importance"]]
              .round(4).to_string(index=False))
    print("\nCaveat: impurity importances are biased toward high-cardinality/continuous features and "
          "split credit among correlated ones; they show what the forest used, not causal effects.")

    # ---- 9. save -----------------------------------------------------------
    res.to_csv(OUT_DIR / "results.csv", index=False)
    pd.concat(imp_all).to_csv(OUT_DIR / "feature_importances.csv", index=False)
    keep = (["filename", "server", "hand", "split", "pose_ok", "pose_status"] +
            META_CAT + META_NUM + [TARGET] + arm_cols + body_cols)
    df[keep].to_csv(OUT_DIR / "features.csv", index=False)
    print(f"\nSaved to {OUT_DIR}: results.csv, features.csv, feature_importances.csv"
          + (", bootstrap_differences.csv" if N_BOOTSTRAP > 0 else ""))
    print("(features.csv includes speed_kmh for convenience; it is never used as a predictor.)")


if __name__ == "__main__":
    main()
