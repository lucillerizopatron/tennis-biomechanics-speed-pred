

#!/usr/bin/env python3
"""
Random Forest pipeline: serving-arm vs full-body pose for serve-speed
prediction (2024 US Open serve dataset).

All models use:
- The same player-level train/test split.
- The same pose-valid evaluation serves.
- Preprocessing fitted on training data only.

Models:
  1. Train-mean predictor
  2. Linear regression: metadata + height (original baseline)
  3. RF: metadata + height
  4. RF: metadata without height

  Additional Linear Regression metadata analysis:
  5. LR: gender only
  6. LR: gender + serve_number
  7. LR: gender + serve_number + height

  Original pose models:
  8. RF: metadata + height + arm pose
  9. RF: metadata + height + full-body pose
  10. RF: metadata + height + arm pose + acceleration
  11. RF: metadata + height + full-body pose + acceleration

  Additional RF models:
  12. RF: metadata without height + arm pose
  13. RF: metadata without height + full-body pose
  14. RF: arm pose only
  15. RF: full-body pose only

Run:
    python speed_pooled.py        (from src/random_forest/)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import warnings
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
HELPER_DIR = SCRIPT_DIR.parent   # src/, where serve_data.py lives
DATA_DIR_OVERRIDE = None
OUT_DIR = SCRIPT_DIR.parent.parent / "results" / "random_forest" / "speed_pooled"

SMOOTHING_MODULE = "smoothing"
HANDEDNESS_MODULE = "serve_handedness"
SERVE_DATA_MODULE = "serve_data"

USE_FEATURE_CACHE = True
SMOOTH_CUTOFF_HZ = None
SMOOTH_ORDER = 2

SPLIT_SEED = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 2000
TOP_N_IMPORTANCES = 15

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

REF_COUNTS = dict(
    train_players=87,
    test_players=22,
    train_serves=4729,
    test_serves=1237,
)

REF_LINEAR = dict(MAE=10.597, RMSE=13.291, R2=0.649)

FALLBACK_LEFT_HANDED = {
    "Denis Shapovalov",
    "Beatriz Haddad Maia",
}

META_CAT = ["gender", "serve_number"]
META_NUM = ["height_cm"]
META_NO_HEIGHT_NUM = []

TARGET = "speed_kmh"

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

def require(module: str, pip_name: str | None = None):
    try:
        return importlib.import_module(module)
    except ImportError:
        sys.exit(
            f"ERROR: missing dependency '{module}'. "
            f"Install with: pip install {pip_name or module}"
        )


np = require("numpy")
pd = require("pandas")
require("scipy")
require("sklearn", "scikit-learn")

try:
    importlib.import_module("pyarrow")
except ImportError:
    try:
        importlib.import_module("fastparquet")
    except ImportError:
        sys.exit(
            "ERROR: reading metadata.parquet needs pyarrow "
            "(or fastparquet). Install with: pip install pyarrow"
        )

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
    sys.exit(f"ERROR importing {SERVE_DATA_MODULE}: {e}")

try:
    sm = importlib.import_module(SMOOTHING_MODULE)
except ImportError as e:
    sys.exit(f"ERROR importing {SMOOTHING_MODULE}: {e}")

for fn in ("hip_relative", "smooth", "speed"):
    if not hasattr(sm, fn):
        sys.exit(f"ERROR: smoothing module has no function '{fn}'.")

try:
    hm = importlib.import_module(HANDEDNESS_MODULE)
    LEFT_HANDED = set(hm.LEFT_HANDED)
    hand = hm.hand
    HAND_SOURCE = HANDEDNESS_MODULE
except (ImportError, AttributeError):
    warnings.warn(
        "Handedness module unavailable; using fallback list. "
        "Handedness is NOT FULLY VERIFIED."
    )
    LEFT_HANDED = set(FALLBACK_LEFT_HANDED)
    hand = lambda s: "L" if s in LEFT_HANDED else "R"
    HAND_SOURCE = "embedded fallback"

DATA_DIR = (
    Path(DATA_DIR_OVERRIDE)
    if DATA_DIR_OVERRIDE
    else Path(sd.DATA)
)

FPS = float(sd.FPS)

# ============================================================================
# JOINTS / FEATURE DEFINITIONS
# ============================================================================

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

assert JOINT_NAMES[sd.L_SHOULDER] == "l_shoulder"
assert JOINT_NAMES[sd.R_SHOULDER] == "r_shoulder"
assert JOINT_NAMES[sd.L_ELBOW] == "l_elbow"
assert JOINT_NAMES[sd.R_ELBOW] == "r_elbow"
assert JOINT_NAMES[sd.L_WRIST] == "l_wrist"
assert JOINT_NAMES[sd.R_WRIST] == "r_wrist"

BASE_FEATURE_SUFFIXES = [
    "path_length",
    "peak_speed",
    "peak_time_norm",
    "mean_speed",
]

ACCEL_FEATURE_SUFFIXES = [
    "peak_acceleration",
    "mean_acceleration",
    "peak_acceleration_time_norm",
]

FEATURE_SUFFIXES = BASE_FEATURE_SUFFIXES + ACCEL_FEATURE_SUFFIXES
ARM_ROLES = ["shoulder", "elbow", "wrist"]

_b, _a = butter(
    SMOOTH_ORDER,
    (SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ) / (FPS / 2),
)

MIN_FRAMES = 3 * max(len(_a), len(_b)) + 1
CUTOFF = SMOOTH_CUTOFF_HZ or sm.DEFAULT_CUTOFF_HZ

if not (0 < CUTOFF < FPS / 2):
    sys.exit(f"ERROR: invalid smoothing cutoff {CUTOFF} Hz.")

JOINT_FEATURE_COLS = [
    f"{j}_{s}"
    for j in JOINT_NAMES
    for s in FEATURE_SUFFIXES
]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def joint_features(k: np.ndarray) -> dict:
    """Extract position-derived speed and acceleration features."""

    if k.ndim != 3 or k.shape[1:] != (17, 3):
        raise ValueError(f"unexpected shape {k.shape}, expected (T,17,3)")

    if k.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"too_short: {k.shape[0]} frames < {MIN_FRAMES}"
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

        velocity = np.diff(x[:, j, :], axis=0) * FPS
        speed = np.linalg.norm(velocity, axis=1)

        out[f"{name}_path_length"] = float(speed.sum() / FPS)
        out[f"{name}_peak_speed"] = float(speed.max())
        out[f"{name}_peak_time_norm"] = float(
            np.argmax(speed) / (len(speed) - 1)
        )
        out[f"{name}_mean_speed"] = float(speed.mean())

        acceleration = np.diff(velocity, axis=0) * FPS
        acceleration_magnitude = np.linalg.norm(
            acceleration, axis=1
        )

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


def compute_or_load_joint_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute or load cached joint features."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_csv = OUT_DIR / "joint_features_cache.csv"
    cache_meta = OUT_DIR / "joint_features_cache.json"

    key = dict(
        fps=FPS,
        cutoff_hz=CUTOFF,
        order=SMOOTH_ORDER,
        n=len(rows),
        files_md5=hashlib.md5(
            "\n".join(rows["filename"]).encode()
        ).hexdigest(),
        feature_suffixes=FEATURE_SUFFIXES,
    )

    if USE_FEATURE_CACHE and cache_csv.exists() and cache_meta.exists():

        if json.loads(cache_meta.read_text()) == key:
            print(f"Loading cached features from {cache_csv}")
            cached = pd.read_csv(cache_csv, index_col=0)

            if cached.index.equals(rows.index):
                return cached

        print("Cache present but stale; recomputing.")

    kp_dir = DATA_DIR / "keypoints"
    records, statuses = [], []

    for n, (idx, fname) in enumerate(
        zip(rows.index, rows["filename"]), 1
    ):
        try:
            k = np.load(kp_dir / fname)
            records.append(joint_features(k))
            statuses.append("ok")

        except FileNotFoundError:
            records.append({})
            statuses.append("missing_file")

        except ValueError as e:
            msg = str(e)

            tag = (
                "bad_shape"
                if msg.startswith("unexpected shape")
                else "too_short"
                if msg.startswith("too_short")
                else "non_finite"
                if msg.startswith("non_finite")
                else "value_error"
            )

            records.append({})
            statuses.append(tag)

            if tag in ("bad_shape", "value_error"):
                print(f"WARNING {fname}: {msg}")

        except Exception as e:
            records.append({})
            statuses.append(f"load_error_{type(e).__name__}")
            print(f"WARNING {fname}: {type(e).__name__}: {e}")

        if n % 1000 == 0:
            print(f"Processed {n}/{len(rows)} serves")

    feats = pd.DataFrame(
        records,
        index=rows.index,
    ).reindex(columns=JOINT_FEATURE_COLS)

    feats["pose_status"] = statuses

    if USE_FEATURE_CACHE:
        feats.to_csv(cache_csv)
        cache_meta.write_text(json.dumps(key))

    return feats


def build_representation_columns(
    df: pd.DataFrame,
):
    """Create dominant-arm and full-body pose representations."""

    new = {}
    is_left = (df["hand"] == "L").to_numpy()

    for role in ARM_ROLES:
        for suffix in FEATURE_SUFFIXES:
            new[f"arm_{role}_{suffix}"] = np.where(
                is_left,
                df[f"l_{role}_{suffix}"],
                df[f"r_{role}_{suffix}"],
            )

    for joint in JOINT_NAMES:
        for suffix in FEATURE_SUFFIXES:
            new[f"body_{joint}_{suffix}"] = (
                df[f"{joint}_{suffix}"].to_numpy()
            )

    new = pd.DataFrame(new, index=df.index)

    arm_cols = [c for c in new if c.startswith("arm_")]
    body_cols = [c for c in new if c.startswith("body_")]

    arm_base_cols = [
        c for c in arm_cols
        if not any(
            c.endswith("_" + s)
            for s in ACCEL_FEATURE_SUFFIXES
        )
    ]

    body_base_cols = [
        c for c in body_cols
        if not any(
            c.endswith("_" + s)
            for s in ACCEL_FEATURE_SUFFIXES
        )
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

def make_pipeline(
    regressor,
    cat_cols: list[str],
    num_cols: list[str],
    pose_cols: list[str],
) -> Pipeline:
    """
    Build a pipeline with explicitly selected metadata and pose inputs.

    Categorical variables are one-hot encoded.
    Imputation and encoding are fitted on training data only.
    """

    transformers = []

    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                cat_cols,
            )
        )

    numeric_cols = num_cols + pose_cols

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
                ]),
                numeric_cols,
            )
        )

    pre = ColumnTransformer(
        transformers,
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("pre", pre),
        ("reg", regressor),
    ])


def metrics(y, p) -> dict:
    return dict(
        MAE=mean_absolute_error(y, p),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        R2=r2_score(y, p),
    )


def player_bootstrap_diff(
    players,
    y,
    p_new,
    p_ref,
    n_boot,
    seed=0,
):
    """Paired bootstrap over test players: new model minus reference."""

    codes, uniq = pd.factorize(players)
    P = len(uniq)

    cnt = np.bincount(codes, minlength=P).astype(float)

    def sums(p):
        e = y - p

        return (
            np.bincount(
                codes,
                weights=np.abs(e),
                minlength=P,
            ),
            np.bincount(
                codes,
                weights=e ** 2,
                minlength=P,
            ),
        )

    (a_abs, a_sq) = sums(p_new)
    (r_abs, r_sq) = sums(p_ref)

    draws = np.random.default_rng(seed).integers(
        0, P, size=(n_boot, P)
    )

    c = cnt[draws].sum(1)

    mae = (
        a_abs[draws].sum(1) / c
        - r_abs[draws].sum(1) / c
    )

    rmse = (
        np.sqrt(a_sq[draws].sum(1) / c)
        - np.sqrt(r_sq[draws].sum(1) / c)
    )

    return mae, rmse

# ============================================================================
# MAIN
# ============================================================================

def main():

    pd.set_option("display.width", 220)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------------

    print(f"Data directory: {DATA_DIR}")

    try:
        meta = sd.load_metadata(DATA_DIR)
    except Exception as e:
        sys.exit(f"ERROR reading metadata: {type(e).__name__}: {e}")

    needed = (
        ["filename", "server"]
        + META_CAT
        + META_NUM
        + [TARGET]
    )

    absent = [c for c in needed if c not in meta.columns]

    if absent:
        sys.exit(f"ERROR: metadata missing columns: {absent}")

    print(f"Metadata rows: {len(meta)}")

    df = meta[needed].copy()

    for col in ["serve_number", "height_cm", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)

    df = df.dropna(subset=[TARGET, "server"])
    n1 = len(df)

    df = df[
        df["serve_number"].isin([1, 2])
        & df["gender"].isin(["M", "F"])
    ].copy()

    print(
        f"\nFiltering: {n0} -> {n1} -> {len(df)} "
        "(valid target/server, serve number and gender)"
    )

    print(
        f"Serves: {len(df)}, players: {df['server'].nunique()}, "
        f"missing heights: {int(df['height_cm'].isna().sum())}"
    )

    # ------------------------------------------------------------------------
    # 2. Handedness
    # ------------------------------------------------------------------------

    df["hand"] = df["server"].map(hand)

    present_left = sorted(LEFT_HANDED & set(df["server"]))
    missing_left = sorted(LEFT_HANDED - set(df["server"]))

    print(f"\nHandedness source: {HAND_SOURCE}")
    print(f"Left-handed players in data: {present_left}")

    if missing_left:
        print(f"WARNING: listed left-handers not found: {missing_left}")

    # ------------------------------------------------------------------------
    # 3. Player-level split
    # ------------------------------------------------------------------------

    players = (
        df.groupby("server", as_index=False)
        .agg(height_cm=("height_cm", "median"))
    )

    players["height_for_split"] = (
        players["height_cm"]
        .fillna(players["height_cm"].median())
    )

    players["height_group"] = pd.qcut(
        players["height_for_split"],
        q=4,
        labels=False,
        duplicates="drop",
    )

    train_players, test_players = train_test_split(
        players["server"],
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=players["height_group"],
    )

    train_set = set(train_players)
    test_set = set(test_players)

    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(df["server"])

    df["split"] = np.where(
        df["server"].isin(test_set),
        "test",
        "train",
    )

    got = dict(
        train_players=len(train_set),
        test_players=len(test_set),
        train_serves=int((df["split"] == "train").sum()),
        test_serves=int((df["split"] == "test").sum()),
    )

    print("\nPLAYER-LEVEL SPLIT")
    print(got)
    print(f"Reference counts: {REF_COUNTS}")

    if got != REF_COUNTS:
        print(
            "WARNING: split counts differ from reference. "
            "Check filtering and split settings."
        )

    # ------------------------------------------------------------------------
    # 4. Pose feature extraction
    # ------------------------------------------------------------------------

    print("\nExtracting pose features...")
    print(
        f"FPS={FPS}, cutoff={CUTOFF} Hz, "
        f"order={SMOOTH_ORDER}, min frames={MIN_FRAMES}"
    )

    jf = compute_or_load_joint_features(df)
    df = df.join(jf)

    df["pose_ok"] = df["pose_status"].eq("ok")

    n_bad = int((~df["pose_ok"]).sum())

    print(
        f"\nPOSE EXTRACTION: {int(df['pose_ok'].sum())} usable, "
        f"{n_bad} excluded"
    )

    if n_bad:
        print(df.loc[~df["pose_ok"], "pose_status"].value_counts())

        if n_bad / len(df) > 0.02:
            print("WARNING: more than 2% of poses excluded.")

    if df["pose_ok"].sum() == 0:
        sys.exit("ERROR: no usable pose sequences.")

    (
        rep,
        arm_cols,
        body_cols,
        arm_base_cols,
        body_base_cols,
    ) = build_representation_columns(df)

    df = pd.concat([df, rep], axis=1)

    # Every model is evaluated on exactly the same pose-valid serves.
    pose_df = df[df["pose_ok"]].copy()

    train = pose_df[pose_df["split"] == "train"].copy()
    test = pose_df[pose_df["split"] == "test"].copy()

    assert set(train["server"]).isdisjoint(set(test["server"]))

    assert not train[arm_cols + body_cols].isna().any().any()
    assert not test[arm_cols + body_cols].isna().any().any()

    print("\nFINAL EVALUATION SAMPLE (identical for all models)")
    print(
        f"Train: {len(train)} serves / "
        f"{train['server'].nunique()} players"
    )
    print(
        f"Test:  {len(test)} serves / "
        f"{test['server'].nunique()} players"
    )
    print(
        f"Arm features: {len(arm_cols)}, "
        f"full-body features: {len(body_cols)}"
    )

    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    # ------------------------------------------------------------------------
    # 5. Define model specifications
    # ------------------------------------------------------------------------

    specs = {

        "Mean predictor (train mean)": {
            "reg": DummyRegressor(strategy="mean"),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": [],
            "type": "reference",
        },

        # Original linear regression baseline
        "Linear regression: metadata + height": {
            "reg": LinearRegression(),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": [],
            "type": "baseline",
        },

        # ------------------------------------------------------------
        # NEW: Linear Regression metadata comparison
        # ------------------------------------------------------------

        "LR: gender only": {
            "reg": LinearRegression(),
            "cat": ["gender"],
            "num": [],
            "pose": [],
            "type": "metadata_ablation",
        },

        "LR: gender + serve_number": {
            "reg": LinearRegression(),
            "cat": ["gender", "serve_number"],
            "num": [],
            "pose": [],
            "type": "metadata_ablation",
        },

        "LR: gender + serve_number + height": {
            "reg": LinearRegression(),
            "cat": ["gender", "serve_number"],
            "num": ["height_cm"],
            "pose": [],
            "type": "metadata_ablation",
        },

        # ------------------------------------------------------------
        # Original RF baselines
        # ------------------------------------------------------------

        "RF: metadata + height": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": [],
            "type": "baseline",
        },

        "RF: metadata without height": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NO_HEIGHT_NUM,
            "pose": [],
            "type": "baseline",
        },

        # ------------------------------------------------------------
        # Original RF models: metadata + height + pose
        # ------------------------------------------------------------

        "RF: metadata + height + arm": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": arm_base_cols,
            "type": "pose",
        },

        "RF: metadata + height + body": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": body_base_cols,
            "type": "pose",
        },

        "RF: metadata + height + arm + acceleration": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": arm_cols,
            "type": "pose",
        },

        "RF: metadata + height + body + acceleration": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NUM,
            "pose": body_cols,
            "type": "pose",
        },

        # ------------------------------------------------------------
        # RF models: metadata without height + pose
        # ------------------------------------------------------------

        "RF: metadata without height + arm": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NO_HEIGHT_NUM,
            "pose": arm_base_cols,
            "type": "pose",
        },

        "RF: metadata without height + body": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": META_CAT,
            "num": META_NO_HEIGHT_NUM,
            "pose": body_base_cols,
            "type": "pose",
        },

        # ------------------------------------------------------------
        # RF pose-only models
        # ------------------------------------------------------------

        "RF: arm only": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": [],
            "num": [],
            "pose": arm_base_cols,
            "type": "pose_only",
        },

        "RF: body only": {
            "reg": RandomForestRegressor(**RF_PARAMS),
            "cat": [],
            "num": [],
            "pose": body_base_cols,
            "type": "pose_only",
        },
    }

    # ------------------------------------------------------------------------
    # 6. Fit and evaluate all models
    # ------------------------------------------------------------------------

    fitted = {}
    preds = {}
    rows = []

    for name, spec in specs.items():

        cat_cols = spec["cat"]
        num_cols = spec["num"]
        pose_cols = spec["pose"]

        cols = cat_cols + num_cols + pose_cols

        pipe = make_pipeline(
            spec["reg"],
            cat_cols,
            num_cols,
            pose_cols,
        )

        pipe.fit(train[cols], y_train)
        p = pipe.predict(test[cols])

        fitted[name] = pipe
        preds[name] = p

        rows.append({
            "Model": name,
            "n_features_metadata": len(cat_cols) + len(num_cols),
            "n_features_pose": len(pose_cols),
            **metrics(y_test, p),
        })

    res = pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # 7. Comparisons against appropriate baselines
    # ------------------------------------------------------------------------

    comparison_pairs = [

        # New LR comparisons: incremental metadata contribution
        (
            "LR: gender + serve_number",
            "LR: gender only",
        ),
        (
            "LR: gender + serve_number + height",
            "LR: gender + serve_number",
        ),

        # Original RF comparisons
        (
            "RF: metadata + height + arm",
            "RF: metadata + height",
        ),
        (
            "RF: metadata + height + body",
            "RF: metadata + height",
        ),
        (
            "RF: metadata + height + arm + acceleration",
            "RF: metadata + height",
        ),
        (
            "RF: metadata + height + body + acceleration",
            "RF: metadata + height",
        ),
        (
            "RF: metadata without height + arm",
            "RF: metadata without height",
        ),
        (
            "RF: metadata without height + body",
            "RF: metadata without height",
        ),
        (
            "RF: arm only",
            "Mean predictor (train mean)",
        ),
        (
            "RF: body only",
            "Mean predictor (train mean)",
        ),
    ]

    res_idx = res.set_index("Model")

    print("\n" + "=" * 90)
    print("RESULTS (test players only; MAE/RMSE in km/h)")
    print("=" * 90)

    print(
        res[
            [
                "Model",
                "n_features_metadata",
                "n_features_pose",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].round(3).to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # 7A. NEW: Descriptive speed by gender and serve number
    # ------------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("TEST SET: MEAN SPEED BY GENDER AND SERVE NUMBER")
    print("=" * 90)

    descriptive = (
        test.groupby(["gender", "serve_number"], observed=True)[TARGET]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    print(descriptive.round(2).to_string(index=False))

    descriptive.to_csv(
        OUT_DIR / "speed_by_gender_and_serve_number.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 7B. Height contribution: original RF baseline comparison
    # ------------------------------------------------------------------------

    print("\nHEIGHT CONTRIBUTION: RF metadata baseline comparison")

    height_diff = {
        "MAE": (
            res_idx.loc["RF: metadata + height", "MAE"]
            - res_idx.loc["RF: metadata without height", "MAE"]
        ),
        "RMSE": (
            res_idx.loc["RF: metadata + height", "RMSE"]
            - res_idx.loc["RF: metadata without height", "RMSE"]
        ),
        "R2": (
            res_idx.loc["RF: metadata + height", "R2"]
            - res_idx.loc["RF: metadata without height", "R2"]
        ),
    }

    print(
        "Difference (with height - without height): "
        + ", ".join(f"{k}={v:.3f}" for k, v in height_diff.items())
    )

    # ------------------------------------------------------------------------
    # 7C. Model comparisons
    # ------------------------------------------------------------------------

    print("\nMODEL COMPARISONS (new model minus reference)")
    print("Negative MAE/RMSE differences indicate improvement.")

    comparison_rows = []

    for new, ref in comparison_pairs:

        new_row = res_idx.loc[new]
        ref_row = res_idx.loc[ref]

        comparison_rows.append({
            "Model": new,
            "Reference": ref,
            "dMAE": new_row["MAE"] - ref_row["MAE"],
            "dRMSE": new_row["RMSE"] - ref_row["RMSE"],
            "dR2": new_row["R2"] - ref_row["R2"],
        })

    comparisons = pd.DataFrame(comparison_rows)

    print(comparisons.round(3).to_string(index=False))

    # ------------------------------------------------------------------------
    # 8. Sanity check: reproduce original linear baseline
    # ------------------------------------------------------------------------

    full_tr = df[df["split"] == "train"]
    full_te = df[df["split"] == "test"]

    chk = make_pipeline(
        LinearRegression(),
        META_CAT,
        META_NUM,
        [],
    )

    chk.fit(full_tr[META_CAT + META_NUM], full_tr[TARGET])

    cm = metrics(
        full_te[TARGET].to_numpy(),
        chk.predict(full_te[META_CAT + META_NUM]),
    )

    print("\nSANITY CHECK: linear baseline on ALL filtered serves")
    print(
        "This run: "
        + ", ".join(f"{k}={v:.3f}" for k, v in cm.items())
    )
    print(
        "Reference: "
        + ", ".join(f"{k}={v:.3f}" for k, v in REF_LINEAR.items())
    )

    ok = all(
        abs(cm[k] - REF_LINEAR[k]) < 0.01
        for k in REF_LINEAR
    )

    print(
        "-> reproduces existing baseline"
        if ok
        else "-> DOES NOT match reference; check filtering/split"
    )

    # ------------------------------------------------------------------------
    # 9. Player-level bootstrap
    # ------------------------------------------------------------------------

    if N_BOOTSTRAP > 0:

        print(
            f"\nPLAYER-LEVEL BOOTSTRAP "
            f"({N_BOOTSTRAP} resamples)"
        )
        print(
            "New model minus reference; negative MAE/RMSE "
            "differences indicate improvement."
        )

        boot_rows = []

        for i, (new, ref) in enumerate(comparison_pairs):

            dmae, drmse = player_bootstrap_diff(
                test["server"].to_numpy(),
                y_test,
                preds[new],
                preds[ref],
                N_BOOTSTRAP,
                seed=i,
            )

            boot_rows.append({
                "new": new,
                "reference": ref,
                "dMAE_mean": dmae.mean(),
                "dMAE_lo95": np.percentile(dmae, 2.5),
                "dMAE_hi95": np.percentile(dmae, 97.5),
                "share_MAE_better": float((dmae < 0).mean()),
                "dRMSE_mean": drmse.mean(),
                "dRMSE_lo95": np.percentile(drmse, 2.5),
                "dRMSE_hi95": np.percentile(drmse, 97.5),
            })

        boot = pd.DataFrame(boot_rows)

        print(boot.round(3).to_string(index=False))

        boot.to_csv(
            OUT_DIR / "bootstrap_differences.csv",
            index=False,
        )

    # ------------------------------------------------------------------------
    # 10. Feature importances
    # ------------------------------------------------------------------------

    imp_all = []

    for name, spec in specs.items():

        if spec["type"] not in ("pose", "pose_only"):
            continue

        pipe = fitted[name]

        names = pipe.named_steps["pre"].get_feature_names_out()

        imp = pd.DataFrame({
            "model": name,
            "feature": names,
            "importance": (
                pipe.named_steps["reg"].feature_importances_
            ),
        })

        imp["group"] = np.where(
            imp["feature"].str.startswith(("arm_", "body_")),
            "pose",
            "metadata",
        )

        imp = imp.sort_values(
            "importance",
            ascending=False,
        )

        imp_all.append(imp)

        print(f"\nFEATURE IMPORTANCES: {name}")

        print(
            "Total importance by group: "
            + ", ".join(
                f"{g}={v:.3f}"
                for g, v in (
                    imp.groupby("group")["importance"].sum().items()
                )
            )
        )

        print(
            imp.head(TOP_N_IMPORTANCES)[
                ["group", "feature", "importance"]
            ].round(4).to_string(index=False)
        )

    print(
        "\nCaveat: impurity-based feature importances can be biased "
        "towards continuous or high-cardinality features and can "
        "split importance across correlated features."
    )

    # ------------------------------------------------------------------------
    # 11. Save outputs
    # ------------------------------------------------------------------------

    res.to_csv(OUT_DIR / "results.csv", index=False)

    comparisons.to_csv(
        OUT_DIR / "model_comparisons.csv",
        index=False,
    )

    if imp_all:
        pd.concat(imp_all).to_csv(
            OUT_DIR / "feature_importances.csv",
            index=False,
        )

    keep = (
        [
            "filename",
            "server",
            "hand",
            "split",
            "pose_ok",
            "pose_status",
        ]
        + META_CAT
        + META_NUM
        + [TARGET]
        + arm_cols
        + body_cols
    )

    df[keep].to_csv(
        OUT_DIR / "features.csv",
        index=False,
    )

    print(f"\nSaved outputs to: {OUT_DIR}")
    print(
        "Files: results.csv, model_comparisons.csv, "
        "features.csv, feature_importances.csv, "
        "bootstrap_differences.csv, "
        "speed_by_gender_and_serve_number.csv"
    )
    print(
        "Note: speed_kmh is the target and is never used "
        "as a predictor."
    )


if __name__ == "__main__":
    main()
