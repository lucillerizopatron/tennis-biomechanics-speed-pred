

"""
Stratified Random Forest pipeline for tennis serve-speed prediction.

Four independent analyses:
    F - First serve
    F - Second serve
    M - First serve
    M - Second serve

Within each group, compare:
    1. Mean baseline: training subgroup mean
    2. RF: arm only
    3. RF: full body only
    4. RF: height only

Player-level split:
    Players are split separately for each gender.
    The SAME player split is used for first and second serves
    within that gender.

All models in each group use exactly the same pose-valid serves.

Run:
    python3 src/random_forest_stratified.py
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

DATA_DIR_OVERRIDE = None

OUT_DIR = SCRIPT_DIR / "pose_rf_stratified_outputs"

SMOOTHING_MODULE = "smoothing"
HANDEDNESS_MODULE = "serve_handedness"
SERVE_DATA_MODULE = "serve_data"

USE_FEATURE_CACHE = True

SMOOTH_CUTOFF_HZ = None
SMOOTH_ORDER = 2

SPLIT_SEED = 42
TEST_SIZE = 0.20

N_BOOTSTRAP = 2000
TOP_N_IMPORTANCES = 15

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

FALLBACK_LEFT_HANDED = {
    "Denis Shapovalov",
    "Beatriz Haddad Maia",
}

TARGET = "speed_kmh"

# ============================================================================
# DEPENDENCIES
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ============================================================================
# PROJECT HELPERS
# ============================================================================

sys.path.insert(0, str(SCRIPT_DIR))

try:
    sd = importlib.import_module(SERVE_DATA_MODULE)
    sm = importlib.import_module(SMOOTHING_MODULE)
except ImportError as e:
    sys.exit(f"ERROR importing project helper: {e}")

for fn in ("hip_relative", "smooth"):
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
# JOINTS AND FEATURES
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

FEATURE_SUFFIXES = (
    BASE_FEATURE_SUFFIXES + ACCEL_FEATURE_SUFFIXES
)

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
    f"{joint}_{suffix}"
    for joint in JOINT_NAMES
    for suffix in FEATURE_SUFFIXES
]

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def joint_features(k: np.ndarray) -> dict:

    if k.ndim != 3 or k.shape[1:] != (17, 3):
        raise ValueError(f"unexpected shape {k.shape}")

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
            acceleration,
            axis=1,
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


def compute_or_load_joint_features(rows: pd.DataFrame):

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

    records = []
    statuses = []

    for n, (idx, fname) in enumerate(
        zip(rows.index, rows["filename"]),
        1,
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
            statuses.append(
                f"load_error_{type(e).__name__}"
            )

            print(
                f"WARNING {fname}: "
                f"{type(e).__name__}: {e}"
            )

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


def build_representation_columns(df: pd.DataFrame):

    new = {}

    is_left = (df["hand"] == "L").to_numpy()

    # Dominant arm: shoulder, elbow and wrist
    for role in ARM_ROLES:
        for suffix in FEATURE_SUFFIXES:

            new[f"arm_{role}_{suffix}"] = np.where(
                is_left,
                df[f"l_{role}_{suffix}"],
                df[f"r_{role}_{suffix}"],
            )

    # Full body: all 17 joints
    for joint in JOINT_NAMES:
        for suffix in FEATURE_SUFFIXES:

            new[f"body_{joint}_{suffix}"] = (
                df[f"{joint}_{suffix}"].to_numpy()
            )

    new = pd.DataFrame(new, index=df.index)

    arm_cols = [
        c for c in new.columns
        if c.startswith("arm_")
    ]

    body_cols = [
        c for c in new.columns
        if c.startswith("body_")
    ]

    # Main comparison uses base features only.
    # Acceleration features are excluded.
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
# MODELS AND METRICS
# ============================================================================

def make_pipeline(num_cols: list[str]) -> Pipeline:

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                num_cols,
            )
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("pre", pre),
        ("reg", RandomForestRegressor(**RF_PARAMS)),
    ])


def calculate_metrics(y, p):

    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": (
            r2_score(y, p)
            if len(y) >= 2
            else np.nan
        ),
    }


def player_bootstrap_diff(
    players,
    y,
    p_new,
    p_ref,
    n_boot,
    seed=0,
):
    """Paired bootstrap over test players."""

    codes, unique_players = pd.factorize(players)
    n_players = len(unique_players)

    if n_players < 2:
        return None

    counts = np.bincount(
        codes,
        minlength=n_players,
    ).astype(float)

    def error_sums(pred):

        errors = y - pred

        absolute = np.bincount(
            codes,
            weights=np.abs(errors),
            minlength=n_players,
        )

        squared = np.bincount(
            codes,
            weights=errors ** 2,
            minlength=n_players,
        )

        return absolute, squared

    new_abs, new_sq = error_sums(p_new)
    ref_abs, ref_sq = error_sums(p_ref)

    draws = np.random.default_rng(seed).integers(
        0,
        n_players,
        size=(n_boot, n_players),
    )

    n_observations = counts[draws].sum(axis=1)

    d_mae = (
        new_abs[draws].sum(axis=1) / n_observations
        - ref_abs[draws].sum(axis=1) / n_observations
    )

    d_rmse = (
        np.sqrt(
            new_sq[draws].sum(axis=1) / n_observations
        )
        - np.sqrt(
            ref_sq[draws].sum(axis=1) / n_observations
        )
    )

    return d_mae, d_rmse

# ============================================================================
# PLAYER-LEVEL SPLITS
# ============================================================================

def make_gender_player_split(df_gender: pd.DataFrame, gender: str):
    """
    Create one player-level split per gender.

    The same player split is subsequently used for both
    first and second serves of that gender.
    """

    players = (
        df_gender.groupby("server", as_index=False)
        .agg(height_cm=("height_cm", "median"))
    )

    n_players = len(players)

    if n_players < 5:
        raise ValueError(
            f"Only {n_players} players for gender {gender}; "
            "not enough for a reliable player-level split."
        )

    players["height_for_split"] = (
        players["height_cm"]
        .fillna(players["height_cm"].median())
    )

    stratify = None

    try:

        players["height_group"] = pd.qcut(
            players["height_for_split"],
            q=4,
            labels=False,
            duplicates="drop",
        )

        counts = players["height_group"].value_counts()

        n_test = int(np.ceil(TEST_SIZE * n_players))

        if (
            players["height_group"].nunique() >= 2
            and counts.min() >= 2
            and n_test >= players["height_group"].nunique()
        ):
            stratify = players["height_group"]

    except ValueError:
        stratify = None

    try:

        train_players, test_players = train_test_split(
            players["server"],
            test_size=TEST_SIZE,
            random_state=SPLIT_SEED,
            stratify=stratify,
        )

    except ValueError as e:

        print(
            f"WARNING: stratified split failed for {gender}: {e}"
        )

        print("Using unstratified player-level split.")

        train_players, test_players = train_test_split(
            players["server"],
            test_size=TEST_SIZE,
            random_state=SPLIT_SEED,
        )

    train_set = set(train_players)
    test_set = set(test_players)

    assert train_set.isdisjoint(test_set)
    assert train_set | test_set == set(players["server"])

    print(f"\n{gender} PLAYER SPLIT")
    print(
        f"Train: {len(train_set)} players | "
        f"Test: {len(test_set)} players"
    )

    return train_set, test_set

# ============================================================================
# RUN ONE SUBGROUP
# ============================================================================

def run_group(
    df_group,
    gender,
    serve_number,
    arm_base_cols,
    body_base_cols,
):

    group_name = (
        f"{gender}_{'first' if serve_number == 1 else 'second'}"
    )

    print("\n" + "=" * 100)
    print(f"ANALYSIS: {group_name.upper()}")
    print("=" * 100)

    # ------------------------------------------------------------
    # Apply the gender-level player split
    # ------------------------------------------------------------

    train = df_group[df_group["split"] == "train"].copy()
    test = df_group[df_group["split"] == "test"].copy()

    train_players = train["server"].nunique()
    test_players = test["server"].nunique()

    print(
        f"Training: {len(train)} serves / "
        f"{train_players} players"
    )

    print(
        f"Test:     {len(test)} serves / "
        f"{test_players} players"
    )

    if len(train) < 10 or len(test) < 2:
        print("WARNING: insufficient observations. Skipping.")
        return None

    if train_players < 2 or test_players < 2:
        print("WARNING: too few distinct players. Skipping.")
        return None

    assert set(train["server"]).isdisjoint(set(test["server"]))

    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    # ------------------------------------------------------------
    # Model specifications
    # ------------------------------------------------------------

    specs = {
        "RF: arm only": arm_base_cols,
        "RF: body only": body_base_cols,
        "RF: height only": ["height_cm"],
    }

    fitted = {}
    predictions = {}
    result_rows = []

    # ------------------------------------------------------------
    # 1. Mean baseline
    # ------------------------------------------------------------

    # Calculate the mean using ONLY the training subgroup.
    # Every test serve receives exactly the same prediction.

    train_mean = float(np.mean(y_train))

    mean_pred = np.full(
        shape=len(y_test),
        fill_value=train_mean,
        dtype=float,
    )

    predictions["Mean baseline"] = mean_pred

    result_rows.append({
        "group": group_name,
        "gender": gender,
        "serve_number": serve_number,
        "Model": "Mean baseline",
        "n_features": 0,
        "training_mean_kmh": train_mean,
        "n_train": len(train),
        "n_test": len(test),
        "n_train_players": train_players,
        "n_test_players": test_players,
        **calculate_metrics(y_test, mean_pred),
    })

    print(f"\nTraining subgroup mean: {train_mean:.3f} km/h")

    # ------------------------------------------------------------
    # 2. Fit and evaluate Random Forest models
    # ------------------------------------------------------------

    for model_name, cols in specs.items():

        pipe = make_pipeline(cols)

        pipe.fit(train[cols], y_train)

        pred = pipe.predict(test[cols])

        fitted[model_name] = pipe
        predictions[model_name] = pred

        result_rows.append({
            "group": group_name,
            "gender": gender,
            "serve_number": serve_number,
            "Model": model_name,
            "n_features": len(cols),
            "training_mean_kmh": train_mean,
            "n_train": len(train),
            "n_test": len(test),
            "n_train_players": train_players,
            "n_test_players": test_players,
            **calculate_metrics(y_test, pred),
        })

    results = pd.DataFrame(result_rows)

    print("\nMODEL RESULTS")
    print(
        results[
            ["Model", "n_features", "MAE", "RMSE", "R2"]
        ].round(3).to_string(index=False)
    )

    # ------------------------------------------------------------
    # 3. Model comparisons
    # ------------------------------------------------------------

    # Negative dMAE / dRMSE means the first model performs better.

    pairs = [
        ("RF: arm only", "Mean baseline"),
        ("RF: body only", "Mean baseline"),
        ("RF: height only", "Mean baseline"),
        ("RF: arm only", "RF: body only"),
    ]

    result_index = results.set_index("Model")

    comparison_rows = []

    for new, reference in pairs:

        a = result_index.loc[new]
        b = result_index.loc[reference]

        comparison_rows.append({
            "group": group_name,
            "Model": new,
            "Reference": reference,
            "dMAE": a["MAE"] - b["MAE"],
            "dRMSE": a["RMSE"] - b["RMSE"],
            "dR2": a["R2"] - b["R2"],
        })

    comparisons = pd.DataFrame(comparison_rows)

    print("\nMODEL COMPARISONS")
    print("Negative dMAE/dRMSE indicates improvement.")

    print(comparisons.round(3).to_string(index=False))

    # ------------------------------------------------------------
    # 4. Player-level paired bootstrap
    # ------------------------------------------------------------

    bootstrap_rows = []

    if N_BOOTSTRAP > 0:

        print(
            f"\nPLAYER-LEVEL BOOTSTRAP "
            f"({N_BOOTSTRAP} resamples)"
        )

        for i, (new, reference) in enumerate(pairs):

            boot = player_bootstrap_diff(
                test["server"].to_numpy(),
                y_test,
                predictions[new],
                predictions[reference],
                N_BOOTSTRAP,
                seed=SPLIT_SEED + i,
            )

            if boot is None:
                continue

            d_mae, d_rmse = boot

            bootstrap_rows.append({
                "group": group_name,
                "Model": new,
                "Reference": reference,

                "dMAE_mean": d_mae.mean(),
                "dMAE_lo95": np.percentile(d_mae, 2.5),
                "dMAE_hi95": np.percentile(d_mae, 97.5),
                "share_MAE_better": (d_mae < 0).mean(),

                "dRMSE_mean": d_rmse.mean(),
                "dRMSE_lo95": np.percentile(d_rmse, 2.5),
                "dRMSE_hi95": np.percentile(d_rmse, 97.5),
            })

        bootstrap = pd.DataFrame(bootstrap_rows)

        if not bootstrap.empty:
            print(bootstrap.round(3).to_string(index=False))

    else:
        bootstrap = pd.DataFrame()

    # ------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------

    importance_rows = []

    for model_name in (
        "RF: arm only",
        "RF: body only",
        "RF: height only",
    ):

        pipe = fitted[model_name]

        names = pipe.named_steps["pre"].get_feature_names_out()

        importances = (
            pipe.named_steps["reg"].feature_importances_
        )

        imp = pd.DataFrame({
            "group": group_name,
            "Model": model_name,
            "feature": names,
            "importance": importances,
        })

        imp["feature_group"] = np.select(
            [
                imp["feature"].str.startswith("arm_"),
                imp["feature"].str.startswith("body_"),
                imp["feature"].eq("height_cm"),
            ],
            [
                "arm",
                "body",
                "height",
            ],
            default="other",
        )

        importance_rows.append(imp)

        print(f"\nFEATURE IMPORTANCE: {model_name}")

        print(
            imp.sort_values(
                "importance",
                ascending=False,
            ).head(TOP_N_IMPORTANCES)[
                ["feature_group", "feature", "importance"]
            ].round(4).to_string(index=False)
        )

    # ------------------------------------------------------------
    # 6. Save predictions for this subgroup
    # ------------------------------------------------------------

    prediction_df = test[
        ["filename", "server", "gender", "serve_number", TARGET]
    ].copy()

    prediction_df["training_mean_kmh"] = train_mean

    for model_name, pred in predictions.items():

        safe_name = (
            model_name.lower()
            .replace(": ", "_")
            .replace(" + ", "_")
            .replace(" ", "_")
        )

        prediction_df[safe_name] = pred

    return {
        "results": results,
        "comparisons": comparisons,
        "bootstrap": bootstrap,
        "importances": pd.concat(
            importance_rows,
            ignore_index=True,
        ),
        "predictions": prediction_df,
        "summary": {
            "group": group_name,
            "gender": gender,
            "serve_number": serve_number,
            "training_mean_kmh": train_mean,
            "n_train": len(train),
            "n_test": len(test),
            "n_train_players": train_players,
            "n_test_players": test_players,
        },
    }

# ============================================================================
# MAIN
# ============================================================================

def main():

    pd.set_option("display.width", 220)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")

    # ------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------

    try:
        meta = sd.load_metadata(DATA_DIR)
    except Exception as e:
        sys.exit(
            f"ERROR reading metadata: {type(e).__name__}: {e}"
        )

    needed = [
        "filename",
        "server",
        "gender",
        "serve_number",
        "height_cm",
        TARGET,
    ]

    absent = [
        col for col in needed
        if col not in meta.columns
    ]

    if absent:
        sys.exit(f"ERROR: metadata missing columns: {absent}")

    df = meta[needed].copy()

    for col in ("serve_number", "height_cm", TARGET):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_initial = len(df)

    df = df.dropna(
        subset=["server", "gender", "serve_number", TARGET]
    )

    df = df[
        df["gender"].isin(["F", "M"])
        & df["serve_number"].isin([1, 2])
    ].copy()

    print("\nDATASET")
    print(f"Initial rows: {n_initial}")
    print(f"Filtered rows: {len(df)}")
    print(f"Players: {df['server'].nunique()}")

    # ------------------------------------------------------------
    # 2. Group sizes before pose filtering
    # ------------------------------------------------------------

    print("\nGROUP SIZES BEFORE POSE FILTERING")

    group_counts = (
        df.groupby(["gender", "serve_number"], observed=True)
        .agg(
            serves=("filename", "count"),
            players=("server", "nunique"),
        )
        .reset_index()
    )

    print(group_counts.to_string(index=False))

    # ------------------------------------------------------------
    # 3. Handedness
    # ------------------------------------------------------------

    df["hand"] = df["server"].map(hand)

    print(f"\nHandedness source: {HAND_SOURCE}")

    print(
        "Left-handed players:",
        sorted(LEFT_HANDED & set(df["server"])),
    )

    # ------------------------------------------------------------
    # 4. Player-level splits, one per gender
    # ------------------------------------------------------------

    split_map = {}

    for gender in ("F", "M"):

        gender_df = df[df["gender"] == gender].copy()

        train_set, test_set = make_gender_player_split(
            gender_df,
            gender,
        )

        split_map[gender] = {
            "train": train_set,
            "test": test_set,
        }

    df["split"] = ""

    for gender in ("F", "M"):

        mask = df["gender"] == gender

        df.loc[
            mask & df["server"].isin(split_map[gender]["train"]),
            "split",
        ] = "train"

        df.loc[
            mask & df["server"].isin(split_map[gender]["test"]),
            "split",
        ] = "test"

    assert df["split"].isin(["train", "test"]).all()

    # ------------------------------------------------------------
    # 5. Extract pose features once for the entire dataset
    # ------------------------------------------------------------

    print("\nEXTRACTING POSE FEATURES")

    print(
        f"FPS={FPS}, cutoff={CUTOFF} Hz, "
        f"order={SMOOTH_ORDER}, min frames={MIN_FRAMES}"
    )

    jf = compute_or_load_joint_features(df)

    df = df.join(jf)

    df["pose_ok"] = df["pose_status"].eq("ok")

    print("\nPOSE EXTRACTION")

    print(
        f"Usable: {int(df['pose_ok'].sum())} / {len(df)}"
    )

    print(
        f"Excluded: {int((~df['pose_ok']).sum())}"
    )

    if (~df["pose_ok"]).any():

        print(
            df.loc[
                ~df["pose_ok"],
                "pose_status",
            ].value_counts()
        )

    if not df["pose_ok"].any():
        sys.exit("ERROR: no usable pose sequences.")

    # ------------------------------------------------------------
    # 6. Build arm and body representations
    # ------------------------------------------------------------

    (
        rep,
        arm_cols,
        body_cols,
        arm_base_cols,
        body_base_cols,
    ) = build_representation_columns(df)

    df = pd.concat([df, rep], axis=1)

    print("\nFEATURES")
    print(f"Arm base features: {len(arm_base_cols)}")
    print(f"Body base features: {len(body_base_cols)}")

    print(
        "Acceleration features are excluded from the main "
        "arm/body comparison."
    )

    # ------------------------------------------------------------
    # 7. Evaluate four independent groups
    # ------------------------------------------------------------

    all_results = []
    all_comparisons = []
    all_bootstraps = []
    all_importances = []
    all_predictions = []
    all_summaries = []

    for gender in ("F", "M"):

        for serve_number in (1, 2):

            group_df = df[
                (df["gender"] == gender)
                & (df["serve_number"] == serve_number)
                & df["pose_ok"]
            ].copy()

            print("\nGROUP DATA")
            print(
                f"{gender}, serve {serve_number}: "
                f"{len(group_df)} valid serves, "
                f"{group_df['server'].nunique()} players"
            )

            if group_df.empty:
                print("WARNING: empty group; skipping.")
                continue

            result = run_group(
                group_df,
                gender,
                serve_number,
                arm_base_cols,
                body_base_cols,
            )

            if result is None:
                continue

            all_results.append(result["results"])
            all_comparisons.append(result["comparisons"])

            if not result["bootstrap"].empty:
                all_bootstraps.append(result["bootstrap"])

            all_importances.append(result["importances"])
            all_predictions.append(result["predictions"])
            all_summaries.append(result["summary"])

    # ------------------------------------------------------------
    # 8. Combine and save outputs
    # ------------------------------------------------------------

    if not all_results:
        sys.exit(
            "ERROR: no subgroup could be evaluated. "
            "Check group sizes and player splits."
        )

    results = pd.concat(all_results, ignore_index=True)

    comparisons = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    importances = pd.concat(
        all_importances,
        ignore_index=True,
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summaries = pd.DataFrame(all_summaries)

    results.to_csv(
        OUT_DIR / "results_by_group.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT_DIR / "model_comparisons_by_group.csv",
        index=False,
    )

    importances.to_csv(
        OUT_DIR / "feature_importances_by_group.csv",
        index=False,
    )

    predictions.to_csv(
        OUT_DIR / "test_predictions_by_group.csv",
        index=False,
    )

    summaries.to_csv(
        OUT_DIR / "group_sample_sizes.csv",
        index=False,
    )

    if all_bootstraps:

        bootstraps = pd.concat(
            all_bootstraps,
            ignore_index=True,
        )

        bootstraps.to_csv(
            OUT_DIR / "bootstrap_differences_by_group.csv",
            index=False,
        )

    # ------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------

    print("\n" + "=" * 100)
    print("FINAL SUMMARY: ALL FOUR GROUPS")
    print("=" * 100)

    print(
        results[
            [
                "group",
                "Model",
                "n_test",
                "n_test_players",
                "MAE",
                "RMSE",
                "R2",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nMAIN MODEL COMPARISONS")
    print("Negative dMAE/dRMSE indicates improvement.")

    print(
        comparisons[
            [
                "group",
                "Model",
                "Reference",
                "dMAE",
                "dRMSE",
                "dR2",
            ]
        ].round(3).to_string(index=False)
    )

    print("\nGROUP SAMPLE SIZES")
    print(summaries.to_string(index=False))

    print(f"\nAll outputs saved to: {OUT_DIR}")

    print(
        "\nFiles:"
        "\n- results_by_group.csv"
        "\n- model_comparisons_by_group.csv"
        "\n- bootstrap_differences_by_group.csv"
        "\n- feature_importances_by_group.csv"
        "\n- test_predictions_by_group.csv"
        "\n- group_sample_sizes.csv"
        "\n- joint_features_cache.csv"
        "\n- joint_features_cache.json"
    )

    print(
        "\nIMPORTANT:"
        "\n1. The mean baseline uses only the training subgroup mean."
        "\n2. All models are evaluated on exactly the same test serves."
        "\n3. Bootstrap comparisons are paired and grouped by test player."
        "\n4. R2 values should not be compared naively across groups, "
        "because target-speed variability differs between groups."
        "\n5. Each test subgroup contains relatively few players; "
        "results should be interpreted with caution."
    )


if __name__ == "__main__":
    main()
