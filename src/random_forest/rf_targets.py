#!/usr/bin/env python3
"""
Random Forest experiments for HYPOTHESIS.txt.

Targets:
    gender        : pose-only, player-level split
    serve_number  : pose and gender+height metadata, player-level split
    identity      : pose-only, match-level split, evaluated only on players seen in train

Representations:
    Arm / Body x
    Shape | Motion | Scaled Motion |
    Shape + Motion | Shape + Scaled Motion

Variants:
    gender / identity: pose only
    serve_number: pose and pose+meta

Bootstrap:
    Player-level for binary targets.
    Player/class-level for identity.

The helper modules are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import butter

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

OUT_DIR = PROJECT_DIR / "rf_targets_outputs"
FOLDS_JSON = PROJECT_DIR / "splits" / "folds.json"

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

N_BOOT = 2000
BOOT_SEED = 11
TOP_N_IMPORTANCES = 8

# Set only before looking at the final results.
EQUIV_MARGIN = None

FEATURE_PREFIXES = (
    "arm_",
    "body_",
    "shape_",
    "scarm_",
    "scbody_",
    "scale_",
)

COMMON_FORBIDDEN = {
    "server",
    "match_id",
    "returner",
    "filename",
    "serve_width",
    "serve_depth",
    "return_depth",
    "ace",
    "point_winner",
    "rally_count",
    "point_number",
    "set_no",
    "game_no",
    "quality_label",
}


# ============================================================
# LOCAL MODULE IMPORTS
# ============================================================

def load_local_module(module_name, filename):
    path = SCRIPT_DIR / filename

    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# Ensure the intended local handedness module is used.
hand_module = load_local_module("handedness", "handedness.py")

if hand_module is None:
    hand_module = load_local_module(
        "handedness",
        "serve_handedness.py",
    )

if hand_module is None:
    sys.exit(
        "ERROR: cannot find handedness.py or serve_handedness.py"
    )

if not hasattr(hand_module, "hand"):
    sys.exit("ERROR: handedness module has no function hand().")


# Core project modules.
try:
    import serve_data as sd
    import smoothing
except ImportError as exc:
    sys.exit(f"ERROR importing project modules: {exc}")


# Feature module can be named features.py or pose_feature.py.
feature_path = SCRIPT_DIR / "features.py"

if not feature_path.exists():
    feature_path = SCRIPT_DIR / "pose_feature.py"

if not feature_path.exists():
    sys.exit(
        "ERROR: cannot find features.py or pose_feature.py"
    )

spec = importlib.util.spec_from_file_location(
    "features",
    feature_path,
)

if spec is None or spec.loader is None:
    sys.exit("ERROR: cannot load feature module.")

features = importlib.util.module_from_spec(spec)
sys.modules["features"] = features
spec.loader.exec_module(features)

for function_name in ("build", "columns"):
    if not hasattr(features, function_name):
        sys.exit(
            f"ERROR: feature module has no {function_name}()"
        )

if not hasattr(features, "hand"):
    features.hand = hand_module.hand


FPS = float(sd.FPS)
CUTOFF = float(smoothing.DEFAULT_CUTOFF_HZ)

_b, _a = butter(2, CUTOFF / (FPS / 2))
MIN_FRAMES = 3 * max(len(_a), len(_b)) + 1


# ============================================================
# TARGET DEFINITIONS
# ============================================================

TARGETS = {
    "gender": dict(
        task="player",
        forbidden={"gender", "gender_M"},
        meta=[],
        baselines=[],
        default_variants=("pose",),
        allowed_variants=("pose",),
    ),

    "serve_number": dict(
        task="player",
        forbidden={"serve_number", "speed_kmh"},
        meta=["gender_M", "height_cm"],
        baselines=[
            (
                "Metadata: gender + height",
                "rf",
                ["gender_M", "height_cm"],
            ),
            (
                "Metadata: gender + height (logistic)",
                "logit",
                ["gender_M", "height_cm"],
            ),
        ],
        hyp="Metadata: gender + height",
        default_variants=("pose", "pose+meta"),
        allowed_variants=("pose", "pose+meta"),
    ),

    "identity": dict(
        task="match",
        forbidden=set(),
        meta=[],
        baselines=[],
        default_variants=("pose",),
        allowed_variants=("pose",),
    ),
}


MAJORITY = "Majority class"

REPRESENTATIONS = [
    ("Arm Shape", "arm", ("shape",)),
    ("Body Shape", "body", ("shape",)),

    ("Arm Motion", "arm", ("motion",)),
    ("Body Motion", "body", ("motion",)),

    ("Arm Scaled Motion", "arm", ("scaled",)),
    ("Body Scaled Motion", "body", ("scaled",)),

    ("Arm Shape + Motion", "arm", ("shape", "motion")),
    ("Body Shape + Motion", "body", ("shape", "motion")),

    ("Arm Shape + Scaled Motion", "arm", ("shape", "scaled")),
    ("Body Shape + Scaled Motion", "body", ("shape", "scaled")),
]

KIND_LABELS = [
    "Shape",
    "Motion",
    "Scaled Motion",
    "Shape + Motion",
    "Shape + Scaled Motion",
]


# ============================================================
# HASHING / CACHE
# ============================================================

def file_hash(path):
    path = Path(path)

    if not path.exists():
        return None

    return hashlib.md5(path.read_bytes()).hexdigest()


def dataframe_hash(df):
    values = pd.util.hash_pandas_object(
        df,
        index=True,
    ).values

    return hashlib.md5(values.tobytes()).hexdigest()


def filename_hash(meta):
    text = "\n".join(meta["filename"].astype(str))
    return hashlib.md5(text.encode()).hexdigest()


def keypoint_fingerprint(meta):
    """
    Fingerprint based on filenames, sizes and modification times.
    This avoids hashing several thousand large arrays on every run.
    """

    records = []

    keypoint_dir = Path(sd.DATA) / "keypoints"

    for filename in meta["filename"]:
        path = keypoint_dir / str(filename)

        if not path.exists():
            records.append((str(filename), None, None))
            continue

        stat = path.stat()

        records.append(
            (
                str(filename),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )

    return hashlib.md5(
        repr(records).encode()
    ).hexdigest()


def cache_key(meta):
    servers = sorted(meta["server"].astype(str).unique())

    hand_hash = hashlib.md5(
        ",".join(
            f"{server}:{features.hand(server)}"
            for server in servers
        ).encode()
    ).hexdigest()

    return {
        "cutoff": CUTOFF,
        "fps": FPS,
        "min_frames": MIN_FRAMES,
        "features_module": file_hash(features.__file__),
        "smoothing_module": file_hash(smoothing.__file__),
        "serve_data_module": file_hash(sd.__file__),
        "handedness_module": file_hash(hand_module.__file__),
        "metadata": dataframe_hash(meta),
        "filename_order": filename_hash(meta),
        "keypoint_files": keypoint_fingerprint(meta),
        "handedness": hand_hash,
    }


# ============================================================
# FEATURE GENERATION
# ============================================================

def compute_features(meta):
    meta = meta.reset_index(drop=True)

    arrays = []
    valid_indices = []
    excluded = {}

    keypoint_dir = Path(sd.DATA) / "keypoints"

    for i, row in meta.iterrows():
        filename = str(row["filename"])
        path = keypoint_dir / filename

        try:
            keypoints = np.load(path)

        except FileNotFoundError:
            excluded[filename] = "missing_file"
            continue

        except Exception as exc:
            excluded[filename] = (
                f"load_error_{type(exc).__name__}"
            )
            continue

        if (
            keypoints.ndim != 3
            or keypoints.shape[1:] != (17, 3)
        ):
            excluded[filename] = "bad_shape"
            continue

        if keypoints.shape[0] < MIN_FRAMES:
            excluded[filename] = "too_short"
            continue

        if not np.isfinite(keypoints).all():
            excluded[filename] = "non_finite"
            continue

        arrays.append(keypoints)
        valid_indices.append(i)

        if (i + 1) % 1000 == 0:
            print(
                f"  loaded {i + 1}/{len(meta)} keypoint files"
            )

    if not arrays:
        sys.exit("ERROR: no valid keypoint sequences.")

    meta_valid = (
        meta.iloc[valid_indices]
        .reset_index(drop=True)
    )

    print(
        f"Building features for {len(arrays)} serves "
        f"(cutoff {CUTOFF:g} Hz)..."
    )

    feature_df = features.build(
        meta_valid,
        arrays,
        CUTOFF,
    )

    feature_columns = [
        c for c in feature_df.columns
        if c.startswith(FEATURE_PREFIXES)
    ]

    finite = np.isfinite(
        feature_df[feature_columns].to_numpy(dtype=float)
    ).all(axis=1)

    for filename in feature_df.loc[
        ~finite, "filename"
    ]:
        excluded[str(filename)] = "non_finite_feature"

    feature_df = (
        feature_df.loc[finite]
        .reset_index(drop=True)
    )

    return feature_df, excluded


def get_features(meta, use_cache=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = OUT_DIR / "features_cache.parquet"
    cache_info = OUT_DIR / "features_cache.json"

    key = cache_key(meta)

    if use_cache and cache_file.exists() and cache_info.exists():
        try:
            saved = json.loads(cache_info.read_text())

            if saved.get("key") == key:
                print(f"Loading feature cache: {cache_file}")

                return (
                    pd.read_parquet(cache_file),
                    saved.get("excluded", {}),
                    "cache",
                )

        except Exception as exc:
            print(f"WARNING: cache could not be read: {exc}")

        print("Feature cache is stale. Recomputing.")

    feature_df, excluded = compute_features(meta)

    if use_cache:
        feature_df.to_parquet(cache_file)

        cache_info.write_text(
            json.dumps(
                {
                    "key": key,
                    "excluded": excluded,
                },
                indent=2,
            )
        )

    return feature_df, excluded, "computed"


# ============================================================
# SHARED SPLITS
# ============================================================

def fallback_fold_ids(meta, rep, task, n_folds=5):
    if task == "player":
        rows = np.arange(len(meta))
        column = "server"

    elif task == "match":
        n_matches = (
            meta.groupby("server")["match_id"]
            .nunique()
        )

        rows = np.where(
            meta["server"].isin(
                n_matches[n_matches >= 2].index
            )
        )[0]

        column = "match_id"

    else:
        raise ValueError(task)

    groups = meta[column].values[rows]
    unique_groups = np.unique(groups)

    order = dict(
        zip(
            unique_groups,
            np.random.RandomState(rep).permutation(
                len(unique_groups)
            ),
        )
    )

    group_ids = np.array([order[g] for g in groups])

    fold = np.full(len(meta), -1)

    splitter = GroupKFold(n_splits=n_folds)

    for k, (_, test_indices) in enumerate(
        splitter.split(rows, groups=group_ids)
    ):
        fold[rows[test_indices]] = k

    return fold


def load_split_module():
    for name in ("splits", "split"):
        try:
            module = importlib.import_module(name)

            if hasattr(module, "fold_ids"):
                return module

        except ImportError:
            continue

    return None


def validate_fold_groups(meta, fold, task):
    if task == "player":
        column = "server"
    else:
        column = "match_id"

    for k in np.unique(fold[fold >= 0]):
        test = meta.loc[fold == k]
        train = meta.loc[
            (fold >= 0) & (fold != k)
        ]

        overlap = (
            set(train[column])
            & set(test[column])
        )

        if overlap:
            raise ValueError(
                f"Group leakage in fold {k}: "
                f"{len(overlap)} overlapping {column} groups"
            )


def load_folds(meta, task, max_reps=None):
    json_path = Path(FOLDS_JSON)

    split_module = load_split_module()

    json_data = None
    json_folds = None
    json_valid = False

    if json_path.exists():
        json_data = json.loads(json_path.read_text())

        info = json_data.get("_info", {})

        if (
            info.get("n_serves") == len(meta)
            and task in json_data
        ):
            candidate = {
                int(r): np.asarray(v, dtype=int)
                for r, v in json_data[task].items()
            }

            fingerprint = info.get("filename_hash")

            if fingerprint is not None:
                json_valid = (
                    fingerprint == filename_hash(meta)
                )
            else:
                print(
                    "WARNING: folds.json has no filename fingerprint. "
                    "Checking against split module."
                )

                if split_module is not None:
                    json_valid = True

                    for r, fold in candidate.items():
                        expected = np.asarray(
                            split_module.fold_ids(
                                task,
                                r,
                                meta,
                            )
                        )

                        if not np.array_equal(fold, expected):
                            json_valid = False
                            break
                else:
                    print(
                        "WARNING: cannot verify legacy folds.json "
                        "against split.py."
                    )
                    json_valid = True

            if json_valid:
                json_folds = candidate

        else:
            print(
                "WARNING: folds.json does not match metadata "
                "or does not contain this task."
            )

    if json_folds is not None:
        folds = json_folds
        source = str(json_path)

    elif split_module is not None:
        n_reps = getattr(split_module, "N_REPS", 3)

        folds = {
            r: np.asarray(
                split_module.fold_ids(task, r, meta)
            )
            for r in range(n_reps)
        }

        source = f"{split_module.__name__}.py"

    else:
        folds = {
            r: fallback_fold_ids(meta, r, task)
            for r in range(3)
        }

        source = "fallback GroupKFold"

    reps = sorted(folds)

    if max_reps is not None:
        reps = reps[:max_reps]

    folds = {
        r: folds[r]
        for r in reps
    }

    for r, fold in folds.items():
        if len(fold) != len(meta):
            raise ValueError(
                f"Fold array length mismatch in repetition {r}"
            )

        if not set(np.unique(fold)).issubset(
            {-1, 0, 1, 2, 3, 4}
        ):
            raise ValueError(
                f"Unexpected fold IDs in repetition {r}"
            )

        validate_fold_groups(meta, fold, task)

    return folds, source


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def make_model(kind):
    if kind == "majority":
        return DummyClassifier(
            strategy="most_frequent"
        )

    if kind == "rf":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "classifier",
                    RandomForestClassifier(**RF_PARAMS),
                ),
            ]
        )

    if kind == "logit":
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000),
                ),
            ]
        )

    raise ValueError(kind)


# ============================================================
# FEATURE SELECTION
# ============================================================

def representation_columns(df, size, kinds):
    columns = []

    for kind in kinds:
        columns.extend(
            features.columns(df, kind, size)
        )

    columns = list(dict.fromkeys(columns))

    if not columns:
        raise ValueError(
            f"No feature columns for {size}, {kinds}"
        )

    return columns


def feature_group(column):
    if column in {
        "height_cm",
        "speed_kmh",
        "serve_number",
        "gender_M",
    }:
        return "metadata"

    if column.startswith("shape_"):
        return "shape"

    if column.startswith(("scarm_", "scbody_")):
        return "scaled_motion"

    return "motion"


# ============================================================
# METRICS
# ============================================================

def safe_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    return tuple(
        np.percentile(values, [2.5, 97.5])
    )


def binary_metrics(confusion):
    """
    confusion shape: (..., true_class, predicted_class)
    """

    confusion = np.asarray(
        confusion,
        dtype=float,
    )

    tp = np.stack(
        [
            confusion[..., 0, 0],
            confusion[..., 1, 1],
        ],
        axis=-1,
    )

    support = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    total = confusion.sum(axis=(-2, -1))

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        recall = np.where(
            support > 0,
            tp / support,
            np.nan,
        )

        precision = np.where(
            predicted > 0,
            tp / predicted,
            0.0,
        )

        denominator = precision + recall

        f1 = np.where(
            np.isnan(recall),
            np.nan,
            np.where(
                denominator > 0,
                2 * precision * recall / denominator,
                0.0,
            ),
        )

        accuracy = tp.sum(axis=-1) / total

    return {
        "accuracy": accuracy,
        "balanced_accuracy": np.nanmean(
            recall,
            axis=-1,
        ),
        "macro_f1": np.nanmean(
            f1,
            axis=-1,
        ),
    }


def multiclass_metrics(tp, support, predicted, draws=None):
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        recall = np.where(
            support > 0,
            tp / support,
            np.nan,
        )

        precision = np.where(
            predicted > 0,
            tp / predicted,
            0.0,
        )

        denominator = precision + recall

        f1 = np.where(
            np.isnan(recall),
            np.nan,
            np.where(
                denominator > 0,
                2 * precision * recall / denominator,
                0.0,
            ),
        )

    if draws is None:
        accuracy = tp.sum() / support.sum()

        return {
            "accuracy": accuracy,
            "balanced_accuracy": np.nanmean(recall),
            "macro_f1": np.nanmean(f1),
        }

    accuracy = (
        tp[draws].sum(axis=1)
        / support[draws].sum(axis=1)
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": np.nanmean(
            recall[draws],
            axis=1,
        ),
        "macro_f1": np.nanmean(
            f1[draws],
            axis=1,
        ),
    }


class BinaryEvaluator:
    def __init__(self, y, player_codes, n_players):
        self.y = y
        self.player_codes = player_codes
        self.n_units = n_players

    def player_confusions(self, predictions):
        index = (
            self.player_codes * 4
            + self.y * 2
            + predictions
        )

        return np.bincount(
            index,
            minlength=self.n_units * 4,
        ).reshape(
            self.n_units,
            2,
            2,
        ).astype(float)

    def aggregate(self, predictions):
        return sum(
            self.player_confusions(p)
            for p in predictions
        )

    def pooled(self, aggregate):
        return {
            key: float(value)
            for key, value in binary_metrics(
                aggregate.sum(axis=0)
            ).items()
        }

    def bootstrap(self, aggregate, draws):
        return binary_metrics(
            aggregate[draws].sum(axis=1)
        )

    def per_rep(self, predictions):
        return [
            {
                "accuracy": accuracy_score(
                    self.y,
                    p,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    self.y,
                    p,
                ),
                "macro_f1": f1_score(
                    self.y,
                    p,
                    average="macro",
                    zero_division=0,
                ),
            }
            for p in predictions
        ]


class IdentityEvaluator:
    def __init__(self, y, n_classes, labels):
        self.y = y
        self.n_classes = n_classes
        self.labels = labels
        self.n_units = n_classes

    def counts(self, prediction, mask):
        true = self.y[mask]
        pred = prediction[mask]

        tp = np.bincount(
            true[true == pred],
            minlength=self.n_classes,
        )

        support = np.bincount(
            true,
            minlength=self.n_classes,
        )

        predicted = np.bincount(
            pred,
            minlength=self.n_classes,
        )

        return np.vstack(
            [tp, support, predicted]
        ).astype(float)

    def aggregate(self, predictions, masks):
        result = np.zeros(
            (3, self.n_classes),
            dtype=float,
        )

        for prediction, mask in zip(
            predictions,
            masks,
        ):
            result += self.counts(
                prediction,
                mask,
            )

        return result

    def pooled(self, aggregate):
        return {
            key: float(value)
            for key, value in multiclass_metrics(
                *aggregate
            ).items()
        }

    def bootstrap(self, aggregate, draws):
        return multiclass_metrics(
            *aggregate,
            draws=draws,
        )

    def per_rep(self, predictions, masks):
        rows = []

        for prediction, mask in zip(
            predictions,
            masks,
        ):
            rows.append(
                multiclass_metrics(
                    *self.counts(prediction, mask)
                )
            )

        return rows


# ============================================================
# PAIRED COMPARISONS
# ============================================================

def build_pairs(names, has_metadata):
    """
    Build paired comparisons with like-for-like references.

    Pose-only models are compared with the majority baseline.
    For serve number, metadata + pose models are compared with the
    metadata-only baseline using exactly gender + height.
    """
    pairs = []

    for representation, _, _ in REPRESENTATIONS:
        if representation in names:
            pairs.append(
                (representation, MAJORITY, "pose only vs majority")
            )

        if has_metadata:
            combined = representation + " + metadata"
            baseline = "Metadata: gender + height"
            if combined in names and baseline in names:
                pairs.append(
                    (
                        combined,
                        baseline,
                        "pose + metadata vs matching metadata baseline",
                    )
                )

    for suffix, description in (
        ("", "pose only"),
        (" + metadata", "pose + metadata"),
    ):
        if suffix and not has_metadata:
            continue

        for label in KIND_LABELS:
            body = f"Body {label}{suffix}"
            arm = f"Arm {label}{suffix}"
            if body in names and arm in names:
                pairs.append(
                    (body, arm, f"body minus arm ({description})")
                )

        for size in ("Arm", "Body"):
            comparisons = [
                (
                    f"{size} Shape + Motion{suffix}",
                    f"{size} Shape{suffix}",
                    f"motion added to shape ({description})",
                ),
                (
                    f"{size} Shape + Scaled Motion{suffix}",
                    f"{size} Shape{suffix}",
                    f"scaled motion added to shape ({description})",
                ),
                (
                    f"{size} Shape + Scaled Motion{suffix}",
                    f"{size} Scaled Motion{suffix}",
                    f"shape added to scaled motion ({description})",
                ),
                (
                    f"{size} Scaled Motion{suffix}",
                    f"{size} Shape{suffix}",
                    f"scaled motion vs shape ({description})",
                ),
            ]
            pairs.extend(
                pair for pair in comparisons
                if pair[0] in names and pair[1] in names
            )

    return pairs


# ============================================================
# ONE TARGET
# ============================================================

def run_target(target, args, meta, all_features, original_positions):
    config = TARGETS[target]

    output_dir = OUT_DIR / target
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    print(
        "\n"
        + "=" * 90
        + f"\nTARGET: {target} | split: {config['task']}\n"
        + "=" * 90
    )

    folds_full, split_source = load_folds(
        meta,
        config["task"],
        args.reps,
    )

    repetitions = sorted(folds_full)

    fold_arrays = {
        rep: folds_full[rep][original_positions]
        for rep in repetitions
    }

    n_folds = 5

    # --------------------------------------------------------
    # Select rows
    # --------------------------------------------------------

    if target == "gender":
        selected = all_features["gender"].isin(
            ["M", "F"]
        ).to_numpy()

    elif target == "serve_number":
        selected = all_features["serve_number"].isin(
            [1, 2]
        ).to_numpy()

    else:
        selected = fold_arrays[repetitions[0]] >= 0

        for rep in repetitions:
            if not np.array_equal(
                fold_arrays[rep] >= 0,
                selected,
            ):
                raise ValueError(
                    "Identity task rows differ between repetitions."
                )

    selected_indices = np.flatnonzero(selected)

    df = (
        all_features.iloc[selected_indices]
        .reset_index(drop=True)
    )

    fold_rows = {
        rep: fold_arrays[rep][selected_indices]
        for rep in repetitions
    }

    for rep in repetitions:
        if (fold_rows[rep] < 0).any():
            raise ValueError(
                f"Unexpected fold -1 in target {target}, rep {rep}"
            )

    group_column = (
        "server"
        if config["task"] == "player"
        else "match_id"
    )

    groups = df[group_column].to_numpy()
    servers = df["server"].to_numpy()

    player_ids, player_codes = np.unique(
        servers,
        return_inverse=True,
    )

    # --------------------------------------------------------
    # Target labels
    # --------------------------------------------------------

    if target == "gender":
        labels = ["F", "M"]

        y = (
            df["gender"] == "M"
        ).astype(int).to_numpy()

        evaluator = BinaryEvaluator(
            y,
            player_codes,
            len(player_ids),
        )

    elif target == "serve_number":
        labels = ["1", "2"]

        y = (
            df["serve_number"] == 2
        ).astype(int).to_numpy()

        evaluator = BinaryEvaluator(
            y,
            player_codes,
            len(player_ids),
        )

    else:
        labels, y = np.unique(
            servers,
            return_inverse=True,
        )

        labels = [str(value) for value in labels]

        evaluator = IdentityEvaluator(
            y,
            len(labels),
            labels,
        )

    n = len(df)

    print(
        f"Sample: {n} serves / {len(player_ids)} players"
    )

    print(
        f"Split source: {split_source} | "
        f"{len(repetitions)} repetitions x {n_folds} folds"
    )

    if target != "identity":
        print(
            "Class counts:",
            dict(zip(labels, np.bincount(y))),
        )
    else:
        print(
            f"Identity classes: {len(labels)}"
        )

    # --------------------------------------------------------
    # Model specifications
    # --------------------------------------------------------

    variants = (
        [value.strip() for value in args.variants.split(",") if value.strip()]
        if args.variants
        else list(config["default_variants"])
    )

    invalid_variants = set(variants) - set(config["allowed_variants"])
    if invalid_variants:
        raise ValueError(
            f"Target '{target}' does not support variants "
            f"{sorted(invalid_variants)}. Allowed: "
            f"{list(config['allowed_variants'])}"
        )

    specifications = [
        {
            "name": MAJORITY,
            "kind": "majority",
            "columns": [],
            "group": "baseline",
            "X": np.zeros((n, 1)),
        }
    ]

    for name, kind, columns in config["baselines"]:
        specifications.append(
            {
                "name": name,
                "kind": kind,
                "columns": columns,
                "group": "baseline",
            }
        )

    for name, size, kinds in REPRESENTATIONS:
        pose_columns = representation_columns(
            df,
            size,
            kinds,
        )

        if "pose" in variants:
            specifications.append(
                {
                    "name": name,
                    "kind": "rf",
                    "columns": pose_columns,
                    "group": size,
                }
            )

        if "pose+meta" in variants:
            specifications.append(
                {
                    "name": f"{name} + metadata",
                    "kind": "rf",
                    "columns": (
                        config["meta"] + pose_columns
                    ),
                    "group": size,
                }
            )

    for spec in specifications:
        if "X" not in spec:
            spec["X"] = df[
                spec["columns"]
            ].to_numpy(dtype=float)

        forbidden = (
            COMMON_FORBIDDEN | config["forbidden"]
        ) & set(spec["columns"])

        if forbidden:
            raise ValueError(
                f"Forbidden predictors in {spec['name']}: "
                f"{sorted(forbidden)}"
            )

        pose_columns = [
            c for c in spec["columns"]
            if c not in {
                "height_cm",
                "speed_kmh",
                "serve_number",
                "gender_M",
            }
        ]

        if pose_columns:
            values = df[pose_columns].to_numpy(
                dtype=float
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite pose values in {spec['name']}"
                )

    print("\nModels:")

    for spec in specifications:
        print(
            f"  {spec['name']}: "
            f"{len(spec['columns'])} features"
        )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    R = len(repetitions)

    predictions = {
        spec["name"]: np.full(
            (R, n),
            -1,
            dtype=int,
        )
        for spec in specifications
    }

    evaluation_masks = np.ones(
        (R, n),
        dtype=bool,
    )

    importances = {
        spec["name"]: []
        for spec in specifications
        if spec["kind"] == "rf"
    }

    identity_coverage = []

    print("\nTraining models...")

    for repetition_index, repetition in enumerate(
        repetitions
    ):
        fold = fold_rows[repetition]

        for k in range(n_folds):
            test = np.flatnonzero(fold == k)

            train = np.flatnonzero(
                (fold >= 0) & (fold != k)
            )

            if len(test) == 0 or len(train) == 0:
                raise ValueError(
                    f"Empty train/test fold: rep={repetition}, fold={k}"
                )

            if set(groups[train]) & set(groups[test]):
                raise ValueError(
                    f"Group leakage in rep={repetition}, fold={k}"
                )

            if target == "identity":
                seen_players = np.unique(y[train])

                evaluation_masks[
                    repetition_index,
                    test,
                ] = np.isin(
                    y[test],
                    seen_players,
                )

                evaluable = evaluation_masks[
                    repetition_index,
                    test,
                ]

                identity_coverage.append(
                    {
                        "rep": repetition,
                        "fold": k,
                        "test_serves": len(test),
                        "evaluable_serves": int(evaluable.sum()),
                        "test_players": len(
                            np.unique(y[test])
                        ),
                        "evaluable_players": len(
                            np.unique(y[test][evaluable])
                        ),
                    }
                )

            for spec in specifications:
                model = make_model(
                    spec["kind"]
                )

                model.fit(
                    spec["X"][train],
                    y[train],
                )

                predicted = model.predict(
                    spec["X"][test]
                )

                predictions[
                    spec["name"]
                ][repetition_index, test] = predicted

                if spec["kind"] == "rf":
                    importances[
                        spec["name"]
                    ].append(
                        model.named_steps[
                            "classifier"
                        ].feature_importances_
                    )

        print(
            f"  repetition {repetition} complete "
            f"({time.time() - start_time:.0f}s)"
        )

    for name, array in predictions.items():
        if (array < 0).any():
            raise ValueError(
                f"Missing OOF predictions for {name}"
            )

    if target == "identity":
        coverage_df = pd.DataFrame(
            identity_coverage
        )

        coverage_df.to_csv(
            output_dir / "identity_fold_coverage.csv",
            index=False,
        )

        print("\nIdentity evaluation coverage:")
        print(
            coverage_df.to_string(index=False)
        )

    # --------------------------------------------------------
    # Bootstrap draws
    # --------------------------------------------------------

    rng = np.random.default_rng(BOOT_SEED)

    draws = rng.integers(
        0,
        evaluator.n_units,
        size=(N_BOOT, evaluator.n_units),
    )

    aggregate = {}
    bootstrap_metrics = {}
    result_rows = []
    confusion_rows = []

    for spec in specifications:
        name = spec["name"]

        if target == "identity":
            aggregate[name] = evaluator.aggregate(
                predictions[name],
                evaluation_masks,
            )
        else:
            aggregate[name] = evaluator.aggregate(
                predictions[name]
            )

        bootstrap_metrics[name] = evaluator.bootstrap(
            aggregate[name],
            draws,
        )

        if target == "identity":
            per_rep = evaluator.per_rep(
                predictions[name],
                evaluation_masks,
            )
        else:
            per_rep = evaluator.per_rep(
                predictions[name]
            )

        per_rep_df = pd.DataFrame(per_rep)

        # Point estimates are computed from the same pooled
        # aggregate used by the bootstrap.
        pooled = evaluator.pooled(
            aggregate[name]
        )

        row = {
            "target": target,
            "model": name,
            "group": spec["group"],
            "n_features": len(spec["columns"]),
        }

        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ):
            row[metric] = pooled[metric]

            row[f"{metric}_mean_reps"] = (
                per_rep_df[metric].mean()
            )

            row[f"{metric}_sd_reps"] = (
                per_rep_df[metric].std(ddof=1)
                if len(per_rep_df) > 1
                else np.nan
            )

            low, high = safe_ci(
                bootstrap_metrics[name][metric]
            )

            row[f"{metric}_lo95"] = low
            row[f"{metric}_hi95"] = high

        result_rows.append(row)

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    for spec in specifications:
        name = spec["name"]

        if target != "identity":
            confusion = aggregate[name].sum(axis=0)

            for true_class in (0, 1):
                for predicted_class in (0, 1):
                    confusion_rows.append(
                        {
                            "model": name,
                            "true": labels[true_class],
                            "predicted": labels[predicted_class],
                            "count": int(
                                confusion[
                                    true_class,
                                    predicted_class,
                                ]
                            ),
                        }
                    )

        else:
            confusion = np.zeros(
                (
                    len(labels),
                    len(labels),
                ),
                dtype=int,
            )

            for rep_index in range(R):
                mask = evaluation_masks[rep_index]

                np.add.at(
                    confusion,
                    (
                        y[mask],
                        predictions[name][rep_index][mask],
                    ),
                    1,
                )

            for i, j in np.argwhere(confusion > 0):
                confusion_rows.append(
                    {
                        "model": name,
                        "true": labels[i],
                        "predicted": labels[j],
                        "count": int(confusion[i, j]),
                    }
                )

    results = pd.DataFrame(result_rows)

    results["n_serves"] = n
    results["n_players"] = len(player_ids)
    results["n_repetitions"] = R
    results["n_folds"] = n_folds
    results["split_source"] = split_source
    results["variants"] = ",".join(variants)

    results.to_csv(
        output_dir / "results.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_rows
    ).to_csv(
        output_dir / "confusion_matrices.csv",
        index=False,
    )

    print("\nRESULTS")
    print(
        results[
            [
                "model",
                "n_features",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "balanced_accuracy_lo95",
                "balanced_accuracy_hi95",
            ]
        ].round(3).to_string(index=False)
    )

    # --------------------------------------------------------
    # Paired bootstrap differences
    # --------------------------------------------------------

    model_names = {
        spec["name"]
        for spec in specifications
    }

    pairs = build_pairs(
        model_names,
        has_metadata=("pose+meta" in variants and bool(config["meta"])),
    )

    difference_rows = []

    for new, reference, relation in pairs:
        new_pooled = evaluator.pooled(
            aggregate[new]
        )

        reference_pooled = evaluator.pooled(
            aggregate[reference]
        )

        row = {
            "target": target,
            "new": new,
            "reference": reference,
            "relation": relation,
        }

        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ):
            difference = (
                bootstrap_metrics[new][metric]
                - bootstrap_metrics[reference][metric]
            )

            low, high = safe_ci(difference)

            row[f"d_{metric}"] = (
                new_pooled[metric]
                - reference_pooled[metric]
            )

            row[f"d_{metric}_lo95"] = low
            row[f"d_{metric}_hi95"] = high

            valid = difference[
                np.isfinite(difference)
            ]

            row[
                f"d_{metric}_share_positive"
            ] = (
                float((valid > 0).mean())
                if len(valid)
                else np.nan
            )

        row["ci_excludes_0_bal_acc"] = bool(
            row["d_balanced_accuracy_lo95"] > 0
            or row["d_balanced_accuracy_hi95"] < 0
        )

        if (
            EQUIV_MARGIN is not None
            and relation.startswith("body minus arm")
        ):
            row["within_margin_bal_acc"] = bool(
                row["d_balanced_accuracy_lo95"]
                > -EQUIV_MARGIN
                and row["d_balanced_accuracy_hi95"]
                < EQUIV_MARGIN
            )

        difference_rows.append(row)

    differences = pd.DataFrame(
        difference_rows
    )

    differences.to_csv(
        output_dir / "bootstrap_differences.csv",
        index=False,
    )

    print("\nPAIRED BOOTSTRAP DIFFERENCES")

    if len(differences):
        print(
            differences[
                [
                    "new",
                    "reference",
                    "relation",
                    "d_balanced_accuracy",
                    "d_balanced_accuracy_lo95",
                    "d_balanced_accuracy_hi95",
                    "ci_excludes_0_bal_acc",
                ]
            ].round(3).to_string(index=False)
        )

    # --------------------------------------------------------
    # Feature importances
    # --------------------------------------------------------

    importance_rows = []

    for name, fitted_importances in importances.items():
        if not fitted_importances:
            continue

        columns = next(
            spec["columns"]
            for spec in specifications
            if spec["name"] == name
        )

        values = np.vstack(
            fitted_importances
        )

        table = pd.DataFrame(
            {
                "target": target,
                "model": name,
                "feature": columns,
                "importance_mean": values.mean(axis=0),
                "importance_sd_fits": (
                    values.std(axis=0, ddof=1)
                    if len(values) > 1
                    else np.nan
                ),
            }
        )

        table["group"] = table["feature"].map(
            feature_group
        )

        importance_rows.append(table)

    if importance_rows:
        importance_df = pd.concat(
            importance_rows,
            ignore_index=True,
        )

        importance_df.to_csv(
            output_dir / "feature_importances.csv",
            index=False,
        )

    # --------------------------------------------------------
    # OOF predictions
    # --------------------------------------------------------

    oof_rows = []

    for rep_index, rep in enumerate(repetitions):
        for i in range(n):
            row = {
                "rep": rep,
                "filename": df.iloc[i]["filename"],
                "server": servers[i],
                "match_id": df.iloc[i]["match_id"],
                "true": labels[y[i]],
                "evaluable": bool(
                    evaluation_masks[rep_index, i]
                ),
            }

            for name in predictions:
                row[name] = labels[
                    predictions[name][rep_index, i]
                ]

            oof_rows.append(row)

    pd.DataFrame(
        oof_rows
    ).to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Identity per-player results
    # --------------------------------------------------------

    if target == "identity":
        rows = []

        for spec in specifications:
            counts = aggregate[spec["name"]]

            tp, support, predicted = counts

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):
                recall = np.where(
                    support > 0,
                    tp / support,
                    np.nan,
                )

                precision = np.where(
                    predicted > 0,
                    tp / predicted,
                    0.0,
                )

            for i, label in enumerate(labels):
                rows.append(
                    {
                        "model": spec["name"],
                        "player": label,
                        "support": support[i],
                        "correct": tp[i],
                        "recall": recall[i],
                        "precision": precision[i],
                    }
                )

        pd.DataFrame(rows).to_csv(
            output_dir / "identity_per_player.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Run information
    # --------------------------------------------------------

    run_info = {
        "target": target,
        "n_serves": n,
        "n_players": len(player_ids),
        "n_repetitions": R,
        "n_folds": n_folds,
        "split_source": split_source,
        "rf_params": RF_PARAMS,
        "cutoff_hz": CUTOFF,
        "fps": FPS,
        "min_frames": MIN_FRAMES,
        "n_boot": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "variants": variants,
        "metadata_features": list(config["meta"]),
        "metadata_baselines": [
            {
                "name": name,
                "kind": kind,
                "columns": columns,
            }
            for name, kind, columns in config["baselines"]
        ],
        "equivalence_margin": EQUIV_MARGIN,
        "feature_module": str(features.__file__),
        "feature_columns": {
            spec["name"]: spec["columns"]
            for spec in specifications
        },
        "excluded_files": int(
            len(args.excluded)
        ),
        "identity_evaluation_coverage": (
            identity_coverage
            if target == "identity"
            else None
        ),
        "seconds": round(
            time.time() - start_time,
            2,
        ),
    }

    (output_dir / "run_info.json").write_text(
        json.dumps(
            run_info,
            indent=2,
            default=str,
        )
    )

    print(
        f"\nSaved results in: {output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        choices=list(TARGETS) + ["all"],
        required=True,
    )

    parser.add_argument(
        "--variants",
        default=None,
        help="pose,pose+meta",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
    )

    args = parser.parse_args()

    global RF_PARAMS
    global N_BOOT
    global EQUIV_MARGIN

    if args.quick:
        RF_PARAMS = {
            **RF_PARAMS,
            "n_estimators": 50,
        }

        N_BOOT = 200
        args.reps = 1

    if args.n_boot is not None:
        if args.n_boot < 1:
            parser.error("--n-boot must be positive")

        N_BOOT = args.n_boot

    if args.reps is not None and args.reps < 1:
        parser.error("--reps must be positive")

    if args.margin is not None and args.margin <= 0:
        parser.error("--margin must be positive")

    EQUIV_MARGIN = args.margin

    if args.variants:
        variants = {
            value.strip()
            for value in args.variants.split(",")
            if value.strip()
        }

        invalid = variants - {"pose", "pose+meta"}
        if invalid:
            parser.error(f"Unknown variants: {sorted(invalid)}")

        if not variants:
            parser.error("At least one variant is required")

        requested_targets = (
            list(TARGETS) if args.target == "all" else [args.target]
        )
        for target in requested_targets:
            unsupported = variants - set(TARGETS[target]["allowed_variants"])
            if unsupported:
                parser.error(
                    f"Target '{target}' does not support variants "
                    f"{sorted(unsupported)}. Allowed: "
                    f"{list(TARGETS[target]['allowed_variants'])}"
                )

    pd.set_option(
        "display.width",
        220,
    )

    pd.set_option(
        "display.max_columns",
        40,
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    try:
        meta = sd.load_metadata().reset_index(drop=True)

    except Exception as exc:
        sys.exit(
            f"ERROR loading metadata: {exc}"
        )

    required = [
        "filename",
        "server",
        "match_id",
        "gender",
        "serve_number",
    ]

    # Height is needed only for the serve-number metadata baseline.
    if args.target in ("serve_number", "all"):
        required.append("height_cm")

    missing = [
        column for column in required
        if column not in meta.columns
    ]

    if missing:
        sys.exit(
            f"ERROR: metadata missing columns {missing}"
        )

    if meta["filename"].duplicated().any():
        sys.exit(
            "ERROR: duplicate filenames in metadata"
        )

    print(
        f"Metadata: {len(meta)} serves | "
        f"{meta['server'].nunique()} players | "
        f"{meta['match_id'].nunique()} matches"
    )

    # --------------------------------------------------------
    # Feature generation
    # --------------------------------------------------------

    feature_df, excluded, cache_status = get_features(
        meta,
        use_cache=not args.no_cache,
    )

    print(
        f"Features: {cache_status} | "
        f"excluded serves: {len(excluded)}"
    )

    if excluded:
        print(
            pd.Series(excluded)
            .value_counts()
            .to_string()
        )

    if len(excluded) / len(meta) > 0.02:
        print(
            "WARNING: more than 2% of serves were excluded."
        )

    # Numeric metadata.
    for column in ("height_cm", "serve_number"):
        if column in feature_df.columns:
            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            )

    feature_df["gender_M"] = np.where(
        feature_df["gender"] == "M",
        1.0,
        np.where(
            feature_df["gender"] == "F",
            0.0,
            np.nan,
        ),
    )

    # Map valid feature rows back to the original metadata order.
    positions = pd.Series(
        np.arange(len(meta)),
        index=meta["filename"],
    ).loc[
        feature_df["filename"]
    ].to_numpy()

    if not (
        set(feature_df["filename"])
        <= set(meta["filename"])
    ):
        sys.exit(
            "ERROR: feature filenames do not match metadata."
        )

    args.excluded = excluded

    # --------------------------------------------------------
    # Run requested targets
    # --------------------------------------------------------

    targets = (
        list(TARGETS)
        if args.target == "all"
        else [args.target]
    )

    for target in targets:
        run_target(
            target,
            args,
            meta,
            feature_df,
            positions,
        )

    print("\nAll requested experiments completed.")


if __name__ == "__main__":
    main()
