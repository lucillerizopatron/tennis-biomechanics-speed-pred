"""Shared Random Forest code: paths, features (cached), shared folds, models, bootstrap.

Used by rf_classify.py and rf_speed.py. Folds come from splits/splits.py, the same
shared 5-fold splits (3 repeats) as the transformer.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent            # src/random_forest
SRC = HERE.parent                                  # src
ROOT = SRC.parent                                  # repository root
for p in (HERE, SRC, ROOT / "splits"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import serve_data as sd   # noqa: E402
import splits             # noqa: E402
import rf_features        # noqa: E402

RESULTS = ROOT / "results" / "random_forest"
CACHE = ROOT / "results" / "cache"

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
N_BOOT, BOOT_SEED = 2000, 11

# the four models shown on the slides, plus motion-only and scaled-motion variants
REPRESENTATIONS = [
    ("Arm Shape", "arm", ("shape",)), ("Body Shape", "body", ("shape",)),
    ("Arm Motion", "arm", ("motion",)), ("Body Motion", "body", ("motion",)),
    ("Arm Scaled Motion", "arm", ("scaled",)), ("Body Scaled Motion", "body", ("scaled",)),
    ("Arm Shape + Motion", "arm", ("shape", "motion")), ("Body Shape + Motion", "body", ("shape", "motion")),
    ("Arm Shape + Scaled Motion", "arm", ("shape", "scaled")),
    ("Body Shape + Scaled Motion", "body", ("shape", "scaled")),
]
KIND_LABELS = ["Shape", "Motion", "Scaled Motion", "Shape + Motion", "Shape + Scaled Motion"]


def representation_columns(df, size, kinds):
    cols = []
    for kind in kinds:
        cols += rf_features.columns(df, kind, size)
    return list(dict.fromkeys(cols))


# ------------------------------------------------------------------ data
def load_data():
    """Metadata + features for all serves, in metadata order (features cached)."""
    meta = sd.load_metadata().reset_index(drop=True)
    key = hashlib.md5(Path(rf_features.__file__).read_bytes()
                      + "\n".join(meta["filename"]).encode()).hexdigest()
    cache, info = CACHE / "rf_features.parquet", CACHE / "rf_features.json"
    if cache.exists() and info.exists() and json.loads(info.read_text()).get("key") == key:
        df = pd.read_parquet(cache)
    else:
        print(f"Computing features for {len(meta)} serves...", flush=True)
        df = rf_features.build(meta, sd.load_keypoints(meta))
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        info.write_text(json.dumps({"key": key}))
    df["gender_M"] = (df["gender"] == "M").astype(int)
    df["second"] = (df["serve_number"] == 2).astype(int)
    return df


def shared_folds(df, task, n_reps=None):
    """{rep: test fold (0-4, -1 = not in task) of every serve} from splits/splits.py."""
    meta = df[["server", "match_id"]]
    reps = range(splits.N_REPS if n_reps is None else n_reps)
    out = {r: np.asarray(splits.fold_ids(task, r, meta)) for r in reps}
    col = "server" if task == "player" else "match_id"
    for r, f in out.items():                      # no group in both training and test
        for k in range(splits.N_FOLDS):
            assert not set(df[col][(f >= 0) & (f != k)]) & set(df[col][f == k]), "group leakage"
    return out


# ------------------------------------------------------------------ models
def rf_classifier(class_weight=None):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(**RF_PARAMS, class_weight=class_weight))])


def logistic(class_weight=None):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight))])


def rf_regressor():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("reg", RandomForestRegressor(**RF_PARAMS))])


def linear():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", LinearRegression())])


# ------------------------------------------------------------------ bootstrap
def draw_weights(n_units, rng=None):
    """N_BOOT x n_units matrix: how often each unit (player) is drawn in each resample.
    Pass one generator to several calls to continue the same random stream."""
    rng = np.random.default_rng(BOOT_SEED) if rng is None else rng
    draws = rng.integers(0, n_units, size=(N_BOOT, n_units))
    return np.stack([np.bincount(d, minlength=n_units) for d in draws]).astype(float)


def ci(values):
    return np.nanpercentile(values, 2.5), np.nanpercentile(values, 97.5)
