"""Loading helpers for the tennis serve dataset.

Real metadata columns (they differ from the README/docs): filename, match_id,
server, returner, gender, height_cm, point_number, point_winner, serve_number,
speed_kmh, serve_width, serve_depth, return_depth, n_frames, duration_sec,
quality_label, ace, rally_count, set_no, game_no.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tennis_serve_data"

FPS = 60.0       # n_frames / duration_sec is exactly 60 for every serve


def load_metadata(data: Path = DATA) -> pd.DataFrame:
    """Metadata with one row per serve; raises if a keypoint file is missing."""
    meta = pd.read_parquet(data / "metadata.parquet")
    present = set(p.name for p in (data / "keypoints").glob("*.npy"))
    missing = ~meta["filename"].isin(present)
    if missing.any():
        raise FileNotFoundError(f"{missing.sum()} metadata rows have no keypoint file")
    return meta.reset_index(drop=True)


def load_keypoints(meta: pd.DataFrame, data: Path = DATA) -> list:
    """List of (T, 17, 3) float32 arrays, in the same order as meta rows."""
    return [np.load(data / "keypoints" / f) for f in meta["filename"]]
