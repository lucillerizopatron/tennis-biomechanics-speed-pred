"""Joint layout and serving-arm choice for the transformer inputs.

The 3D keypoints are in Human3.6M joint order (MotionBERT output), not COCO:
  0 pelvis, 1 R hip, 2 R knee, 3 R foot, 4 L hip, 5 L knee, 6 L foot,
  7 spine, 8 thorax, 9 neck, 10 head,
  11 L shoulder, 12 L elbow, 13 L wrist, 14 R shoulder, 15 R elbow, 16 R wrist.
Check: bones read this way are plausible and left/right match (thigh 0.35 m, upper arm 0.21 m).

The left/right labels do not follow handedness, so the serving arm is chosen per serve as the
side whose wrist has the higher peak speed (smoothed, hip-relative positions).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
import serve_data as sd
from smoothing import smooth

R_HIP, L_HIP = 1, 4
ARM = {"L": (11, 12, 13), "R": (14, 15, 16)}  # shoulder, elbow, wrist
TORSO = {"L": (11, L_HIP), "R": (14, R_HIP)}  # shoulder to hip


def hip_relative(k):
    centre = (k[:, L_HIP] + k[:, R_HIP]) / 2
    return k - centre[:, None, :]


def _bone(k, a, b):
    return float(np.median(np.linalg.norm(k[:, a] - k[:, b], axis=1)))


def serve_info(k):
    """k: (T,17,3) raw keypoints. Returns smoothed hip-relative positions and per-serve extras."""
    r = smooth(hip_relative(k))
    spd = {}
    for s in "LR":
        w = ARM[s][2]
        v = np.linalg.norm(np.diff(r[:, w], axis=0), axis=1) * sd.FPS
        spd[s] = (v.max(), v.mean(), v.sum() / sd.FPS)  # peak speed, mean speed, path length
    side = "L" if spd["L"][0] >= spd["R"][0] else "R"
    s, e, w = ARM[side]
    scale_arm = _bone(k, s, e) + _bone(k, e, w)
    scale_torso = (_bone(k, *TORSO["L"]) + _bone(k, *TORSO["R"])) / 2
    alt = [("L" if spd["L"][i] >= spd["R"][i] else "R") for i in (1, 2)]
    return r, side, scale_arm, scale_torso, alt
