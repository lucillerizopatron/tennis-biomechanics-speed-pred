"""Per-serve Random Forest features from the 3D keypoints.

The keypoints are in Human3.6M joint order (the output of MotionBERT), not COCO.
For every serve this module computes:
  - shape  : median bone lengths over the serve (body proportions; the 3D lifting
             largely normalises skeleton size, so these are proportions, not size)
  - motion : per joint path length, peak and mean speed, timing of the peak speed,
             peak and mean acceleration (hip-centred, 8 Hz low-pass filtered)
  - scaled motion : the distance/speed/acceleration features divided by arm length
             (serving arm) or torso length (full body)
for the serving arm (shoulder, elbow, wrist; 3 joints) and the full body (17 joints).

Serving arm: chosen per serve as the side whose wrist reaches the higher peak speed.
The left/right labels in the 3D data do not follow the players' handedness (they look
mirrored in about half of the serves), so a handedness table cannot be used.
The skeleton is relabelled so the serving side is always "s" and the other side "o",
which keeps full-body features comparable across serves.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FPS = 60.0
CUTOFF_HZ = 8.0

# Human3.6M joint indices
SIDE = {"r": {"hip": 1, "knee": 2, "ankle": 3, "shoulder": 14, "elbow": 15, "wrist": 16},
        "l": {"hip": 4, "knee": 5, "ankle": 6, "shoulder": 11, "elbow": 12, "wrist": 13}}
MID = {"pelvis": 0, "spine": 7, "thorax": 8, "neck": 9, "head": 10}
PARTS = ["hip", "knee", "ankle", "shoulder", "elbow", "wrist"]
JOINTS = list(MID) + [f"{s}_{p}" for s in ("s", "o") for p in PARTS]     # 17 joints

MOTION = ["path_length", "peak_speed", "mean_speed", "peak_time_norm", "peak_acc", "mean_acc"]
SCALED = ["path_length", "peak_speed", "mean_speed", "peak_acc", "mean_acc"]
BONES = {"s_upper_arm": ("s_shoulder", "s_elbow"), "o_upper_arm": ("o_shoulder", "o_elbow"),
         "s_forearm": ("s_elbow", "s_wrist"), "o_forearm": ("o_elbow", "o_wrist"),
         "s_thigh": ("s_hip", "s_knee"), "o_thigh": ("o_hip", "o_knee"),
         "s_shank": ("s_knee", "s_ankle"), "o_shank": ("o_knee", "o_ankle"),
         "shoulder_width": ("s_shoulder", "o_shoulder"), "hip_width": ("s_hip", "o_hip"),
         "s_torso": ("s_shoulder", "s_hip"), "o_torso": ("o_shoulder", "o_hip")}


def hip_relative(k):
    """Subtract the hip centre (mean of the two H36M hips) from every joint."""
    centre = (k[:, 1] + k[:, 4]) / 2
    return k - centre[:, None, :]


def smooth(k, cutoff=CUTOFF_HZ):
    """Zero-phase 2nd-order Butterworth low-pass along time."""
    b, a = butter(2, cutoff / (FPS / 2))
    return filtfilt(b, a, k, axis=0)


def peak_speed(x):
    return (np.linalg.norm(np.diff(x, axis=0), axis=1) * FPS).max()


def joint_index(serving):
    """Joint name -> H36M index, with the serving side labelled 's'."""
    other = "l" if serving == "r" else "r"
    idx = dict(MID)
    for p in PARTS:
        idx[f"s_{p}"] = SIDE[serving][p]
        idx[f"o_{p}"] = SIDE[other][p]
    return idx


def motion(r, j):
    v = np.diff(r[:, j], axis=0) * FPS
    spd = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(np.diff(v, axis=0), axis=1) * FPS
    return [spd.sum() / FPS, spd.max(), spd.mean(), spd.argmax() / len(spd), acc.max(), acc.mean()]


def bone(k, a, b):
    return float(np.median(np.linalg.norm(k[:, a] - k[:, b], axis=1)))


def serve_row(k, cutoff=CUTOFF_HZ):
    """All features of one serve; k has shape (frames, 17, 3)."""
    r = smooth(hip_relative(k.astype(float)), cutoff)
    serving = "r" if peak_speed(r[:, 16]) >= peak_speed(r[:, 13]) else "l"
    idx = joint_index(serving)
    row = {"serving_side": serving}
    for name in JOINTS:
        for f, v in zip(MOTION, motion(r, idx[name])):
            row[f"body_{name}_{f}"] = v
    for part in ("shoulder", "elbow", "wrist"):
        for f, v in zip(MOTION, motion(r, idx[f"s_{part}"])):
            row[f"arm_{part}_{f}"] = v
    for bn, (a, b) in BONES.items():
        row[f"shape_{bn}"] = bone(k, idx[a], idx[b])
    row["shape_arm_upper"] = row["shape_s_upper_arm"]
    row["shape_arm_fore"] = row["shape_s_forearm"]
    row["scale_arm"] = row["shape_arm_upper"] + row["shape_arm_fore"]
    row["scale_torso"] = (row["shape_s_torso"] + row["shape_o_torso"]) / 2
    return row


def build(meta, keypoints, cutoff=CUTOFF_HZ):
    """Metadata rows + features, one row per serve, in the order of `meta`."""
    df = pd.DataFrame([serve_row(k, cutoff) for k in keypoints])
    scaled = {}
    for prefix, scale in (("arm", "scale_arm"), ("body", "scale_torso")):
        for c in [c for c in df if c.startswith(prefix + "_")
                  and any(c.endswith("_" + f) for f in SCALED)]:
            scaled["sc" + c] = df[c] / df[scale]
    df = pd.concat([df, pd.DataFrame(scaled)], axis=1)
    return pd.concat([meta.reset_index(drop=True), df], axis=1)


def columns(df, kind, size):
    """Feature columns. kind: shape | motion | scaled ; size: arm | body."""
    if kind == "shape":
        if size == "arm":
            return ["shape_arm_upper", "shape_arm_fore"]
        return [c for c in df if c.startswith("shape_") and not c.startswith("shape_arm_")]
    if kind == "scaled":
        base = [c for c in df if c.startswith(size + "_") and c.endswith("peak_time_norm")]
        return [c for c in df if c.startswith("sc" + size + "_")] + base
    return [c for c in df if c.startswith(size + "_")]
