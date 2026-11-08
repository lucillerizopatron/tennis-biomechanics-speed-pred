"""Per-serve features: motion summaries, bone lengths and size-scaled motion.

Motion features come from smoothed, hip-relative keypoints. Bone lengths are
body shape. Scaled motion divides the distance, speed and acceleration features
by a limb size (arm length for the serving arm, torso length for the body), so
that motion can be compared with shape without being a disguised size feature.
"""
import numpy as np
import pandas as pd

import serve_data as sd
from serve_data import FPS
from smoothing import hip_relative, smooth
from handedness import hand

JOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
          "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
          "l_knee", "r_knee", "l_ankle", "r_ankle"]
MOTION = ["path_length", "peak_speed", "mean_speed", "peak_time_norm",
          "peak_acc", "mean_acc"]
SCALED = ["path_length", "peak_speed", "mean_speed", "peak_acc", "mean_acc"]
# (shoulder, elbow, wrist) indices for the left and right arm
ARM = {"L": (5, 7, 9), "R": (6, 8, 10)}
BONES = {"l_upper_arm": (5, 7), "r_upper_arm": (6, 8), "l_forearm": (7, 9),
         "r_forearm": (8, 10), "l_thigh": (11, 13), "r_thigh": (12, 14),
         "l_shank": (13, 15), "r_shank": (14, 16), "shoulder_width": (5, 6),
         "hip_width": (11, 12), "l_torso": (5, 11), "r_torso": (6, 12)}


def bone(k, a, b):
    return float(np.median(np.linalg.norm(k[:, a] - k[:, b], axis=1)))


def joint_motion(r):
    """(T,17,3) -> dict joint index -> the MOTION features."""
    vel = np.diff(r, axis=0) * FPS
    spd = np.linalg.norm(vel, axis=2)
    acc = np.linalg.norm(np.diff(vel, axis=0), axis=2) * FPS
    out = {}
    for j in range(17):
        out[j] = [spd[:, j].sum() / FPS, spd[:, j].max(), spd[:, j].mean(),
                  spd[:, j].argmax() / len(spd), acc[:, j].max(), acc[:, j].mean()]
    return out


def serve_row(k, server, cutoff):
    h = hand(server)
    r = smooth(hip_relative(k), cutoff)
    mot = joint_motion(r)
    row = {}
    for j, name in enumerate(JOINTS):
        for f, v in zip(MOTION, mot[j]):
            row[f"body_{name}_{f}"] = v
    for part, j in zip(("shoulder", "elbow", "wrist"), ARM[h]):
        for f, v in zip(MOTION, mot[j]):
            row[f"arm_{part}_{f}"] = v
    for name, (a, b) in BONES.items():
        row[f"shape_{name}"] = bone(k, a, b)
    s, e, w = ARM[h]
    row["shape_arm_upper"] = bone(k, s, e)
    row["shape_arm_fore"] = bone(k, e, w)
    row["scale_arm"] = row["shape_arm_upper"] + row["shape_arm_fore"]
    row["scale_torso"] = (row["shape_l_torso"] + row["shape_r_torso"]) / 2
    # both wrists, so the serving-arm lookup is not needed for the pipeline check
    row["wrist_max_peak_speed"] = max(mot[9][1], mot[10][1])
    row["wrist_peak_time_norm"] = mot[9][3] if mot[9][1] >= mot[10][1] else mot[10][3]
    return row


def build(meta, kps, cutoff=8.0):
    rows = [serve_row(k, s, cutoff) for k, s in zip(kps, meta.server)]
    df = pd.DataFrame(rows)
    # scaled copies of the motion features (arm by arm length, body by torso)
    for prefix, scale in (("arm", "scale_arm"), ("body", "scale_torso")):
        for c in [c for c in df if c.startswith(prefix + "_")
                  and any(c.endswith("_" + f) for f in SCALED)]:
            df["sc" + c] = df[c] / df[scale]
    return pd.concat([meta.reset_index(drop=True), df], axis=1)


def columns(df, kind, size):
    """Feature columns. kind: motion | scaled | shape; size: arm | body."""
    if kind == "shape":
        if size == "arm":
            return ["shape_arm_upper", "shape_arm_fore"]
        return [c for c in df if c.startswith("shape_") and not c.startswith("shape_arm_")]
    if kind == "scaled":
        # timing features are already size-free, so keep them in the scaled set
        base = [c for c in df if c.startswith(size + "_") and c.endswith("peak_time_norm")]
        return [c for c in df if c.startswith("sc" + size + "_")] + base
    return [c for c in df if c.startswith(size + "_")]


if __name__ == "__main__":
    import sys
    meta = sd.load_metadata().reset_index(drop=True)
    kps = sd.load_keypoints(meta)
    for cutoff in [8.0, 6.0, 12.0]:
        df = build(meta, kps, cutoff)
        out = sd.ROOT / "results" / "cache" / f"pose_features_{int(cutoff)}hz.parquet"
        out.parent.mkdir(exist_ok=True)
        df.to_parquet(out)
        print(cutoff, df.shape, flush=True)
