"""Smoothing of keypoint trajectories
Default: zero-phase Butterworth low-pass, 2nd order, 8 Hz cutoff at 60 fps.
"""
import numpy as np
from scipy.signal import butter, filtfilt

import serve_data as sd

DEFAULT_CUTOFF_HZ = 8.0


def hip_relative(k: np.ndarray) -> np.ndarray:
    """Subtract the hip centre from every joint (coordinates are not root-centred)."""
    centre = (k[:, sd.L_HIP] + k[:, sd.R_HIP]) / 2
    return k - centre[:, None, :]


def smooth(k: np.ndarray, cutoff_hz: float = DEFAULT_CUTOFF_HZ, order: int = 2) -> np.ndarray:
    """Low-pass filter a (T, J, 3) trajectory along time (zero phase, no lag)."""
    b, a = butter(order, cutoff_hz / (sd.FPS / 2))
    return filtfilt(b, a, k, axis=0)


def speed(x: np.ndarray) -> np.ndarray:
    """Per-frame speed of a (T, 3) trajectory, in coordinate units per second."""
    return np.linalg.norm(np.diff(x, axis=0), axis=1) * sd.FPS
