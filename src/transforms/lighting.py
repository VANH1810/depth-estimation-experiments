"""Photometric transforms shared by robustness experiments."""

import numpy as np


def apply_brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Apply multiplicative brightness while preserving shape and dtype family."""

    if factor <= 0.0:
        raise ValueError(f"factor must be positive, got {factor}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")

    if np.issubdtype(rgb.dtype, np.integer):
        info = np.iinfo(rgb.dtype)
        bright = np.clip(rgb.astype(np.float32) * factor, info.min, info.max)
        return bright.round().astype(rgb.dtype)

    upper = 1.0 if float(np.nanmax(rgb)) <= 1.0 else 255.0
    bright = np.clip(rgb.astype(np.float32) * factor, 0.0, upper)
    return bright.astype(rgb.dtype, copy=False)
