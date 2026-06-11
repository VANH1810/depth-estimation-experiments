"""Visualization helpers for depth prediction grids."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


def _colorize(array: np.ndarray, vmin: float, vmax: float, cmap: str) -> np.ndarray:
    data = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = colormaps[cmap](norm)[..., :3]
    return (rgb * 255.0).astype(np.uint8)


def _colorize_relative(array: np.ndarray, cmap: str = "magma_r") -> np.ndarray:
    finite = array[np.isfinite(array)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return np.zeros((*array.shape, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    return _colorize(array, float(vmin), float(vmax), cmap)


def _resize_rgb(rgb: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == target_shape:
        return rgb.astype(np.uint8, copy=False)
    width = target_shape[1]
    height = target_shape[0]
    return cv2.resize(rgb.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def _grid(images: list[np.ndarray]) -> np.ndarray:
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def save_depth_visualization(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    output_path: str | Path,
    min_depth: float,
    max_depth: float,
    prediction_type: str,
    alignment: str,
) -> None:
    """Save a 2x2 RGB/GT/prediction/error visualization."""

    target_shape = gt_depth.shape
    rgb_view = _resize_rgb(rgb, target_shape)
    gt_col = _colorize(gt_depth, min_depth, max_depth, "magma_r")

    if prediction_type == "relative" and alignment == "raw":
        pred_col = _colorize_relative(pred_depth)
        error_col = np.zeros_like(pred_col)
    else:
        pred_col = _colorize(pred_depth, min_depth, max_depth, "magma_r")
        valid = np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > min_depth) & (pred_depth > min_depth)
        error = np.zeros_like(gt_depth, dtype=np.float32)
        error[valid] = np.abs(gt_depth[valid] - pred_depth[valid]) / np.maximum(gt_depth[valid], 1e-6)
        error_col = _colorize(error, 0.0, 0.3, "coolwarm")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_grid([rgb_view, gt_col, pred_col, error_col])).save(output_path)
