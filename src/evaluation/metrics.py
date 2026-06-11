"""Depth evaluation metrics shared by all model runners."""

import math
from collections.abc import Iterable

import numpy as np


METRIC_NAMES = [
    "AbsRel",
    "SqRel",
    "RMSE",
    "RMSE_log",
    "MAE",
    "delta1",
    "delta2",
    "delta3",
    "log10",
    "SILog",
]

METRICS_PER_IMAGE_COLUMNS = [
    "sample_id",
    "scene_id",
    "image_path",
    "depth_path",
    "prediction_type",
    "alignment",
    "alignment_scale",
    "alignment_shift",
    "alignment_fit_pixels",
    "valid_pixels",
    "gt_mean_depth",
    "gt_min_depth",
    "gt_max_depth",
    "gt_depth_range",
    "inference_time_s",
    *METRIC_NAMES,
]


def valid_depth_mask(
    gt: np.ndarray,
    pred: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> np.ndarray:
    """Return the common valid mask used for metrics across all models."""

    mask = np.isfinite(gt) & np.isfinite(pred) & (gt > min_depth) & (pred > min_depth)
    if max_depth is not None:
        mask &= gt <= max_depth
    return mask


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> dict[str, float]:
    """Compute standard depth metrics on a shared valid mask."""

    mask = valid_depth_mask(gt, pred, min_depth=min_depth, max_depth=max_depth)
    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        return {"valid_pixels": 0}

    gt_valid = gt[mask].astype(np.float64)
    pred_valid = np.clip(pred[mask].astype(np.float64), min_depth, None)
    diff = gt_valid - pred_valid
    diff_log = np.log(pred_valid) - np.log(gt_valid)
    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)

    return {
        "valid_pixels": valid_pixels,
        "gt_mean_depth": float(np.mean(gt_valid)),
        "gt_min_depth": float(np.min(gt_valid)),
        "gt_max_depth": float(np.max(gt_valid)),
        "gt_depth_range": float(np.max(gt_valid) - np.min(gt_valid)),
        "AbsRel": float(np.mean(np.abs(diff) / gt_valid)),
        "SqRel": float(np.mean((diff**2) / gt_valid)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "RMSE_log": float(np.sqrt(np.mean(diff_log**2))),
        "MAE": float(np.mean(np.abs(diff))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25**2)),
        "delta3": float(np.mean(thresh < 1.25**3)),
        "log10": float(np.mean(np.abs(np.log10(pred_valid) - np.log10(gt_valid)))),
        "SILog": float(100.0 * np.std(diff_log)),
    }


def summarize_rows(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Return mean and standard deviation for numeric metric columns."""

    rows = list(rows)
    names = [
        "valid_pixels",
        "gt_mean_depth",
        "gt_min_depth",
        "gt_max_depth",
        "gt_depth_range",
        "inference_time_s",
        *METRIC_NAMES,
    ]
    averages: dict[str, float] = {}
    stddevs: dict[str, float] = {}
    for name in names:
        values = []
        for row in rows:
            value = row.get(name)
            if value in (None, ""):
                continue
            value = float(value)
            if math.isfinite(value):
                values.append(value)
        if values:
            averages[name] = float(np.mean(values))
            stddevs[name] = float(np.std(values))
    return {"avg_metrics": averages, "std_metrics": stddevs}
