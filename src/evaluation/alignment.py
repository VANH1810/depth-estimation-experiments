"""Alignment utilities for relative-depth evaluation."""

from dataclasses import dataclass

import numpy as np


ALIGNMENT_MODES = ("raw", "median_aligned", "scale_shift_aligned")


class AlignmentError(RuntimeError):
    """Raised when a requested alignment cannot be estimated safely."""


@dataclass(frozen=True)
class AlignmentResult:
    prediction: np.ndarray
    scale: float | None = None
    shift: float | None = None
    valid_pixels: int = 0
    applied: bool = False


def alignment_valid_mask(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> np.ndarray:
    """Return pixels valid for fitting alignment parameters."""

    mask = np.isfinite(gt) & np.isfinite(pred) & (gt > min_depth) & (pred > 0.0)
    if max_depth is not None:
        mask &= gt <= max_depth
    return mask


def median_scale_align(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    min_valid_pixels: int = 10,
) -> AlignmentResult:
    """Align relative depth with a single median scale factor."""

    valid_pixels = int(mask.sum())
    if valid_pixels < min_valid_pixels:
        raise AlignmentError(f"Not enough valid pixels for median alignment: {valid_pixels}")
    pred_valid = pred[mask].astype(np.float64)
    gt_valid = gt[mask].astype(np.float64)
    pred_median = float(np.median(pred_valid))
    gt_median = float(np.median(gt_valid))
    if not np.isfinite(pred_median) or abs(pred_median) < 1e-12:
        raise AlignmentError("Predicted median is zero or non-finite; cannot median-align.")
    scale = gt_median / pred_median
    return AlignmentResult(
        prediction=(pred.astype(np.float32) * np.float32(scale)).astype(np.float32),
        scale=float(scale),
        shift=0.0,
        valid_pixels=valid_pixels,
        applied=True,
    )


def scale_shift_align(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    min_valid_pixels: int = 10,
) -> AlignmentResult:
    """Fit gt ~= scale * pred + shift over valid pixels."""

    valid_pixels = int(mask.sum())
    if valid_pixels < min_valid_pixels:
        raise AlignmentError(f"Not enough valid pixels for scale-shift alignment: {valid_pixels}")
    pred_valid = pred[mask].astype(np.float64)
    gt_valid = gt[mask].astype(np.float64)
    design = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, gt_valid, rcond=None)
    if rank < 2:
        raise AlignmentError("Scale-shift alignment is rank deficient.")
    scale, shift = float(solution[0]), float(solution[1])
    if not np.isfinite(scale) or not np.isfinite(shift):
        raise AlignmentError("Scale-shift alignment produced non-finite parameters.")
    aligned = pred.astype(np.float32) * np.float32(scale) + np.float32(shift)
    return AlignmentResult(
        prediction=aligned.astype(np.float32),
        scale=scale,
        shift=shift,
        valid_pixels=valid_pixels,
        applied=True,
    )


def apply_alignment(
    pred: np.ndarray,
    gt: np.ndarray,
    mode: str,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    min_valid_pixels: int = 10,
) -> AlignmentResult:
    """Apply one supported alignment mode to a prediction."""

    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"Unknown alignment mode '{mode}'. Choices: {', '.join(ALIGNMENT_MODES)}")
    if mode == "raw":
        return AlignmentResult(prediction=pred.astype(np.float32, copy=True), applied=False)

    mask = alignment_valid_mask(pred, gt, min_depth=min_depth, max_depth=max_depth)
    if mode == "median_aligned":
        return median_scale_align(pred, gt, mask, min_valid_pixels=min_valid_pixels)
    if mode == "scale_shift_aligned":
        return scale_shift_align(pred, gt, mask, min_valid_pixels=min_valid_pixels)
    raise AssertionError(f"Unhandled alignment mode: {mode}")
