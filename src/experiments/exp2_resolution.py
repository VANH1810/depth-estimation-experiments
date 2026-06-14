"""Resolution sensitivity experiment for metric monocular depth models."""

from __future__ import annotations

from pathlib import Path

from src.experiments.robustness_common import plot_sensitivity, run_robustness_loop, write_readme
from src.transforms.resolution import resize_rgb_and_intrinsics


CSV_COLUMNS = ["dataset", "model", "resolution_scale", "num_samples", "AbsRel", "RMSE", "delta1"]


def _resolution_transform(rgb, intrinsics, scale: float):
    return resize_rgb_and_intrinsics(rgb, intrinsics=intrinsics, scale=scale)


def _readme(
    *,
    dataset_name: str,
    model_names: list[str],
    resolution_scales: list[float],
    metrics_path: Path,
    figure_path: Path,
    errors_count: int,
) -> str:
    models = ", ".join(model_names)
    scales = ", ".join(f"{scale:g}x" for scale in resolution_scales)
    notes = "No inference errors were recorded." if errors_count == 0 else f"{errors_count} inference errors are logged in errors.json."
    return f"""
# exp2_resolution

## Purpose

This experiment evaluates how stable metric depth predictions are when the input RGB image is externally resized before inference.

## Setup

- Dataset: `{dataset_name}`
- Models: {models}
- Resolution scales: {scales}
- CSV: `{metrics_path.name}`
- Figure: `{figure_path.as_posix()}`

For each sample, the RGB input is resized by the selected scale, the model predicts depth from that resized RGB, and the prediction is resized back to the original ground-truth depth resolution before metrics are computed. The ground-truth depth and valid-depth mask remain at the original resolution.

## Metrics

- AbsRel: lower is better.
- RMSE: lower is better.
- delta1: higher is better.

## Figure Description

`resolution_sensitivity.png` contains three metric panels: AbsRel, RMSE, and delta1 versus external input resolution scale. Each curve is one model.

## Intrinsics and Internal Resizing

If a sample provides a 3x3 camera intrinsic matrix, fx and cx are scaled by the realized width scale, and fy and cy are scaled by the realized height scale before inference. If intrinsics are unavailable, the model wrapper uses its normal behavior. Some wrappers, including ZoeDepth and Depth Anything V2 through Transformers and UniDepth's own preprocessing, may still resize internally; this experiment measures robustness to the external resize applied before the wrapper.

## Interpretation Guide

A robust metric-depth model should show small changes in AbsRel, RMSE, and delta1 around the 1.0x baseline. Large metric shifts at lower resolution indicate sensitivity to input resolution, preprocessing, or image statistics.

{notes}
"""


def run_resolution_experiment(
    dataset_name: str,
    model_names: list[str],
    resolution_scales: list[float],
    output_root: str,
    device: str = "cpu",
    max_samples: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run exp2_resolution and save aggregate CSV, figure, and README."""

    variants = [(f"{scale:g}x", float(scale)) for scale in resolution_scales]
    aggregate_rows, experiment_root, errors = run_robustness_loop(
        dataset_name=dataset_name,
        model_names=model_names,
        variants=variants,
        output_root=output_root,
        experiment_dir_name="exp2_resolution",
        group_columns=["dataset", "model", "variant_value"],
        transform_fn=_resolution_transform,
        device=device,
        max_samples=max_samples,
        overwrite=overwrite,
    )

    for row in aggregate_rows:
        row["resolution_scale"] = row.pop("variant_value", row.get("resolution_scale"))

    # Re-save after renaming the grouped x column to the requested public schema.
    from src.utils.io import write_csv

    metrics_path = experiment_root / "metrics_resolution.csv"
    write_csv(metrics_path, aggregate_rows, CSV_COLUMNS)
    figure_path = experiment_root / "figures" / "resolution_sensitivity.png"
    plot_sensitivity(
        aggregate_rows,
        x_column="resolution_scale",
        model_column="model",
        output_path=figure_path,
        x_label="Resolution scale",
        overwrite=overwrite,
    )
    write_readme(
        experiment_root / "README.md",
        _readme(
            dataset_name=dataset_name,
            model_names=model_names,
            resolution_scales=resolution_scales,
            metrics_path=metrics_path,
            figure_path=figure_path,
            errors_count=len(errors),
        ),
        overwrite=overwrite,
    )
