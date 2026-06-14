"""Lighting sensitivity experiment for metric monocular depth models."""

from __future__ import annotations

from pathlib import Path

from src.experiments.robustness_common import plot_sensitivity, run_robustness_loop, write_readme
from src.transforms.lighting import apply_brightness


CSV_COLUMNS = [
    "dataset",
    "model",
    "perturbation_type",
    "perturbation_value",
    "num_samples",
    "AbsRel",
    "RMSE",
    "delta1",
]


def _lighting_transform(rgb, intrinsics, factor: float):
    return apply_brightness(rgb, factor=factor), intrinsics


def _readme(
    *,
    dataset_name: str,
    model_names: list[str],
    brightness_factors: list[float],
    metrics_path: Path,
    figure_path: Path,
    errors_count: int,
) -> str:
    models = ", ".join(model_names)
    factors = ", ".join(f"{factor:g}" for factor in brightness_factors)
    notes = "No inference errors were recorded." if errors_count == 0 else f"{errors_count} inference errors are logged in errors.json."
    return f"""
# exp3_lighting

## Purpose

This experiment evaluates how stable metric depth predictions are when the input RGB image is darkened or brightened before inference.

## Setup

- Dataset: `{dataset_name}`
- Models: {models}
- Brightness factors: {factors}
- CSV: `{metrics_path.name}`
- Figure: `{figure_path.as_posix()}`

For each sample, brightness is applied directly to the RGB input before model inference. A factor below 1.0 darkens the image, 1.0 is the original baseline, and a factor above 1.0 brightens it.

## Metrics

- AbsRel: lower is better.
- RMSE: lower is better.
- delta1: higher is better.

## Figure Description

`lighting_sensitivity.png` contains three metric panels: AbsRel, RMSE, and delta1 versus brightness factor. Each curve is one model.

## Ground Truth and Intrinsics

Lighting perturbation changes only RGB pixel values. Ground-truth depth, the valid-depth mask, and camera intrinsics are unchanged because photometric changes do not alter scene geometry or camera projection.

## Internal Resizing

Some wrappers, including ZoeDepth and Depth Anything V2 through Transformers and UniDepth's own preprocessing, may still resize internally; this experiment measures robustness to the external photometric perturbation applied before the wrapper.

## Interpretation Guide

A robust metric-depth model should keep AbsRel, RMSE, and delta1 close to the 1.0 brightness baseline. Large degradation for darker or brighter inputs indicates sensitivity to photometric domain shift.

{notes}
"""


def run_lighting_experiment(
    dataset_name: str,
    model_names: list[str],
    brightness_factors: list[float],
    output_root: str,
    device: str = "cpu",
    max_samples: int | None = None,
    overwrite: bool = False,
) -> None:
    """Run exp3_lighting and save aggregate CSV, figure, and README."""

    variants = [(f"brightness_{factor:g}", float(factor)) for factor in brightness_factors]
    aggregate_rows, experiment_root, errors = run_robustness_loop(
        dataset_name=dataset_name,
        model_names=model_names,
        variants=variants,
        output_root=output_root,
        experiment_dir_name="exp3_lighting",
        group_columns=["dataset", "model", "variant_value"],
        transform_fn=_lighting_transform,
        device=device,
        max_samples=max_samples,
        overwrite=overwrite,
    )

    for row in aggregate_rows:
        row["perturbation_type"] = "brightness"
        row["perturbation_value"] = row.pop("variant_value", row.get("perturbation_value"))

    from src.utils.io import write_csv

    metrics_path = experiment_root / "metrics_lighting.csv"
    write_csv(metrics_path, aggregate_rows, CSV_COLUMNS)
    figure_path = experiment_root / "figures" / "lighting_sensitivity.png"
    plot_sensitivity(
        aggregate_rows,
        x_column="perturbation_value",
        model_column="model",
        output_path=figure_path,
        x_label="Brightness factor",
        overwrite=overwrite,
    )
    write_readme(
        experiment_root / "README.md",
        _readme(
            dataset_name=dataset_name,
            model_names=model_names,
            brightness_factors=brightness_factors,
            metrics_path=metrics_path,
            figure_path=figure_path,
            errors_count=len(errors),
        ),
        overwrite=overwrite,
    )
