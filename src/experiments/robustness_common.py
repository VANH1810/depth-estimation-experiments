"""Shared implementation for test-time robustness experiments."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.pipeline import resize_depth_to
from src.datasets.depth_test import DepthTestDataset
from src.datasets.hypersim import HypersimDataset
from src.evaluation.metrics import compute_depth_metrics
from src.models.factory import create_model_runner, normalize_model_key
from src.utils.io import prepare_output_dirs
from src.visualization.style import GRID_COLOR, apply_ieee_style, bar_color_for_model


DATASET_DEFAULTS = {
    "depth_test": {
        "factory": DepthTestDataset,
        "data_root": "data/depth_test",
        "sample_every": 5,
        "max_depth": 5.0,
    },
    "hypersim": {
        "factory": HypersimDataset,
        "data_root": "data/hypersim/samples",
        "sample_every": 1,
        "max_depth": 50.0,
    },
}

METRIC_COLUMNS = ["AbsRel", "RMSE", "delta1"]
MODEL_LABELS = {
    "unidepth": "UniDepth",
    "unidepthv2": "UniDepth",
    "unidepth_v2": "UniDepth",
    "unidepth-v2": "UniDepth",
    "lpiccinelli/unidepth-v2-vitl14": "UniDepth",
    "zoedepth": "ZoeDepth",
    "depth_anything_v2_metric_indoor_small": "DA-V2 Metric Small",
    "depth_anything_v2_metric_indoor_base": "DA-V2 Metric Base",
    "depth_anything_v2_small": "DA-V2 Small",
}


def model_label(model_key: str) -> str:
    normalized = normalize_model_key(str(model_key).strip())
    lookup_key = normalized.lower().replace(" ", "_")
    return MODEL_LABELS.get(lookup_key, MODEL_LABELS.get(normalized, str(model_key)))


def load_dataset(dataset_name: str, max_samples: int | None):
    if dataset_name not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choices: {sorted(DATASET_DEFAULTS)}")
    defaults = DATASET_DEFAULTS[dataset_name]
    return defaults["factory"](
        defaults["data_root"],
        sample_every=defaults["sample_every"],
        max_samples=max_samples,
    )


def dataset_max_depth(dataset_name: str) -> float | None:
    return DATASET_DEFAULTS[dataset_name]["max_depth"]


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_metrics(rows: list[dict], group_columns: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        groups.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = dict(zip(group_columns, key, strict=True))
        out["num_samples"] = len(group)
        for metric in METRIC_COLUMNS:
            out[metric] = finite_mean([row[metric] for row in group if metric in row])
        aggregated.append(out)
    return aggregated


def plot_sensitivity(
    rows: list[dict],
    *,
    x_column: str,
    model_column: str,
    output_path: str | Path,
    x_label: str,
    overwrite: bool,
) -> Path | None:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Figure already exists: {output_path}. Pass --overwrite to replace it.")
    if not rows:
        return None

    apply_ieee_style()
    models = sorted({str(row[model_column]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True)

    plotted_any = False
    for ax, metric in zip(axes, METRIC_COLUMNS, strict=True):
        for model in models:
            model_rows = [row for row in rows if row[model_column] == model and np.isfinite(float(row[metric]))]
            model_rows = sorted(model_rows, key=lambda row: float(row[x_column]))
            if not model_rows:
                continue
            x_values = [float(row[x_column]) for row in model_rows]
            y_values = [float(row[metric]) for row in model_rows]
            label = model_label(model)
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=5.0,
                label=label,
                color=bar_color_for_model(model),
            )
            plotted_any = True
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)
        ax.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.7)
        if metric == "delta1":
            ax.set_ylim(0.0, 1.02)

    if not plotted_any:
        plt.close(fig)
        return None

    axes[-1].legend(loc="best", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_readme(path: str | Path, content: str, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"README already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def write_errors(path: str | Path, errors: list[dict]) -> None:
    if not errors:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(errors, indent=2), encoding="utf-8")


def run_robustness_loop(
    *,
    dataset_name: str,
    model_names: list[str],
    variants: list[tuple[str, float]],
    output_root: str | Path,
    experiment_dir_name: str,
    group_columns: list[str],
    transform_fn: Callable,
    device: str,
    max_samples: int | None,
    overwrite: bool,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    skip_errors: bool = True,
) -> tuple[list[dict], Path, list[dict]]:
    dataset = load_dataset(dataset_name, max_samples=max_samples)
    experiment_root = Path(output_root) / dataset_name / experiment_dir_name
    prepare_output_dirs([experiment_root], overwrite=overwrite)
    max_depth = dataset_max_depth(dataset_name) if max_depth is None else max_depth

    per_sample_rows: list[dict] = []
    errors: list[dict] = []
    for model_name in model_names:
        canonical_model = normalize_model_key(model_name)
        runner = create_model_runner(model_name, device=device)
        print(f"Loading {model_label(canonical_model)} for {dataset_name}/{experiment_dir_name}...")
        runner.load()

        for index, sample in enumerate(dataset, start=1):
            print(f"[{index}/{len(dataset)}] {canonical_model} {sample.sample_id}")
            for variant_name, variant_value in variants:
                try:
                    rgb_aug, intrinsics_aug = transform_fn(sample.image, sample.intrinsics, variant_value)
                    start = time.time()
                    prediction = runner.predict(rgb_aug, intrinsics=intrinsics_aug)
                    elapsed = time.time() - start
                    depth_pred = resize_depth_to(prediction.depth, sample.depth.shape)
                    depth_pred = np.nan_to_num(depth_pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    metrics = compute_depth_metrics(
                        depth_pred,
                        sample.depth,
                        min_depth=min_depth,
                        max_depth=max_depth,
                    )
                    if int(metrics.get("valid_pixels", 0)) == 0:
                        raise RuntimeError("No valid pixels for metric computation.")
                    row = {
                        "dataset": dataset_name,
                        "model": canonical_model,
                        "sample_id": sample.sample_id,
                        "variant_name": variant_name,
                        "variant_value": float(variant_value),
                        "inference_time_s": elapsed,
                        **metrics,
                    }
                    per_sample_rows.append(row)
                except Exception as exc:
                    error = {
                        "dataset": dataset_name,
                        "model": canonical_model,
                        "sample_id": sample.sample_id,
                        "variant_name": variant_name,
                        "variant_value": variant_value,
                        "error": str(exc),
                    }
                    errors.append(error)
                    print(f"  ERROR {variant_name}: {exc}")
                    if not skip_errors:
                        raise

    aggregate_rows = aggregate_metrics(per_sample_rows, group_columns=group_columns)
    write_errors(experiment_root / "errors.json", errors)
    return aggregate_rows, experiment_root, errors
