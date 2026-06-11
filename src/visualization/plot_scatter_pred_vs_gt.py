"""Predicted-depth versus ground-truth scatter figure."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.visualization.figure_utils import (
    ResultEntry,
    entry_label,
    figure_suffix,
    get_valid_mask,
    load_gt_depth,
    load_prediction,
    metric_text,
    resize_to_match,
    sample_valid_points,
    save_figure,
)


def _stable_seed(seed: int, dataset: str, model: str, alignment: str) -> int:
    payload = f"{seed}:{dataset}:{model}:{alignment}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _collect_points(
    entry: ResultEntry,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "sample_id" not in entry.metrics_df.columns or "depth_path" not in entry.metrics_df.columns:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    rows = entry.metrics_df.dropna(subset=["sample_id", "depth_path"]).drop_duplicates(subset=["sample_id"])
    if rows.empty:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    per_image = max(1, math.ceil(max_points / len(rows)))
    rng = np.random.default_rng(_stable_seed(seed, entry.dataset, entry.model, entry.alignment))
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []

    for _, row in rows.iterrows():
        sample_id = str(row["sample_id"])
        pred_path = entry.predictions_dir / f"{sample_id}.npy"
        if not pred_path.exists():
            continue
        try:
            gt = load_gt_depth(str(row["depth_path"]))
            pred = resize_to_match(load_prediction(entry, sample_id), gt.shape)
        except Exception as exc:
            print(f"WARNING skip scatter sample {entry_label(entry)} {sample_id}: {exc}")
            continue
        valid = get_valid_mask(
            gt,
            pred,
            min_depth=entry.min_depth_m,
            max_depth=entry.max_depth_m,
            require_positive_pred=True,
        )
        gt_sample, pred_sample = sample_valid_points(gt, pred, valid, max_points=per_image, rng=rng)
        if gt_sample.size:
            gt_chunks.append(gt_sample)
            pred_chunks.append(pred_sample)

    if not gt_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    gt_all = np.concatenate(gt_chunks)
    pred_all = np.concatenate(pred_chunks)
    if gt_all.size > max_points:
        indices = rng.choice(gt_all.size, size=max_points, replace=False)
        gt_all = gt_all[indices]
        pred_all = pred_all[indices]
    return gt_all, pred_all


def plot_scatter_pred_vs_gt(
    entries: list[ResultEntry],
    *,
    dataset: str,
    output_dir: str | Path,
    protocol: str,
    include_reference: bool,
    max_points: int,
    log_depth: bool,
    seed: int,
    dpi: int,
    overwrite: bool,
) -> Path | None:
    plot_data: list[tuple[ResultEntry, np.ndarray, np.ndarray]] = []
    for entry in entries:
        if entry.protocol_group == "relative_raw":
            continue
        gt_points, pred_points = _collect_points(entry, max_points=max_points, seed=seed)
        if gt_points.size == 0:
            print(f"WARNING no scatter points for {dataset} {entry_label(entry)}")
            continue
        if log_depth:
            positive = (gt_points > 0) & (pred_points > 0)
            gt_points = gt_points[positive]
            pred_points = pred_points[positive]
        if gt_points.size:
            plot_data.append((entry, gt_points, pred_points))

    if not plot_data:
        print(f"WARNING no scatter_pred_vs_gt data: {dataset} {protocol}")
        return None

    all_values = np.concatenate([np.concatenate([gt, pred]) for _, gt, pred in plot_data])
    all_values = all_values[np.isfinite(all_values)]
    if log_depth:
        all_values = all_values[all_values > 0]
    if all_values.size == 0:
        return None
    axis_min = float(np.min(all_values))
    axis_max = float(np.max(all_values))
    if log_depth:
        axis_min = max(axis_min, 1e-3)
    margin = max(1e-6, 0.025 * (axis_max - axis_min))
    axis_min = max(1e-6 if log_depth else 0.0, axis_min - margin)
    axis_max = axis_max + margin

    n = len(plot_data)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 4.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()
    for ax, (entry, gt_points, pred_points) in zip(axes_flat, plot_data):
        alpha = 0.07 if gt_points.size > 20_000 else 0.18
        ax.scatter(gt_points, pred_points, s=2.2, alpha=alpha, color="#2f6fbb", linewidths=0, rasterized=True)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#222222", linewidth=1.0, linestyle="--")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        if log_depth:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_title(entry_label(entry, multiline=False), fontsize=10, fontweight="bold")
        ax.set_xlabel("GT depth (m)")
        ax.set_ylabel("Predicted depth (m)")
        ax.text(
            0.04,
            0.96,
            metric_text(entry, sampled_points=int(gt_points.size)),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.86, "boxstyle": "round,pad=0.25"},
        )
    for ax in axes_flat[len(plot_data) :]:
        ax.axis("off")

    title = f"{dataset}: predicted depth vs GT"
    if protocol != "primary" or include_reference:
        title += f" ({protocol}{' + reference' if include_reference else ''})"
    if log_depth:
        title += " - log scale"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    suffix = figure_suffix(protocol, include_reference)
    return save_figure(
        fig,
        Path(output_dir) / f"{dataset}_scatter_pred_vs_gt{suffix}.png",
        dpi=dpi,
        overwrite=overwrite,
    )
