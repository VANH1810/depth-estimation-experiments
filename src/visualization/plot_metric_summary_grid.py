"""Metric summary grid figure for depth comparison tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.visualization.figure_utils import ResultEntry, entry_label, figure_suffix, save_figure
from src.visualization.style import (
    EDGE_COLOR,
    FIGURE_FONT_SIZE,
    GRID_COLOR,
    apply_ieee_style,
    bar_color_for_model,
    hatch_for_model,
)


apply_ieee_style()


METRICS = [
    ("AbsRel", "AbsRel ↓"),
    ("RMSE", "RMSE ↓"),
    ("delta1", "δ1 ↑"),
    ("MAE", "MAE ↓"),
]


def _model_labels(entries: list[ResultEntry]) -> list[str]:
    return [
        entry_label(entry, multiline=True, include_alignment=False).replace("DA-V2 Metric Base", "DA-V2 Metric\nBase")
        for entry in entries
    ]


def _style_bars(bars, entries: list[ResultEntry], valid: np.ndarray) -> None:
    valid_entries = [entry for entry, is_valid in zip(entries, valid) if is_valid]
    for bar, entry in zip(bars, valid_entries):
        color = bar_color_for_model(entry.model)
        bar.set_facecolor(color)
        bar.set_edgecolor(EDGE_COLOR)
        bar.set_linewidth(0.45)
        hatch = hatch_for_model(entry.model)
        if hatch:
            bar.set_hatch(hatch)


def plot_metric_summary_grid(
    entries: list[ResultEntry],
    *,
    dataset: str,
    output_dir: str | Path,
    protocol: str,
    include_reference: bool,
    dpi: int,
    overwrite: bool,
) -> Path | None:
    if not entries:
        print(f"WARNING no entries for metric_summary_grid: {dataset} {protocol}")
        return None

    labels = _model_labels(entries)
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(9.4, len(labels) * 2.55), 7.8), constrained_layout=True)
    axes = np.ravel(axes)

    plotted_any = False
    for ax, (metric, label) in zip(axes, METRICS):
        values = [entry.metric_value(metric) for entry in entries]
        numeric = np.array([np.nan if value is None else float(value) for value in values], dtype=float)
        valid = np.isfinite(numeric)
        if not valid.any():
            ax.axis("off")
            continue
        x = np.arange(len(entries))
        bars = ax.bar(x[valid], numeric[valid])
        _style_bars(bars, entries, valid)
        ax.set_title(label, fontsize=FIGURE_FONT_SIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=FIGURE_FONT_SIZE)
        ax.set_ylabel(metric)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.25, linewidth=0.7)
        ymax = float(np.nanmax(numeric[valid]))
        ymin = float(np.nanmin(numeric[valid]))
        if metric == "delta1":
            ax.set_ylim(max(0.0, ymin - 0.08), min(1.02, max(1.0, ymax + 0.04)))
        else:
            ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)
        for idx, value in enumerate(numeric):
            if not np.isfinite(value):
                continue
            offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(idx, value + offset, f"{value:.3g}", ha="center", va="bottom", fontsize=FIGURE_FONT_SIZE)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"WARNING metric_summary_grid has no numeric metrics: {dataset} {protocol}")
        return None

    suffix = figure_suffix(protocol, include_reference)
    return save_figure(
        fig,
        Path(output_dir) / f"{dataset}_metric_summary_grid{suffix}.png",
        dpi=dpi,
        overwrite=overwrite,
    )


def plot_legacy_metric_figures(
    entries: list[ResultEntry],
    *,
    dataset: str,
    output_dir: str | Path,
    protocol: str,
    include_reference: bool,
    dpi: int,
    overwrite: bool,
) -> list[Path]:
    paths: list[Path] = []
    if not entries:
        return paths
    labels = _model_labels(entries)
    suffix = figure_suffix(protocol, include_reference)
    for metric, label in METRICS[:3]:
        values = np.array(
            [np.nan if entry.metric_value(metric) is None else float(entry.metric_value(metric)) for entry in entries],
            dtype=float,
        )
        valid = np.isfinite(values)
        if not valid.any():
            continue
        fig, ax = plt.subplots(figsize=(max(9.0, len(labels) * 2.45), 5.8), constrained_layout=True)
        x = np.arange(len(entries))
        bars = ax.bar(x[valid], values[valid])
        _style_bars(bars, entries, valid)
        ax.set_title(label, fontsize=FIGURE_FONT_SIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=FIGURE_FONT_SIZE)
        ax.set_ylabel(metric)
        ax.grid(axis="y", color=GRID_COLOR, alpha=0.25)
        out = save_figure(
            fig,
            Path(output_dir) / f"{dataset}_{metric.lower()}_comparison{suffix}.png",
            dpi=dpi,
            overwrite=overwrite,
        )
        if out is not None:
            paths.append(out)
    return paths
