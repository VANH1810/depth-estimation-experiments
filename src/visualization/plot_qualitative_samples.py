"""Qualitative overview figure for depth experiments."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.visualization.figure_utils import (
    ResultEntry,
    colorize_array,
    compute_absrel_error_map,
    entry_label,
    figure_suffix,
    get_valid_mask,
    load_gt_depth,
    load_prediction,
    load_rgb,
    resize_to_match,
    save_figure,
    short_model_label,
)
from src.visualization.style import (
    ANNOTATION_FACE_COLOR,
    DEPTH_CMAP,
    ERROR_CMAP,
    FIGURE_FONT_SIZE,
    GRID_COLOR,
    TEXT_COLOR,
    apply_ieee_style,
)


apply_ieee_style()


DATASET_LABELS = {
    "depth_test": "DepthTest",
    "hypersim": "Hypersim",
}



def _sample_ids_for_entry(entry: ResultEntry) -> set[str]:
    if "sample_id" not in entry.metrics_df.columns:
        return set()
    ids = set(str(value) for value in entry.metrics_df["sample_id"].dropna())
    return {sample_id for sample_id in ids if (entry.predictions_dir / f"{sample_id}.npy").exists()}


def _candidate_sample_ids(entries: list[ResultEntry]) -> list[str]:
    id_sets = [_sample_ids_for_entry(entry) for entry in entries]
    id_sets = [ids for ids in id_sets if ids]
    if not id_sets:
        return []
    common = set.intersection(*id_sets)
    if common:
        return sorted(common)
    return sorted(set.union(*id_sets))


def _reference_entry(entries: list[ResultEntry]) -> ResultEntry:
    for entry in entries:
        if entry.model == "unidepth" and entry.alignment == "raw":
            return entry
    return entries[0]


def _select_sample_ids(
    entries: list[ResultEntry],
    *,
    num_samples: int,
    strategy: str,
    seed: int,
    fixed_sample_ids: list[str] | None,
) -> list[str]:
    candidates = _candidate_sample_ids(entries)
    if not candidates:
        return []
    if strategy == "fixed_ids":
        requested = fixed_sample_ids or []
        return [sample_id for sample_id in requested if sample_id in candidates][:num_samples]

    ref = _reference_entry(entries)
    df = ref.metrics_df.copy()
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str)
        df = df[df["sample_id"].isin(candidates)]
    else:
        df = pd.DataFrame()

    rng = np.random.default_rng(seed)
    if df.empty or "AbsRel" not in df.columns:
        count = min(num_samples, len(candidates))
        return list(rng.choice(candidates, size=count, replace=False))

    df["AbsRel"] = pd.to_numeric(df["AbsRel"], errors="coerce")
    df = df.dropna(subset=["AbsRel"])
    if df.empty:
        count = min(num_samples, len(candidates))
        return list(rng.choice(candidates, size=count, replace=False))

    if strategy == "best":
        ordered = df.sort_values("AbsRel", ascending=True)["sample_id"].tolist()
    elif strategy == "worst":
        ordered = df.sort_values("AbsRel", ascending=False)["sample_id"].tolist()
    elif strategy == "random":
        count = min(num_samples, len(candidates))
        return list(rng.choice(candidates, size=count, replace=False))
    else:
        median = float(df["AbsRel"].median())
        ordered = df.assign(_distance=(df["AbsRel"] - median).abs()).sort_values("_distance")["sample_id"].tolist()

    selected: list[str] = []
    for sample_id in ordered:
        if sample_id not in selected:
            selected.append(sample_id)
        if len(selected) >= num_samples:
            return selected
    for sample_id in candidates:
        if sample_id not in selected:
            selected.append(sample_id)
        if len(selected) >= num_samples:
            break
    return selected


def _row_for_sample(entry: ResultEntry, sample_id: str) -> pd.Series | None:
    if "sample_id" not in entry.metrics_df.columns:
        return None
    rows = entry.metrics_df[entry.metrics_df["sample_id"].astype(str) == sample_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def _short_sample_id(sample_id: str) -> str:
    if len(sample_id) <= 34:
        return sample_id
    return "..." + sample_id[-31:]


def _format_metric_value(value: object, precision: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{parsed:.{precision}f}" if math.isfinite(parsed) else "-"


def _sample_metric_text(entries: list[ResultEntry], sample_id: str) -> str:
    blocks: list[str] = []
    for entry in entries:
        row = _row_for_sample(entry, sample_id)
        if row is None:
            continue
        blocks.append(
            "\n".join(
                [
                    short_model_label(entry.model),
                    f"AbsRel {_format_metric_value(row.get('AbsRel'))}",
                    f"RMSE {_format_metric_value(row.get('RMSE'))}",
                    f"δ1 {_format_metric_value(row.get('delta1'))}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "metrics\nunavailable"


def _load_sample_context(entries: list[ResultEntry], sample_id: str) -> tuple[np.ndarray, np.ndarray, pd.Series] | None:
    for entry in entries:
        row = _row_for_sample(entry, sample_id)
        if row is None or "image_path" not in row or "depth_path" not in row:
            continue
        try:
            rgb = load_rgb(str(row["image_path"]))
            gt = load_gt_depth(str(row["depth_path"]))
            return rgb, gt, row
        except Exception as exc:
            print(f"WARNING skip qualitative sample context {sample_id}: {exc}")
    return None


def _prepare_sample_panels(
    entries: list[ResultEntry],
    sample_id: str,
    gt: np.ndarray,
) -> tuple[list[dict], float, float, float]:
    gt_valid = get_valid_mask(gt, min_depth=entries[0].min_depth_m, max_depth=entries[0].max_depth_m)
    if not gt_valid.any():
        return [], 0.0, 1.0, 0.3
    vmin = float(np.nanpercentile(gt[gt_valid], 2))
    vmax = float(np.nanpercentile(gt[gt_valid], 98))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(gt[gt_valid])), float(np.nanmax(gt[gt_valid]))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    panels: list[dict] = []
    errors: list[np.ndarray] = []
    for entry in entries:
        pred_path = entry.predictions_dir / f"{sample_id}.npy"
        if not pred_path.exists():
            panels.append({"entry": entry, "pred": None, "valid": None, "error": None})
            continue
        try:
            pred = resize_to_match(load_prediction(entry, sample_id), gt.shape)
        except Exception as exc:
            print(f"WARNING skip qualitative prediction {entry_label(entry)} {sample_id}: {exc}")
            panels.append({"entry": entry, "pred": None, "valid": None, "error": None})
            continue
        valid = get_valid_mask(
            gt,
            pred,
            min_depth=entry.min_depth_m,
            max_depth=entry.max_depth_m,
            require_positive_pred=True,
        )
        error = compute_absrel_error_map(gt, pred, valid) if valid.any() else np.full(gt.shape, np.nan, dtype=np.float32)
        if np.isfinite(error).any():
            errors.append(error[np.isfinite(error)])
        panels.append({"entry": entry, "pred": pred, "valid": valid, "error": error})

    if errors:
        err_max = float(np.nanpercentile(np.concatenate(errors), 98))
        err_max = min(max(err_max, 0.05), 2.0)
    else:
        err_max = 0.3
    return panels, vmin, vmax, err_max


def _add_group_separators(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    sample_count: int,
    group_sample_counts: list[int] | None = None,
) -> None:
    """Draw subtle visual group dividers in figure coordinates."""
    fig.canvas.draw()

    line_style = {
        "color": GRID_COLOR,
        "alpha": 0.24,
        "linewidth": 0.75,
        "solid_capstyle": "butt",
        "transform": fig.transFigure,
        "clip_on": False,
    }
    dataset_line_style = {**line_style, "alpha": 0.36, "linewidth": 0.95}

    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    top = axes[0, 0].get_position().y1
    bottom = axes[-1, 0].get_position().y0

    if axes.shape[1] > 1:
        x = 0.5 * (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0)
        fig.add_artist(Line2D([x, x], [bottom, top], **line_style))

    dataset_boundaries: set[int] = set()
    if group_sample_counts:
        total = 0
        for count in group_sample_counts[:-1]:
            total += count
            dataset_boundaries.add(total)

    for sample_index in range(1, sample_count):
        row_above = sample_index * 2 - 1
        row_below = sample_index * 2
        if row_below >= axes.shape[0]:
            continue
        y = 0.5 * (axes[row_above, 0].get_position().y0 + axes[row_below, 0].get_position().y1)
        style = dataset_line_style if sample_index in dataset_boundaries else line_style
        fig.add_artist(Line2D([left, right], [y, y], **style))


def _collect_qualitative_contexts(
    entries: list[ResultEntry],
    *,
    dataset: str,
    protocol: str,
    num_samples: int,
    sample_strategy: str,
    fixed_sample_ids: list[str] | None,
    seed: int,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if not entries:
        print(f"WARNING no entries for qualitative_samples: {dataset} {protocol}")
        return []

    sample_ids = _select_sample_ids(
        entries,
        num_samples=num_samples,
        strategy=sample_strategy,
        seed=seed,
        fixed_sample_ids=fixed_sample_ids,
    )
    if not sample_ids:
        print(f"WARNING no qualitative sample ids: {dataset} {protocol}")
        return []

    contexts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sample_id in sample_ids:
        context = _load_sample_context(entries, sample_id)
        if context is None:
            continue
        rgb, gt, _ = context
        contexts.append((sample_id, rgb, gt))
    return contexts


def _render_qualitative_figure(
    groups: list[dict],
    *,
    output_path: Path,
    dpi: int,
    overwrite: bool,
    show_dataset_labels: bool,
    show_range_column: bool,
    show_metric_column: bool,
    show_range_header: bool,
    visual_width_ratio: float = 1.1,
) -> Path | None:
    groups = [group for group in groups if group["contexts"]]
    if not groups:
        return None

    entries = groups[0]["entries"]
    n_models = len(entries)
    if show_range_column and show_metric_column:
        raise ValueError("show_range_column and show_metric_column are mutually exclusive")
    ncols = n_models + 1 + int(show_range_column) + int(show_metric_column)
    sample_count = sum(len(group["contexts"]) for group in groups)
    nrows = sample_count * 2
    width_ratios = [visual_width_ratio] + [1.0] * n_models
    if show_range_column:
        width_ratios.append(0.65)
    if show_metric_column:
        width_ratios.append(0.95)
    figure_width = max(4.25 * (ncols - 1) + 2.2, sum(width_ratios) * 3.6)
    figure_height = 3.25 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
        constrained_layout=True,
    )

    sample_index = 0
    for group in groups:
        dataset = group["dataset"]
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        group_entries = group["entries"]
        for group_sample_index, (sample_id, rgb, gt) in enumerate(group["contexts"]):
            top = sample_index * 2
            bottom = top + 1
            panels, depth_vmin, depth_vmax, err_max = _prepare_sample_panels(group_entries, sample_id, gt)
            gt_valid = get_valid_mask(gt, min_depth=group_entries[0].min_depth_m, max_depth=group_entries[0].max_depth_m)

            axes[top, 0].imshow(rgb)
            if sample_index == 0:
                axes[top, 0].set_title("RGB & GT", fontsize=FIGURE_FONT_SIZE, fontweight="bold", pad=8)
            if show_dataset_labels and group_sample_index == 0:
                axes[top, 0].text(
                    0.02,
                    0.98,
                    dataset_label,
                    transform=axes[top, 0].transAxes,
                    va="top",
                    ha="left",
                    fontsize=FIGURE_FONT_SIZE,
                    fontweight="bold",
                    color=TEXT_COLOR,
                    bbox={"facecolor": ANNOTATION_FACE_COLOR, "alpha": 0.72, "edgecolor": "none", "pad": 1.8},
                )
            axes[top, 0].axis("off")

            axes[bottom, 0].imshow(
                colorize_array(gt, vmin=depth_vmin, vmax=depth_vmax, cmap=DEPTH_CMAP, valid_mask=gt_valid)
            )
            axes[bottom, 0].axis("off")

            for col, panel in enumerate(panels, start=1):
                entry = panel["entry"]
                label = short_model_label(entry.model)
                if panel["pred"] is None:
                    axes[top, col].text(
                        0.5,
                        0.5,
                        "missing\nprediction",
                        ha="center",
                        va="center",
                        fontsize=FIGURE_FONT_SIZE,
                    )
                    axes[bottom, col].text(
                        0.5,
                        0.5,
                        "missing\nprediction",
                        ha="center",
                        va="center",
                        fontsize=FIGURE_FONT_SIZE,
                    )
                else:
                    axes[top, col].imshow(
                        colorize_array(
                            panel["error"],
                            vmin=0.0,
                            vmax=err_max,
                            cmap=ERROR_CMAP,
                            valid_mask=panel["valid"],
                        )
                    )
                    axes[bottom, col].imshow(
                        colorize_array(
                            panel["pred"],
                            vmin=depth_vmin,
                            vmax=depth_vmax,
                            cmap=DEPTH_CMAP,
                            valid_mask=panel["valid"],
                        )
                    )
                if sample_index == 0:
                    axes[top, col].set_title(label, fontsize=FIGURE_FONT_SIZE, fontweight="bold", pad=8)
                axes[top, col].axis("off")
                axes[bottom, col].axis("off")

            if show_range_column:
                err_ax = axes[top, -1]
                depth_ax = axes[bottom, -1]
                err_sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=err_max), cmap=ERROR_CMAP)
                depth_sm = ScalarMappable(norm=Normalize(vmin=depth_vmin, vmax=depth_vmax), cmap=DEPTH_CMAP)
                err_cbar = fig.colorbar(err_sm, cax=err_ax)
                depth_cbar = fig.colorbar(depth_sm, cax=depth_ax)
                err_cbar.set_label(f"AbsRel\n[0, {err_max:.2f}]", fontsize=FIGURE_FONT_SIZE)
                depth_cbar.set_label(
                    f"Depth (m)\n[{depth_vmin:.2f}, {depth_vmax:.2f}]",
                    fontsize=FIGURE_FONT_SIZE,
                )
                if show_range_header and sample_index == 0:
                    err_ax.set_title("Range", fontsize=FIGURE_FONT_SIZE, fontweight="bold", pad=8)
                err_ax.tick_params(labelsize=FIGURE_FONT_SIZE)
                depth_ax.tick_params(labelsize=FIGURE_FONT_SIZE)

            if show_metric_column:
                metric_top_ax = axes[top, -1]
                metric_bottom_ax = axes[bottom, -1]
                metric_top_ax.axis("off")
                metric_bottom_ax.axis("off")
                if sample_index == 0:
                    metric_top_ax.set_title("Metric", fontsize=FIGURE_FONT_SIZE, fontweight="bold", pad=8)

                fig.canvas.draw()
                top_box = metric_top_ax.get_position()
                bottom_box = metric_bottom_ax.get_position()
                fig.text(
                    0.5 * (top_box.x0 + top_box.x1),
                    0.5 * (top_box.y1 + bottom_box.y0),
                    _sample_metric_text(group_entries, sample_id),
                    ha="center",
                    va="center",
                    fontsize=FIGURE_FONT_SIZE,
                    color=TEXT_COLOR,
                    linespacing=1.25,
                )

            sample_index += 1

    _add_group_separators(
        fig,
        axes,
        sample_count=sample_count,
        group_sample_counts=[len(group["contexts"]) for group in groups],
    )
    return save_figure(fig, output_path, dpi=dpi, overwrite=overwrite)


def plot_qualitative_samples(
    entries: list[ResultEntry],
    *,
    dataset: str,
    output_dir: str | Path,
    protocol: str,
    include_reference: bool,
    num_samples: int,
    sample_strategy: str,
    fixed_sample_ids: list[str] | None,
    seed: int,
    dpi: int,
    overwrite: bool,
) -> Path | None:
    contexts = _collect_qualitative_contexts(
        entries,
        dataset=dataset,
        protocol=protocol,
        num_samples=num_samples,
        sample_strategy=sample_strategy,
        seed=seed,
        fixed_sample_ids=fixed_sample_ids,
    )
    if not contexts:
        return None

    suffix = figure_suffix(protocol, include_reference)
    return _render_qualitative_figure(
        [{"dataset": dataset, "entries": entries, "contexts": contexts}],
        output_path=Path(output_dir) / f"{dataset}_qualitative_samples{suffix}.png",
        dpi=dpi,
        overwrite=overwrite,
        show_dataset_labels=False,
        show_range_column=True,
        show_metric_column=False,
        show_range_header=True,
    )


def plot_combined_qualitative_samples(
    dataset_entries: list[tuple[str, list[ResultEntry]]],
    *,
    output_dir: str | Path,
    protocol: str,
    include_reference: bool,
    num_samples: int,
    sample_strategy: str,
    fixed_sample_ids: list[str] | None,
    seed: int,
    dpi: int,
    overwrite: bool,
) -> Path | None:
    groups: list[dict] = []
    for dataset, entries in dataset_entries:
        contexts = _collect_qualitative_contexts(
            entries,
            dataset=dataset,
            protocol=protocol,
            num_samples=1,
            sample_strategy=sample_strategy,
            seed=seed,
            fixed_sample_ids=fixed_sample_ids,
        )
        if contexts:
            groups.append({"dataset": dataset, "entries": entries, "contexts": contexts})

    suffix = figure_suffix(protocol, include_reference)
    return _render_qualitative_figure(
        groups,
        output_path=Path(output_dir) / f"depth_hypersim_qualitative_samples{suffix}.png",
        dpi=dpi,
        overwrite=overwrite,
        show_dataset_labels=True,
        show_range_column=True,
        show_metric_column=False,
        show_range_header=False,
        visual_width_ratio=1.1,
    )
