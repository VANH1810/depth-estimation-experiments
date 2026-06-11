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
)


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
    if not entries:
        print(f"WARNING no entries for qualitative_samples: {dataset} {protocol}")
        return None

    sample_ids = _select_sample_ids(
        entries,
        num_samples=num_samples,
        strategy=sample_strategy,
        seed=seed,
        fixed_sample_ids=fixed_sample_ids,
    )
    if not sample_ids:
        print(f"WARNING no qualitative sample ids: {dataset} {protocol}")
        return None

    contexts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for sample_id in sample_ids:
        context = _load_sample_context(entries, sample_id)
        if context is None:
            continue
        rgb, gt, _ = context
        contexts.append((sample_id, rgb, gt))
    if not contexts:
        return None

    n_models = len(entries)
    ncols = n_models + 2
    nrows = len(contexts) * 2
    width_ratios = [1.1] + [1.0] * n_models + [0.18]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * (ncols - 1) + 0.9, 2.15 * nrows),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
        constrained_layout=True,
    )

    for sample_index, (sample_id, rgb, gt) in enumerate(contexts):
        top = sample_index * 2
        bottom = top + 1
        panels, depth_vmin, depth_vmax, err_max = _prepare_sample_panels(entries, sample_id, gt)
        gt_valid = get_valid_mask(gt, min_depth=entries[0].min_depth_m, max_depth=entries[0].max_depth_m)

        axes[top, 0].imshow(rgb)
        axes[top, 0].set_title("RGB", fontsize=9, fontweight="bold")
        axes[top, 0].text(
            0.02,
            0.98,
            _short_sample_id(sample_id),
            transform=axes[top, 0].transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )
        axes[top, 0].axis("off")

        axes[bottom, 0].imshow(
            colorize_array(gt, vmin=depth_vmin, vmax=depth_vmax, cmap="magma", valid_mask=gt_valid)
        )
        axes[bottom, 0].set_title("GT depth", fontsize=9, fontweight="bold")
        axes[bottom, 0].axis("off")

        for col, panel in enumerate(panels, start=1):
            entry = panel["entry"]
            label = entry_label(entry, multiline=True)
            if panel["pred"] is None:
                axes[top, col].text(0.5, 0.5, "missing\nprediction", ha="center", va="center", fontsize=8)
                axes[bottom, col].text(0.5, 0.5, "missing\nprediction", ha="center", va="center", fontsize=8)
                axes[top, col].set_title(f"Error map\n{label}", fontsize=8.5)
                axes[bottom, col].set_title(f"Pred depth\n{label}", fontsize=8.5)
            else:
                axes[top, col].imshow(
                    colorize_array(panel["error"], vmin=0.0, vmax=err_max, cmap="coolwarm", valid_mask=panel["valid"])
                )
                axes[top, col].set_title(f"Error map\n{label}", fontsize=8.5)
                axes[bottom, col].imshow(
                    colorize_array(
                        panel["pred"],
                        vmin=depth_vmin,
                        vmax=depth_vmax,
                        cmap="magma",
                        valid_mask=panel["valid"],
                    )
                )
                axes[bottom, col].set_title(f"Pred depth\n{label}", fontsize=8.5)
            axes[top, col].axis("off")
            axes[bottom, col].axis("off")

        err_ax = axes[top, -1]
        depth_ax = axes[bottom, -1]
        err_sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=err_max), cmap="coolwarm")
        depth_sm = ScalarMappable(norm=Normalize(vmin=depth_vmin, vmax=depth_vmax), cmap="magma")
        err_cbar = fig.colorbar(err_sm, cax=err_ax)
        depth_cbar = fig.colorbar(depth_sm, cax=depth_ax)
        err_cbar.set_label(f"AbsRel\n[0, {err_max:.2f}]", fontsize=7)
        depth_cbar.set_label(f"Depth (m)\n[{depth_vmin:.2f}, {depth_vmax:.2f}]", fontsize=7)
        err_ax.set_title("Ranges", fontsize=8.5, fontweight="bold")
        err_ax.tick_params(labelsize=6)
        depth_ax.tick_params(labelsize=6)

    title = f"{dataset}: qualitative samples - fallback error-colored prediction maps"
    if protocol != "primary" or include_reference:
        title += f" ({protocol}{' + reference' if include_reference else ''})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    suffix = figure_suffix(protocol, include_reference)
    return save_figure(
        fig,
        Path(output_dir) / f"{dataset}_qualitative_samples{suffix}.png",
        dpi=dpi,
        overwrite=overwrite,
    )
