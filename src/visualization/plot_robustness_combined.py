"""Combined robustness figures for resolution and lighting experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.experiments.robustness_common import METRIC_COLUMNS, model_label
from src.models.factory import normalize_model_key
from src.visualization.style import FIGURE_FONT_SIZE, GRID_COLOR, apply_ieee_style


DATASET_LABELS = {
    "depth_test": "DepthTest",
    "hypersim": "Hypersim",
}

METRIC_LABELS = {
    "AbsRel": "AbsRel",
    "RMSE": "RMSE",
    "delta1": r"$\delta_1$",
}

LEGACY_ROBUSTNESS_COLORS = {
    "unidepth": "#0072B2",
    "zoedepth": "#009E73",
    "depth_anything_v2_metric_indoor_base": "#D55E00",
    "depth_anything_v2_metric_indoor_small": "#D55E00",
}

LEGACY_MODEL_ORDER = {
    "unidepth": 0,
    "zoedepth": 1,
    "depth_anything_v2_metric_indoor_base": 2,
    "depth_anything_v2_metric_indoor_small": 2,
}


def _legacy_robustness_color(model: str) -> str:
    normalized = normalize_model_key(str(model).strip())
    return LEGACY_ROBUSTNESS_COLORS.get(normalized, "#666666")


def _legacy_model_sort_key(model: str) -> tuple[int, str]:
    normalized = normalize_model_key(str(model).strip())
    return LEGACY_MODEL_ORDER.get(normalized, 99), normalized


def _load_experiment_csv(results_root: Path, dataset: str, experiment: str) -> pd.DataFrame:
    filename = "metrics_resolution.csv" if experiment == "exp2_resolution" else "metrics_lighting.csv"
    path = results_root / dataset / experiment / filename
    if not path.exists():
        print(f"WARNING missing robustness CSV: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = dataset
    return df


def _plot_dataset_metric(
    ax,
    df: pd.DataFrame,
    *,
    dataset: str,
    metric: str,
    x_column: str,
    x_label: str,
) -> None:
    metric_label = METRIC_LABELS.get(metric, metric)
    models = sorted((str(model) for model in df["model"].dropna().unique()), key=_legacy_model_sort_key)
    for model in models:
        model_rows = df[df["model"].astype(str) == model].copy()
        model_rows[metric] = pd.to_numeric(model_rows[metric], errors="coerce")
        model_rows[x_column] = pd.to_numeric(model_rows[x_column], errors="coerce")
        model_rows = model_rows[np.isfinite(model_rows[metric]) & np.isfinite(model_rows[x_column])]
        model_rows = model_rows.sort_values(x_column)
        if model_rows.empty:
            continue
        ax.plot(
            model_rows[x_column].to_numpy(dtype=float),
            model_rows[metric].to_numpy(dtype=float),
            marker="o",
            linewidth=1.9,
            markersize=5.5,
            label=model_label(model),
            color=_legacy_robustness_color(model),
        )

    ax.set_title(
        f"{DATASET_LABELS.get(dataset, dataset)} - {metric_label}",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.grid(True, color=GRID_COLOR, alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=FIGURE_FONT_SIZE)
    if metric == "delta1":
        ax.set_ylim(0.0, 1.02)


def plot_combined_robustness(
    *,
    results_root: str | Path,
    datasets: list[str],
    experiment: str,
    output_path: str | Path,
    x_column: str,
    x_label: str,
    overwrite: bool,
    dpi: int = 300,
) -> Path | None:
    apply_ieee_style()
    results_root = Path(results_root)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"skip existing figure: {output_path}")
        return None

    frames = [_load_experiment_csv(results_root, dataset, experiment) for dataset in datasets]
    frames = [df for df in frames if not df.empty]
    if not frames:
        print(f"WARNING no data for combined robustness figure: {experiment}")
        return None
    data = pd.concat(frames, ignore_index=True)

    present_datasets = [dataset for dataset in datasets if dataset in set(data["dataset"].astype(str))]
    if not present_datasets:
        present_datasets = sorted(data["dataset"].astype(str).unique())

    nrows = len(present_datasets)
    ncols = len(METRIC_COLUMNS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.4 * ncols, 5.4 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.86, hspace=0.36, wspace=0.18)
    for row_idx, dataset in enumerate(present_datasets):
        dataset_df = data[data["dataset"].astype(str) == dataset]
        for col_idx, metric in enumerate(METRIC_COLUMNS):
            _plot_dataset_metric(
                axes[row_idx, col_idx],
                dataset_df,
                dataset=dataset,
                metric=metric,
                x_column=x_column,
                x_label=x_label,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(unique),
            frameon=False,
            fontsize=FIGURE_FONT_SIZE,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create combined robustness figures from experiment CSVs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets", nargs="+", default=["depth_test", "hypersim"])
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    plot_combined_robustness(
        results_root=args.results_root,
        datasets=args.datasets,
        experiment="exp2_resolution",
        output_path=output_dir / "resolution_sensitivity_combined.png",
        x_column="resolution_scale",
        x_label="Resolution scale",
        overwrite=args.overwrite,
        dpi=args.dpi,
    )
    plot_combined_robustness(
        results_root=args.results_root,
        datasets=args.datasets,
        experiment="exp3_lighting",
        output_path=output_dir / "lighting_robustness_combined.png",
        x_column="perturbation_value",
        x_label="Brightness factor",
        overwrite=args.overwrite,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
