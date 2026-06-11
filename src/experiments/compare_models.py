"""Aggregate per-model metric summaries into comparison tables."""

import argparse
import csv
import json
from pathlib import Path

from src.models.factory import MODEL_CHOICES, normalize_model_key


COMPARISON_COLUMNS = [
    "dataset",
    "model",
    "checkpoint",
    "prediction_type",
    "depth_unit",
    "alignment",
    "protocol_group",
    "is_primary",
    "is_fewshot_valid",
    "has_train_overlap_risk",
    "AbsRel",
    "SqRel",
    "RMSE",
    "RMSE_log",
    "MAE",
    "delta1",
    "delta2",
    "delta3",
    "num_images",
]


METRIC_ALIASES = {
    "AbsRel": ("AbsRel", "arel"),
    "SqRel": ("SqRel", "sqrel"),
    "RMSE": ("RMSE", "rmse"),
    "RMSE_log": ("RMSE_log", "rmselog"),
    "MAE": ("MAE", "mae"),
    "delta1": ("delta1", "d1"),
    "delta2": ("delta2", "d2"),
    "delta3": ("delta3", "d3"),
}


def _metric(avg_metrics: dict, name: str) -> float | str:
    for key in METRIC_ALIASES.get(name, (name,)):
        if key in avg_metrics:
            return avg_metrics[key]
    return ""


def _read_summary(path: Path, dataset: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    avg = summary.get("avg_metrics", {})
    row = {
        "dataset": summary.get("dataset", dataset),
        "model": summary.get("model_key") or summary.get("model") or path.parents[1].name,
        "checkpoint": summary.get("checkpoint") or summary.get("model_id") or summary.get("model") or "",
        "prediction_type": summary.get("prediction_type", "unknown"),
        "depth_unit": summary.get("depth_unit", "unknown"),
        "alignment": summary.get("alignment", "none"),
        "num_images": summary.get("num_images", summary.get("num_samples", "")),
    }
    canonical_model = normalize_model_key(str(row["model"]))
    if row["depth_unit"] == "unknown":
        if canonical_model in {"unidepth", "zoedepth"} or canonical_model.startswith("depth_anything_v2_metric_indoor"):
            row["depth_unit"] = "meter"
        elif canonical_model == "depth_anything_v2_small":
            row["depth_unit"] = "relative"
    row.update(_protocol_tags(row))
    for metric in COMPARISON_COLUMNS:
        if metric in {
            "dataset",
            "model",
            "checkpoint",
            "prediction_type",
            "depth_unit",
            "alignment",
            "protocol_group",
            "is_primary",
            "is_fewshot_valid",
            "has_train_overlap_risk",
            "num_images",
        }:
            continue
        row[metric] = _metric(avg, metric)
    return row


def _protocol_tags(row: dict) -> dict:
    """Classify a summary row into the comparison protocol groups."""

    dataset = row.get("dataset", "")
    model = normalize_model_key(str(row.get("model", "")))
    alignment = row.get("alignment", "")
    prediction_type = row.get("prediction_type", "unknown")

    if model.startswith("depth_anything_v2_metric_indoor") and dataset == "hypersim":
        return {
            "protocol_group": "train_overlap_reference",
            "is_primary": False,
            "is_fewshot_valid": False,
            "has_train_overlap_risk": True,
        }
    if model == "depth_anything_v2_small":
        if alignment in {"median_aligned", "scale_shift_aligned"}:
            protocol_group = "relative_aligned"
        elif alignment == "raw":
            protocol_group = "relative_raw"
        else:
            protocol_group = "relative_aligned"
        return {
            "protocol_group": protocol_group,
            "is_primary": False,
            "is_fewshot_valid": True,
            "has_train_overlap_risk": False,
        }
    if prediction_type == "metric" and alignment == "raw":
        return {
            "protocol_group": "primary_metric_raw",
            "is_primary": True,
            "is_fewshot_valid": True,
            "has_train_overlap_risk": False,
        }
    return {
        "protocol_group": "other",
        "is_primary": False,
        "is_fewshot_valid": True,
        "has_train_overlap_risk": False,
    }


def discover_summaries(results_root: Path, datasets: list[str], models: list[str]) -> list[tuple[str, Path]]:
    summaries: list[tuple[str, Path]] = []
    for dataset in datasets:
        dataset_root = results_root / dataset
        for model in models:
            model = normalize_model_key(model)
            model_root = dataset_root / model
            if not model_root.exists():
                continue
            summaries.extend((dataset, path) for path in sorted(model_root.glob("*/metrics_summary.json")))
    return summaries


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv_table(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Comparison file exists: {path}. Pass --overwrite to update it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in COMPARISON_COLUMNS})


def write_markdown_table(path: Path, rows: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Comparison file exists: {path}. Pass --overwrite to update it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(COMPARISON_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COMPARISON_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(name, "")) for name in COMPARISON_COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparisons(
    results_root: Path,
    datasets: list[str],
    models: list[str],
    overwrite: bool = False,
) -> list[dict]:
    discovered = discover_summaries(results_root, datasets, models)
    rows = [_read_summary(path, dataset) for dataset, path in discovered]
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["protocol_group"],
            row["alignment"],
            float(row["AbsRel"]) if row.get("AbsRel") not in ("", None) else float("inf"),
            float(row["RMSE"]) if row.get("RMSE") not in ("", None) else float("inf"),
            -(float(row["delta1"]) if row.get("delta1") not in ("", None) else -float("inf")),
        )
    )

    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        comparison_dir = results_root / dataset / "comparison"
        write_csv_table(comparison_dir / "comparison_summary.csv", dataset_rows, overwrite=overwrite)
        write_markdown_table(comparison_dir / "comparison_summary.md", dataset_rows, overwrite=overwrite)
        primary_metric_rows = [row for row in dataset_rows if row.get("protocol_group") == "primary_metric_raw"]
        aligned_rows = [row for row in dataset_rows if row.get("protocol_group") == "relative_aligned"]
        relative_raw_rows = [row for row in dataset_rows if row.get("protocol_group") == "relative_raw"]
        train_overlap_rows = [row for row in dataset_rows if row.get("protocol_group") == "train_overlap_reference"]
        write_csv_table(
            comparison_dir / "primary_metric_raw_comparison.csv",
            primary_metric_rows,
            overwrite=overwrite,
        )
        write_markdown_table(
            comparison_dir / "primary_metric_raw_comparison.md",
            primary_metric_rows,
            overwrite=overwrite,
        )
        write_csv_table(
            comparison_dir / "aligned_relative_comparison.csv",
            aligned_rows,
            overwrite=overwrite,
        )
        write_markdown_table(
            comparison_dir / "aligned_relative_comparison.md",
            aligned_rows,
            overwrite=overwrite,
        )
        write_csv_table(
            comparison_dir / "relative_raw_comparison.csv",
            relative_raw_rows,
            overwrite=overwrite,
        )
        write_markdown_table(
            comparison_dir / "relative_raw_comparison.md",
            relative_raw_rows,
            overwrite=overwrite,
        )
        write_csv_table(
            comparison_dir / "train_overlap_reference_comparison.csv",
            train_overlap_rows,
            overwrite=overwrite,
        )
        write_markdown_table(
            comparison_dir / "train_overlap_reference_comparison.md",
            train_overlap_rows,
            overwrite=overwrite,
        )
        # Legacy filenames retained for older notebooks/scripts.
        metric_raw_rows = primary_metric_rows
        write_csv_table(comparison_dir / "metric_raw_comparison.csv", metric_raw_rows, overwrite=overwrite)
        write_markdown_table(comparison_dir / "metric_raw_comparison.md", metric_raw_rows, overwrite=overwrite)
        write_csv_table(comparison_dir / "aligned_comparison.csv", aligned_rows, overwrite=overwrite)
        write_markdown_table(comparison_dir / "aligned_comparison.md", aligned_rows, overwrite=overwrite)

    all_dir = results_root / "comparison"
    write_csv_table(all_dir / "all_models_summary.csv", rows, overwrite=overwrite)
    write_markdown_table(all_dir / "all_models_summary.md", rows, overwrite=overwrite)
    write_csv_table(all_dir / "all_results_with_protocol_tags.csv", rows, overwrite=overwrite)
    write_markdown_table(all_dir / "all_results_with_protocol_tags.md", rows, overwrite=overwrite)
    write_csv_table(
        all_dir / "metric_raw_comparison.csv",
        [row for row in rows if row.get("protocol_group") == "primary_metric_raw"],
        overwrite=overwrite,
    )
    write_markdown_table(
        all_dir / "metric_raw_comparison.md",
        [row for row in rows if row.get("protocol_group") == "primary_metric_raw"],
        overwrite=overwrite,
    )
    write_csv_table(
        all_dir / "primary_metric_raw_comparison.csv",
        [row for row in rows if row.get("protocol_group") == "primary_metric_raw"],
        overwrite=overwrite,
    )
    write_markdown_table(
        all_dir / "primary_metric_raw_comparison.md",
        [row for row in rows if row.get("protocol_group") == "primary_metric_raw"],
        overwrite=overwrite,
    )
    write_csv_table(
        all_dir / "aligned_comparison.csv",
        [row for row in rows if row.get("protocol_group") == "relative_aligned"],
        overwrite=overwrite,
    )
    write_markdown_table(
        all_dir / "aligned_comparison.md",
        [row for row in rows if row.get("protocol_group") == "relative_aligned"],
        overwrite=overwrite,
    )
    write_csv_table(
        all_dir / "aligned_relative_comparison.csv",
        [row for row in rows if row.get("protocol_group") == "relative_aligned"],
        overwrite=overwrite,
    )
    write_markdown_table(
        all_dir / "aligned_relative_comparison.md",
        [row for row in rows if row.get("protocol_group") == "relative_aligned"],
        overwrite=overwrite,
    )
    write_csv_table(
        all_dir / "train_overlap_reference_comparison.csv",
        [row for row in rows if row.get("protocol_group") == "train_overlap_reference"],
        overwrite=overwrite,
    )
    write_markdown_table(
        all_dir / "train_overlap_reference_comparison.md",
        [row for row in rows if row.get("protocol_group") == "train_overlap_reference"],
        overwrite=overwrite,
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build comparison tables from metrics_summary.json files.")
    parser.add_argument("--results_root", default="results", help="Root result directory.")
    parser.add_argument("--datasets", nargs="+", default=["depth_test", "hypersim"], help="Datasets to include.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CHOICES), help="Model keys to include.")
    parser.add_argument("--overwrite", action="store_true", help="Update existing comparison files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = build_comparisons(
        results_root=Path(args.results_root),
        datasets=args.datasets,
        models=args.models,
        overwrite=args.overwrite,
    )
    print(f"Wrote comparison tables with {len(rows)} rows.")


if __name__ == "__main__":
    main()
