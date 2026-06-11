"""Generate compact paper-style figures for depth experiment reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.figure_utils import ResultEntry, load_results_index, resolve_models_for_protocol
from src.visualization.plot_metric_summary_grid import plot_legacy_metric_figures, plot_metric_summary_grid
from src.visualization.plot_qualitative_samples import plot_qualitative_samples
from src.visualization.plot_scatter_pred_vs_gt import plot_scatter_pred_vs_gt


DEFAULT_FIGURES = ["metric_summary_grid", "scatter_pred_vs_gt", "qualitative_samples"]


def _write_report(
    *,
    results_root: Path,
    output_dir: Path,
    datasets: list[str],
    protocol: str,
    include_reference: bool,
    overwrite: bool,
) -> None:
    path = results_root / "report_experiments.md"
    if path.exists() and not overwrite:
        print(f"skip existing report: {path}")
        return
    lines = [
        "# Depth Experiment Report",
        "",
        "## Experiment Setup",
        "- Datasets: " + ", ".join(datasets),
        "- Models: UniDepth, ZoeDepth, Depth Anything V2 Small, and available Depth Anything V2 Metric variants.",
        "- Metrics: AbsRel, SqRel, RMSE, RMSE_log, MAE, delta1, delta2, delta3, log10, SILog.",
        "- Protocol groups: `primary_metric_raw`, `relative_raw`, `relative_aligned`, `train_overlap_reference`.",
        "- Default report figures use `--protocol primary`, so raw relative-depth outputs are not mixed with metric-depth plots.",
        "",
        "## Primary Metric Results",
    ]
    for dataset in datasets:
        comparison_dir = results_root / dataset / "comparison"
        summary = comparison_dir / "comparison_summary.md"
        primary = comparison_dir / "primary_metric_raw_comparison.md"
        aligned = comparison_dir / "aligned_relative_comparison.md"
        relative_raw = comparison_dir / "relative_raw_comparison.md"
        train_overlap = comparison_dir / "train_overlap_reference_comparison.md"
        lines += [
            f"### {dataset}",
            f"- Full summary: `{summary}`",
            f"- Primary metric raw comparison: `{primary}`",
            f"- Relative aligned comparison: `{aligned}`",
            f"- Relative raw comparison: `{relative_raw}`",
            f"- Train-overlap reference comparison: `{train_overlap}`",
        ]
        if primary.exists():
            lines.append("")
            lines.extend(primary.read_text(encoding="utf-8").splitlines()[:12])
            lines.append("")

    lines += [
        "## Visualization Figures",
        f"- Figure output directory: `{output_dir}`",
        f"- Active protocol for this report run: `{protocol}`.",
        f"- Include train-overlap/reference rows in figures: `{include_reference}`.",
        "",
        "### `metric_summary_grid`",
        "- The previous per-metric files such as `absrel_comparison.png`, `rmse_comparison.png`, and `delta1_comparison.png` are replaced by one multi-panel figure per dataset/protocol.",
        "- Subplots use separate y-axes and mark metric direction in the title: AbsRel ↓, RMSE ↓, delta1 ↑, and MAE ↓.",
        "",
        "### `scatter_pred_vs_gt`",
        "- Scatter plots show valid GT depth on the x-axis and predicted depth on the y-axis with a `y = x` reference line.",
        "- Points are sampled reproducibly across images instead of drawing every pixel.",
        "- Raw relative-depth predictions are excluded from metric-depth scatter figures; aligned relative predictions are labeled with their alignment mode when requested.",
        "",
        "### `qualitative_samples`",
        "- Zero-shot qualitative results. Each pair of consecutive rows corresponds to one test sample. Each odd row shows the input RGB image and the fallback prediction map color-coded with coolwarm based on the absolute relative error. Each even row shows GT depth and predicted depth. The last column represents the colormap ranges for depth and error.",
        "- The current implementation uses an error-colored depth-map fallback, not a true 3D pointcloud render, because there is no shared pointcloud renderer/intrinsics path wired into the report figure generator.",
        "",
        "## Relative / Aligned Results",
        "- Depth Anything V2 Small is a relative-depth checkpoint. Its raw output has no meter unit.",
        "- `median_aligned` and `scale_shift_aligned` use ground-truth valid pixels for fitting and should be reported as supplementary shape/order analysis.",
        "",
        "## Hypersim Train-overlap Reference",
        "- Depth Anything V2 Metric Indoor is tagged as `train_overlap_reference` on HyperSim and is excluded from primary figures unless `--include-reference` is set.",
        "",
        "## Reproducibility",
        "- Main figure command:",
        "  `python -m src.visualization.make_report_figures --results-root results --datasets depth_test hypersim --output-dir results/figures --figures metric_summary_grid scatter_pred_vs_gt qualitative_samples --max-points 100000 --num-qualitative-samples 4 --overwrite`",
        "- Supplementary aligned figure command:",
        "  `python -m src.visualization.make_report_figures --results-root results --datasets depth_test hypersim --output-dir results/figures/supplementary --protocol aligned --figures metric_summary_grid scatter_pred_vs_gt qualitative_samples --overwrite`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {path}")


def _figure_entries(entries: list[ResultEntry], figure_name: str) -> list[ResultEntry]:
    if figure_name in {"scatter_pred_vs_gt", "qualitative_samples"}:
        return [entry for entry in entries if entry.protocol_group != "relative_raw"]
    return entries


def run(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_qualitative_samples = args.num_qualitative_samples
    if args.max_samples is not None:
        num_qualitative_samples = args.max_samples

    generated: list[Path] = []
    for dataset in args.datasets:
        index = load_results_index(results_root, dataset)
        selected = resolve_models_for_protocol(
            index,
            protocol=args.protocol,
            include_reference=args.include_reference,
        )
        if not selected:
            print(f"WARNING no selected result rows for dataset={dataset}, protocol={args.protocol}")
            continue

        if "metric_summary_grid" in args.figures:
            out = plot_metric_summary_grid(
                _figure_entries(selected, "metric_summary_grid"),
                dataset=dataset,
                output_dir=output_dir,
                protocol=args.protocol,
                include_reference=args.include_reference,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
            if out is not None:
                generated.append(out)
            if args.legacy_separate_metric_figures:
                generated.extend(
                    plot_legacy_metric_figures(
                        _figure_entries(selected, "metric_summary_grid"),
                        dataset=dataset,
                        output_dir=output_dir,
                        protocol=args.protocol,
                        include_reference=args.include_reference,
                        dpi=args.dpi,
                        overwrite=args.overwrite,
                    )
                )

        if "scatter_pred_vs_gt" in args.figures:
            out = plot_scatter_pred_vs_gt(
                _figure_entries(selected, "scatter_pred_vs_gt"),
                dataset=dataset,
                output_dir=output_dir,
                protocol=args.protocol,
                include_reference=args.include_reference,
                max_points=args.max_points,
                log_depth=args.log_depth,
                seed=args.seed,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
            if out is not None:
                generated.append(out)

        if "qualitative_samples" in args.figures:
            out = plot_qualitative_samples(
                _figure_entries(selected, "qualitative_samples"),
                dataset=dataset,
                output_dir=output_dir,
                protocol=args.protocol,
                include_reference=args.include_reference,
                num_samples=num_qualitative_samples,
                sample_strategy=args.sample_strategy,
                fixed_sample_ids=args.fixed_sample_ids,
                seed=args.seed,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
            if out is not None:
                generated.append(out)

    _write_report(
        results_root=results_root,
        output_dir=output_dir,
        datasets=args.datasets,
        protocol=args.protocol,
        include_reference=args.include_reference,
        overwrite=args.overwrite,
    )
    print(f"generated {len(generated)} figure(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create compact report figures from depth experiment results.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets", nargs="+", default=["depth_test", "hypersim"])
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--figures", nargs="+", default=DEFAULT_FIGURES, choices=DEFAULT_FIGURES)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--num-qualitative-samples", type=int, default=4)
    parser.add_argument(
        "--sample-strategy",
        default="median_absrel",
        choices=["median_absrel", "best", "worst", "random", "fixed_ids"],
    )
    parser.add_argument("--fixed-sample-ids", nargs="*", default=None)
    parser.add_argument("--log-depth", action="store_true")
    parser.add_argument("--legacy-separate-metric-figures", action="store_true")
    parser.add_argument("--protocol", default="primary", choices=["primary", "aligned", "all"])
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Backward-compatible alias for --num-qualitative-samples.",
    )
    parser.add_argument("--alignment-for-relative", default="scale_shift_aligned", help=argparse.SUPPRESS)
    parser.add_argument("--format", nargs="+", default=["png"], help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
