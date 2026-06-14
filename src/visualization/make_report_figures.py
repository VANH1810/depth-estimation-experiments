"""Generate compact paper-style figures for depth experiment reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.figure_utils import DEFAULT_REPORT_MODELS, ResultEntry, load_results_index, select_metric_raw_entries
from src.visualization.plot_metric_summary_grid import plot_legacy_metric_figures, plot_metric_summary_grid
from src.visualization.plot_qualitative_samples import plot_combined_qualitative_samples, plot_qualitative_samples
from src.visualization.plot_scatter_pred_vs_gt import plot_scatter_pred_vs_gt
from src.visualization.style import apply_ieee_style


apply_ieee_style()


DEFAULT_FIGURES = [
    "metric_summary_grid",
    "scatter_pred_vs_gt",
    "qualitative_samples",
    "qualitative_samples_combined",
]


def _write_figures_readme(*, output_dir: Path, datasets: list[str], overwrite: bool) -> None:
    path = output_dir / "README.md"
    if path.exists() and not overwrite:
        print(f"skip existing figure README: {path}")
        return

    lines = [
        "# Figure Captions",
        "",
        "These figures use raw metric-depth predictions from UniDepth, ZoeDepth, and DA-V2 Metric Base.",
        "Depth Anything V2 Small relative outputs and aligned variants are excluded from the main figures.",
        "",
    ]
    for dataset in datasets:
        if dataset == "depth_test":
            metric_caption = (
                "Metric comparison on depth_test using raw metric-depth predictions from UniDepth, ZoeDepth, "
                "and DA-V2 Metric Base."
            )
            qualitative_caption = (
                "Qualitative results on depth_test. Each pair of consecutive rows corresponds to one test sample. "
                "The odd row shows the RGB image and error-colored prediction maps using coolwarm based on absolute "
                "relative error. The even row shows GT depth and predicted depth. The last column shows the colormap "
                "ranges for depth and error."
            )
            scatter_caption = (
                "Predicted depth versus GT depth scatter plots for the three metric-depth models on depth_test. "
                "The diagonal line indicates perfect metric prediction."
            )
        elif dataset == "hypersim":
            metric_caption = (
                "Metric comparison on hypersim using raw metric-depth predictions from UniDepth, ZoeDepth, and "
                "DA-V2 Metric Base. DA-V2 Metric Base is included as a reference, not as a main few-shot baseline, "
                "if its checkpoint has Hypersim train/fine-tune overlap."
            )
            qualitative_caption = (
                "Qualitative results on hypersim using UniDepth, ZoeDepth, and DA-V2 Metric Base. DA-V2 Metric Base "
                "is shown only as a reference if the checkpoint has Hypersim train/fine-tune overlap."
            )
            scatter_caption = (
                "Predicted depth versus GT depth scatter plots for the three metric-depth models on hypersim. "
                "DA-V2 Metric Base should be interpreted carefully if it has train/fine-tune overlap with Hypersim."
            )
        else:
            metric_caption = (
                f"Metric comparison on {dataset} using raw metric-depth predictions from UniDepth, ZoeDepth, "
                "and DA-V2 Metric Base."
            )
            qualitative_caption = (
                f"Qualitative results on {dataset}. Each pair of consecutive rows corresponds to one test sample."
            )
            scatter_caption = (
                f"Predicted depth versus GT depth scatter plots for the three metric-depth models on {dataset}."
            )

        lines += [
            f"## {dataset}_metric_summary_grid.png",
            "Caption:",
            metric_caption,
            "",
            f"## {dataset}_qualitative_samples.png",
            "Caption:",
            qualitative_caption,
            "",
            f"## {dataset}_scatter_pred_vs_gt.png",
            "Caption:",
            scatter_caption,
            "",
        ]

    if {"depth_test", "hypersim"}.issubset(set(datasets)):
        lines += [
            "## depth_hypersim_qualitative_samples.png",
            "Caption:",
            "Combined qualitative results for depth_test and hypersim using raw metric-depth predictions from "
            "UniDepth, ZoeDepth, and DA-V2 Metric Base. The figure contains one representative sample from "
            "depth_test and one representative sample from hypersim while keeping the same qualitative layout. "
            "Columns show RGB/GT, UniDepth, ZoeDepth, DA-V2 Metric Base, and the colormap ranges for absolute "
            "relative error and depth. Dataset labels mark each dataset group, a light horizontal separator "
            "distinguishes the datasets, and a subtle vertical separator separates the RGB/GT column from the "
            "model prediction columns.",
            "",
        ]

    lines += [
        "## Reproducibility",
        "```bash",
        "python -m src.visualization.make_report_figures \\",
        "  --results-root results \\",
        "  --datasets depth_test hypersim \\",
        "  --output-dir results/figures \\",
        "  --figures metric_summary_grid scatter_pred_vs_gt qualitative_samples qualitative_samples_combined \\",
        "  --models unidepth zoedepth depth_anything_v2_metric_indoor_base \\",
        "  --num-qualitative-samples 2 \\",
        "  --max-points 100000 \\",
        "  --overwrite",
        "```",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote figure README: {path}")


def _figure_entries(entries: list[ResultEntry], figure_name: str) -> list[ResultEntry]:
    if figure_name in {"scatter_pred_vs_gt", "qualitative_samples", "qualitative_samples_combined"}:
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
    qualitative_groups: list[tuple[str, list[ResultEntry]]] = []
    for dataset in args.datasets:
        index = load_results_index(results_root, dataset)
        selected = select_metric_raw_entries(index, args.models)
        if not selected:
            print(f"WARNING no selected metric raw result rows for dataset={dataset}, models={args.models}")
            continue
        qualitative_groups.append((dataset, _figure_entries(selected, "qualitative_samples_combined")))

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

    if "qualitative_samples_combined" in args.figures and len(qualitative_groups) >= 2:
        preferred_order = {"depth_test": 0, "hypersim": 1}
        qualitative_groups = sorted(qualitative_groups, key=lambda item: preferred_order.get(item[0], 99))
        out = plot_combined_qualitative_samples(
            qualitative_groups,
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

    _write_figures_readme(output_dir=output_dir, datasets=args.datasets, overwrite=args.overwrite)
    print(f"generated {len(generated)} figure(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create compact report figures from depth experiment results.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--datasets", nargs="+", default=["depth_test", "hypersim"])
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--figures", nargs="+", default=DEFAULT_FIGURES, choices=DEFAULT_FIGURES)
    parser.add_argument("--models", nargs="+", default=DEFAULT_REPORT_MODELS)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--num-qualitative-samples", type=int, default=2)
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
