"""Shared CLI helpers for dataset experiment entry points."""

import argparse
from pathlib import Path

from src.evaluation.alignment import ALIGNMENT_MODES
from src.evaluation.evaluator import evaluate_dataset
from src.models.factory import DEFAULT_MODEL_IDS, MODEL_CLI_CHOICES, create_model_runner, normalize_model_key


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    default_data_root: str,
    default_output_root: str,
    default_sample_every: int,
    default_max_depth: float,
    default_viz_max_depth: float,
) -> None:
    parser.add_argument("--data_root", default=default_data_root, help="Dataset root.")
    parser.add_argument("--output_root", default=default_output_root, help="Dataset result root.")
    parser.add_argument("--model", default="unidepth", choices=MODEL_CLI_CHOICES, help="Model runner key.")
    parser.add_argument("--model_id", default=None, help="Optional model/checkpoint id for the selected runner.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--resolution_level", type=int, default=7, help="UniDepth V2 resolution level.")
    parser.add_argument("--unidepth_root", default=None, help="Path to local UniDepth source directory.")
    parser.add_argument("--limit", type=int, default=None, help="Alias for --max_samples.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--sample_every", type=int, default=default_sample_every, help="Use every Nth sample/frame.")
    parser.add_argument("--min_depth", type=float, default=1e-3, help="Minimum valid depth in meters.")
    parser.add_argument("--max_depth", type=float, default=default_max_depth, help="Maximum valid GT depth in meters.")
    parser.add_argument("--viz_max_depth", type=float, default=default_viz_max_depth, help="Visualization max depth.")
    parser.add_argument(
        "--alignment",
        nargs="+",
        choices=ALIGNMENT_MODES,
        default=None,
        help="Alignment modes to evaluate. Defaults depend on model prediction type.",
    )
    parser.add_argument("--min_alignment_pixels", type=int, default=10, help="Minimum pixels for fitted alignment.")
    parser.add_argument("--no_visualizations", action="store_true", help="Skip visualization PNGs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow updating files in existing output dirs.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop instead of skipping failed samples.")
    parser.add_argument(
        "--allow-train-overlap-reference",
        action="store_true",
        help=(
            "Acknowledge models whose checkpoint may overlap the evaluation dataset. "
            "Currently relevant for Depth Anything V2 Metric Indoor on HyperSim."
        ),
    )
    parser.epilog = "Default model ids: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_MODEL_IDS.items())


def run_dataset_cli(args: argparse.Namespace, dataset_name: str, dataset_factory) -> dict:
    max_samples = args.limit if args.limit is not None else args.max_samples
    canonical_model = normalize_model_key(args.model)
    if dataset_name == "hypersim" and canonical_model.startswith("depth_anything_v2_metric_indoor"):
        message = (
            "WARNING: Depth Anything V2 Metric Indoor is fine-tuned on synthetic Hypersim indoor metric depth data. "
            "This run will be tagged as train_overlap_reference and must not be used as a primary few-shot/zero-shot "
            "Hypersim leaderboard row."
        )
        if args.allow_train_overlap_reference:
            print(message)
        else:
            print(message + " Pass --allow-train-overlap-reference to make this acknowledgement explicit.")
    dataset = dataset_factory(args.data_root, sample_every=args.sample_every, max_samples=max_samples)
    runner = create_model_runner(
        args.model,
        device=args.device,
        model_id=args.model_id,
        resolution_level=args.resolution_level,
        unidepth_root=args.unidepth_root,
    )
    return evaluate_dataset(
        dataset_name=dataset_name,
        dataset=dataset,
        runner=runner,
        output_root=Path(args.output_root),
        alignment_modes=args.alignment,
        device_name=args.device,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        save_visuals=not args.no_visualizations,
        viz_max_depth=args.viz_max_depth,
        overwrite=args.overwrite,
        skip_errors=not args.stop_on_error,
        min_alignment_pixels=args.min_alignment_pixels,
    )
