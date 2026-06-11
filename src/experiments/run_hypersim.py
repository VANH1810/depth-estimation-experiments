"""Run a selected depth model on extracted HyperSim samples."""

import argparse

from src.datasets.hypersim import HypersimDataset
from src.experiments.runner_cli import add_experiment_args, run_dataset_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-model depth experiments on extracted HyperSim.")
    add_experiment_args(
        parser,
        default_data_root="data/hypersim/samples",
        default_output_root="results/hypersim",
        default_sample_every=1,
        default_max_depth=50.0,
        default_viz_max_depth=5.0,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dataset_cli(args, dataset_name="hypersim", dataset_factory=HypersimDataset)


if __name__ == "__main__":
    main()
