"""Run a selected depth model on Depth_Test."""

import argparse

from src.datasets.depth_test import DepthTestDataset
from src.experiments.runner_cli import add_experiment_args, run_dataset_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-model depth experiments on Depth_Test.")
    add_experiment_args(
        parser,
        default_data_root="data/depth_test",
        default_output_root="results/depth_test",
        default_sample_every=5,
        default_max_depth=5.0,
        default_viz_max_depth=5.0,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dataset_cli(args, dataset_name="depth_test", dataset_factory=DepthTestDataset)


if __name__ == "__main__":
    main()
