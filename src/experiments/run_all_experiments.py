"""Orchestrate multi-dataset, multi-model depth experiments."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.experiments.exp2_resolution import run_resolution_experiment
from src.experiments.exp3_lighting import run_lighting_experiment
from src.models.factory import MODEL_CHOICES, MODEL_CLI_CHOICES


DATASET_MODULES = {
    "depth_test": "src.experiments.run_depth_test",
    "hypersim": "src.experiments.run_hypersim",
}
EXPERIMENT_CHOICES = ("baseline", "exp2_resolution", "exp3_lighting")


def _summary_exists(results_root: Path, dataset: str, model: str) -> bool:
    model_root = results_root / dataset / model
    return model_root.exists() and any(model_root.glob("*/metrics_summary.json"))


def _command_for(args: argparse.Namespace, dataset: str, model: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        DATASET_MODULES[dataset],
        "--model",
        model,
        "--device",
        args.device,
        "--output_root",
        str(Path(args.output_root) / dataset),
    ]
    sample_limit = args.limit if args.limit is not None else args.max_samples
    if sample_limit is not None:
        cmd += ["--limit", str(sample_limit)]
    if args.alignments:
        cmd += ["--alignment", *args.alignments]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.skip_figures:
        cmd.append("--no_visualizations")
    if args.allow_train_overlap_reference:
        cmd.append("--allow-train-overlap-reference")
    return cmd


def _write_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def run(args: argparse.Namespace) -> list[dict]:
    results_root = Path(args.output_root)
    log_path = results_root / "logs" / "run_all_experiments.log"
    statuses: list[dict] = []
    _write_log(log_path, f"# run started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    experiments = args.experiments or ["baseline"]
    run_baseline = "baseline" in experiments

    if run_baseline:
        for dataset in args.datasets:
            for model in args.models:
                status = {"experiment": "baseline", "dataset": dataset, "model": model, "status": "pending", "command": ""}
                if args.resume and not args.overwrite and _summary_exists(results_root, dataset, model):
                    status["status"] = "skipped_existing"
                    statuses.append(status)
                    _write_log(log_path, json.dumps(status))
                    print(f"SKIP existing: {dataset}/{model}")
                    continue
                if args.skip_inference or args.skip_eval:
                    status["status"] = "skipped_by_flag"
                    statuses.append(status)
                    _write_log(log_path, json.dumps(status))
                    print(f"SKIP by flag: {dataset}/{model}")
                    continue

                cmd = _command_for(args, dataset, model)
                status["command"] = " ".join(cmd)
                _write_log(log_path, "$ " + status["command"])
                print(f"RUN: {dataset}/{model}")
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=Path.cwd(),
                        text=True,
                        timeout=args.job_timeout_seconds if args.job_timeout_seconds > 0 else None,
                    )
                    status["returncode"] = completed.returncode
                    status["status"] = "success" if completed.returncode == 0 else "failed"
                except subprocess.TimeoutExpired as exc:
                    status["returncode"] = None
                    status["status"] = "timeout"
                    status["error"] = f"Timed out after {exc.timeout} seconds"
                statuses.append(status)
                _write_log(log_path, json.dumps(status))

        compare_cmd = [
            sys.executable,
            "-m",
            "src.experiments.compare_models",
            "--results_root",
            str(results_root),
            "--datasets",
            *args.datasets,
            "--models",
            *args.models,
            "--overwrite",
        ]
        _write_log(log_path, "$ " + " ".join(compare_cmd))
        subprocess.run(compare_cmd, cwd=Path.cwd(), text=True)

    robustness_max_samples = args.max_samples if args.max_samples is not None else args.limit
    if args.experiments is not None and robustness_max_samples is None:
        robustness_max_samples = 20

    for dataset in args.datasets:
        if "exp2_resolution" in experiments:
            status = {"experiment": "exp2_resolution", "dataset": dataset, "models": args.models, "status": "pending"}
            print(f"RUN: {dataset}/exp2_resolution")
            try:
                run_resolution_experiment(
                    dataset_name=dataset,
                    model_names=args.models,
                    resolution_scales=args.resolution_scales,
                    output_root=args.output_root,
                    device=args.device,
                    max_samples=robustness_max_samples,
                    overwrite=args.overwrite,
                )
                status["status"] = "success"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = str(exc)
                if args.stop_on_error:
                    raise
            statuses.append(status)
            _write_log(log_path, json.dumps(status))

        if "exp3_lighting" in experiments:
            status = {"experiment": "exp3_lighting", "dataset": dataset, "models": args.models, "status": "pending"}
            print(f"RUN: {dataset}/exp3_lighting")
            try:
                run_lighting_experiment(
                    dataset_name=dataset,
                    model_names=args.models,
                    brightness_factors=args.brightness_factors,
                    output_root=args.output_root,
                    device=args.device,
                    max_samples=robustness_max_samples,
                    overwrite=args.overwrite,
                )
                status["status"] = "success"
            except Exception as exc:
                status["status"] = "failed"
                status["error"] = str(exc)
                if args.stop_on_error:
                    raise
            statuses.append(status)
            _write_log(log_path, json.dumps(status))

    status_path = results_root / "logs" / "run_all_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"Wrote status log: {status_path}")
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all configured depth experiments.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_MODULES), default=["depth_test", "hypersim"])
    parser.add_argument("--models", nargs="+", choices=MODEL_CLI_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_CHOICES,
        default=None,
        help="Experiments to run. Omit to preserve the legacy baseline behavior.",
    )
    parser.add_argument("--alignments", nargs="+", default=None, help="Optional override passed to each dataset runner.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--resolution-scales", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--brightness-factors", nargs="+", type=float, default=[0.6, 0.8, 1.0, 1.2, 1.4])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", dest="stop_on_error", action="store_true")
    parser.add_argument("--skip-inference", dest="skip_inference", action="store_true")
    parser.add_argument("--skip-eval", dest="skip_eval", action="store_true")
    parser.add_argument("--skip-figures", dest="skip_figures", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1, help="Reserved for future parallel execution.")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--allow-train-overlap-reference",
        action="store_true",
        help="Pass through acknowledgement for train-overlap reference models such as DA-V2 Metric Indoor on HyperSim.",
    )
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=0,
        help="Per dataset/model timeout. 0 disables timeout.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    statuses = run(args)
    succeeded = sum(item["status"] == "success" for item in statuses)
    failed = [item for item in statuses if item["status"] == "failed"]
    print(f"Success: {succeeded}; failed: {len(failed)}; total jobs: {len(statuses)}")
    for item in failed:
        print(f"FAILED {item['dataset']}/{item['model']}: {item.get('returncode')}")


if __name__ == "__main__":
    main()
