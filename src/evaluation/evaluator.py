"""Dataset evaluator for multi-model depth experiments."""

import time
from pathlib import Path

import numpy as np

from src.common.pipeline import resize_depth_to
from src.evaluation.alignment import ALIGNMENT_MODES, AlignmentError, apply_alignment
from src.evaluation.metrics import METRICS_PER_IMAGE_COLUMNS, compute_depth_metrics, summarize_rows
from src.models.base import DepthModelRunner
from src.utils.io import prepare_output_dirs, write_csv, write_json
from src.utils.visualization import save_depth_visualization


def sanitize_for_metric(pred: np.ndarray, min_depth: float) -> np.ndarray:
    """Replace non-finite values and clamp non-positive depths for metric use."""

    clean = np.nan_to_num(pred.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    clean[clean <= min_depth] = min_depth
    return clean


def _summary_payload(
    dataset_name: str,
    runner: DepthModelRunner,
    mode: str,
    rows: list[dict],
    errors: list[dict],
    output_dir: Path,
    min_depth: float,
    max_depth: float | None,
    device_name: str,
    save_visuals: bool,
) -> dict:
    paths = {
        "predictions": str(output_dir / "predictions"),
        "visualizations": str(output_dir / "visualizations") if save_visuals else None,
        "metrics_per_image_csv": str(output_dir / "metrics_per_image.csv"),
        "metrics_csv": str(output_dir / "metrics.csv"),
        "metrics_summary_json": str(output_dir / "metrics_summary.json"),
        "summary_json": str(output_dir / "summary.json"),
    }
    return {
        "dataset": dataset_name,
        "model": runner.display_name,
        "model_key": runner.key,
        "model_id": runner.model_id,
        "checkpoint": runner.model_id,
        "prediction_type": runner.prediction_type,
        "depth_unit": getattr(runner, "depth_unit", "unknown"),
        "alignment": mode,
        "num_images": len(rows),
        "num_errors": len(errors),
        "device": device_name,
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "notes": runner.notes,
        "training_data_note": getattr(runner, "training_data_note", ""),
        "output_structure": paths,
        **summarize_rows(rows),
    }


def evaluate_dataset(
    dataset_name: str,
    dataset,
    runner: DepthModelRunner,
    output_root: str | Path,
    alignment_modes: tuple[str, ...] | list[str] | None,
    device_name: str,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    save_visuals: bool = True,
    viz_max_depth: float = 5.0,
    overwrite: bool = False,
    skip_errors: bool = True,
    min_alignment_pixels: int = 10,
) -> dict[str, dict]:
    """Run one model over one dataset and save per-alignment results."""

    modes = tuple(alignment_modes or runner.default_alignment_modes)
    unknown = [mode for mode in modes if mode not in ALIGNMENT_MODES]
    if unknown:
        raise ValueError(f"Unknown alignment modes: {unknown}. Choices: {ALIGNMENT_MODES}")

    output_root = Path(output_root)
    mode_dirs = {mode: output_root / runner.key / mode for mode in modes}
    prepare_output_dirs(mode_dirs.values(), overwrite=overwrite)
    for mode_dir in mode_dirs.values():
        (mode_dir / "predictions").mkdir(parents=True, exist_ok=True)
        if save_visuals:
            (mode_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset)}")
    print(f"Model: {runner.display_name} ({runner.model_id})")
    print(f"Prediction type: {runner.prediction_type}")
    print(f"Alignments: {', '.join(modes)}")
    print(f"Output root: {output_root}")
    print("Loading model...")
    runner.load()

    rows_by_mode: dict[str, list[dict]] = {mode: [] for mode in modes}
    errors: list[dict] = []

    for index, sample in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {sample.sample_id}")
        try:
            start = time.time()
            raw_prediction = runner.predict(sample.image)
            elapsed = time.time() - start
            raw_depth = resize_depth_to(raw_prediction.depth, sample.depth.shape)
            raw_depth = np.nan_to_num(raw_depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            for mode in modes:
                try:
                    aligned = apply_alignment(
                        raw_depth,
                        sample.depth,
                        mode=mode,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        min_valid_pixels=min_alignment_pixels,
                    )
                    eval_depth = sanitize_for_metric(aligned.prediction, min_depth=min_depth)
                    mode_dir = mode_dirs[mode]
                    pred_path = mode_dir / "predictions" / f"{sample.sample_id}.npy"
                    np.save(pred_path, eval_depth.astype(np.float32))

                    metrics = compute_depth_metrics(
                        eval_depth,
                        sample.depth,
                        min_depth=min_depth,
                        max_depth=max_depth,
                    )
                    row = {
                        "sample_id": sample.sample_id,
                        "scene_id": sample.scene_id,
                        "image_path": sample.image_path,
                        "depth_path": sample.depth_path,
                        "prediction_type": raw_prediction.prediction_type,
                        "alignment": mode,
                        "alignment_scale": aligned.scale,
                        "alignment_shift": aligned.shift,
                        "alignment_fit_pixels": aligned.valid_pixels,
                        "inference_time_s": elapsed,
                        **metrics,
                    }
                    rows_by_mode[mode].append(row)

                    if save_visuals:
                        save_depth_visualization(
                            rgb=sample.image,
                            gt_depth=sample.depth,
                            pred_depth=eval_depth,
                            output_path=mode_dir / "visualizations" / f"{sample.sample_id}.png",
                            min_depth=min_depth,
                            max_depth=viz_max_depth,
                            prediction_type=raw_prediction.prediction_type,
                            alignment=mode,
                        )
                except AlignmentError as exc:
                    errors.append({"sample_id": sample.sample_id, "alignment": mode, "error": str(exc)})
                    print(f"  {mode} skipped: {exc}")
                    if not skip_errors:
                        raise
        except Exception as exc:
            errors.append({"sample_id": sample.sample_id, "alignment": "*", "error": str(exc)})
            print(f"  ERROR: {exc}")
            if not skip_errors:
                raise

    summaries: dict[str, dict] = {}
    for mode, rows in rows_by_mode.items():
        mode_dir = mode_dirs[mode]
        write_csv(mode_dir / "metrics_per_image.csv", rows, METRICS_PER_IMAGE_COLUMNS)
        write_csv(mode_dir / "metrics.csv", rows, METRICS_PER_IMAGE_COLUMNS)
        summary = _summary_payload(
            dataset_name=dataset_name,
            runner=runner,
            mode=mode,
            rows=rows,
            errors=errors,
            output_dir=mode_dir,
            min_depth=min_depth,
            max_depth=max_depth,
            device_name=device_name,
            save_visuals=save_visuals,
        )
        write_json(mode_dir / "metrics_summary.json", summary)
        write_json(mode_dir / "summary.json", summary)
        if errors:
            write_json(mode_dir / "errors.json", errors)
        summaries[mode] = summary
        print(f"{mode}: wrote {len(rows)} rows to {mode_dir}")

    return summaries
