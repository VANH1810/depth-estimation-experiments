"""Shared utilities for paper-style depth report figures."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_LABELS = {
    "unidepth": "UniDepth",
    "zoedepth": "ZoeDepth",
    "depth_anything_v2_small": "DA-V2 Small",
    "depth_anything_v2_metric_indoor": "DA-V2 Metric",
    "depth_anything_v2_metric_indoor_small": "DA-V2 Metric",
    "depth_anything_v2_metric_indoor_base": "DA-V2 Metric Base",
}

MODEL_ORDER = {
    "unidepth": 0,
    "zoedepth": 1,
    "depth_anything_v2_metric_indoor_small": 2,
    "depth_anything_v2_metric_indoor_base": 3,
    "depth_anything_v2_small": 4,
}

ALIGNMENT_LABELS = {
    "raw": "raw",
    "median_aligned": "median aligned",
    "scale_shift_aligned": "S+S aligned",
}

ALIGNMENT_ORDER = {
    "raw": 0,
    "scale_shift_aligned": 1,
    "median_aligned": 2,
}


@dataclass(frozen=True)
class ResultEntry:
    dataset: str
    model: str
    alignment: str
    result_dir: Path
    metrics_path: Path
    predictions_dir: Path
    metrics_df: pd.DataFrame
    summary: dict[str, Any]
    comparison_row: dict[str, Any]

    @property
    def protocol_group(self) -> str:
        return str(self.comparison_row.get("protocol_group") or self.summary.get("protocol_group") or "other")

    @property
    def prediction_type(self) -> str:
        return str(
            self.comparison_row.get("prediction_type")
            or self.summary.get("prediction_type")
            or _first_nonempty(self.metrics_df, "prediction_type")
            or "unknown"
        )

    @property
    def depth_unit(self) -> str:
        return str(self.comparison_row.get("depth_unit") or self.summary.get("depth_unit") or "unknown")

    @property
    def max_depth_m(self) -> float | None:
        value = self.summary.get("max_depth_m")
        if value in (None, ""):
            if "gt_max_depth" in self.metrics_df.columns:
                values = pd.to_numeric(self.metrics_df["gt_max_depth"], errors="coerce")
                finite = values[np.isfinite(values)]
                if not finite.empty:
                    return float(finite.max())
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @property
    def min_depth_m(self) -> float:
        value = self.summary.get("min_depth_m", 1e-3)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1e-3
        return value if math.isfinite(value) and value > 0 else 1e-3

    def metric_value(self, metric: str) -> float | None:
        for source in (self.comparison_row, self.summary.get("avg_metrics", {})):
            value = source.get(metric) if isinstance(source, dict) else None
            parsed = _to_float(value)
            if parsed is not None:
                return parsed
        if metric in self.metrics_df.columns:
            values = pd.to_numeric(self.metrics_df[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            if not finite.empty:
                return float(finite.mean())
        return None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_nonempty(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _infer_protocol(dataset: str, model: str, alignment: str, summary: dict[str, Any], metrics_df: pd.DataFrame) -> dict:
    prediction_type = summary.get("prediction_type") or _first_nonempty(metrics_df, "prediction_type") or "unknown"
    depth_unit = summary.get("depth_unit", "unknown")
    if model.startswith("depth_anything_v2_metric_indoor") and dataset == "hypersim":
        return {
            "protocol_group": "train_overlap_reference",
            "is_primary": False,
            "is_fewshot_valid": False,
            "has_train_overlap_risk": True,
            "prediction_type": prediction_type,
            "depth_unit": depth_unit,
        }
    if model == "depth_anything_v2_small":
        return {
            "protocol_group": "relative_aligned" if alignment in {"median_aligned", "scale_shift_aligned"} else "relative_raw",
            "is_primary": False,
            "is_fewshot_valid": True,
            "has_train_overlap_risk": False,
            "prediction_type": prediction_type,
            "depth_unit": depth_unit,
        }
    if prediction_type == "metric" and alignment == "raw":
        return {
            "protocol_group": "primary_metric_raw",
            "is_primary": True,
            "is_fewshot_valid": True,
            "has_train_overlap_risk": False,
            "prediction_type": prediction_type,
            "depth_unit": depth_unit,
        }
    return {
        "protocol_group": "other",
        "is_primary": False,
        "is_fewshot_valid": True,
        "has_train_overlap_risk": False,
        "prediction_type": prediction_type,
        "depth_unit": depth_unit,
    }


def load_results_index(results_root: str | Path, dataset: str) -> list[ResultEntry]:
    """Load result directories and comparison tags for one dataset."""

    results_root = Path(results_root)
    dataset_root = results_root / dataset
    comparison = _read_csv(dataset_root / "comparison" / "comparison_summary.csv")
    comparison_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if not comparison.empty:
        for _, row in comparison.iterrows():
            comparison_by_key[(str(row.get("model", "")), str(row.get("alignment", "")))] = row.to_dict()

    entries: list[ResultEntry] = []
    if not dataset_root.exists():
        return entries
    for mode_dir in sorted(path for path in dataset_root.glob("*/*") if path.is_dir()):
        metrics_path = mode_dir / "metrics_per_image.csv"
        if not metrics_path.exists():
            metrics_path = mode_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        model = mode_dir.parent.name
        alignment = mode_dir.name
        metrics_df = _read_csv(metrics_path)
        if metrics_df.empty:
            continue
        summary_path = mode_dir / "metrics_summary.json"
        summary = _read_json(summary_path)
        comparison_row = comparison_by_key.get((model, alignment), {})
        if not comparison_row:
            comparison_row = _infer_protocol(dataset, model, alignment, summary, metrics_df)
        entries.append(
            ResultEntry(
                dataset=dataset,
                model=model,
                alignment=alignment,
                result_dir=mode_dir,
                metrics_path=metrics_path,
                predictions_dir=mode_dir / "predictions",
                metrics_df=metrics_df,
                summary=summary,
                comparison_row=comparison_row,
            )
        )
    return entries


def resolve_models_for_protocol(
    entries: list[ResultEntry],
    protocol: str = "primary",
    include_reference: bool = False,
) -> list[ResultEntry]:
    """Return entries that are meaningful for the selected report protocol."""

    selected: list[ResultEntry] = []
    for entry in entries:
        group = entry.protocol_group
        if protocol == "primary" and group == "primary_metric_raw":
            selected.append(entry)
        elif protocol == "aligned" and group == "relative_aligned":
            selected.append(entry)
        elif protocol == "all" and group in {"primary_metric_raw", "relative_aligned"}:
            selected.append(entry)
        if include_reference and group == "train_overlap_reference":
            selected.append(entry)

    unique: dict[tuple[str, str], ResultEntry] = {}
    for entry in selected:
        unique[(entry.model, entry.alignment)] = entry
    return sorted(unique.values(), key=lambda item: _sort_key(item))


def _sort_key(entry: ResultEntry) -> tuple[int, int, float]:
    metric = entry.metric_value("AbsRel")
    return (
        MODEL_ORDER.get(entry.model, 99),
        ALIGNMENT_ORDER.get(entry.alignment, 99),
        metric if metric is not None else float("inf"),
    )


def short_model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def entry_label(entry: ResultEntry, *, multiline: bool = False, include_alignment: bool = True) -> str:
    label = short_model_label(entry.model)
    if not include_alignment:
        return label
    alignment = ALIGNMENT_LABELS.get(entry.alignment, entry.alignment.replace("_", " "))
    if entry.model == "depth_anything_v2_small":
        if entry.alignment == "scale_shift_aligned":
            suffix = "S+S"
        elif entry.alignment == "median_aligned":
            suffix = "median"
        else:
            suffix = "raw relative"
        return f"{label}{chr(10) if multiline else ' '}{suffix}"
    if entry.alignment != "raw":
        return f"{label}{chr(10) if multiline else ' '}{alignment}"
    if entry.protocol_group == "train_overlap_reference":
        return f"{label}{chr(10) if multiline else ' '}reference"
    return label


def figure_suffix(protocol: str, include_reference: bool = False) -> str:
    if protocol == "primary" and not include_reference:
        return ""
    if protocol == "primary":
        return "_primary_with_reference"
    if include_reference:
        return f"_{protocol}_with_reference"
    return f"_{protocol}"


def resolve_data_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return repo_root() / path


def load_rgb(image_path: str) -> np.ndarray:
    if "#frame=" in image_path:
        video_path_text, frame_part = image_path.split("#frame=", maxsplit=1)
        video_path = resolve_data_path(video_path_text)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_part))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read RGB frame {frame_part} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(resolve_data_path(image_path)).convert("RGB"))


def _depth_scale_for_npz(npz_path: Path) -> float:
    camera_path = npz_path.parent.parent / "camera_parameters.json"
    if camera_path.exists():
        try:
            with open(camera_path, encoding="utf-8") as handle:
                return float(json.load(handle).get("depth_scale", 0.001))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0.001
    return 0.001


def load_gt_depth(depth_path: str) -> np.ndarray:
    if "#index=" in depth_path:
        npz_path_text, index_part = depth_path.split("#index=", maxsplit=1)
        npz_path = resolve_data_path(npz_path_text)
        with np.load(npz_path) as batch:
            depth = batch["depth_frames"][int(index_part)].astype(np.float32)
        return depth * _depth_scale_for_npz(npz_path)
    path = resolve_data_path(depth_path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    return np.array(Image.open(path)).astype(np.float32) / 1000.0


def load_prediction(entry: ResultEntry, sample_id: str) -> np.ndarray:
    return np.load(entry.predictions_dir / f"{sample_id}.npy").astype(np.float32)


def resize_to_match(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array.astype(np.float32, copy=False)
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def get_valid_mask(
    gt: np.ndarray,
    pred: np.ndarray | None = None,
    *,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
    require_positive_pred: bool = True,
) -> np.ndarray:
    mask = np.isfinite(gt) & (gt > min_depth)
    if max_depth is not None:
        mask &= gt < max_depth
    if pred is not None:
        mask &= np.isfinite(pred)
        if require_positive_pred:
            mask &= pred > 0
    return mask


def compute_absrel_error_map(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    error = np.full(gt.shape, np.nan, dtype=np.float32)
    denom = np.maximum(gt[valid_mask].astype(np.float32), 1e-6)
    error[valid_mask] = np.abs(pred[valid_mask].astype(np.float32) - gt[valid_mask].astype(np.float32)) / denom
    return error


def sample_valid_points(
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(valid_mask.ravel())
    if indices.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    count = min(int(max_points), int(indices.size))
    if indices.size > count:
        indices = rng.choice(indices, size=count, replace=False)
    gt_flat = gt.ravel()
    pred_flat = pred.ravel()
    return gt_flat[indices].astype(np.float32), pred_flat[indices].astype(np.float32)


def colorize_array(
    data: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not math.isfinite(vmin):
        vmin = 0.0
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    clean = np.nan_to_num(data.astype(np.float32), nan=vmin, posinf=vmax, neginf=vmin)
    norm = np.clip((clean - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = plt.get_cmap(cmap)(norm)[..., :3]
    if valid_mask is not None:
        rgb = rgb.copy()
        rgb[~valid_mask] = 1.0
    return rgb


def save_figure(fig, path: Path, *, dpi: int = 300, overwrite: bool = False) -> Path | None:
    if path.exists() and not overwrite:
        print(f"skip existing figure: {path}")
        plt.close(fig)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        print(f"WARNING blank/empty figure: {path}")
    else:
        print(f"wrote figure: {path}")
    return path


def metric_text(entry: ResultEntry, sampled_points: int | None = None) -> str:
    absrel = entry.metric_value("AbsRel")
    rmse = entry.metric_value("RMSE")
    delta1 = entry.metric_value("delta1")
    lines = []
    if absrel is not None:
        lines.append(f"AbsRel={absrel:.3f}")
    if rmse is not None:
        lines.append(f"RMSE={rmse:.3f}")
    if delta1 is not None:
        lines.append(f"δ1={delta1:.3f}")
    if sampled_points is not None:
        lines.append(f"n={sampled_points:,}")
    return "\n".join(lines)
