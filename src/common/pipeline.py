import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.visualization.style import DEPTH_CMAP, ERROR_CMAP


METRIC_COLUMNS = [
    "sample_id",
    "scene_id",
    "image_path",
    "depth_path",
    "valid_pixels",
    "inference_time_s",
    "d1",
    "d2",
    "d3",
    "arel",
    "sqrel",
    "rmse",
    "rmselog",
    "log10",
    "silog",
    "mae",
    "d1_ssi",
    "arel_ssi",
]


@dataclass(frozen=True)
class DepthSample:
    image: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray | None
    image_path: str
    depth_path: str
    scene_id: str
    sample_id: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_unidepth_root() -> Path:
    return repo_root() / "UniDepth"


def ensure_unidepth_import(unidepth_root: str | Path | None = None) -> None:
    root = Path(unidepth_root) if unidepth_root else default_unidepth_root()
    if not root.exists():
        raise FileNotFoundError(f"UniDepth source directory not found: {root}")
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_unidepth_v2(
    model_name: str,
    device: torch.device,
    resolution_level: int | None,
    unidepth_root: str | Path | None = None,
):
    ensure_unidepth_import(unidepth_root)
    from unidepth.models import UniDepthV2

    model = UniDepthV2.from_pretrained(model_name)
    model.interpolation_mode = "bilinear"
    if resolution_level is not None:
        model.resolution_level = resolution_level
    return model.to(device).eval()


def rgb_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(device)


def predict_depth(
    model,
    rgb: np.ndarray,
    device: torch.device,
    intrinsics: np.ndarray | None = None,
) -> np.ndarray:
    rgb_tensor = rgb_to_tensor(rgb, device)
    camera = None
    if intrinsics is not None:
        camera = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).to(device)
    with torch.inference_mode():
        # camera=None keeps the legacy zero-shot setup where UniDepth predicts intrinsics.
        pred = model.infer(rgb_tensor, camera=camera, normalize=True)
    depth = pred["depth"].squeeze().detach().float().cpu()
    return depth.numpy().astype(np.float32)


def resize_depth_to(depth_pred: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if depth_pred.shape == target_shape:
        return depth_pred.astype(np.float32, copy=False)
    tensor = torch.from_numpy(depth_pred).float()[None, None]
    resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy().astype(np.float32)


def valid_depth_mask(
    depth_gt: np.ndarray,
    depth_pred: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> np.ndarray:
    mask = np.isfinite(depth_gt) & (depth_gt > min_depth)
    if max_depth is not None:
        mask &= depth_gt <= max_depth
    if depth_pred is not None:
        mask &= np.isfinite(depth_pred) & (depth_pred > min_depth)
    return mask


def ssi_align(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    stability_mat = 1e-9 * torch.eye(2, device=pred.device)
    pred_one = torch.stack([pred, torch.ones_like(pred)], dim=1)
    scale_shift = torch.inverse(pred_one.T @ pred_one + stability_mat) @ (pred_one.T @ gt.unsqueeze(1))
    scale, shift = scale_shift.squeeze().chunk(2, dim=0)
    return pred * scale + shift


def compute_depth_metrics(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float | None = None,
) -> dict[str, float]:
    mask = valid_depth_mask(depth_gt, depth_pred, min_depth=min_depth, max_depth=max_depth)
    valid_pixels = int(mask.sum())
    if valid_pixels == 0:
        return {"valid_pixels": 0}

    gt = torch.from_numpy(depth_gt[mask].astype(np.float32))
    pred = torch.from_numpy(depth_pred[mask].astype(np.float32)).clamp(min=min_depth)

    thresh = torch.maximum(gt / pred, pred / gt)
    diff = gt - pred
    diff_log = torch.log(pred) - torch.log(gt)

    pred_ssi = ssi_align(gt, pred).clamp(min=min_depth)
    thresh_ssi = torch.maximum(gt / pred_ssi, pred_ssi / gt)

    return {
        "valid_pixels": valid_pixels,
        "d1": (thresh < 1.25).float().mean().item(),
        "d2": (thresh < 1.25**2).float().mean().item(),
        "d3": (thresh < 1.25**3).float().mean().item(),
        "arel": (diff.abs() / gt).mean().item(),
        "sqrel": ((diff**2) / gt).mean().item(),
        "rmse": torch.sqrt((diff**2).mean()).item(),
        "rmselog": torch.sqrt((diff_log**2).mean()).item(),
        "log10": (torch.log10(pred) - torch.log10(gt)).abs().mean().item(),
        "silog": (100.0 * torch.std(diff_log)).item(),
        "mae": diff.abs().mean().item(),
        "d1_ssi": (thresh_ssi < 1.25).float().mean().item(),
        "arel_ssi": ((gt - pred_ssi).abs() / gt).mean().item(),
    }


def make_sample_id(scene_id: str, frame_id: str | int) -> str:
    frame = f"frame_{frame_id:06d}" if isinstance(frame_id, int) else frame_id.replace(".", "_")
    return f"{scene_id}_{frame}"


def intrinsics_from_camera_json(camera_json: dict) -> np.ndarray | None:
    intr = camera_json.get("color_intrinsics") or camera_json.get("depth_intrinsics")
    if not intr:
        return None
    return np.array(
        [[intr["fx"], 0.0, intr["ppx"]], [0.0, intr["fy"], intr["ppy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class DepthTestDataset:
    """Loader for Depth_Test recordings.

    Depth_Test stores RealSense depth as uint16 millimeters in NPZ batches. The
    camera metadata provides depth_scale, usually 0.001, so samples expose meters.
    """

    def __init__(self, data_root: str | Path, sample_every: int = 5, max_samples: int | None = None):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Depth_Test root not found: {self.data_root}")
        self.sample_every = max(1, sample_every)
        self.records = self._find_recordings()
        self.samples = self._index_samples(max_samples)

    def _find_recordings(self) -> list[Path]:
        if (self.data_root / "color.avi").exists():
            return [self.data_root]
        recordings = sorted(p for p in self.data_root.glob("recording_*") if (p / "color.avi").exists())
        if not recordings:
            raise FileNotFoundError(f"No Depth_Test recording folders found under {self.data_root}")
        return recordings

    def _index_samples(self, max_samples: int | None) -> list[dict]:
        indexed = []
        for rec_path in self.records:
            meta_path = rec_path / "depth_metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing depth metadata: {meta_path}")
            with open(meta_path, encoding="utf-8") as f:
                depth_meta = json.load(f)
            cap = cv2.VideoCapture(str(rec_path / "color.avi"))
            n_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            n_depth = int(depth_meta.get("total_frames", 0))
            n_frames = min(n_rgb, n_depth)
            batch_ranges = []
            start = 0
            for batch in depth_meta["batches"]:
                count = int(batch["frame_count"])
                batch_ranges.append((start, start + count, batch["filename"]))
                start += count
            for idx in range(0, n_frames, self.sample_every):
                for start_idx, end_idx, filename in batch_ranges:
                    if start_idx <= idx < end_idx:
                        indexed.append(
                            {
                                "recording": rec_path,
                                "frame_index": idx,
                                "batch_file": filename,
                                "batch_offset": idx - start_idx,
                            }
                        )
                        break
                if max_samples is not None and len(indexed) >= max_samples:
                    return indexed
        return indexed

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterable[DepthSample]:
        for item in self.samples:
            rec_path = item["recording"]
            scene_id = rec_path.name
            frame_index = item["frame_index"]

            cam_path = rec_path / "camera_parameters.json"
            with open(cam_path, encoding="utf-8") as f:
                camera_json = json.load(f)
            depth_scale = float(camera_json.get("depth_scale", 0.001))
            intrinsics = intrinsics_from_camera_json(camera_json)

            cap = cv2.VideoCapture(str(rec_path / "color.avi"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"Could not read RGB frame {frame_index} from {rec_path / 'color.avi'}")
            image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            depth_file = rec_path / "depth_data" / item["batch_file"]
            with np.load(depth_file) as batch:
                depth = batch["depth_frames"][item["batch_offset"]].astype(np.float32) * depth_scale

            yield DepthSample(
                image=image,
                depth=depth,
                intrinsics=intrinsics,
                image_path=f"{rec_path / 'color.avi'}#frame={frame_index}",
                depth_path=f"{depth_file}#index={item['batch_offset']}",
                scene_id=scene_id,
                sample_id=make_sample_id(scene_id, frame_index),
            )


class HypersimDataset:
    """Loader for extracted HyperSim RGB/depth pairs.

    The extractor writes HyperSim depth_meters HDF5 values directly to .npy, so
    this loader keeps depth values in meters and does not apply an extra scale.
    """

    def __init__(self, data_root: str | Path, sample_every: int = 1, max_samples: int | None = None):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"HyperSim root not found: {self.data_root}")
        self.sample_every = max(1, sample_every)
        self.rgb_dir = self.data_root / "rgb"
        self.depth_dir = self.data_root / "depth"
        if not self.rgb_dir.exists() or not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Expected extracted HyperSim folders '{self.rgb_dir}' and '{self.depth_dir}'. "
                "Run python scripts/extract_hypersim.py first for raw HDF5 data."
            )
        self.intrinsics = self._load_intrinsics()
        image_files = sorted(self.rgb_dir.glob("*.png")) + sorted(self.rgb_dir.glob("*.jpg"))
        self.image_files = image_files[:: self.sample_every]
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]

    def _load_intrinsics(self) -> dict:
        path = self.data_root / "intrinsics.json"
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self) -> Iterable[DepthSample]:
        scene_id = self.data_root.name
        for image_path in self.image_files:
            frame_id = image_path.stem
            depth_path = self.depth_dir / f"{frame_id}.npy"
            if not depth_path.exists():
                depth_path = self.depth_dir / f"{frame_id}.png"
            if not depth_path.exists():
                print(f"Skipping {image_path.name}: missing depth file in {self.depth_dir}")
                continue

            image = np.array(Image.open(image_path).convert("RGB"))
            if depth_path.suffix.lower() == ".npy":
                depth = np.load(depth_path).astype(np.float32)
            else:
                depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0

            k_value = (
                self.intrinsics.get(image_path.name)
                or self.intrinsics.get(frame_id)
                or self.intrinsics.get(str(image_path))
            )
            intrinsics = np.array(k_value, dtype=np.float32) if k_value is not None else None

            yield DepthSample(
                image=image,
                depth=depth,
                intrinsics=intrinsics,
                image_path=str(image_path),
                depth_path=str(depth_path),
                scene_id=scene_id,
                sample_id=make_sample_id(scene_id, frame_id),
            )


def colorize_depth(array: np.ndarray, vmin: float, vmax: float, cmap: str = DEPTH_CMAP) -> np.ndarray:
    ensure_unidepth_import()
    from unidepth.utils import colorize

    safe = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return colorize(safe, vmin=vmin, vmax=vmax, cmap=cmap)


def save_visualization(
    sample: DepthSample,
    depth_pred: np.ndarray,
    output_path: Path,
    depth_vmin: float,
    depth_vmax: float,
) -> None:
    ensure_unidepth_import()
    from unidepth.utils import image_grid

    valid = valid_depth_mask(sample.depth, depth_pred, max_depth=depth_vmax)
    error = np.zeros_like(sample.depth, dtype=np.float32)
    error[valid] = np.abs(sample.depth[valid] - depth_pred[valid]) / np.maximum(sample.depth[valid], 1e-6)

    gt_col = colorize_depth(sample.depth, vmin=depth_vmin, vmax=depth_vmax, cmap=DEPTH_CMAP)
    pred_col = colorize_depth(depth_pred, vmin=depth_vmin, vmax=depth_vmax, cmap=DEPTH_CMAP)
    err_col = colorize_depth(error, vmin=0.0, vmax=0.3, cmap=ERROR_CMAP)
    grid = image_grid([sample.image, gt_col, pred_col, err_col], 2, 2)
    Image.fromarray(grid).save(output_path)


def summarize(rows: list[dict]) -> dict:
    metric_names = [c for c in METRIC_COLUMNS if c not in {"sample_id", "scene_id", "image_path", "depth_path"}]
    averages = {}
    stddevs = {}
    for name in metric_names:
        vals = [float(r[name]) for r in rows if r.get(name) not in (None, "")]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            averages[name] = float(np.mean(vals))
            stddevs[name] = float(np.std(vals))
    return {"avg_metrics": averages, "std_metrics": stddevs}


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


def run_dataset(
    dataset_name: str,
    dataset,
    output_dir: str | Path,
    model_name: str,
    device_name: str,
    resolution_level: int | None,
    min_depth: float,
    max_depth: float | None,
    save_visuals: bool,
    viz_max_depth: float,
    skip_errors: bool,
    unidepth_root: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    viz_dir = output_dir / "visualizations"
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if save_visuals:
        viz_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_name)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(dataset)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Loading model: {model_name}")
    model = load_unidepth_v2(model_name, device, resolution_level, unidepth_root=unidepth_root)

    rows: list[dict] = []
    errors: list[dict] = []
    for index, sample in enumerate(dataset, start=1):
        print(f"[{index}/{len(dataset)}] {sample.sample_id}")
        try:
            t0 = time.time()
            depth_pred = predict_depth(model, sample.image, device)
            elapsed = time.time() - t0
            depth_pred = resize_depth_to(depth_pred, sample.depth.shape)

            if not np.isfinite(depth_pred).any() or np.nanmax(np.abs(depth_pred)) <= min_depth:
                raise RuntimeError("Predicted depth is empty, non-finite, or all zeros.")

            np.save(pred_dir / f"{sample.sample_id}.npy", depth_pred.astype(np.float32))
            metrics = compute_depth_metrics(depth_pred, sample.depth, min_depth=min_depth, max_depth=max_depth)
            row = {
                "sample_id": sample.sample_id,
                "scene_id": sample.scene_id,
                "image_path": sample.image_path,
                "depth_path": sample.depth_path,
                "inference_time_s": elapsed,
                **metrics,
            }
            rows.append(row)

            if save_visuals:
                save_visualization(
                    sample,
                    depth_pred,
                    viz_dir / f"{sample.sample_id}.png",
                    depth_vmin=min_depth,
                    depth_vmax=viz_max_depth,
                )
        except Exception as exc:
            error = {"sample_id": sample.sample_id, "error": str(exc)}
            errors.append(error)
            print(f"  ERROR: {exc}")
            if not skip_errors:
                raise

    write_metrics_csv(metrics_path, rows)
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": len(rows),
        "num_errors": len(errors),
        "device": str(device),
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "output_structure": {
            "predictions": str(pred_dir),
            "visualizations": str(viz_dir) if save_visuals else None,
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        **summarize(rows),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if errors:
        with open(output_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")
    return summary


def add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_root", default=None, help="Dataset root.")
    parser.add_argument("--output_dir", default=None, help="Output directory.")
    parser.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14", help="HuggingFace model id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--resolution_level", type=int, default=7, help="UniDepth V2 resolution level [0, 10).")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for a quick run.")
    parser.add_argument("--sample_every", type=int, default=1, help="Use every Nth sample/frame.")
    parser.add_argument("--min_depth", type=float, default=1e-3, help="Minimum valid depth in meters.")
    parser.add_argument("--max_depth", type=float, default=None, help="Maximum valid depth in meters.")
    parser.add_argument("--viz_max_depth", type=float, default=5.0, help="Visualization depth upper bound in meters.")
    parser.add_argument("--no_visualizations", action="store_true", help="Do not write visualization PNG files.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop instead of skipping failed samples.")
    parser.add_argument("--unidepth_root", default=None, help="Path to local UniDepth source.")
