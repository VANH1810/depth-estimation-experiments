"""ZoeDepth runner for metric-depth inference."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.common.pipeline import resolve_device
from src.models.base import DepthPrediction


DEFAULT_MODEL_ID = "Intel/zoedepth-nyu-kitti"


class ZoeDepthRunner:
    """Run ZoeDepth through Hugging Face Transformers."""

    key = "zoedepth"
    display_name = "ZoeDepth"
    prediction_type = "metric"
    default_alignment_modes = ("raw",)
    notes = (
        "ZoeDepth combines relative and metric depth estimation and outputs metric-scale depth. "
        "The default checkpoint is Intel/zoedepth-nyu-kitti."
    )
    depth_unit = "meter"
    training_data_note = "Default ZoeDepth checkpoint is treated as a metric-depth baseline in this pipeline."

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        self.model_id = model_id
        self.device_name = device
        self.device: torch.device | None = None
        self.processor = None
        self.model = None

    def load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "ZoeDepth requires transformers with ZoeDepth support. "
                "Install with: pip install transformers>=4.45.0"
            ) from exc

        self.device = resolve_device(self.device_name)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device).eval()

    def predict(self, rgb: np.ndarray) -> DepthPrediction:
        if self.model is None or self.processor is None or self.device is None:
            raise RuntimeError("ZoeDepthRunner.load() must be called before predict().")
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        depth = self._post_process(outputs, target_shape=rgb.shape[:2])
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth <= 0.0] = 0.0
        return DepthPrediction(
            depth=depth,
            prediction_type=self.prediction_type,
            model_id=self.model_id,
            notes=self.notes,
        )

    def _post_process(self, outputs, target_shape: tuple[int, int]) -> np.ndarray:
        if hasattr(self.processor, "post_process_depth_estimation"):
            processed = self.processor.post_process_depth_estimation(outputs, source_sizes=[target_shape])
            first = processed[0]
            if isinstance(first, dict):
                if "predicted_depth" in first:
                    tensor = first["predicted_depth"]
                elif "depth" in first:
                    tensor = first["depth"]
                else:
                    raise RuntimeError(f"Unexpected ZoeDepth post-process keys: {sorted(first)}")
            else:
                tensor = first
            return tensor.detach().float().cpu().numpy()

        predicted_depth = outputs.predicted_depth
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=target_shape,
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze().detach().float().cpu().numpy()
