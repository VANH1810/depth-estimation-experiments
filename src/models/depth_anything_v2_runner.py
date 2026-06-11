"""Depth Anything V2 Small runner.

The requested checkpoint, depth-anything/Depth-Anything-V2-Small, is a relative
depth model. For Transformers inference we use the official converted
depth-anything/Depth-Anything-V2-Small-hf checkpoint unless the caller passes a
different model id.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.common.pipeline import resolve_device
from src.models.base import DepthPrediction


DEFAULT_RAW_MODEL_ID = "depth-anything/Depth-Anything-V2-Small"
DEFAULT_TRANSFORMERS_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_METRIC_INDOOR_SMALL_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_METRIC_INDOOR_BASE_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


def transformers_model_id_for(model_id: str) -> str:
    """Map the official raw checkpoint id to its official Transformers id."""

    if model_id == DEFAULT_RAW_MODEL_ID:
        return DEFAULT_TRANSFORMERS_MODEL_ID
    return model_id


class DepthAnythingV2SmallRunner:
    """Run Depth Anything V2 Small as relative-depth inference."""

    key = "depth_anything_v2_small"
    display_name = "Depth Anything V2 Small"
    prediction_type = "relative"
    default_alignment_modes = ("raw", "median_aligned", "scale_shift_aligned")
    relative_notes = (
        "depth-anything/Depth-Anything-V2-Small is a relative-depth checkpoint. "
        "Raw metrics are not an absolute metric-depth comparison; aligned modes "
        "fit scale using the ground truth."
    )
    metric_notes = (
        "This Depth Anything V2 checkpoint name indicates a metric-depth variant. "
        "It is evaluated as metric depth in raw mode by default."
    )
    depth_unit = "relative"
    training_data_note = "Depth Anything V2 Small is a relative-depth checkpoint, not metric depth."

    def __init__(self, model_id: str = DEFAULT_RAW_MODEL_ID, device: str = "auto") -> None:
        self.model_id = model_id
        self.transformers_model_id = transformers_model_id_for(model_id)
        self.device_name = device
        if "metric" in model_id.lower():
            self.prediction_type = "metric"
            self.default_alignment_modes = ("raw",)
            self.notes = self.metric_notes
            self.depth_unit = "meter"
            self.training_data_note = (
                "Metric Depth Anything V2 variants are fine-tuned for metric depth; "
                "check the selected checkpoint for dataset/domain overlap."
            )
        else:
            self.prediction_type = "relative"
            self.default_alignment_modes = ("raw", "median_aligned", "scale_shift_aligned")
            self.notes = self.relative_notes
            self.depth_unit = "relative"
            self.training_data_note = "Depth Anything V2 Small is a relative-depth checkpoint, not metric depth."
        self.device: torch.device | None = None
        self.processor = None
        self.model = None

    def load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "Depth Anything V2 Small requires transformers>=4.45.0. "
                "Install with: pip install transformers>=4.45.0"
            ) from exc

        self.device = resolve_device(self.device_name)
        self.processor = AutoImageProcessor.from_pretrained(self.transformers_model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.transformers_model_id)
        self.model.to(self.device).eval()

    def predict(self, rgb: np.ndarray) -> DepthPrediction:
        if self.model is None or self.processor is None or self.device is None:
            raise RuntimeError("DepthAnythingV2SmallRunner.load() must be called before predict().")
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )

        depth = prediction.squeeze().detach().float().cpu().numpy().astype(np.float32)
        return DepthPrediction(
            depth=depth,
            prediction_type=self.prediction_type,
            model_id=self.model_id,
            notes=self.notes,
        )


class DepthAnythingV2MetricIndoorSmallRunner(DepthAnythingV2SmallRunner):
    """Run Depth Anything V2 Metric Indoor Small as metric-depth inference."""

    key = "depth_anything_v2_metric_indoor_small"
    display_name = "Depth Anything V2 Metric Indoor Small"
    prediction_type = "metric"
    default_alignment_modes = ("raw",)
    depth_unit = "meter"
    training_data_note = (
        "Depth Anything V2 Metric Indoor Small is fine-tuned for indoor metric depth "
        "using synthetic Hypersim data; do not use it as a primary few-shot/zero-shot "
        "baseline on Hypersim."
    )
    notes = (
        "Depth Anything V2 Metric Indoor Small outputs metric indoor depth in meters. "
        "The checkpoint is trained/fine-tuned with Hypersim indoor metric data, so "
        "Hypersim results are train-overlap/in-domain references only."
    )

    def __init__(self, model_id: str = DEFAULT_METRIC_INDOOR_SMALL_MODEL_ID, device: str = "auto") -> None:
        super().__init__(model_id=model_id, device=device)
        self.prediction_type = "metric"
        self.default_alignment_modes = ("raw",)
        self.depth_unit = "meter"
        self.training_data_note = self.__class__.training_data_note
        self.notes = self.__class__.notes


class DepthAnythingV2MetricIndoorBaseRunner(DepthAnythingV2MetricIndoorSmallRunner):
    """Run Depth Anything V2 Metric Indoor Base as metric-depth inference."""

    key = "depth_anything_v2_metric_indoor_base"
    display_name = "Depth Anything V2 Metric Indoor Base"
    training_data_note = (
        "Depth Anything V2 Metric Indoor Base is fine-tuned for indoor metric depth "
        "using synthetic Hypersim data; do not use it as a primary few-shot/zero-shot "
        "baseline on Hypersim."
    )
    notes = (
        "Depth Anything V2 Metric Indoor Base outputs metric indoor depth in meters. "
        "The checkpoint is trained/fine-tuned with Hypersim indoor metric data, so "
        "Hypersim results are train-overlap/in-domain references only."
    )

    def __init__(self, model_id: str = DEFAULT_METRIC_INDOOR_BASE_MODEL_ID, device: str = "auto") -> None:
        super().__init__(model_id=model_id, device=device)
        self.training_data_note = self.__class__.training_data_note
        self.notes = self.__class__.notes
