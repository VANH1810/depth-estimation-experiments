"""UniDepth runner used by the multi-model experiment pipeline."""

from pathlib import Path

import numpy as np
import torch

from src.common.pipeline import load_unidepth_v2, predict_depth, resolve_device
from src.models.base import DepthPrediction


class UniDepthRunner:
    """Run UniDepth V2 in the same zero-shot mode as the legacy scripts."""

    key = "unidepth"
    display_name = "UniDepth V2"
    prediction_type = "metric"
    default_alignment_modes = ("raw",)
    notes = "UniDepth predicts metric depth and camera intrinsics from a single RGB image."
    depth_unit = "meter"
    training_data_note = "No target-dataset fine-tuning is applied in this experiment pipeline."

    def __init__(
        self,
        model_id: str = "lpiccinelli/unidepth-v2-vitl14",
        device: str = "auto",
        resolution_level: int | None = 7,
        unidepth_root: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.resolution_level = resolution_level
        self.unidepth_root = unidepth_root
        self.device: torch.device | None = None
        self.model = None

    def load(self) -> None:
        self.device = resolve_device(self.device_name)
        self.model = load_unidepth_v2(
            self.model_id,
            self.device,
            self.resolution_level,
            unidepth_root=self.unidepth_root,
        )

    def predict(self, rgb: np.ndarray) -> DepthPrediction:
        if self.model is None or self.device is None:
            raise RuntimeError("UniDepthRunner.load() must be called before predict().")
        depth = predict_depth(self.model, rgb, self.device)
        return DepthPrediction(
            depth=depth.astype(np.float32, copy=False),
            prediction_type=self.prediction_type,
            model_id=self.model_id,
            notes=self.notes,
        )
