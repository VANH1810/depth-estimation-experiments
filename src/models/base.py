"""Shared interfaces for depth model runners."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


PredictionType = str


@dataclass(frozen=True)
class DepthPrediction:
    """Raw output from a depth model before dataset-specific evaluation alignment."""

    depth: np.ndarray
    prediction_type: PredictionType
    model_id: str
    notes: str = ""


class DepthModelRunner(Protocol):
    """Minimal interface expected by the evaluator."""

    key: str
    display_name: str
    model_id: str
    prediction_type: PredictionType
    default_alignment_modes: tuple[str, ...]
    notes: str
    depth_unit: str
    training_data_note: str

    def load(self) -> None:
        """Load model weights and move the model to the configured device."""

    def predict(self, rgb: np.ndarray, intrinsics: np.ndarray | None = None) -> DepthPrediction:
        """Return a single-image depth prediction as a float32 HxW array."""
