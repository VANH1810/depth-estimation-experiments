"""Dataset entry points for multi-model depth experiments."""

from src.datasets.depth_test import DepthTestDataset
from src.datasets.hypersim import HypersimDataset

__all__ = ["DepthTestDataset", "HypersimDataset"]
