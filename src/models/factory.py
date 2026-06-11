"""Factory for supported depth model runners."""

from pathlib import Path

from src.models.depth_anything_v2_runner import DEFAULT_METRIC_INDOOR_BASE_MODEL_ID
from src.models.depth_anything_v2_runner import DEFAULT_METRIC_INDOOR_SMALL_MODEL_ID
from src.models.depth_anything_v2_runner import DEFAULT_RAW_MODEL_ID as DEFAULT_DAV2_MODEL_ID
from src.models.depth_anything_v2_runner import DepthAnythingV2MetricIndoorBaseRunner
from src.models.depth_anything_v2_runner import DepthAnythingV2MetricIndoorSmallRunner
from src.models.depth_anything_v2_runner import DepthAnythingV2SmallRunner
from src.models.unidepth_runner import UniDepthRunner
from src.models.zoedepth_runner import DEFAULT_MODEL_ID as DEFAULT_ZOEDEPTH_MODEL_ID
from src.models.zoedepth_runner import ZoeDepthRunner


MODEL_CHOICES = (
    "unidepth",
    "depth_anything_v2_small",
    "depth_anything_v2_metric_indoor_small",
    "depth_anything_v2_metric_indoor_base",
    "zoedepth",
)
MODEL_ALIASES = {
    "depth_anything_v2_metric_indoor": "depth_anything_v2_metric_indoor_small",
}
MODEL_CLI_CHOICES = (*MODEL_CHOICES, *MODEL_ALIASES)


DEFAULT_MODEL_IDS = {
    "unidepth": "lpiccinelli/unidepth-v2-vitl14",
    "depth_anything_v2_small": DEFAULT_DAV2_MODEL_ID,
    "depth_anything_v2_metric_indoor": DEFAULT_METRIC_INDOOR_SMALL_MODEL_ID,
    "depth_anything_v2_metric_indoor_small": DEFAULT_METRIC_INDOOR_SMALL_MODEL_ID,
    "depth_anything_v2_metric_indoor_base": DEFAULT_METRIC_INDOOR_BASE_MODEL_ID,
    "zoedepth": DEFAULT_ZOEDEPTH_MODEL_ID,
}


def normalize_model_key(model_key: str) -> str:
    """Return the canonical model key used for result directories."""

    return MODEL_ALIASES.get(model_key, model_key)


def create_model_runner(
    model_key: str,
    device: str,
    model_id: str | None = None,
    resolution_level: int | None = 7,
    unidepth_root: str | Path | None = None,
):
    """Create a model runner by stable experiment key."""

    canonical_key = normalize_model_key(model_key)
    if canonical_key not in MODEL_CHOICES:
        raise ValueError(f"Unsupported model '{model_key}'. Choices: {', '.join(MODEL_CLI_CHOICES)}")
    resolved_model_id = model_id or DEFAULT_MODEL_IDS[model_key]
    if canonical_key == "unidepth":
        return UniDepthRunner(
            model_id=resolved_model_id,
            device=device,
            resolution_level=resolution_level,
            unidepth_root=unidepth_root,
        )
    if canonical_key == "depth_anything_v2_small":
        return DepthAnythingV2SmallRunner(model_id=resolved_model_id, device=device)
    if canonical_key == "depth_anything_v2_metric_indoor_small":
        return DepthAnythingV2MetricIndoorSmallRunner(model_id=resolved_model_id, device=device)
    if canonical_key == "depth_anything_v2_metric_indoor_base":
        return DepthAnythingV2MetricIndoorBaseRunner(model_id=resolved_model_id, device=device)
    if canonical_key == "zoedepth":
        return ZoeDepthRunner(model_id=resolved_model_id, device=device)
    raise AssertionError(f"Unhandled model key: {model_key}")
