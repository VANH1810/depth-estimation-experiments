"""Shared Matplotlib style for report figures."""

from __future__ import annotations

import re

import matplotlib
from cycler import cycler


FONT_SERIF = ["Times New Roman", "Times"]
FIGURE_FONT_SIZE = 18

BAR_COLORS = {
    "unidepth": "#00129A",
    "zoedepth": "#75190A",
    "depth_anything_v2": "#4D4C19",
    "depth_anything_v2_small": "#4D4C19",
    "depth_anything_v2_metric": "#4D4C19",
    "depth_anything_v2_metric_indoor": "#4D4C19",
    "depth_anything_v2_metric_indoor_small": "#4D4C19",
    "depth_anything_v2_metric_indoor_base": "#4D4C19",
    "da-v2": "#4D4C19",
    "da-v2 metric": "#4D4C19",
    "da-v2 metric base": "#4D4C19",
}

MODEL_COLORS = {
    "UniDepth": "#00129A",
    "ZoeDepth": "#75190A",
    "DA-V2": "#4D4C19",
    "Depth Anything V2": "#4D4C19",
    "DA-V2 Metric": "#4D4C19",
    "DA-V2 Metric Base": "#4D4C19",
}

SCATTER_COLOR = "#00129A"
REFERENCE_LINE_COLOR = "#666666"
GRID_COLOR = "#999999"
TEXT_COLOR = "#20242A"
EDGE_COLOR = "#333333"
FIGURE_FACE_COLOR = "#FFFFFF"
AXES_FACE_COLOR = "#FFFFFF"
ANNOTATION_FACE_COLOR = "#FFFFFF"
ANNOTATION_EDGE_COLOR = "#BBBBBB"
DEPTH_CMAP = "magma_r"
ERROR_CMAP = "coolwarm"

DA_V2_HATCHES = {
    "depth_anything_v2_small": "//",
    "depth_anything_v2_metric_indoor": "\\\\",
    "depth_anything_v2_metric_indoor_small": "xx",
    "depth_anything_v2_metric_indoor_base": "",
}


def apply_ieee_style() -> None:
    """Apply a Times-first, IEEE-friendly Matplotlib style."""

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": FONT_SERIF,
            "font.size": FIGURE_FONT_SIZE,
            "axes.labelsize": FIGURE_FONT_SIZE,
            "axes.titlesize": FIGURE_FONT_SIZE,
            "figure.titlesize": FIGURE_FONT_SIZE,
            "xtick.labelsize": FIGURE_FONT_SIZE,
            "ytick.labelsize": FIGURE_FONT_SIZE,
            "legend.fontsize": FIGURE_FONT_SIZE,
            "legend.title_fontsize": FIGURE_FONT_SIZE,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": EDGE_COLOR,
            "axes.facecolor": AXES_FACE_COLOR,
            "figure.facecolor": FIGURE_FACE_COLOR,
            "savefig.facecolor": FIGURE_FACE_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "axes.prop_cycle": cycler(
                color=[
                    MODEL_COLORS["UniDepth"],
                    MODEL_COLORS["ZoeDepth"],
                    MODEL_COLORS["DA-V2"],
                ]
            ),
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _normalize_key(value: str) -> str:
    return re.sub(r"[\s_]+", "_", value.strip().lower().replace("-", "_"))


def bar_color_for_model(model: str) -> str:
    """Return the required bar color for a model key or display label."""

    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    normalized = _normalize_key(model)
    if normalized in BAR_COLORS:
        return BAR_COLORS[normalized]
    if normalized.startswith("depth_anything_v2") or normalized.startswith("da_v2"):
        return BAR_COLORS["depth_anything_v2"]
    return MODEL_COLORS["UniDepth"]


def bar_colors_for_models(models: list[str]) -> list[str]:
    return [bar_color_for_model(model) for model in models]


def hatch_for_model(model: str) -> str:
    normalized = _normalize_key(model)
    if normalized in DA_V2_HATCHES:
        return DA_V2_HATCHES[normalized]
    if normalized.startswith("depth_anything_v2") or normalized.startswith("da_v2"):
        return ".."
    return ""
