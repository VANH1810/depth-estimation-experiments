"""Path helpers for local experiment scripts."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_root() -> Path:
    return repo_root() / "results"
