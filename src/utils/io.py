"""File IO helpers for experiment outputs."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path


def prepare_output_dirs(paths: Iterable[Path], overwrite: bool = False) -> None:
    """Create output directories, refusing to reuse non-empty dirs unless requested."""

    for path in paths:
        path = Path(path)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {path}. "
                "Pass --overwrite to update files in this directory."
            )
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
