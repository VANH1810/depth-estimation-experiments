"""Extract HyperSim HDF5 data to the flat dataset format.

The multi-model pipeline expects:
  <output_root>/rgb/*.png
  <output_root>/depth/*.npy
  <output_root>/intrinsics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def estimate_intrinsics(width: int, height: int) -> list[list[float]]:
    """Return a simple pinhole intrinsics estimate for extracted HyperSim frames."""
    fx = fy = 0.7 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def extract_hypersim(scene_root: Path, output_root: Path, max_samples: int | None = None) -> None:
    """Extract one HyperSim scene from HDF5 RGB/depth files to PNG/NPY pairs."""
    rgb_dir_hdf5 = scene_root / "images" / "scene_cam_00_final_hdf5"
    depth_dir_hdf5 = scene_root / "images" / "scene_cam_00_geometry_hdf5"
    out_rgb = output_root / "rgb"
    out_depth = output_root / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    if not rgb_dir_hdf5.exists():
        raise FileNotFoundError(f"Missing HyperSim RGB HDF5 directory: {rgb_dir_hdf5}")
    if not depth_dir_hdf5.exists():
        raise FileNotFoundError(f"Missing HyperSim depth HDF5 directory: {depth_dir_hdf5}")

    rgb_files = sorted(rgb_dir_hdf5.glob("frame.*.color.hdf5"))
    depth_files = sorted(depth_dir_hdf5.glob("frame.*.depth_meters.hdf5"))
    if max_samples is not None:
        rgb_files = rgb_files[:max_samples]
        depth_files = depth_files[:max_samples]

    print(f"Found {len(rgb_files)} RGB files, {len(depth_files)} depth files")
    intrinsics_map = {}

    for count, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files), start=1):
        frame_id = rgb_file.stem.replace(".color", "")

        with h5py.File(rgb_file, "r") as handle:
            rgb = handle["dataset"][:]
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        with h5py.File(depth_file, "r") as handle:
            depth = handle["dataset"][:].astype(np.float32)

        Image.fromarray(rgb_uint8).save(out_rgb / f"{frame_id}.png")
        np.save(out_depth / f"{frame_id}.npy", depth)
        height, width = depth.shape
        intrinsics_map[f"{frame_id}.png"] = estimate_intrinsics(width, height)

        if count % 20 == 0:
            print(f"  Extracted {count}/{len(rgb_files)} frames")

    intrinsics_out = output_root / "intrinsics.json"
    with open(intrinsics_out, "w", encoding="utf-8") as handle:
        json.dump(intrinsics_map, handle, indent=2)

    print(f"Extraction complete: {len(intrinsics_map)} frames")
    print(f"RGB: {out_rgb}")
    print(f"Depth: {out_depth}")
    print(f"Intrinsics: {intrinsics_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract HyperSim HDF5 RGB/depth files to the flat dataset format.")
    parser.add_argument("--scene_root", default="data/hypersim/raw/ai_001_001", help="HyperSim scene root.")
    parser.add_argument("--output_root", default="data/hypersim/samples", help="Output flat dataset root.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extract_hypersim(Path(args.scene_root), Path(args.output_root), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
