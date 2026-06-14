"""Resolution transforms shared by robustness experiments."""

import cv2
import numpy as np


def resize_rgb_and_intrinsics(
    rgb: np.ndarray,
    intrinsics: np.ndarray | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize RGB by scale and scale pinhole intrinsics to the actual output size."""

    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")

    height, width = rgb.shape[:2]
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    scale_x = new_width / float(width)
    scale_y = new_height / float(height)

    if new_width == width and new_height == height:
        resized = rgb.copy()
    else:
        resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(rgb.dtype, copy=False)

    resized_intrinsics = None
    if intrinsics is not None:
        resized_intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
        if resized_intrinsics.shape != (3, 3):
            raise ValueError(f"Expected 3x3 intrinsics, got {resized_intrinsics.shape}")
        resized_intrinsics[0, 0] *= scale_x
        resized_intrinsics[0, 2] *= scale_x
        resized_intrinsics[1, 1] *= scale_y
        resized_intrinsics[1, 2] *= scale_y

    return resized, resized_intrinsics
