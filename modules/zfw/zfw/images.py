__all__ = [
    "convert_srgb_to_linear",
    "load_image_data",
]

from pathlib import Path

import PIL.Image
import numpy as np


def load_image_data(file_path: Path | str) -> np.ndarray:
    srgb = np.array(PIL.Image.open(file_path).convert("RGBA"))
    srgb_normalized = srgb.astype(np.float32) / 255.0
    linear = convert_srgb_to_linear(srgb_normalized[..., :3])
    alpha = srgb_normalized[..., 3:4]
    return np.concatenate((linear, alpha), axis=-1)


def convert_srgb_to_linear(srgb_normalized: np.ndarray) -> np.ndarray:
    """Convert an sRGB image to linear color space."""
    threshold = 0.04045
    below_threshold = srgb_normalized <= threshold
    above_threshold = srgb_normalized > threshold

    linear = np.zeros_like(srgb_normalized)
    linear[below_threshold] = srgb_normalized[below_threshold] / 12.92
    linear[above_threshold] = (
        (srgb_normalized[above_threshold] + 0.055) / 1.055
    ) ** 2.4

    return linear
