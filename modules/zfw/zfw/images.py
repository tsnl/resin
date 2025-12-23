__all__ = [
    "convert_linear_to_srgb",
    "convert_srgb_to_linear",
    "convert_srgb_to_linear",
    "load_rgba_image",
    "compute_psnr",
]

from pathlib import Path

import PIL.Image
import numpy as np

from .basic import ColorSpace


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    :param img1: First image array.
    :param img2: Second image array.
    :return: PSNR value in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    # Ensure we work with floats to avoid overflow/wrapping with uint8
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)

    mse = np.mean((img1_f - img2_f) ** 2)
    if mse == 0:
        return float("inf")

    # Determine max_i based on data type of original images
    if img1.dtype == np.uint8:
        max_i = 255.0
    else:
        # Assume float 0-1 if not uint8
        max_i = 1.0

    return 20 * np.log10(max_i / np.sqrt(mse))


def load_rgba_image(
    file_path: Path | str,
    file_color_space: ColorSpace = "srgb",
    output_color_space: ColorSpace = "linear",
) -> np.ndarray:
    """
    Loads an RGBA image as a normalized NumPy array in linear color space.

    :param file_path: The path to the image file to load.
    """

    src = np.array(PIL.Image.open(file_path).convert("RGBA"))
    src_normalized = src.astype(np.float32) / 255.0
    dst_rgb_linear = convert_color(
        src_normalized[..., :3],
        src_color_space=file_color_space,
        dst_color_space=output_color_space,
    )
    dst_alpha = src_normalized[..., 3:4]
    return np.concatenate((dst_rgb_linear, dst_alpha), axis=-1)


def save_rgba_image(*, file_path: Path | str, data: np.ndarray):
    """
    Saves an RGBA image from a normalized NumPy array in linear color space.

    :param file_path: The path to save the image file to.
    :param data: The image data as a NumPy array.
    """

    assert data.ndim == 3, "Data must be a 3D array: (height, width, channels)"
    assert data.shape[-1] == 4, "Data must have 4 channels (RGBA)"

    linear = data[..., :3]
    alpha = data[..., 3:4]
    srgb_normalized = convert_linear_to_srgb(linear)
    srgb_normalized = np.concatenate((srgb_normalized, alpha), axis=-1)
    srgb = (srgb_normalized * 255.0).clip(0, 255).astype(np.uint8)
    PIL.Image.fromarray(srgb, mode="RGBA").save(file_path)


def convert_color(
    data: np.ndarray, src_color_space: ColorSpace, dst_color_space: ColorSpace
) -> np.ndarray:
    """
    Convert an image between color spaces.

    :param data: The image data as a NumPy array.
    :param src_color_space: The source color space of the image.
    :param dst_color_space: The destination color space of the image.
    """

    if src_color_space == dst_color_space:
        return data

    match (src_color_space, dst_color_space):
        case ("srgb", "linear"):
            return convert_srgb_to_linear(data)
        case ("linear", "srgb"):
            return convert_linear_to_srgb(data)
        case _:
            raise ValueError(
                f"Unsupported color space conversion: {repr(src_color_space)} to {repr(dst_color_space)}"
            )


def convert_srgb_to_linear(srgb_normalized: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image to linear color space.
    """

    threshold = 0.04045
    below_threshold = srgb_normalized <= threshold
    above_threshold = srgb_normalized > threshold

    linear = np.zeros_like(srgb_normalized)
    linear[below_threshold] = srgb_normalized[below_threshold] / 12.92
    linear[above_threshold] = (
        (srgb_normalized[above_threshold] + 0.055) / 1.055
    ) ** 2.4

    return linear


def convert_linear_to_srgb(linear_normalized: np.ndarray) -> np.ndarray:
    """
    Convert a linear color space image to sRGB.
    """

    threshold = 0.0031308
    below_threshold = linear_normalized <= threshold
    above_threshold = linear_normalized > threshold

    srgb_normalized = np.zeros_like(linear_normalized)
    srgb_normalized[below_threshold] = linear_normalized[below_threshold] * 12.92
    srgb_normalized[above_threshold] = (
        1.055 * (linear_normalized[above_threshold] ** (1.0 / 2.4)) - 0.055
    )

    return srgb_normalized
