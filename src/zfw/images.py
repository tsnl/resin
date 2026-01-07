__all__ = [
    "ImageFormat",
    "compute_psnr",
    "convert_color",
    "convert_image_format",
    "convert_linear_to_srgb",
    "convert_rgb_to_grayscale",
    "convert_srgb_to_linear",
    "debug_save_rgba_image",
    "encode_bc1",
    "encode_bc4",
    "encode_bc5",
    "normalize_image_to_f32",
]

from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import numpy as np

from .basic import ColorSpace, logger
from .images_bcn import encode_bc1, encode_bc4, encode_bc5

# ImageFormat: subset of WebGPU texture formats for images
type ImageFormat = Literal[
    "rgba32float",
    "rgba16float",
    "rgba8unorm",
    "r32float",
    "r16float",
    "r8unorm",
    "rgba8unorm-srgb",
    "rgb32float",
    "rgb16float",
    "bc1-rgba-unorm",
    "bc4-r-unorm",
    "bc5-rg-unorm",
    "bc5-rg-snorm",
]


LOG = logger(__name__)


#
# PSNR:
#


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


#
# RGB -> Grayscale:
#


def convert_rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using luminance-preserving weights.
    :param rgb: Input RGB image as a NumPy array of shape (H, W, 3).
    :returns: Grayscale image as a NumPy array of shape (H, W, 1).
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    grayscale = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return grayscale[..., np.newaxis]


def debug_save_rgba_image(*, file_path: Path | str, data: np.ndarray):
    """
    Saves an RGBA image from a normalized NumPy array in linear color space.

    :param file_path: The path to save the image file to.
    :param data: The image data as a NumPy array.
    """
    assert data.ndim == 3, "Data must be a 3D array: (height, width, channels)"
    assert data.shape[-1] == 4, "Data must have 4 channels (RGBA)"

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    linear = data[..., :3]
    alpha = data[..., 3:4]
    srgb_normalized = convert_linear_to_srgb(linear)
    srgb_normalized = np.concatenate((srgb_normalized, alpha), axis=-1)
    srgb = (srgb_normalized * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(file_path, srgb)


#
# convert_image_format
#


def convert_image_format(
    data: np.ndarray,
    input_format: ImageFormat,
    output_format: ImageFormat,
) -> np.ndarray:
    """
    Convert from one ImageFormat to another.

    Steps:
    1. Normalize input to float32
    2. Adjust channel count if needed (e.g., RGBA -> RGB by dropping alpha)
    3. Apply color space conversion if needed (srgb<->linear)
    4. Re-encode to output bit-depth
    """

    input_channels = image_format_channel_count(input_format)
    output_channels = image_format_channel_count(output_format)

    # Normalize to float32
    f32_data = normalize_image_to_f32(data)

    # Handle channel count mismatch
    if input_channels != output_channels:
        if input_channels == 4 and output_channels == 3:
            # Drop alpha channel
            f32_data = f32_data[:, :, :3]
        elif input_channels == 3 and output_channels == 4:
            # Add alpha channel (all 1.0)
            alpha = np.ones((*f32_data.shape[:2], 1), dtype=f32_data.dtype)
            f32_data = np.concatenate((f32_data, alpha), axis=-1)
        elif input_channels == 1 and output_channels == 4:
            # Replicate grayscale to RGBA
            f32_data = np.repeat(f32_data, 4, axis=-1)
        elif input_channels == 4 and output_channels == 1:
            # Convert RGBA to grayscale (use luminance formula)
            f32_data = (
                0.299 * f32_data[:, :, 0:1]
                + 0.587 * f32_data[:, :, 1:2]
                + 0.114 * f32_data[:, :, 2:3]
            )
        else:
            raise ValueError(
                f"Unsupported channel conversion: {input_channels} -> {output_channels}"
            )

    # Apply color space conversion if needed
    input_is_srgb = "srgb" in input_format
    output_is_srgb = "srgb" in output_format

    if input_is_srgb and not output_is_srgb:
        # sRGB -> linear
        f32_data = convert_color(
            f32_data,
            src_color_space="srgb",
            dst_color_space="linear",
            src_channels=f32_data.shape[-1],
        )
    elif not input_is_srgb and output_is_srgb:
        # linear -> sRGB
        f32_data = convert_color(
            f32_data,
            src_color_space="linear",
            dst_color_space="srgb",
            src_channels=f32_data.shape[-1],
        )

    # Re-encode to target format
    return _encode_f32_to_format(f32_data, output_format)


def normalize_image_to_f32(
    data: np.ndarray,
) -> np.ndarray:
    """Normalize image data to float32 in [0, 1] range."""
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    elif data.dtype == np.uint16:
        return data.astype(np.float32) / 65535.0
    elif np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {data.dtype}")


def _encode_f32_to_format(
    data: np.ndarray,
    image_format: ImageFormat,
) -> np.ndarray:
    """Re-encode normalized float32 image to target format bit-depth."""
    match image_format:
        case "rgba32float" | "rgb32float" | "r32float":
            return data.astype(np.float32)
        case "rgba16float" | "rgb16float" | "r16float":
            return data.astype(np.float16)
        case "rgba8unorm" | "rgba8unorm-srgb" | "r8unorm":
            return (data * 255.0).clip(0, 255).astype(np.uint8)
        case _:
            raise ValueError(f"Unknown image format: {image_format}")


def image_format_channel_count(image_format: ImageFormat) -> int:
    """Get number of channels for image format."""
    match image_format:
        case "rgba32float" | "rgba16float" | "rgba8unorm" | "rgba8unorm-srgb":
            return 4
        case "rgb32float" | "rgb16float":
            return 3
        case "r32float" | "r16float" | "r8unorm":
            return 1
        case "bc4-r-unorm":
            return 1
        case _:
            raise ValueError(f"Unknown image format: {image_format}")


#
# convert_color: deprecated API
#


def convert_color(
    data: np.ndarray,
    src_color_space: ColorSpace,
    dst_color_space: ColorSpace,
    src_channels: int | None = None,
) -> np.ndarray:
    """
    Convert an image between color spaces.

    NOTE: Deprecated: use convert_image_format() instead.

    If src_channels is provided, will validate that the image has that many channels.
    If mismatch, raises an error (no implicit channel conversion).

    :param data: The image data as a NumPy array.
    :param src_color_space: The source color space of the image.
    :param dst_color_space: The destination color space of the image.
    :param src_channels: Optional expected number of channels. If provided and mismatched, raises.
    :raises ValueError: If channel count mismatch, or unsupported color space conversion.
    """

    if src_channels is not None:
        if data.shape[-1] != src_channels:
            raise ValueError(
                f"Channel mismatch: expected {src_channels} channels, got {data.shape[-1]}"
            )

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
