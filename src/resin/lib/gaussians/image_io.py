"""Image helpers for 3DGS render outputs."""

from collections.abc import Sequence


def rgb_f32_to_rgb888_bytes(
    pixels: Sequence[float], *, width: int, height: int
) -> bytes:
    """Pack linear RGB floats into contiguous 8-bit RGB bytes."""
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"expected {expected} floats, got {len(pixels)}")
    out = bytearray(expected)
    for i, value in enumerate(pixels):
        out[i] = _clamp_u8(value)
    return bytes(out)


def _clamp_u8(x: float) -> int:
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 255
    return int(x * 255.0 + 0.5)
