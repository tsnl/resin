"""Image I/O helpers for 3DGS render outputs."""

import struct
import zlib
from collections.abc import Sequence
from pathlib import Path


def rgb_f32_to_rgb888_bytes(pixels: Sequence[float], *, width: int, height: int) -> bytes:
    """Pack linear RGB floats into contiguous 8-bit RGB bytes."""
    expected = width * height * 3
    if len(pixels) != expected:
        msg = f"expected {expected} floats, got {len(pixels)}"
        raise ValueError(msg)
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


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def save_rgb_f32_png(
    path: str | Path,
    pixels: Sequence[float],
    *,
    width: int,
    height: int,
) -> Path:
    """Write a ``width*height*3`` linear RGB float buffer as an 8-bit sRGB PNG."""
    out = Path(path)
    expected = width * height * 3
    if len(pixels) != expected:
        msg = f"expected {expected} floats, got {len(pixels)}"
        raise ValueError(msg)

    rows: list[bytes] = []
    for y in range(height):
        row_start = y * width * 3
        row = bytearray(width * 3 + 1)
        row[0] = 0
        for x in range(width):
            off = row_start + x * 3
            dst = 1 + x * 3
            row[dst] = _clamp_u8(pixels[off])
            row[dst + 1] = _clamp_u8(pixels[off + 1])
            row[dst + 2] = _clamp_u8(pixels[off + 2])
        rows.append(bytes(row))

    compressed = zlib.compress(b"".join(rows), level=9)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")
    _ = out.write_bytes(png)
    return out