"""Tests for 3DGS image output helpers."""

import struct
import zlib
from pathlib import Path

import pytest

from resin.lib.gs.image_io import rgb_f32_to_rgb888_bytes, save_rgb_f32_png
from resin.lib.gs.reference import render_gnomen_cpu


def _png_ihdr_size(data: bytes) -> tuple[int, int]:
    ihdr_off = data.index(b"IHDR") + 4
    width, height = struct.unpack(">II", data[ihdr_off : ihdr_off + 8])
    return width, height


def test_save_rgb_f32_png_writes_valid_header(tmp_path: Path) -> None:
    width = height = 8
    pixels = render_gnomen_cpu(width=width, height=height)
    out = save_rgb_f32_png(tmp_path / "gnomen.png", pixels, width=width, height=height)

    data = out.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IDAT" in data
    assert data.endswith(b"IEND\xaeB`\x82")
    assert _png_ihdr_size(data) == (width, height)


def test_save_rgb_f32_png_center_pixel_matches_buffer(tmp_path: Path) -> None:
    width = height = 4
    pixels = [0.0, 0.0, 0.0] * (width * height)
    pixels[(1 * width + 2) * 3] = 1.0
    out = save_rgb_f32_png(tmp_path / "spot.png", pixels, width=width, height=height)

    raw = out.read_bytes()
    idat_start = raw.index(b"IDAT") + 4
    idat_len = struct.unpack(">I", raw[idat_start - 4 : idat_start])[0]
    idat = raw[idat_start : idat_start + idat_len]
    inflated = zlib.decompress(idat)
    row_stride = 1 + width * 3
    red = inflated[row_stride * 1 + 1 + 2 * 3]
    assert red == 255


def test_rgb_f32_to_rgb888_bytes_clamps_linear_values() -> None:
    rgb = rgb_f32_to_rgb888_bytes([0.0, 0.5, 1.5], width=1, height=1)
    assert rgb == bytes([0, 128, 255])


def test_save_rgb_f32_png_rejects_wrong_buffer_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 12 floats"):
        _ = save_rgb_f32_png(tmp_path / "bad.png", [0.0, 0.0], width=2, height=2)