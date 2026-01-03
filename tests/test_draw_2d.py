from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pytest

from tests.image_ref_tests import assert_image_matches_reference
from zfw import (
    setup_logging,
    Draw2dRenderer,
    Draw2dFrame,
    Draw2dQuad,
    ReadbackBuffer,
    Rgba8UnormTexture,
    convert_color,
    load_rgba_image,
)

from conftest import GpuFixture


def test_basic_draw_2d(gpu: GpuFixture):
    """Test basic 2D rendering with quads and textures."""
    device = gpu.device
    queue = gpu.queue

    # Load rainbow test image
    rainbow_image_data = load_rgba_image(Path("tests/data/rainbow-512x512.png"))
    assert rainbow_image_data.shape == (512, 512, 4)

    # Convert from linear to sRGB and to uint8 for rgba8unorm texture
    rainbow_image_data_uint8 = (np.clip(rainbow_image_data, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )

    rainbow_texture = Rgba8UnormTexture(
        device=device,
        size_wh=(512, 512),
        label="TestRainbowImage",
    )

    queue.write_texture(
        rainbow_texture.texel_copy_texture_info(),
        rainbow_image_data_uint8.tobytes(),
        rainbow_texture.texel_copy_buffer_layout(),
        rainbow_texture.size(),
    )

    renderer = Draw2dRenderer(device, queue, (1024, 1024))
    frame = Draw2dFrame(device=device, target_size_wh=(1024, 1024))
    readback_buffer = ReadbackBuffer(
        device=device,
        count=1024 * 1024 * 4,
        label="BasicDraw2dTest.ReadbackBuffer",
        dtype=np.uint8,
    )

    command_encoder = device.create_command_encoder(
        label="BasicDraw2dTest.CommandEncoder"
    )

    quads = [
        Draw2dQuad(
            dst_xy_px=(10, 10),
            dst_wh_px=(497, 497),
            fill_color_rgba=(1.0, 0.0, 0.0, 1.0),
        ),
        Draw2dQuad(
            dst_xy_px=(517, 10),
            dst_wh_px=(497, 497),
            fill_texture=rainbow_texture.wgpu_texture(),
            fill_color_rgba=(1.0, 1.0, 1.0, 1.0),
        ),
        Draw2dQuad(
            dst_xy_px=(20, 527),
            dst_wh_px=(477, 477),
            fill_color_rgba=(0.0, 1.0, 0.0, 1.0),
            border_thickness_px=(5, 10, 15, 20),
            border_color_rgba=(0.0, 0.5, 0.0, 1.0),
        ),
    ]

    renderer.record(quads, frame, command_encoder)
    readback_buffer.copy_from_texture(frame.get_output_image(), command_encoder)

    queue.submit([command_encoder.finish()])

    # Readback and convert from linear to sRGB
    data_raw = readback_buffer.read()
    data_raw_f32 = data_raw.astype(np.float32) / 255.0
    data_srgb_f32 = convert_color(
        data=data_raw_f32,
        src_color_space="linear",
        dst_color_space="srgb",
    )
    data_srgb = (np.clip(data_srgb_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = data_srgb.reshape((1024, 1024, 4))

    assert_image_matches_reference(
        image,
        "test_basic_draw_2d",
        psnr_threshold=65.0,
        test_subdir="draw_2d_test",
    )


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    pytest.main([__file__])
