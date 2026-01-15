import logging

import numpy as np
import numpy.typing as npt
import pytest
import wgpu

from image_ref_tests import assert_image_matches_reference

from resin import (
    setup_logging,
    Draw2dRenderer,
    Draw2dQuad,
    convert_color,
)


def test_basic_draw_2d(
    gpu_device: wgpu.GPUDevice, rainbow_512x512_image: npt.NDArray[np.float32]
):
    """Test basic 2D rendering with quads and textures."""
    device = gpu_device
    queue = gpu_device.queue

    # Convert from linear to sRGB and to uint8 for rgba8unorm texture
    rainbow_image_data_uint8 = (
        np.clip(rainbow_512x512_image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

    rainbow_texture = device.create_texture(
        size=(512, 512, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        label="TestRainbowImage",
    )

    queue.write_texture(
        destination=wgpu.TexelCopyTextureInfo(
            texture=rainbow_texture,
            origin=(0, 0, 0),
            mip_level=0,
            aspect=wgpu.TextureAspect.all,
        ),
        data=rainbow_image_data_uint8.tobytes(),
        data_layout=wgpu.TexelCopyBufferLayout(
            offset=0,
            bytes_per_row=512 * 4,
            rows_per_image=512,
        ),
        size=rainbow_texture.size,
    )

    renderer = Draw2dRenderer(device, queue, (1024, 1024))
    readback_buffer = device.create_buffer(
        size=1024 * 1024 * 4,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="BasicDraw2dTest.ReadbackBuffer",
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
            fill_texture=rainbow_texture,
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

    renderer.record(quads, command_encoder)
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=renderer.get_output_image(),
            mip_level=0,
            origin=(0, 0, 0),
            aspect=wgpu.TextureAspect.all,
        ),
        destination=wgpu.TexelCopyBufferInfo(
            bytes_per_row=1024 * 4,
            rows_per_image=1024,
            buffer=readback_buffer,
        ),
        copy_size=renderer.get_output_image().size,
    )

    queue.submit([command_encoder.finish()])

    # Readback and convert from linear to sRGB
    readback_buffer.map_sync(mode=wgpu.MapMode.READ)
    data_raw = np.asarray(readback_buffer.read_mapped()).view(dtype=np.uint8)
    readback_buffer.unmap()

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
        test_subdir="test_draw_2d",
    )


if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    pytest.main([__file__])
