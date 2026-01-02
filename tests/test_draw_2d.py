import wgpu
import wgpu.backends.wgpu_native
from zfw import (
    Draw2dRenderer,
    Draw2dFrame,
    Draw2dQuad,
    ReadbackBuffer,
    Rgba8UnormTexture,
)
from PIL import Image
import os
import ctypes


def test_basic_draw_2d():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(label="TestDevice")
    queue = device.queue

    rainbow_image = Image.open("tests/data/rainbow-512x512.png").convert("RGBA")
    rainbow_texture = Rgba8UnormTexture(device, (512, 512), "TestRainbowImage")

    queue.write_texture(
        rainbow_texture.texel_copy_texture_info(),
        rainbow_image.tobytes(),
        rainbow_texture.texel_copy_buffer_layout(),
        rainbow_texture.size(),
    )

    renderer = Draw2dRenderer.create(device, queue, (1024, 1024))
    frame = Draw2dFrame(device, (1024, 1024))
    readback_buffer = ReadbackBuffer(
        device, 1024 * 1024, "BasicDraw2dTest.ReadbackBuffer", ctypes.c_uint8 * 4
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

    # Readback
    data = readback_buffer.read()

    output_path = "output/draw_2d/basic_render_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.frombytes("RGBA", (1024, 1024), data)
    img.save(output_path)
