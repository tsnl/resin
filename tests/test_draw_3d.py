import wgpu
from zfw import Draw3dRenderer, Draw3dFrame, Draw3dScene
from PIL import Image
import os
import ctypes
import numpy as np


FRAME_W = 1024
FRAME_H = 1024


def test_basic_draw_3d():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(label="BasicDraw3dTest.Device")
    queue = device.queue

    renderer = Draw3dRenderer(device, queue, (FRAME_W, FRAME_H))
    frame = Draw3dFrame(renderer)
    readback_buffer = device.create_buffer(
        size=FRAME_W * FRAME_H * 4 * ctypes.sizeof(ctypes.c_float),
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        label="BasicDraw3dTest.ReadbackBuffer",
    )

    scene = Draw3dScene()

    command_encoder = device.create_command_encoder(
        label="BasicDraw2dTest.CommandEncoder"
    )

    renderer.record(scene, frame, command_encoder)
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=frame.get_output_image(),
            mip_level=0,
            origin=(0, 0, 0),
            aspect=wgpu.TextureAspect.all,
        ),
        destination=wgpu.TexelCopyBufferInfo(
            bytes_per_row=FRAME_W * 4 * ctypes.sizeof(ctypes.c_float),
            rows_per_image=FRAME_H,
            buffer=readback_buffer,
        ),
        copy_size=frame.get_output_image().size,
    )

    queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    data = np.asarray(readback_buffer.read_mapped()).view(dtype=np.float32)
    readback_buffer.unmap()

    # Reshape to (1024, 1024, 4)
    data = data.reshape((1024, 1024, 4))

    # Apply tonemapping
    img_data = (data * 255.0).astype(np.uint8)
    output_path = "output/draw_3d/basic_render_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.fromarray(img_data, "RGBA")
    img.save(output_path)
