import wgpu
from zfw import Draw3dRenderer, Draw3dFrame, Draw3dScene, ReadbackBuffer
from PIL import Image
import os
import ctypes
import numpy as np


def test_basic_draw_3d():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(label="BasicDraw3dTest.Device")
    queue = device.queue

    renderer = Draw3dRenderer(device, queue, (1024, 1024))
    frame = Draw3dFrame(renderer)
    readback_buffer = ReadbackBuffer(
        device,
        1024 * 1024,
        "BasicDraw3dTest.ReadbackBuffer",
        ctypes.c_float * 4,
    )

    scene = Draw3dScene()

    command_encoder = device.create_command_encoder(
        label="BasicDraw2dTest.CommandEncoder"
    )

    renderer.record(scene, frame, command_encoder)
    readback_buffer.copy_from_texture(frame.get_output_image(), command_encoder)

    queue.submit([command_encoder.finish()])

    data = readback_buffer.read()

    float_data = np.frombuffer(data, dtype=np.float32)

    # Reshape to (1024, 1024, 4)
    float_data = float_data.reshape((1024, 1024, 4))

    # Apply tonemapping
    img_data = (float_data * 255.0).astype(np.uint8)

    output_path = "output/draw_3d/basic_render_test.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.fromarray(img_data, "RGBA")
    img.save(output_path)
