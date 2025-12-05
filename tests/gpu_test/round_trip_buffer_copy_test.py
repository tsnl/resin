import numpy as np

from zero.gpu import (
    GpuBufferMeta,
    GpuBufferUsage,
    GpuContext,
    GpuDevice,
    GpuImageMeta,
)


def make_context() -> tuple[GpuContext, GpuDevice]:
    ctx = GpuContext(
        app_name="gpu-tests",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = ctx.create_device(physical_device=phys, surface=None)
    return ctx, dev


def test_buffer_roundtrip():
    _, dev = make_context()

    data0 = np.random.randn(1024).astype(np.float32)
    meta = GpuBufferMeta.from_array(data0)

    device_buf_usage: list[GpuBufferUsage] = ["copy-src", "copy-dst", "storage"]
    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    device_buf = dev.create_buffer(usages=device_buf_usage, meta=meta)
    host_buf_1 = dev.create_buffer(usages=host_buf_usage, meta=meta)
    host_buf_2 = dev.create_buffer(usages=host_buf_usage, meta=meta)

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to device_buf
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.copy_buffer_to_buffer(src=host_buf_1, dst=device_buf, size=meta.size)
    cmd.submit().wait()

    # copy device_buf to host_buf_2
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.copy_buffer_to_buffer(src=device_buf, dst=host_buf_2, size=meta.size)
    cmd.submit().wait()

    # read host_buf_2
    data1 = host_buf_2.read()

    # Test:
    assert np.allclose(data0, data1)


def test_image_roundtrip():
    _, dev = make_context()

    data0 = np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
    meta = GpuImageMeta.from_array(data0)

    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    image = dev.create_image(usages=["texture-binding"], meta=meta)
    host_buf_1 = dev.create_buffer(usages=host_buf_usage, meta=meta.into_buffer_meta())
    host_buf_2 = dev.create_buffer(usages=host_buf_usage, meta=meta.into_buffer_meta())

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to image
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=image, layout="copy-dst")
    cmd.copy_buffer_to_image(dst=image, src=host_buf_1)
    cmd.submit().wait()

    # copy image to host_buf_2
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=image, layout="copy-src")
    cmd.copy_image_to_buffer(src=image, dst=host_buf_2)
    cmd.submit().wait()

    # read host_buf_2
    data1 = host_buf_2.read()

    # reshape data1
    data1 = data1.reshape(data0.shape)

    # Test:
    assert np.array_equal(data0, data1)
