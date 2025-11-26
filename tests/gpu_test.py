import torch

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

    data0 = torch.randn(1024, dtype=torch.float32)
    meta = GpuBufferMeta.from_tensor(data0)

    device_buf_usage: list[GpuBufferUsage] = ["copy-dst", "storage"]
    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    device_buf = dev.create_buffer(usages=device_buf_usage, meta=meta)
    host_buf_1 = dev.create_buffer(usages=host_buf_usage, meta=meta)
    host_buf_2 = dev.create_buffer(usages=host_buf_usage, meta=meta)

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to device_buf
    with dev.command(queue_type="transfer") as cmd:
        cmd.copy_buffer_to_buffer(src=host_buf_1, dst=device_buf, size=meta.size)

    # copy device_buf to host_buf_2
    with dev.command(queue_type="transfer") as cmd:
        cmd.copy_buffer_to_buffer(src=device_buf, dst=host_buf_2, size=meta.size)

    # read host_buf_2
    data1 = host_buf_2.read()

    # Test:
    assert torch.allclose(data0, data1)


def test_image_roundtrip():
    _, dev = make_context()

    data0 = torch.randint(0, 256, (1024, 1024, 4), dtype=torch.uint8)
    meta = GpuImageMeta.from_tensor(data0)

    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    image = dev.create_image(usages=["texture-binding"], meta=meta)
    host_buf_1 = dev.create_buffer(usages=host_buf_usage, meta=meta.into_buffer_meta())
    host_buf_2 = dev.create_buffer(usages=host_buf_usage, meta=meta.into_buffer_meta())

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to image
    with dev.command(queue_type="transfer") as cmd:
        cmd.copy_buffer_to_image(dst=image, src=host_buf_1)

    # copy image to host_buf_2
    with dev.command(queue_type="transfer") as cmd:
        cmd.copy_image_to_buffer(src=image, dst=host_buf_2)

    # read host_buf_2
    data1 = host_buf_2.read()

    # reshape data1
    data1 = data1.reshape(data0.shape)

    # Test:
    assert torch.equal(data0, data1)
