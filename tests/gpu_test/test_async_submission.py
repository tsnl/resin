"""Test async command submission with manual fence synchronization."""

import torch

from zero.gpu import (
    GpuBufferMeta,
    GpuBufferUsage,
    GpuContext,
    GpuDevice,
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


def test_async_buffer_copy():
    """Test non-blocking submission with manual fence wait."""
    _, dev = make_context()

    data0 = torch.randn(1024, dtype=torch.float32)
    meta = GpuBufferMeta.from_tensor(data0)

    device_buf_usage: list[GpuBufferUsage] = ["copy-src", "copy-dst", "storage"]
    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    device_buf = dev.create_buffer(usages=device_buf_usage, meta=meta)
    host_buf_1 = dev.create_buffer(usages=host_buf_usage, meta=meta)
    host_buf_2 = dev.create_buffer(usages=host_buf_usage, meta=meta)

    # Fill host_buf_1
    host_buf_1.write(data=data0)

    # Submit work without blocking
    with dev.command(queue_type="transfer", block=False) as cmd:
        cmd.copy_buffer_to_buffer(src=host_buf_1, dst=device_buf, size=meta.size)
        fence1 = cmd.get_fence()

    with dev.command(queue_type="transfer", block=False) as cmd:
        cmd.copy_buffer_to_buffer(src=device_buf, dst=host_buf_2, size=meta.size)
        fence2 = cmd.get_fence()

    # Both commands submitted without waiting
    # Now manually wait for completion
    fence1.wait()
    fence2.wait()

    # Read host_buf_2
    data1 = host_buf_2.read()

    # Test
    assert torch.allclose(data0, data1)


def test_blocking_vs_nonblocking():
    """Test that blocking and non-blocking modes both work."""
    _, dev = make_context()

    data0 = torch.randn(512, dtype=torch.float32)
    meta = GpuBufferMeta.from_tensor(data0)

    buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    buf1 = dev.create_buffer(usages=buf_usage, meta=meta)
    buf2 = dev.create_buffer(usages=buf_usage, meta=meta)
    buf3 = dev.create_buffer(usages=buf_usage, meta=meta)

    buf1.write(data=data0)

    # Blocking mode (default)
    with dev.command(queue_type="transfer", block=True) as cmd:
        cmd.copy_buffer_to_buffer(src=buf1, dst=buf2, size=meta.size)
    # Command completed and cleaned up

    # Non-blocking mode
    with dev.command(queue_type="transfer", block=False) as cmd:
        cmd.copy_buffer_to_buffer(src=buf1, dst=buf3, size=meta.size)
        fence = cmd.get_fence()
    # Command submitted but not necessarily completed
    fence.wait()

    # Both should have same result
    data2 = buf2.read()
    data3 = buf3.read()

    assert torch.allclose(data0, data2)
    assert torch.allclose(data0, data3)
