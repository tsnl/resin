import os
import torch
import pytest

from zero.gpu import (
    GpuContext,
    GpuBufferMeta,
    GpuImageMeta,
    LogicError,
)


@pytest.fixture(autouse=True, scope="session")
def ensure_vulkan_sdk_loaded():
    """Skip tests unless Vulkan SDK env is active.

    The repo's `AGENTS.md` requires sourcing the Vulkan SDK before running.
    This guard prevents hard crashes when the Vulkan loader or validation
    layers aren't available in the environment.
    """
    sdk = os.environ.get("VULKAN_SDK")
    if not sdk:
        pytest.skip(
            "Vulkan SDK not active. Run: source ~/VulkanSDK/1.4.328.1/setup-env.sh"
        )


def make_device():
    ctx = GpuContext(
        app_name="gpu-tests",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = ctx.create_device(physical_device=phys, surface=None)
    return ctx, dev


def test_buffer_roundtrip():
    ctx, dev = make_device()
    tensor = torch.randn(1024, dtype=torch.float32)
    meta = GpuBufferMeta.from_tensor(tensor)
    device_buffer = dev.create_buffer(usages=["copy-dst", "storage"], meta=meta)
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-src", "copy-dst"], meta=meta
    )

    with dev.command(queue_type="transfer") as cmd:
        cmd.write_buffer(
            dst=device_buffer, tensor=tensor, staging_buffer=staging_buffer
        )
    with dev.command(queue_type="transfer") as cmd:
        res = cmd.read_buffer(src=device_buffer, staging_buffer=staging_buffer)
        assert res is None  # deferred
    with dev.command(queue_type="transfer") as cmd:  # use encoder helper finalize
        roundtrip = cmd.finalize_read_buffer(staging_buffer)
    assert torch.allclose(tensor, roundtrip)


def test_image_roundtrip():
    ctx, dev = make_device()
    tensor = torch.randint(0, 256, (16, 32, 4), dtype=torch.uint8)
    meta = GpuImageMeta.from_tensor(tensor)
    image = dev.create_image(usages=["texture-binding"], meta=meta)
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-src", "copy-dst", "storage"],
        meta=meta.into_buffer_meta(),
    )

    with dev.command(queue_type="transfer") as cmd:
        cmd.write_image(dst=image, tensor=tensor, staging_buffer=staging_buffer)
    with dev.command(queue_type="transfer") as cmd:
        cmd.read_image(src=image, staging_buffer=staging_buffer)
    with dev.command(queue_type="transfer") as cmd:
        roundtrip = cmd.finalize_read_image(staging_buffer=staging_buffer, image=image)
    assert torch.equal(tensor, roundtrip)


def test_write_without_staging_raises():
    ctx, dev = make_device()
    tensor = torch.randn(64, dtype=torch.float32)
    meta = GpuBufferMeta.from_tensor(tensor)
    device_buffer = dev.create_buffer(usages=["copy-dst", "storage"], meta=meta)
    with pytest.raises(LogicError):
        with dev.command(queue_type="transfer") as cmd:
            cmd.write_buffer(dst=device_buffer, tensor=tensor, staging_buffer=None)


if __name__ == "__main__":
    sdk = os.environ.get("VULKAN_SDK")
    if not sdk:
        print(
            "Vulkan SDK not active. Run: source ~/VulkanSDK/1.4.328.1/setup-env.sh",
            flush=True,
        )
        raise SystemExit(2)

    print("[gpu_test] Running buffer roundtrip…", flush=True)
    test_buffer_roundtrip()
    print("[gpu_test] Buffer roundtrip OK", flush=True)

    print("[gpu_test] Running image roundtrip…", flush=True)
    test_image_roundtrip()
    print("[gpu_test] Image roundtrip OK", flush=True)

    print("[gpu_test] Checking staging guard…", flush=True)
    try:
        test_write_without_staging_raises()
        print("[gpu_test] Staging guard OK", flush=True)
    except AssertionError as e:
        print(f"[gpu_test] Staging guard failed: {e}", flush=True)
        raise
