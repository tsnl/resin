from dataclasses import dataclass
import logging
import os
from pathlib import Path

import pytest
import wgpu

from zfw import (
    load_image,
    setup_logging,
    request_wgpu_device,
    ImageResource,
    convert_rgb_to_grayscale,
)


@dataclass
class GpuFixture:
    adapter: wgpu.GPUAdapter
    device: wgpu.GPUDevice
    queue: wgpu.GPUQueue


@pytest.fixture(scope="session")
def gpu() -> GpuFixture:
    """Fixture that provides a GPU device and queue."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = request_wgpu_device(adapter, label="TestDevice")
    queue = device.queue
    return GpuFixture(adapter=adapter, device=device, queue=queue)


@pytest.fixture(scope="session")
def rainbow_512x512_image() -> ImageResource:
    """Load the rainbow test image as a float32 linear RGBA array."""
    image_resource = load_image(
        Path("tests/data/rainbow-512x512.png"),
        image_format="rgba8unorm-srgb",
        expected_format="rgba32float",
    )
    image_data = image_resource.data
    assert image_data.shape == (512, 512, 4)
    image_data.setflags(write=False)
    return image_resource


@pytest.fixture(scope="session")
def rainbow_512x512_image_greyscale(
    rainbow_512x512_image: ImageResource,
) -> ImageResource:
    image_data_rgb = rainbow_512x512_image.data[..., :3]
    image_data_gray = convert_rgb_to_grayscale(rgb=image_data_rgb)
    return ImageResource(
        data=image_data_gray,
        width=rainbow_512x512_image.width,
        height=rainbow_512x512_image.height,
        depth=1,
        image_format="r32float",
    )


@pytest.fixture(scope="session", autouse=True)
def _setup_logging():
    log_level = os.environ.get("ZFW_TEST_LOG_LEVEL", "WARNING")
    setup_logging(level=logging.getLevelNamesMapping()[log_level])
