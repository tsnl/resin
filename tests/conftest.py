from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pytest
import wgpu

from zfw import load_rgba_image, setup_logging


@dataclass
class GpuFixture:
    adapter: wgpu.GPUAdapter
    device: wgpu.GPUDevice
    queue: wgpu.GPUQueue


@pytest.fixture(scope="session")
def gpu() -> GpuFixture:
    """Fixture that provides a GPU device and queue."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(label="TestDevice")
    queue = device.queue
    return GpuFixture(adapter=adapter, device=device, queue=queue)


@pytest.fixture(scope="session")
def rainbow_512x512_image() -> np.ndarray:
    """Load the rainbow test image as a float32 linear RGBA array."""
    image_data = load_rgba_image(Path("tests/data/rainbow-512x512.png"))
    assert image_data.shape == (512, 512, 4)
    image_data.setflags(write=False)
    return image_data


@pytest.fixture(scope="session", autouse=True)
def _setup_logging():
    setup_logging(level=logging.WARNING)
