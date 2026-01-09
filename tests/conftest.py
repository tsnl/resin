import logging
import os

import numpy as np
import numpy.typing as npt
import pytest
import wgpu

from zfw import (
    load_image,
    setup_logging,
    request_wgpu_device,
    convert_rgb_to_grayscale,
)
from zfw.images import convert_srgb_to_linear


@pytest.fixture(scope="session")
def gpu_device() -> wgpu.GPUDevice:
    """Fixture that provides a GPU device and queue."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return request_wgpu_device(adapter, label="ZfwTestDevice")


@pytest.fixture(scope="session")
def rainbow_512x512_image() -> npt.NDArray[np.float32]:
    """Load the rainbow test image as a float32 linear RGBA array."""
    image_data = load_image("tests/data/rainbow-512x512.png")
    image_data = convert_srgb_to_linear(image_data)
    assert image_data.shape == (512, 512, 4)
    image_data.setflags(write=False)
    return image_data


@pytest.fixture(scope="session")
def rainbow_512x512_image_grayscale(
    rainbow_512x512_image: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    image_data_rgb = rainbow_512x512_image[..., :3]
    image_data_gray = convert_rgb_to_grayscale(rgb=image_data_rgb)
    return image_data_gray


@pytest.fixture(scope="session", autouse=True)
def _setup_logging():
    log_level = os.environ.get("ZFW_TEST_LOG_LEVEL", "WARNING")
    setup_logging(level=logging.getLevelNamesMapping()[log_level])
