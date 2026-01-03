from dataclasses import dataclass

import pytest
import wgpu


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
