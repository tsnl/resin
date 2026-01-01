from pathlib import Path

import numpy as np

from .basic import BaseResource
from .compositor import Compositor, CompositorInput
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
    VK_FORMAT_R32G32B32A32_SFLOAT,
)
from .loader import load_rgba_image


class CompositorTestHarness(BaseResource):
    """Minimal harness to exercise the compositor without a swapchain."""

    gpu_context: GpuContext
    gpu_device: GpuDevice
    compositor: Compositor
    compositor_input: CompositorInput
    source_image: GpuImage
    dest_image: GpuImage

    def __init__(self) -> None:
        super().__init__(parent_resource=None)

        self.gpu_context = GpuContext(
            app_name="zfw compositor_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )
        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )

        # Load linear-space source image
        src = load_rgba_image(Path("tests_data/rainbow-512x512.png"))
        self.source_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding"],
            data=src,
        )

        # Destination render target (linear float)
        self.dest_image = GpuImage(
            device=self.gpu_device,
            usages=["color-attachment", "transfer-src"],
            meta=GpuImageMeta(shape=src.shape, dtype=np.float32),
        )

        self.compositor = Compositor(
            gpu_device=self.gpu_device,
            viewport_width=self.dest_image.width,
            viewport_height=self.dest_image.height,
            vk_color_format=VK_FORMAT_R32G32B32A32_SFLOAT,
            parent_resource=self,
        )
        self.compositor_input = CompositorInput(
            compositor=self.compositor,
            image=self.source_image,
        )

    def _on_dispose(self) -> None:
        if hasattr(self, "compositor_input"):
            self.compositor_input.dispose()
        if hasattr(self, "compositor"):
            self.compositor.dispose()
        if hasattr(self, "dest_image"):
            self.dest_image.dispose()
        if hasattr(self, "source_image"):
            self.source_image.dispose()
        if hasattr(self, "gpu_device"):
            self.gpu_device.dispose()
        if hasattr(self, "gpu_context"):
            self.gpu_context.dispose()

    def round_trip(self) -> np.ndarray:
        encoder = GpuCommandEncoder(device=self.gpu_device, queue_type="graphics")
        self.compositor.record(
            command_encoder=encoder,
            output_image=self.dest_image,
            input=self.compositor_input,
        )
        encoder.submit().wait()

        # Read back the destination image as float32 linear RGBA
        buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["copy-dst", "staging"],
            meta=GpuBufferMeta(
                element_count=(self.dest_image.height * self.dest_image.width * 4),
                element_dtype=np.float32,
            ),
        )
        readback_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="transfer",
        )
        readback_encoder.transition_image_layout(
            image=self.dest_image,
            layout="transfer-src-optimal",
        )
        readback_encoder.copy_image_to_buffer(src=self.dest_image, dst=buffer)
        readback_encoder.submit().wait()

        data = buffer.memory.read(dtype=np.float32).reshape(
            (self.dest_image.height, self.dest_image.width, 4)
        )
        buffer.dispose()
        return data


def test_compositor_round_trip_matches_source():
    harness = CompositorTestHarness()
    try:
        actual = harness.round_trip()
        expected = load_rgba_image(Path("tests_data/rainbow-512x512.png"))
        np.testing.assert_allclose(actual, expected, atol=1e-5)
    finally:
        harness.dispose()
