from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from .basic import BaseResource
from .gpu import (
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuBuffer,
    GpuCommandEncoder,
    GpuFence,
    GpuImageMeta,
    GpuBufferMeta,
)
from .renderer import (
    RendererImage,
    RendererContext,
    Renderer,
    RendererQuadArray,
    RendererQuad,
)
from .images import load_rgba_image

TEST_IMAGE_W, TEST_IMAGE_H = 800, 600


class RendererTestEngine(BaseResource):
    def __init__(self):
        super().__init__(parent=None)
        self.gpu_context = GpuContext(
            app_name="zfw renderer_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )
        self.renderer_context = RendererContext(
            gpu_context=self.gpu_context,
        )

        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )
        self.renderer = Renderer(
            context=self.renderer_context,
            gpu_device=self.gpu_device,
        )

        self.target = GpuImage(
            device=self.gpu_device,
            usages=["color-attachment", "transfer-src"],
            meta=GpuImageMeta(
                shape=(TEST_IMAGE_H, TEST_IMAGE_W, 4),
                dtype=np.uint8,
                color_space="srgb",
            ),
        )

    def _on_dispose(self) -> None:
        self.target.dispose()

        self.renderer.dispose()
        self.gpu_device.dispose()

        self.renderer_context.dispose()
        self.gpu_context.dispose()

    def readback(self) -> np.ndarray:
        buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["copy-dst", "staging"],
            meta=GpuBufferMeta(
                element_count=(TEST_IMAGE_H * TEST_IMAGE_W * 4),
                element_dtype=np.uint8,
            ),
        )
        encoder = GpuCommandEncoder(device=self.gpu_device, queue_type="transfer")
        encoder.transition_image_layout(
            image=self.target,
            layout="transfer-src-optimal",
        )
        encoder.copy_image_to_buffer(src=self.target, dst=buffer)
        encoder.submit().wait()

        return buffer.memory.read(dtype=np.uint8).reshape(
            (TEST_IMAGE_H, TEST_IMAGE_W, 4)
        )

    def draw(self, quads: RendererQuadArray | list[RendererQuad]):
        fence = GpuFence(device=self.gpu_device)
        self.renderer.draw(
            quads=quads,
            target=self.target,
            fence=fence,
            wait_semaphores=[],
            done_semaphores=[],
        )
        fence.wait()


def test_renderer_quads():
    engine = RendererTestEngine()

    engine.draw(
        quads=[
            RendererQuad(
                dst_xy=(32, 64),
                dst_wh=(512, 256),
                color=(1.0, 1.0, 1.0, 1.0),
                border_thickness_px=(0, 0, 8, 0),
                border_color=(0.0, 0.1, 0.8, 1.0),
            ),
            RendererQuad(
                dst_xy=(40, 72),
                dst_wh=(64, 64),
                color=(0.0, 0.2, 0.0, 1.0),
            ),
            RendererQuad(
                dst_xy=(112, 72),
                dst_wh=(64, 64),
                color=(0.0, 0.2, 0.0, 0.5),
            ),
        ]
    )
    image = engine.readback()

    output_path = Path("output/zfw/renderer_test/test_renderer_quads.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(image).save(output_path)


def test_renderer_image():
    engine = RendererTestEngine()

    image_data = load_rgba_image("tests_data/rainbow-512x512.png")
    assert image_data.shape == (512, 512, 4)

    image = RendererImage(renderer=engine.renderer, data=image_data)

    border_thickness_px = 8

    engine.draw(
        quads=[
            RendererQuad(
                dst_xy=(
                    (TEST_IMAGE_W - image_data.shape[1] - border_thickness_px) // 2,
                    (TEST_IMAGE_H - image_data.shape[0] - border_thickness_px) // 2,
                ),
                color=(1.0, 1.0, 1.0, 1.0),
                border_thickness_px=(8, 8, 8, 8),
                border_color=(1.0, 1.0, 0.0, 1.0),
                image=image,
            ),
        ]
    )
    output_image = engine.readback()

    output_path = Path("output/zfw/renderer_test/test_renderer_image.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(output_image).save(output_path)


def test_renderer_atlas_smoketest():
    gpu_context = GpuContext(
        app_name="zfw.renderer_test.test_renderer_atlas",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    renderer_context = RendererContext(
        gpu_context=gpu_context,
    )

    gpu_device = GpuDevice(
        context=gpu_context,
        physical_device=gpu_context.enumerate_physical_devices()[0],
        surface=None,
    )
    renderer = Renderer(
        context=renderer_context,
        gpu_device=gpu_device,
    )

    orig_image_data = np.empty((128, 128, 4), dtype=np.float32)
    xs, ys = np.meshgrid(
        np.linspace(0.0, 1.0, num=128, endpoint=False),
        np.linspace(0.0, 1.0, num=128, endpoint=False),
        indexing="xy",
    )
    orig_image_data[..., 0] = xs
    orig_image_data[..., 1] = ys
    orig_image_data[..., 2] = 0.0
    orig_image_data[..., 3] = 1.0

    image = RendererImage(renderer=renderer, data=orig_image_data)
    renderer._atlases[4].insert(image)
    renderer._atlases[4].flush()

    # Note: x=0 because it's larger than default_white_image (1x1)
    assert image.allocation_px_xywh[2:] == (128, 128)

    renderer.dispose()
    gpu_device.dispose()
    renderer_context.dispose()
    gpu_context.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
