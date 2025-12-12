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
    RendererAtlas2,
    RendererContext,
    Renderer,
    RendererQuadArray,
    PageRectAllocator,
    UvRectArray,
)

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
            usages=["color-attachment"],
            meta=GpuImageMeta(
                shape=(TEST_IMAGE_H, TEST_IMAGE_W, 4),
                dtype=np.uint8,
                color_space="srgb",
            ),
        )

        # Initialize empty quad array
        self.quads: RendererQuadArray = RendererQuadArray((0,))

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
        encoder.copy_image_to_buffer(src=self.target, dst=buffer)
        encoder.submit().wait()

        return buffer.memory.read(dtype=np.uint8).reshape(
            (TEST_IMAGE_H, TEST_IMAGE_W, 4)
        )

    def add_quad(
        self,
        *,
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ):
        """Add a quad to the render buffer."""
        # Use default white image
        image = self.renderer.default_white_image

        # Create a new quad entry
        quad = RendererQuadArray((1,))

        # Compute destination coordinates
        dst_x0_px, dst_y0_px = dst_xy
        dst_w_px, dst_h_px = dst_wh
        dst_x1_px = dst_x0_px + dst_w_px
        dst_y1_px = dst_y0_px + dst_h_px

        quad[0]["dst_px"] = (
            (dst_x0_px, dst_y0_px),  # TL
            (dst_x1_px, dst_y0_px),  # TR
            (dst_x1_px, dst_y1_px),  # BR
            (dst_x0_px, dst_y1_px),  # BL
        )

        # Get UV coordinates from image
        (src_x0_uv, src_y0_uv), (src_x1_uv, src_y1_uv) = image.rect_xy_xy_uv
        quad[0]["src_uv"] = (
            (src_x0_uv, src_y0_uv),  # TL
            (src_x1_uv, src_y0_uv),  # TR
            (src_x1_uv, src_y1_uv),  # BR
            (src_x0_uv, src_y1_uv),  # BL
        )

        quad[0]["color"] = color
        quad[0]["border_color"] = border_color
        quad[0]["border_thickness_px"] = border_thickness_px
        quad[0]["height"] = len(self.quads)
        quad[0]["atlas_id"] = image.atlas.atlas_id

        # Append to quad buffer
        self.quads = np.concatenate([self.quads, quad]).view(RendererQuadArray)

    def draw(self):
        self.add_quad(
            dst_xy=(32, 64),
            dst_wh=(512, 256),
            color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_px=(0, 0, 8, 0),
            border_color=(0.0, 0.1, 0.8, 1.0),
        )
        self.add_quad(
            dst_xy=(40, 72),
            dst_wh=(64, 64),
            color=(0.0, 0.2, 0.0, 1.0),
        )
        self.add_quad(
            dst_xy=(112, 72),
            dst_wh=(64, 64),
            color=(0.0, 0.2, 0.0, 0.5),
        )

    def show(self):
        fence = GpuFence(device=self.gpu_device)
        self.renderer.draw(
            quads=self.quads,
            target=self.target,
            fence=fence,
            wait_semaphores=[],
            done_semaphores=[],
        )
        fence.wait()


def test_renderer_quads():
    engine = RendererTestEngine()
    engine.draw()
    engine.show()
    image = engine.readback()

    output_path = Path("output/zfw/renderer_test/test_renderer_quads.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(image).save(output_path)


def test_page_rect_allocator():
    allocator = PageRectAllocator(max_pages=2, max_rects=7)

    # First, add a single page.
    allocator.add_page()

    # Check that we can insert.
    r0 = allocator.alloc(w=0.9, h=0.5)
    assert r0 is not None
    assert (allocator.rects[r0] == UvRectArray.of((0.0, 0.0, 0.9, 0.5))).all()

    # Check that we can fill out the first row.
    r1 = allocator.alloc(w=0.1, h=0.4)
    assert r1 is not None
    assert (allocator.rects[r1] == UvRectArray.of([(0.9, 0.0, 0.1, 0.4)])).all()

    # The next allocation should move to the second row, advancing by the tallest rect
    # in the previous row.
    r2 = allocator.alloc(w=0.1, h=0.1)
    assert r2 is not None
    assert (allocator.rects[r2] == UvRectArray.of([(0.0, 0.5, 0.1, 0.1)])).all()

    # Even if the second row is not full, we can't fit the next rect in it. Check that
    # we move to the third row.
    r3 = allocator.alloc(w=0.95, h=0.1)
    assert r3 is not None
    assert (allocator.rects[r3] == UvRectArray.of([(0.0, 0.6, 0.95, 0.1)])).all()

    # Check that we'd exhaust the first page with a gigantic allocation.
    n0 = allocator.alloc(w=1.0, h=0.5)
    assert n0 is None

    # Check that we can still insert into the first page in the third row.
    r4 = allocator.alloc(w=0.05, h=0.1)
    assert r4 is not None
    assert (allocator.rects[r4] == UvRectArray.of([(0.95, 0.6, 0.05, 0.1)])).all()

    # Add a second page...
    allocator.add_page()

    # ...and check that the gigantic allocation now works.
    r5 = allocator.alloc(w=1.0, h=0.5)
    assert r5 is not None
    assert (allocator.rects[r5] == UvRectArray.of([(0.0, 1.0, 1.0, 0.5)])).all()

    # Ensure that allocations in the second page have the expected offset Y coordinate.
    r6 = allocator.alloc(w=0.5, h=0.5)
    assert r6 is not None
    assert (allocator.rects[r6] == UvRectArray.of([(0.0, 1.5, 0.5, 0.5)])).all()

    # Ensure any further allocations fail with MemoryError.
    with pytest.raises(MemoryError):
        allocator.alloc(w=0.1, h=0.1)

    # Ensure that adding another page fails with MemoryError.
    with pytest.raises(MemoryError):
        allocator.add_page()


def test_renderer_atlas():
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

    atlas = RendererAtlas2(renderer=renderer, channels=4)

    orig_image_data = np.empty((128, 128, 4), dtype=np.float32)
    xs, ys = np.meshgrid(
        np.linspace(0.0, 1.0, num=128, endpoint=False),
        np.linspace(0.0, 1.0, num=128, endpoint=False),
        indexing="xy",
    )
    orig_image_data[..., 0] = xs
    orig_image_data[..., 1] = ys
    orig_image_data[..., 2] = 0.0

    image = atlas.insert(data=orig_image_data)

    # TODO: Read back the atlas data and verify that the image was inserted correctly.

    renderer.dispose()
    gpu_device.dispose()
    renderer_context.dispose()
    gpu_context.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
