from pathlib import Path

import numpy as np
import PIL.Image
import zfw

TEST_IMAGE_W, TEST_IMAGE_H = 800, 600


class Renderer2dFixture(zfw.BaseResource):
    def __init__(self):
        super().__init__(parent=None)
        self.gpu_context = zfw.GpuContext(
            app_name="zfw.tests.renderer_2d_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )
        self.renderer_context = zfw.RendererContext(
            gpu_context=self.gpu_context,
        )

        self.gpu_device = zfw.GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )
        self.renderer = zfw.Renderer(
            context=self.renderer_context,
            gpu_device=self.gpu_device,
        )

        self.target = zfw.GpuImage(
            device=self.gpu_device,
            usages=["color-attachment"],
            meta=zfw.GpuImageMeta(
                shape=(TEST_IMAGE_H, TEST_IMAGE_W, 4),
                dtype=np.uint8,
                color_space="srgb",
            ),
        )

        # Initialize empty quad array
        self.quads: zfw.RendererQuadArray = zfw.RendererQuadArray((0,))

    def _on_dispose(self) -> None:
        self.target.dispose()

        self.renderer.dispose()
        self.gpu_device.dispose()

        self.renderer_context.dispose()
        self.gpu_context.dispose()

    def readback(self) -> np.ndarray:
        buffer = zfw.GpuBuffer(
            device=self.gpu_device,
            usages=["copy-dst", "staging"],
            meta=zfw.GpuBufferMeta(
                element_count=(TEST_IMAGE_H * TEST_IMAGE_W * 4),
                element_dtype=np.uint8,
            ),
        )
        encoder = zfw.GpuCommandEncoder(device=self.gpu_device, queue_type="transfer")
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
        quad = zfw.RendererQuadArray((1,))

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
        self.quads = np.concatenate([self.quads, quad]).view(zfw.RendererQuadArray)

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
        fence = zfw.GpuFence(device=self.gpu_device)
        self.renderer.draw(
            quads=self.quads,
            target=self.target,
            fence=fence,
            wait_semaphores=[],
            done_semaphores=[],
        )
        fence.wait()


def test_renderer_2d():
    fixture = Renderer2dFixture()
    fixture.draw()
    fixture.show()
    image = fixture.readback()

    output_path = Path(__file__).parent / "renderer_2d_test_output.png"
    PIL.Image.fromarray(image).save(output_path)


if __name__ == "__main__":
    test_renderer_2d()
