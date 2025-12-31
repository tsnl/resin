from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from .basic import BaseResource, Font, logger
from .gpu import (
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuBuffer,
    GpuCommandEncoder,
    GpuImageMeta,
    GpuBufferMeta,
    GpuBufferImageCopyRegion,
)
from .draw_2d import Draw2dRenderer, Draw2dTarget, Draw2dQuad
from .images import compute_psnr
from .loader import load_rgba_image

TEST_IMAGE_W, TEST_IMAGE_H = 1280, 720

LOG = logger(__name__)


def assert_image_matches_reference(
    actual_image: np.ndarray,
    test_name: str,
    psnr_threshold: float,
) -> None:
    """
    Compare an actual rendered image against an expected reference image.

    If no reference exists, creates it and passes the test.
    If reference exists, compares using PSNR matching.
    On mismatch, saves the actual image as <name>.actual.png and raises AssertionError.
    """
    expect_dir = Path("tests/expect/zfw/draw_2d_test")
    expect_dir.mkdir(parents=True, exist_ok=True)

    expect_path = expect_dir / f"{test_name}.png"
    actual_path = expect_dir / f"{test_name}.actual.png"

    # Also save to output directory for convenience
    output_path = Path("output/zfw/draw_2d_test") / f"{test_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(actual_image).save(output_path)

    if not expect_path.exists():
        # No reference exists - create it and pass
        PIL.Image.fromarray(actual_image).save(expect_path)
        LOG.warning(f"{test_name}: no expect found: image created: {expect_path}")
        return

    # Load expected image
    expected_image = np.array(PIL.Image.open(expect_path))

    # Check shapes match
    if actual_image.shape != expected_image.shape:
        PIL.Image.fromarray(actual_image).save(actual_path)
        raise AssertionError(
            f"Image shape mismatch for {test_name}: "
            f"expected {expected_image.shape}, got {actual_image.shape}. "
            f"Actual image saved to {actual_path}"
        )

    # Compare images
    psnr = compute_psnr(actual_image, expected_image)
    if psnr < psnr_threshold:
        PIL.Image.fromarray(actual_image).save(actual_path)
        raise AssertionError(
            f"PSNR match failed for {test_name}: "
            f"PSNR {psnr:.2f} < {psnr_threshold}. "
            f"Actual image saved to {actual_path}"
        )

    # If we reach here, the images matched
    # Clean up any previous actual image
    if actual_path.exists():
        actual_path.unlink()


class Draw2dTestEngine(BaseResource):
    def __init__(self):
        super().__init__(parent_resource=None)
        self.gpu_context = GpuContext(
            app_name="zfw draw_2d_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )
        self.draw_2d_context = Draw2dContext(gpu_context=self.gpu_context)

        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )
        self.renderer = Draw2dRenderer(
            context=self.draw_2d_context,
            device=self.gpu_device,
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

        self.draw_2d_context.dispose()
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

    def draw(self):
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="graphics",
        )
        self.renderer.draw(
            command_encoder=command_encoder,
            target=self.target,
        )

        command_encoder.submit().wait()


def test_draw_2d_quads():
    engine = Draw2dTestEngine()

    renderer = engine.renderer
    renderer.add_quad(
        dst_xy=(32, 64),
        dst_wh=(512, 256),
        color=(1.0, 1.0, 1.0, 1.0),
        border_thickness=(0, 0, 8, 0),
        border_color=(0.0, 0.1, 0.8, 1.0),
    )
    renderer.add_quad(
        dst_xy=(40, 72),
        dst_wh=(64, 64),
        color=(0.0, 0.2, 0.0, 1.0),
    )
    renderer.add_quad(
        dst_xy=(112, 72),
        dst_wh=(64, 64),
        color=(0.0, 0.2, 0.0, 0.5),
    )

    engine.draw()
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_quads",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_image():
    engine = Draw2dTestEngine()

    image_data = load_rgba_image("tests_data/rainbow-512x512.png")
    assert image_data.shape == (512, 512, 4)

    # Create a GpuImage for the texture
    gpu_image = GpuImage(
        device=engine.gpu_device,
        usages=["texture-binding", "transfer-dst"],
        meta=GpuImageMeta(
            shape=(512, 512, 4),
            dtype=np.float32,
            color_space="linear",
        ),
    )

    # Upload image data
    staging_buf = GpuBuffer(
        device=engine.gpu_device,
        usages=["staging", "copy-src"],
        meta=GpuBufferMeta(
            element_count=image_data.size,
            element_dtype=image_data.dtype,
        ),
    )
    staging_buf.memory.write(data=image_data)

    encoder = GpuCommandEncoder(device=engine.gpu_device, queue_type="transfer")
    encoder.transition_image_layout(image=gpu_image, layout="transfer-dst-optimal")

    encoder.copy_buffer_to_image(
        src=staging_buf,
        dst=gpu_image,
        regions=[
            GpuBufferImageCopyRegion(
                buffer_offset=0,
                image_offset=(0, 0, 0),
                image_extent=(512, 512, 1),
            )
        ],
    )
    encoder.submit().wait()
    staging_buf.dispose()

    border_thickness = 8
    renderer = engine.renderer
    renderer.add_quad(
        dst_xy=(
            (TEST_IMAGE_W - image_data.shape[1] - border_thickness) // 2,
            (TEST_IMAGE_H - image_data.shape[0] - border_thickness) // 2,
        ),
        color=(1.0, 1.0, 1.0, 1.0),
        border_thickness=(8, 8, 8, 8),
        border_color=(1.0, 1.0, 0.0, 1.0),
        image=gpu_image,
    )

    engine.draw()
    output_image = engine.readback()

    assert_image_matches_reference(
        output_image,
        "test_draw_2d_image",
        psnr_threshold=65.0,
    )

    gpu_image.dispose()
    engine.dispose()


def test_draw_2d_text_basic():
    engine = Draw2dTestEngine()
    renderer = engine.renderer

    renderer.add_text(
        text="Hello, world",
        font="sans-serif",
        dst_xy=(50, 50),
        dst_wh=(400, 100),
        font_size_px=48,
        color=(1.0, 1.0, 1.0, 1.0),
    )

    engine.draw()
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_text_basic",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_text_wrap():
    engine = Draw2dTestEngine()
    renderer = engine.renderer

    long_text = "This is a long text that should wrap to the next line because the width is limited."
    renderer.add_text(
        text=long_text,
        font="serif",
        dst_xy=(50, 200),
        dst_wh=(300, 400),
        font_size_px=32,
        color=(1.0, 0.8, 0.2, 1.0),
        wrap=True,
    )

    engine.draw()
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_text_wrap",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_text_clip():
    engine = Draw2dTestEngine()
    renderer = engine.renderer

    # Text that overflows but wrap is False
    renderer.add_text(
        text="This text should be clipped because it is too long for the box.",
        font="sans-serif",
        dst_xy=(50, 400),
        dst_wh=(200, 50),
        font_size_px=32,
        color=(0.5, 0.5, 1.0, 1.0),
        wrap=False,
    )

    engine.draw()
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_text_clip",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_text_matrix():
    engine = Draw2dTestEngine()
    renderer = engine.renderer

    fonts: list[Font] = ["sans-serif", "serif"]
    sizes = [12, 18, 24]
    weights = [100, 400, 700, 900]

    start_x = 20
    start_y = 20
    padding = 10

    current_y = start_y

    for font in fonts:
        for size in sizes:
            row_height = size + 20
            current_x = start_x

            for weight in weights:
                text = f"{font} {size}px w{weight}"

                # Estimate width
                box_w = 280
                box_h = row_height

                # Background quad with border
                renderer.add_quad(
                    dst_xy=(current_x, current_y),
                    dst_wh=(box_w, box_h),
                    color=(0.1, 0.1, 0.1, 1.0),
                    border_color=(0.5, 0.5, 0.5, 1.0),
                    border_thickness=(1, 1, 1, 1),
                )

                renderer.add_text(
                    text=text,
                    font=font,
                    dst_xy=(current_x + 5, current_y + 5),
                    dst_wh=(box_w - 10, box_h - 10),
                    font_size_px=size,
                    font_weight=weight,
                    color=(1.0, 1.0, 1.0, 1.0),
                    wrap=False,
                )

                current_x += box_w + padding

            current_y += row_height + padding

    engine.draw()
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_text_matrix",
        psnr_threshold=65.0,
    )

    engine.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
