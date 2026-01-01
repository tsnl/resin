from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from .basic import BaseResource, logger
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuContext,
    GpuDevice,
    GpuImage,
)
from .draw_2d import Draw2dRenderer, Draw2dTarget, Draw2dQuad
from .images import compute_psnr, convert_color
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
    """Test harness for Draw2d tests."""

    gpu_context: GpuContext
    gpu_device: GpuDevice
    renderer: Draw2dRenderer
    target: Draw2dTarget

    def __init__(self) -> None:
        super().__init__(parent_resource=None)

        self.gpu_context = GpuContext(
            app_name="zfw draw_2d_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )

        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )

        self.renderer = Draw2dRenderer(
            gpu_device=self.gpu_device,
            target_width_px=TEST_IMAGE_W,
            target_height_px=TEST_IMAGE_H,
            clear_color="black",
        )

        self.target = Draw2dTarget(renderer=self.renderer)

    def _on_dispose(self) -> None:
        self.renderer.dispose()
        self.gpu_device.dispose()
        self.gpu_context.dispose()

    def readback(self) -> np.ndarray:
        """Read back the rendered image as a uint8 RGBA array in sRGB color space."""
        color_image = self.target.color_image
        buffer = GpuBuffer(
            device=self.gpu_device,
            usages=["copy-dst", "staging"],
            meta=GpuBufferMeta(
                element_count=(color_image.height * color_image.width * 4),
                element_dtype=np.float32,
            ),
        )
        encoder = GpuCommandEncoder(device=self.gpu_device, queue_type="transfer")
        encoder.transition_image_layout(
            image=color_image,
            layout="transfer-src-optimal",
        )
        encoder.copy_image_to_buffer(src=color_image, dst=buffer)
        encoder.submit().wait()

        # Read as float32:
        data_f32 = buffer.memory.read(dtype=np.float32).reshape(
            (color_image.height, color_image.width, 4)
        )

        # Convert from linear to sRGB:
        data_f32_srgb = convert_color(
            data=data_f32,
            src_color_space="linear",
            dst_color_space="srgb",
        )

        # Clamp to [0, 1] and convert to uint8
        data_u8_srgb = (np.clip(data_f32_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Cleanup and return:
        buffer.dispose()
        return data_u8_srgb

    def draw(self, quads: list[Draw2dQuad]) -> None:
        """Render the given quads."""
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="graphics",
        )
        self.renderer.record_gpu_commands(
            command_encoder=command_encoder,
            target=self.target,
            quads=quads,
        )
        command_encoder.submit().wait()


def test_draw_2d_quads():
    """Test rendering colored quads with borders."""
    engine = Draw2dTestEngine()

    quads = [
        # Large white quad with blue bottom border
        Draw2dQuad(
            dst_xywh_px=(32, 64, 512, 256),
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_px=(0, 0, 8, 0),
            border_color=(0.0, 0.1, 0.8, 1.0),
        ),
        # Small green opaque quad
        Draw2dQuad(
            dst_xywh_px=(40, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 1.0),
        ),
        # Small green semi-transparent quad
        Draw2dQuad(
            dst_xywh_px=(112, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 0.5),
        ),
    ]

    engine.draw(quads)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_quads",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_image():
    """Test rendering a textured quad with an image."""
    engine = Draw2dTestEngine()

    # Load test image
    image_data = load_rgba_image("tests_data/rainbow-512x512.png")
    assert image_data.shape == (512, 512, 4)

    # Create a GpuImage for the texture
    gpu_image = GpuImage(
        device=engine.gpu_device,
        usages=["texture-binding"],
        data=image_data,
    )

    border_thickness = 8
    quads = [
        Draw2dQuad(
            dst_xywh_px=(
                (TEST_IMAGE_W - image_data.shape[1] - border_thickness) // 2,
                (TEST_IMAGE_H - image_data.shape[0] - border_thickness) // 2,
                image_data.shape[1],
                image_data.shape[0],
            ),
            fill_image=gpu_image,
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_px=(8, 8, 8, 8),
            border_color=(1.0, 1.0, 0.0, 1.0),
        ),
    ]

    engine.draw(quads)
    output_image = engine.readback()

    assert_image_matches_reference(
        output_image,
        "test_draw_2d_image",
        psnr_threshold=65.0,
    )

    gpu_image.dispose()
    engine.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
