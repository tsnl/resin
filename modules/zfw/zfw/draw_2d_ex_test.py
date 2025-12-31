from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from .basic import BaseResource, Font, FontSize, FontWeight, logger
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuContext,
    GpuDevice,
    GpuImage,
)
from .draw_2d import Draw2dTarget
from .draw_2d_ex import (
    Draw2dExCanvas,
    Draw2dExQuad,
    Draw2dExRenderer,
)
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
    expect_dir = Path("tests/expect/zfw/draw_2d_ex_test")
    expect_dir.mkdir(parents=True, exist_ok=True)

    expect_path = expect_dir / f"{test_name}.png"
    actual_path = expect_dir / f"{test_name}.actual.png"

    # Also save to output directory for convenience
    output_path = Path("output/zfw/draw_2d_ex_test") / f"{test_name}.png"
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


class Draw2dExTestEngine(BaseResource):
    """Test harness for Draw2dEx tests."""

    gpu_context: GpuContext
    gpu_device: GpuDevice
    renderer: Draw2dExRenderer

    def __init__(self, *, scale: float = 1.0) -> None:
        super().__init__(parent_resource=None)

        self._scale = scale

        self.gpu_context = GpuContext(
            app_name="zfw draw_2d_ex_test",
            enable_debug_layer_support=True,
            enable_present_support=False,
        )

        self.gpu_device = GpuDevice(
            context=self.gpu_context,
            physical_device=self.gpu_context.enumerate_physical_devices()[0],
            surface=None,
        )

        self.renderer = Draw2dExRenderer(
            gpu_device=self.gpu_device,
            target_width_px=TEST_IMAGE_W,
            target_height_px=TEST_IMAGE_H,
            clear_color="black",
        )
        self.target = Draw2dTarget(renderer=self.renderer.inner)

    def _on_dispose(self) -> None:
        self.renderer.dispose()
        self.gpu_device.dispose()
        self.gpu_context.dispose()

    def create_canvas(self) -> Draw2dExCanvas:
        """Create a new canvas with the engine's scale factor."""
        return Draw2dExCanvas(renderer=self.renderer, scale=self._scale)

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

    def draw(self, canvas: Draw2dExCanvas) -> None:
        """Render the given canvas."""
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="graphics",
        )
        self.renderer.record_gpu_commands(
            command_encoder=command_encoder,
            canvas=canvas,
            target=self.target,
        )
        command_encoder.submit().wait()


# -----------------------------------------------------------------------------
# Parity Tests (mirror draw_2d_test.py)
# -----------------------------------------------------------------------------


def test_draw_2d_ex_quads():
    """Test rendering colored quads with borders (parity with draw_2d)."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    # Large white quad with blue bottom border
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(32, 64, 512, 256),
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_dip=(0, 0, 8, 0),
            border_color=(0.0, 0.1, 0.8, 1.0),
        )
    )
    # Small green opaque quad
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(40, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 1.0),
        )
    )
    # Small green semi-transparent quad
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(112, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 0.5),
        )
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_quads",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_ex_image():
    """Test rendering a textured quad with an image (parity with draw_2d)."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

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
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(
                (TEST_IMAGE_W - image_data.shape[1] - border_thickness) // 2,
                (TEST_IMAGE_H - image_data.shape[0] - border_thickness) // 2,
                image_data.shape[1],
                image_data.shape[0],
            ),
            fill_image=gpu_image,
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_dip=(8, 8, 8, 8),
            border_color=(1.0, 1.0, 0.0, 1.0),
        )
    )

    engine.draw(canvas)
    output_image = engine.readback()

    assert_image_matches_reference(
        output_image,
        "test_draw_2d_ex_image",
        psnr_threshold=65.0,
    )

    gpu_image.dispose()
    engine.dispose()


# -----------------------------------------------------------------------------
# Text Tests
# -----------------------------------------------------------------------------


def test_draw_2d_ex_text_basic():
    """Test basic text rendering."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    canvas.add_text(
        text="Hello, world!",
        font="sans-serif",
        dst_xy_dip=(50, 50),
        dst_wh_dip=(400, 100),
        font_size="large",
        font_weight="regular",
        color=(1.0, 1.0, 1.0, 1.0),
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_text_basic",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_ex_text_wrap():
    """Test text wrapping."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    long_text = (
        "This is a long text that should wrap to the next line "
        "because the width is limited."
    )
    canvas.add_text(
        text=long_text,
        font="serif",
        dst_xy_dip=(50, 200),
        dst_wh_dip=(300, 400),
        font_size="large",
        font_weight="regular",
        color=(1.0, 0.8, 0.2, 1.0),
        wrap=True,
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_text_wrap",
        psnr_threshold=65.0,
    )

    engine.dispose()


# -----------------------------------------------------------------------------
# Text on Quads (New Functionality)
# -----------------------------------------------------------------------------


def test_draw_2d_ex_text_on_quad():
    """Test text rendered on top of a colored quad (z-ordering)."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    # Background quad
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(100, 100, 400, 200),
            fill_color=(0.2, 0.2, 0.5, 1.0),
            border_thickness_dip=(4, 4, 4, 4),
            border_color=(0.8, 0.8, 0.2, 1.0),
        )
    )

    # Text on top of the quad
    canvas.add_text(
        text="Text on Quad",
        font="sans-serif",
        dst_xy_dip=(110, 110),
        dst_wh_dip=(380, 180),
        font_size="extra-large",
        font_weight="bold",
        color=(1.0, 1.0, 1.0, 1.0),
        horizontal_alignment="center",
        vertical_alignment="middle",
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_text_on_quad",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_ex_layered_quads_and_text():
    """Test multiple layers of quads and text with correct z-ordering."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    # First layer: large background quad
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(50, 50, 600, 400),
            fill_color=(0.1, 0.1, 0.1, 1.0),
        )
    )

    # Second layer: text label
    canvas.add_text(
        text="Background Layer",
        font="sans-serif",
        dst_xy_dip=(60, 60),
        dst_wh_dip=(200, 30),
        font_size="regular",
        font_weight="light",
        color=(0.5, 0.5, 0.5, 1.0),
    )

    # Third layer: overlapping colored quad
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(100, 150, 300, 150),
            fill_color=(0.8, 0.2, 0.2, 0.9),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Fourth layer: text on top of the red quad
    canvas.add_text(
        text="Red Panel",
        font="serif",
        dst_xy_dip=(110, 160),
        dst_wh_dip=(280, 130),
        font_size="large",
        font_weight="bold",
        color=(1.0, 1.0, 1.0, 1.0),
        horizontal_alignment="center",
        vertical_alignment="middle",
    )

    # Fifth layer: another quad that overlaps the red one
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(250, 200, 300, 150),
            fill_color=(0.2, 0.6, 0.2, 0.9),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Sixth layer: text on top of the green quad
    canvas.add_text(
        text="Green Panel",
        font="serif",
        dst_xy_dip=(260, 210),
        dst_wh_dip=(280, 130),
        font_size="large",
        font_weight="bold",
        color=(1.0, 1.0, 1.0, 1.0),
        horizontal_alignment="center",
        vertical_alignment="middle",
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_layered_quads_and_text",
        psnr_threshold=65.0,
    )

    engine.dispose()


# -----------------------------------------------------------------------------
# Configuration Matrix Test
# -----------------------------------------------------------------------------


def test_draw_2d_ex_font_matrix():
    """Test matrix of font configurations: fonts × sizes × weights."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    fonts: list[Font] = ["sans-serif", "serif", "monospaced"]
    sizes: list[FontSize] = ["regular", "large", "extra-large"]
    weights: list[FontWeight] = ["light", "regular", "bold"]

    start_x = 20
    start_y = 20
    padding = 10

    current_y = start_y

    for font in fonts:
        for size in sizes:
            # Estimate row height based on size
            row_height = {"regular": 28, "large": 40, "extra-large": 52}[size]
            current_x = start_x

            for weight in weights:
                text = f"{font[:4]} {size[:3]} {weight[:3]}"

                box_w = 200
                box_h = row_height

                # Background quad with border
                canvas.add_quad(
                    Draw2dExQuad(
                        dst_xywh_dip=(current_x, current_y, box_w, box_h),
                        fill_color=(0.15, 0.15, 0.15, 1.0),
                        border_thickness_dip=(1, 1, 1, 1),
                        border_color=(0.4, 0.4, 0.4, 1.0),
                    )
                )

                # Text label
                canvas.add_text(
                    text=text,
                    font=font,
                    dst_xy_dip=(current_x + 5, current_y + 2),
                    dst_wh_dip=(box_w - 10, box_h - 4),
                    font_size=size,
                    font_weight=weight,
                    color=(1.0, 1.0, 1.0, 1.0),
                    wrap=False,
                    vertical_alignment="middle",
                )

                current_x += box_w + padding

            current_y += row_height + padding

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_font_matrix",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_ex_scale_matrix():
    """Test rendering at different scale factors."""
    # Test at scale 2.0
    engine = Draw2dExTestEngine(scale=2.0)
    canvas = engine.create_canvas()

    # At scale 2.0, a 100x50 DIP quad becomes 200x100 physical pixels
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(50, 50, 200, 100),
            fill_color=(0.3, 0.3, 0.6, 1.0),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(1.0, 1.0, 0.0, 1.0),
        )
    )

    canvas.add_text(
        text="Scale 2.0",
        font="sans-serif",
        dst_xy_dip=(60, 60),
        dst_wh_dip=(180, 80),
        font_size="large",
        font_weight="bold",
        color=(1.0, 1.0, 1.0, 1.0),
        horizontal_alignment="center",
        vertical_alignment="middle",
    )

    # Another quad at a different position (in DIP space)
    canvas.add_quad(
        Draw2dExQuad(
            dst_xywh_dip=(50, 180, 200, 100),
            fill_color=(0.6, 0.3, 0.3, 1.0),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(0.0, 1.0, 1.0, 1.0),
        )
    )

    canvas.add_text(
        text="HiDPI Test",
        font="serif",
        dst_xy_dip=(60, 190),
        dst_wh_dip=(180, 80),
        font_size="regular",
        font_weight="regular",
        color=(1.0, 1.0, 1.0, 1.0),
        horizontal_alignment="center",
        vertical_alignment="middle",
    )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_scale_matrix",
        psnr_threshold=65.0,
    )

    engine.dispose()


def test_draw_2d_ex_text_alignment():
    """Test text alignment options."""
    engine = Draw2dExTestEngine()
    canvas = engine.create_canvas()

    alignments = [
        ("left", "top"),
        ("center", "top"),
        ("right", "top"),
        ("left", "middle"),
        ("center", "middle"),
        ("right", "middle"),
        ("left", "bottom"),
        ("center", "bottom"),
        ("right", "bottom"),
    ]

    box_w, box_h = 200, 80
    padding = 10
    start_x, start_y = 20, 20

    for i, (h_align, v_align) in enumerate(alignments):
        row = i // 3
        col = i % 3

        x = start_x + col * (box_w + padding)
        y = start_y + row * (box_h + padding)

        # Background box
        canvas.add_quad(
            Draw2dExQuad(
                dst_xywh_dip=(x, y, box_w, box_h),
                fill_color=(0.2, 0.2, 0.2, 1.0),
                border_thickness_dip=(1, 1, 1, 1),
                border_color=(0.5, 0.5, 0.5, 1.0),
            )
        )

        # Aligned text
        canvas.add_text(
            text=f"{h_align[0].upper()}{v_align[0].upper()}",
            font="monospaced",
            dst_xy_dip=(x + 5, y + 5),
            dst_wh_dip=(box_w - 10, box_h - 10),
            font_size="large",
            font_weight="regular",
            color=(1.0, 1.0, 1.0, 1.0),
            horizontal_alignment=h_align,  # type: ignore
            vertical_alignment=v_align,  # type: ignore
        )

    engine.draw(canvas)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ex_text_alignment",
        psnr_threshold=65.0,
    )

    engine.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
