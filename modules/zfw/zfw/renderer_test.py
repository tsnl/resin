from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import PIL.Image
import pytest
from skimage.metrics import structural_similarity as ssim

from .basic import BaseResource, Font
from .gpu import (
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuBuffer,
    GpuCommandEncoder,
    GpuImageMeta,
    GpuBufferMeta,
)
from .renderer import (
    Image,
    RendererContext,
    Renderer,
    Canvas,
)
from .images import load_rgba_image

TEST_IMAGE_W, TEST_IMAGE_H = 1280, 720


def assert_image_matches_reference(
    actual_image: np.ndarray,
    test_name: str,
    match_type: Literal["exact", "ssim"] = "ssim",
    ssim_threshold: float = 0.99,
) -> None:
    """
    Compare an actual rendered image against an expected reference image.

    If no reference exists, creates it and passes the test.
    If reference exists, compares using exact or SSIM matching.
    On mismatch, saves the actual image as <name>.actual.png and raises AssertionError.

    Args:
        actual_image: The rendered image to test (numpy array)
        test_name: Name of the test (used for file naming)
        match_type: Either "exact" for pixel-perfect matching or "ssim" for structural similarity
        ssim_threshold: Minimum SSIM score required (0-1), only used when match_type="ssim"
    """
    expect_dir = Path("tests/expect/zfw/renderer_test")
    expect_dir.mkdir(parents=True, exist_ok=True)

    expect_path = expect_dir / f"{test_name}.png"
    actual_path = expect_dir / f"{test_name}.actual.png"

    # Also save to output directory for convenience
    output_path = Path("output/zfw/renderer_test") / f"{test_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(actual_image).save(output_path)

    if not expect_path.exists():
        # No reference exists - create it and pass
        PIL.Image.fromarray(actual_image).save(expect_path)
        warnings.warn(f"{test_name}: no expect found: image created: {expect_path}")
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
    if match_type == "exact":
        if not np.array_equal(actual_image, expected_image):
            PIL.Image.fromarray(actual_image).save(actual_path)
            diff_pixels = np.sum(actual_image != expected_image)
            total_pixels = actual_image.size
            raise AssertionError(
                f"Exact pixel match failed for {test_name}: "
                f"{diff_pixels}/{total_pixels} pixels differ. "
                f"Actual image saved to {actual_path}"
            )
    elif match_type == "ssim":
        # Convert to float32 for SSIM calculation
        actual_float = actual_image.astype(np.float32) / 255.0
        expected_float = expected_image.astype(np.float32) / 255.0

        # Calculate SSIM per channel and average
        ssim_score = ssim(
            expected_float,
            actual_float,
            channel_axis=2,
            data_range=1.0,
        )

        if ssim_score < ssim_threshold:
            PIL.Image.fromarray(actual_image).save(actual_path)
            raise AssertionError(
                f"SSIM match failed for {test_name}: "
                f"score {ssim_score:.4f} < threshold {ssim_threshold}. "
                f"Actual image saved to {actual_path}"
            )
    else:
        raise ValueError(f"Unknown match_type: {match_type}")


class RendererTestEngine(BaseResource):
    def __init__(self):
        super().__init__(parent_resource=None)
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
            max_frames_in_flight=1,
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

    def _on_dispose_resource(self) -> None:
        self.target.dispose_resource()

        self.renderer.dispose_resource()
        self.gpu_device.dispose_resource()

        self.renderer_context.dispose_resource()
        self.gpu_context.dispose_resource()

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

    def draw(self, canvas: Canvas):
        command_encoder = GpuCommandEncoder(
            device=self.gpu_device,
            queue_type="graphics",
        )
        self.renderer.draw(
            command_encoder=command_encoder,
            canvas=canvas,
            target=self.target,
            frame_index=0,
        )

        command_encoder.submit().wait()


def test_renderer_quads():
    engine = RendererTestEngine()

    canvas = Canvas(renderer=engine.renderer)
    canvas.add_quad(
        dst_xy=(32, 64),
        dst_wh=(512, 256),
        color=(1.0, 1.0, 1.0, 1.0),
        border_thickness=(0, 0, 8, 0),
        border_color=(0.0, 0.1, 0.8, 1.0),
    )
    canvas.add_quad(
        dst_xy=(40, 72),
        dst_wh=(64, 64),
        color=(0.0, 0.2, 0.0, 1.0),
    )
    canvas.add_quad(
        dst_xy=(112, 72),
        dst_wh=(64, 64),
        color=(0.0, 0.2, 0.0, 0.5),
    )

    engine.draw(canvas=canvas)
    image = engine.readback()

    assert_image_matches_reference(image, "test_renderer_quads")

    engine.dispose_resource()


def test_renderer_image():
    engine = RendererTestEngine()

    image_data = load_rgba_image("tests_data/rainbow-512x512.png")
    assert image_data.shape == (512, 512, 4)

    image = Image(
        renderer=engine.renderer,
        data=image_data,
        sampler="nearest",
    )

    border_thickness = 8
    canvas = Canvas(renderer=engine.renderer)
    canvas.add_quad(
        dst_xy=(
            (TEST_IMAGE_W - image_data.shape[1] - border_thickness) // 2,
            (TEST_IMAGE_H - image_data.shape[0] - border_thickness) // 2,
        ),
        color=(1.0, 1.0, 1.0, 1.0),
        border_thickness=(8, 8, 8, 8),
        border_color=(1.0, 1.0, 0.0, 1.0),
        image=image,
    )

    engine.draw(canvas=canvas)
    output_image = engine.readback()

    assert_image_matches_reference(output_image, "test_renderer_image")

    engine.dispose_resource()


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
        max_frames_in_flight=1,
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

    image = Image(
        renderer=renderer,
        data=orig_image_data,
        sampler="nearest",
    )
    renderer.atlas.heap(channels=4).insert(image)

    encoder = GpuCommandEncoder(device=gpu_device, queue_type="transfer")
    renderer.atlas.heap(channels=4).flush(command_encoder=encoder, frame_index=0)
    encoder.submit().wait()

    # Note: x=0 because it's larger than default_white_image (1x1)
    assert image.allocation_px_xywh[2:] == (128, 128)

    renderer.dispose_resource()
    gpu_device.dispose_resource()
    renderer_context.dispose_resource()
    gpu_context.dispose_resource()


def test_renderer_text_basic():
    engine = RendererTestEngine()
    canvas = Canvas(renderer=engine.renderer)

    canvas.add_text(
        text="Hello, world",
        font="sans-serif",
        dst_xy=(50, 50),
        dst_wh=(400, 100),
        font_size_px=48,
        color=(1.0, 1.0, 1.0, 1.0),
    )

    engine.draw(canvas=canvas)
    image = engine.readback()

    assert_image_matches_reference(image, "test_renderer_text_basic")

    engine.dispose_resource()


def test_renderer_text_wrap():
    engine = RendererTestEngine()
    canvas = Canvas(renderer=engine.renderer)

    long_text = "This is a long text that should wrap to the next line because the width is limited."
    canvas.add_text(
        text=long_text,
        font="serif",
        dst_xy=(50, 200),
        dst_wh=(300, 400),
        font_size_px=32,
        color=(1.0, 0.8, 0.2, 1.0),
        wrap=True,
    )

    engine.draw(canvas=canvas)
    image = engine.readback()

    assert_image_matches_reference(image, "test_renderer_text_wrap")

    engine.dispose_resource()


def test_renderer_text_clip():
    engine = RendererTestEngine()
    canvas = Canvas(renderer=engine.renderer)

    # Text that overflows but wrap is False
    canvas.add_text(
        text="This text should be clipped because it is too long for the box.",
        font="sans-serif",
        dst_xy=(50, 400),
        dst_wh=(200, 50),
        font_size_px=32,
        color=(0.5, 0.5, 1.0, 1.0),
        wrap=False,
    )

    engine.draw(canvas=canvas)
    image = engine.readback()

    assert_image_matches_reference(image, "test_renderer_text_clip")

    engine.dispose_resource()


def test_renderer_text_matrix():
    engine = RendererTestEngine()
    canvas = Canvas(renderer=engine.renderer)

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
                canvas.add_quad(
                    dst_xy=(current_x, current_y),
                    dst_wh=(box_w, box_h),
                    color=(0.1, 0.1, 0.1, 1.0),
                    border_color=(0.5, 0.5, 0.5, 1.0),
                    border_thickness=(1, 1, 1, 1),
                )

                canvas.add_text(
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

    engine.draw(canvas=canvas)
    image = engine.readback()

    assert_image_matches_reference(image, "test_renderer_text_matrix")

    engine.dispose_resource()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
