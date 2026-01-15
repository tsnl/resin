import logging
from typing import Generator

import numpy as np
import numpy.typing as npt
import pytest
import wgpu

from resin import (
    setup_logging,
    BaseDisposable,
    Font,
    FontSize,
    FontWeight,
    Draw2dRenderer,
    Draw2dExtBasePrimitive,
    Draw2dExtQuadPrimitive,
    Draw2dExtTextPrimitive,
    Draw2dExtCanvas,
    convert_color,
)

from image_ref_tests import assert_image_matches_reference


TEST_IMAGE_W, TEST_IMAGE_H = 1280, 720


class Draw2dExTestEngine(BaseDisposable):
    """Test harness for Draw2dEx tests."""

    device: wgpu.GPUDevice
    queue: wgpu.GPUQueue
    renderer: Draw2dRenderer
    canvas: Draw2dExtCanvas
    readback_buffer: wgpu.GPUBuffer

    def __init__(self, *, gpu: wgpu.GPUDevice, scale: float = 1.0) -> None:
        super().__init__()

        setup_logging(level=logging.ERROR)

        self._scale = scale

        # Initialize WebGPU
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync(label="Draw2dExTestEngine")
        self.queue = self.device.queue

        # Create renderer
        self.renderer = Draw2dRenderer(
            self.device,
            self.queue,
            (int(TEST_IMAGE_W * self._scale), int(TEST_IMAGE_H * self._scale)),
        )
        self.canvas = Draw2dExtCanvas(device=self.device, queue=self.queue)

        # Readback buffer for image data
        self.readback_buffer = self.device.create_buffer(
            size=int(TEST_IMAGE_W * TEST_IMAGE_H * self._scale * self._scale * 4),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            label="Draw2dExTestEngine.ReadbackBuffer",
        )

    def _on_dispose(self) -> None:
        self.canvas.dispose()

    def readback(self) -> np.ndarray:
        """Read back the rendered image as a uint8 RGBA array in sRGB color space."""
        color_image = self.renderer.get_output_image()

        # Create command encoder for readback
        command_encoder = self.device.create_command_encoder(label="Readback")

        # Copy from texture to staging buffer
        command_encoder.copy_texture_to_buffer(
            source=wgpu.TexelCopyTextureInfo(
                texture=color_image,
                mip_level=0,
                origin=(0, 0, 0),
            ),
            destination=wgpu.TexelCopyBufferInfo(
                buffer=self.readback_buffer,
                bytes_per_row=int(TEST_IMAGE_W * self._scale) * 4,
                rows_per_image=int(TEST_IMAGE_H * self._scale),
            ),
            copy_size=color_image.size,
        )

        # Submit and wait
        self.queue.submit([command_encoder.finish()])

        # Read as uint8
        self.readback_buffer.map_sync(wgpu.MapMode.READ)
        data_raw = np.asarray(self.readback_buffer.read_mapped())
        self.readback_buffer.unmap()

        # Convert from linear to sRGB
        data_raw_srgb_f32 = convert_color(
            data=data_raw.astype(np.float32) / 255.0,
            src_color_space="linear",
            dst_color_space="srgb",
        )
        data_raw_srgb = (np.clip(data_raw_srgb_f32, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Reshape and return as uint8 sRGB
        return data_raw_srgb.reshape(
            (int(TEST_IMAGE_H * self._scale), int(TEST_IMAGE_W * self._scale), 4)
        )

    def draw(self, primitives: list[Draw2dExtBasePrimitive]) -> None:
        """Render the given primitives."""
        command_encoder = self.device.create_command_encoder(
            label="Draw2dExTestEngine.CommandEncoder"
        )
        self.renderer.record(
            quads=self.canvas.quads(primitives=primitives, scale=self._scale),
            command_encoder=command_encoder,
        )
        self.queue.submit([command_encoder.finish()])


@pytest.fixture(scope="module")
def engine(gpu_device: wgpu.GPUDevice) -> Generator[Draw2dExTestEngine, None, None]:
    """Fixture that provides a Draw2dExTestEngine."""
    eng = Draw2dExTestEngine(gpu=gpu_device)
    yield eng
    eng.dispose()


def test_draw_2d_ext_quads(engine: Draw2dExTestEngine):
    """Test rendering colored quads with borders (parity with draw_2d)."""
    primitives: list[Draw2dExtBasePrimitive] = []

    # Large white quad with blue bottom border
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(32, 64, 512, 256),
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_dip=(0, 0, 8, 0),
            border_color=(0.0, 0.1, 0.8, 1.0),
        )
    )
    # Small green opaque quad
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(40, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 1.0),
        )
    )
    # Small green semi-transparent quad
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(112, 72, 64, 64),
            fill_color=(0.0, 0.2, 0.0, 0.5),
        )
    )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_quads",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


def test_draw_2d_ext_image(
    engine: Draw2dExTestEngine,
    rainbow_512x512_image: npt.NDArray[np.float32],
):
    """Test rendering a textured quad with an image (parity with draw_2d)."""
    primitives: list[Draw2dExtBasePrimitive] = []

    # Load test image (returns float32 linear color space)
    image_data = rainbow_512x512_image
    assert image_data.shape == (512, 512, 4)

    # Convert from linear to sRGB and to uint8 for rgba8unorm texture
    image_data_uint8 = (np.clip(image_data, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Create a WebGPU texture for the image
    image_texture = engine.device.create_texture(
        label="RainbowImage",
        size=(image_data.shape[1], image_data.shape[0], 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    engine.queue.write_texture(
        destination=wgpu.TexelCopyTextureInfo(
            texture=image_texture,
            origin=(0, 0, 0),
            mip_level=0,
            aspect=wgpu.TextureAspect.all,
        ),
        data=image_data_uint8.tobytes(),
        data_layout=wgpu.TexelCopyBufferLayout(
            offset=0,
            bytes_per_row=image_data.shape[1] * 4,
            rows_per_image=image_data.shape[0],
        ),
        size=image_texture.size,
    )

    border_thickness = 8
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(
                (TEST_IMAGE_W - image_data.shape[1] - border_thickness) // 2,
                (TEST_IMAGE_H - image_data.shape[0] - border_thickness) // 2,
                image_data.shape[1],
                image_data.shape[0],
            ),
            fill_texture=image_texture,
            fill_color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_dip=(8, 8, 8, 8),
            border_color=(1.0, 1.0, 0.0, 1.0),
        )
    )

    engine.draw(primitives)
    output_image = engine.readback()

    assert_image_matches_reference(
        output_image,
        "test_draw_2d_ext_image",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


# -----------------------------------------------------------------------------
# Text Tests
# -----------------------------------------------------------------------------


def test_draw_2d_ext_text_basic(engine: Draw2dExTestEngine):
    """Test basic text rendering."""
    primitives: list[Draw2dExtBasePrimitive] = []

    primitives.append(
        Draw2dExtTextPrimitive(
            text="Hello, world!",
            font="sans-serif",
            dst_xy_dip=(50, 50),
            dst_wh_dip=(400, 100),
            font_size="large",
            font_weight="regular",
            color=(1.0, 1.0, 1.0, 1.0),
            wrap=True,
            horizontal_alignment="left",
            vertical_alignment="top",
        )
    )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_text_basic",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


def test_draw_2d_ext_text_wrap(engine: Draw2dExTestEngine):
    """Test text wrapping."""
    primitives: list[Draw2dExtBasePrimitive] = []

    long_text = (
        "This is a long text that should wrap to the next line "
        "because the width is limited."
    )
    primitives.append(
        Draw2dExtTextPrimitive(
            text=long_text,
            font="serif",
            dst_xy_dip=(50, 200),
            dst_wh_dip=(300, 400),
            font_size="large",
            font_weight="regular",
            color=(1.0, 0.8, 0.2, 1.0),
            wrap=True,
            horizontal_alignment="left",
            vertical_alignment="top",
        )
    )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_text_wrap",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


def test_draw_2d_ext_text_on_quad(engine: Draw2dExTestEngine):
    """Test text rendered on top of a colored quad (z-ordering)."""
    primitives: list[Draw2dExtBasePrimitive] = []

    # Background quad
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(100, 100, 400, 200),
            fill_color=(0.2, 0.2, 0.5, 1.0),
            border_thickness_dip=(4, 4, 4, 4),
            border_color=(0.8, 0.8, 0.2, 1.0),
        )
    )

    # Text on top of the quad
    primitives.append(
        Draw2dExtTextPrimitive(
            text="Text on Quad",
            font="sans-serif",
            dst_xy_dip=(110, 110),
            dst_wh_dip=(380, 180),
            font_size="extra-large",
            font_weight="bold",
            color=(1.0, 1.0, 1.0, 1.0),
            wrap=True,
            horizontal_alignment="center",
            vertical_alignment="middle",
        )
    )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_text_on_quad",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


def test_draw_2d_ext_layered_quads_and_text(engine: Draw2dExTestEngine):
    """Test multiple layers of quads and text with correct z-ordering."""
    primitives: list[Draw2dExtBasePrimitive] = []

    # First layer: large background quad
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(50, 50, 600, 400),
            fill_color=(0.1, 0.1, 0.1, 1.0),
        )
    )

    # Second layer: text label
    primitives.append(
        Draw2dExtTextPrimitive(
            text="Background Layer",
            font="sans-serif",
            dst_xy_dip=(60, 60),
            dst_wh_dip=(200, 30),
            font_size="regular",
            font_weight="light",
            color=(0.5, 0.5, 0.5, 1.0),
            wrap=True,
            horizontal_alignment="left",
            vertical_alignment="top",
        )
    )

    # Third layer: overlapping colored quad
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(100, 150, 300, 150),
            fill_color=(0.8, 0.2, 0.2, 0.9),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Fourth layer: text on top of the red quad
    primitives.append(
        Draw2dExtTextPrimitive(
            text="Red Panel",
            font="serif",
            dst_xy_dip=(110, 160),
            dst_wh_dip=(280, 130),
            font_size="large",
            font_weight="bold",
            color=(1.0, 1.0, 1.0, 1.0),
            wrap=True,
            horizontal_alignment="center",
            vertical_alignment="middle",
        )
    )

    # Fifth layer: another quad that overlaps the red one
    primitives.append(
        Draw2dExtQuadPrimitive(
            dst_xywh_dip=(250, 200, 300, 150),
            fill_color=(0.2, 0.6, 0.2, 0.9),
            border_thickness_dip=(2, 2, 2, 2),
            border_color=(1.0, 1.0, 1.0, 1.0),
        )
    )

    # Sixth layer: text on top of the green quad
    primitives.append(
        Draw2dExtTextPrimitive(
            text="Green Panel",
            font="serif",
            dst_xy_dip=(260, 210),
            dst_wh_dip=(280, 130),
            font_size="large",
            font_weight="bold",
            color=(1.0, 1.0, 1.0, 1.0),
            wrap=True,
            horizontal_alignment="center",
            vertical_alignment="middle",
        )
    )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_layered_quads_and_text",
        psnr_threshold=56.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


# -----------------------------------------------------------------------------
# Configuration Matrix Test
# -----------------------------------------------------------------------------


def test_draw_2d_ext_font_matrix(engine: Draw2dExTestEngine):
    """Test matrix of font configurations: fonts × sizes × weights."""
    primitives: list[Draw2dExtBasePrimitive] = []

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
                primitives.append(
                    Draw2dExtQuadPrimitive(
                        dst_xywh_dip=(current_x, current_y, box_w, box_h),
                        fill_color=(0.15, 0.15, 0.15, 1.0),
                        border_thickness_dip=(1, 1, 1, 1),
                        border_color=(0.4, 0.4, 0.4, 1.0),
                    )
                )

                # Text label
                primitives.append(
                    Draw2dExtTextPrimitive(
                        text=text,
                        font=font,
                        dst_xy_dip=(current_x + 5, current_y + 2),
                        dst_wh_dip=(box_w - 10, box_h - 4),
                        font_size=size,
                        font_weight=weight,
                        color=(1.0, 1.0, 1.0, 1.0),
                        wrap=False,
                        horizontal_alignment="left",
                        vertical_alignment="middle",
                    )
                )

                current_x += box_w + padding

            current_y += row_height + padding

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_font_matrix",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


def test_draw_2d_ext_text_alignment(engine: Draw2dExTestEngine):
    """Test text alignment options."""
    primitives: list[Draw2dExtBasePrimitive] = []

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
        primitives.append(
            Draw2dExtQuadPrimitive(
                dst_xywh_dip=(x, y, box_w, box_h),
                fill_color=(0.2, 0.2, 0.2, 1.0),
                border_thickness_dip=(1, 1, 1, 1),
                border_color=(0.5, 0.5, 0.5, 1.0),
            )
        )

        # Aligned text
        primitives.append(
            Draw2dExtTextPrimitive(
                text=f"{h_align[0].upper()}{v_align[0].upper()}",
                font="monospaced",
                dst_xy_dip=(x + 5, y + 5),
                dst_wh_dip=(box_w - 10, box_h - 10),
                font_size="large",
                font_weight="regular",
                color=(1.0, 1.0, 1.0, 1.0),
                wrap=True,
                horizontal_alignment=h_align,  # type: ignore
                vertical_alignment=v_align,  # type: ignore
            )
        )

    engine.draw(primitives)
    image = engine.readback()

    assert_image_matches_reference(
        image,
        "test_draw_2d_ext_text_alignment",
        psnr_threshold=65.0,
        test_subdir="test_draw_2d_ext",
    )

    engine.dispose()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
