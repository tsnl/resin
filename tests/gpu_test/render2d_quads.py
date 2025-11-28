"""Test 2D quad rendering with the Renderer2d GPU-accelerated renderer.

This test demonstrates:
- Creating a Renderer2d with Renderer2dContext
- Creating R2dQuadBatch instances with various configurations
- Rendering textured quads with tint colors
- Rendering quads with borders
- Rendering quads without atlases (using default white texture)
- Verifying correct rendering order (later quads on top)

The test renders several quads to an offscreen framebuffer, reads back the
result, and saves it as a PNG for visual inspection.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero.gpu import (
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
)
from zero.render2d import (
    R2dQuadBatch,
    R2dQuadList,
    Renderer2d,
    Renderer2dContext,
)


def make_context() -> tuple[GpuContext, GpuDevice]:
    """Create GPU context and device for testing."""
    ctx = GpuContext(
        app_name="gpu-render2d-test",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = ctx.create_device(physical_device=phys, surface=None)
    return ctx, dev


def generate_gradient_texture(width: int, height: int) -> torch.Tensor:
    """Generate a horizontal gradient texture (RGBA8).

    Left side is red, right side is blue, with full opacity.
    """
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * (1.0 - x / (width - 1)))
            g = 0
            b = int(255 * (x / (width - 1)))
            data[y, x] = [r, g, b, 255]
    return torch.from_numpy(data)


def generate_checkerboard_texture(
    width: int, height: int, tile_size: int = 16
) -> torch.Tensor:
    """Generate a checkerboard texture (RGBA8).

    Alternating white and black tiles.
    """
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            tile_x = x // tile_size
            tile_y = y // tile_size
            if (tile_x + tile_y) % 2 == 0:
                data[y, x] = [255, 255, 255, 255]  # White
            else:
                data[y, x] = [0, 0, 0, 255]  # Black
    return torch.from_numpy(data)


def generate_solid_color_texture(
    width: int, height: int, color: tuple[int, int, int, int]
) -> torch.Tensor:
    """Generate a solid color texture (RGBA8)."""
    data = np.zeros((height, width, 4), dtype=np.uint8)
    data[:, :] = color
    return torch.from_numpy(data)


def upload_texture(dev: GpuDevice, texture_data: torch.Tensor) -> GpuImage:
    """Upload texture data to the GPU."""
    height, width, channels = texture_data.shape
    assert channels == 4, "Texture must be RGBA8"

    # Create the GPU image
    texture = dev.create_image(
        usages=["texture-binding"],
        meta=GpuImageMeta(shape=(height, width, 4), dtype=torch.uint8),
    )

    # Create staging buffer for upload
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-src"],
        meta=texture.meta.into_buffer_meta(),
    )

    # Write data to staging buffer
    staging_buffer.write(data=texture_data.contiguous())

    # Copy staging buffer to texture
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=texture, layout="copy-dst")
    cmd.copy_buffer_to_image(src=staging_buffer, dst=texture)
    cmd.transition_image_layout(image=texture, layout="texture-binding")
    cmd.submit().wait()

    return texture


def create_quad_batch(
    atlas: GpuImage | None,
    quads: list[dict],
) -> R2dQuadBatch:
    """Create a quad batch from a list of quad specifications.

    Each quad dict should have:
    - pos: (x, y, w, h) - position and size in pixels
    - texcoord: (u, v, tw, th) - texcoord rect in pixels (optional)
    - tint: (r, g, b, a) - tint color 0-255 (optional, default white)
    - border_color: (r, g, b, a) - border color 0-255 (optional)
    - border: (top, right, bottom, left) - border thickness (optional)
    """
    n = len(quads)

    offset_px = torch.zeros((n, 4, 2), dtype=torch.uint32)
    texcoord_px = torch.zeros((n, 4, 2), dtype=torch.uint32)
    tint_color = torch.zeros((n, 4), dtype=torch.uint32)
    border_color = torch.zeros((n, 4), dtype=torch.uint32)
    border_thickness_px = torch.zeros((n, 4), dtype=torch.uint32)

    for i, quad in enumerate(quads):
        x, y, w, h = quad["pos"]

        # Set corner positions: TL, TR, BR, BL
        offset_px[i, 0] = torch.tensor([x, y], dtype=torch.uint32)  # TL
        offset_px[i, 1] = torch.tensor([x + w, y], dtype=torch.uint32)  # TR
        offset_px[i, 2] = torch.tensor([x + w, y + h], dtype=torch.uint32)  # BR
        offset_px[i, 3] = torch.tensor([x, y + h], dtype=torch.uint32)  # BL

        # Texcoords (default to unit square)
        tc = quad.get("texcoord", (0, 0, 1, 1))
        u, v, tw, th = tc
        texcoord_px[i, 0] = torch.tensor([u, v], dtype=torch.uint32)  # TL
        texcoord_px[i, 1] = torch.tensor([u + tw, v], dtype=torch.uint32)  # TR
        texcoord_px[i, 2] = torch.tensor([u + tw, v + th], dtype=torch.uint32)  # BR
        texcoord_px[i, 3] = torch.tensor([u, v + th], dtype=torch.uint32)  # BL

        # Tint color (default white)
        tint = quad.get("tint", (255, 255, 255, 255))
        tint_color[i] = torch.tensor(tint, dtype=torch.uint32)

        # Border color (default transparent)
        bc = quad.get("border_color", (0, 0, 0, 0))
        border_color[i] = torch.tensor(bc, dtype=torch.uint32)

        # Border thickness (default 0)
        bt = quad.get("border", (0, 0, 0, 0))
        border_thickness_px[i] = torch.tensor(bt, dtype=torch.uint32)

    return R2dQuadBatch(
        atlas=atlas,
        offset_px=offset_px,
        texcoord_px=texcoord_px,
        tint_color=tint_color,
        border_color=border_color,
        border_thickness_px=border_thickness_px,
    )


def test_render2d_quads():
    """Render various quads and save as PNG."""
    ctx, dev = make_context()

    # Framebuffer dimensions
    width, height = 512, 512

    # Create render target
    render_target = dev.create_image(
        usages=["color-attachment", "texture-binding"],
        meta=GpuImageMeta(shape=(height, width, 4), dtype=torch.uint8),
    )

    # Create textures
    print("Creating textures...")

    gradient_tex_data = generate_gradient_texture(64, 64)
    gradient_tex = upload_texture(dev, gradient_tex_data)
    print("  ✓ Created gradient texture (64x64)")

    checker_tex_data = generate_checkerboard_texture(64, 64, tile_size=8)
    checker_tex = upload_texture(dev, checker_tex_data)
    print("  ✓ Created checkerboard texture (64x64)")

    green_tex_data = generate_solid_color_texture(32, 32, (0, 255, 0, 255))
    green_tex = upload_texture(dev, green_tex_data)
    print("  ✓ Created green texture (32x32)")

    # Create Renderer2d
    print("\nCreating Renderer2d...")
    r2d_ctx = Renderer2dContext(device=dev)
    renderer: Renderer2d = r2d_ctx.create_renderer(
        framebuffer_width=width,
        framebuffer_height=height,
        max_quads_per_batch=1024,
    )
    print("  ✓ Created Renderer2d")

    # Create quad list with multiple batches
    quad_list = R2dQuadList()

    # Batch 1: Gradient texture quads
    batch1 = create_quad_batch(
        atlas=gradient_tex,
        quads=[
            # Large gradient quad in top-left
            {
                "pos": (20, 20, 150, 150),
                "texcoord": (0, 0, 64, 64),
                "tint": (255, 255, 255, 255),
            },
            # Smaller tinted gradient quad overlapping
            {
                "pos": (100, 100, 100, 100),
                "texcoord": (0, 0, 64, 64),
                "tint": (255, 255, 0, 255),  # Yellow tint
            },
        ],
    )
    quad_list.add_batch(batch1)
    print("  ✓ Created batch 1: gradient quads")

    # Batch 2: Checkerboard texture with borders
    batch2 = create_quad_batch(
        atlas=checker_tex,
        quads=[
            # Checkerboard with red border
            {
                "pos": (300, 20, 120, 120),
                "texcoord": (0, 0, 64, 64),
                "tint": (255, 255, 255, 255),
                "border": (5, 5, 5, 5),
                "border_color": (255, 0, 0, 255),
            },
            # Smaller checkerboard with thick blue border
            {
                "pos": (320, 160, 80, 80),
                "texcoord": (0, 0, 64, 64),
                "tint": (255, 255, 255, 255),
                "border": (10, 10, 10, 10),
                "border_color": (0, 0, 255, 255),
            },
        ],
    )
    quad_list.add_batch(batch2)
    print("  ✓ Created batch 2: checkerboard quads with borders")

    # Batch 3: No atlas (solid color quads using default white texture)
    batch3 = create_quad_batch(
        atlas=None,  # Will use default 1x1 white texture
        quads=[
            # Red quad
            {
                "pos": (20, 300, 80, 80),
                "texcoord": (0, 0, 1, 1),
                "tint": (255, 0, 0, 255),
            },
            # Green quad overlapping
            {
                "pos": (60, 340, 80, 80),
                "texcoord": (0, 0, 1, 1),
                "tint": (0, 255, 0, 255),
            },
            # Blue quad overlapping
            {
                "pos": (100, 380, 80, 80),
                "texcoord": (0, 0, 1, 1),
                "tint": (0, 0, 255, 255),
            },
        ],
    )
    quad_list.add_batch(batch3)
    print("  ✓ Created batch 3: solid color quads (no atlas)")

    # Batch 4: Green texture with various border configurations
    batch4 = create_quad_batch(
        atlas=green_tex,
        quads=[
            # Green with asymmetric border
            {
                "pos": (250, 300, 100, 100),
                "texcoord": (0, 0, 32, 32),
                "tint": (255, 255, 255, 255),
                "border": (15, 5, 5, 15),  # thick top and left
                "border_color": (128, 0, 128, 255),  # Purple
            },
            # Green with white border, semi-transparent tint
            {
                "pos": (380, 300, 100, 100),
                "texcoord": (0, 0, 32, 32),
                "tint": (255, 255, 255, 180),  # Semi-transparent
                "border": (3, 3, 3, 3),
                "border_color": (255, 255, 255, 255),
            },
        ],
    )
    quad_list.add_batch(batch4)
    print("  ✓ Created batch 4: green quads with borders")

    # Batch 5: Overlapping quads to test z-ordering
    batch5 = create_quad_batch(
        atlas=None,
        quads=[
            # Background quad (rendered first, should be behind)
            {
                "pos": (200, 200, 100, 100),
                "texcoord": (0, 0, 1, 1),
                "tint": (100, 100, 100, 255),  # Gray
            },
            # Middle quad
            {
                "pos": (220, 220, 100, 100),
                "texcoord": (0, 0, 1, 1),
                "tint": (255, 200, 0, 255),  # Orange
            },
            # Front quad (rendered last, should be on top)
            {
                "pos": (240, 240, 100, 100),
                "texcoord": (0, 0, 1, 1),
                "tint": (0, 255, 255, 255),  # Cyan
            },
        ],
    )
    quad_list.add_batch(batch5)
    print("  ✓ Created batch 5: overlapping quads for z-order test")

    # Render
    print(
        f"\nRendering {quad_list.total_quads} quads in {len(quad_list.batches)} batches..."
    )

    # Step 1: Upload quad data using a transfer command encoder
    transfer_cmd = dev.create_command_encoder(queue_type="transfer")
    renderer.upload(quad_list=quad_list, cmd=transfer_cmd)
    transfer_cmd.submit().wait()

    # Step 2: Render using a graphics command encoder
    graphics_cmd = dev.create_command_encoder(queue_type="graphics")
    graphics_cmd.transition_image_layout(
        image=render_target, layout="color-attachment-optimal"
    )

    with graphics_cmd.render(
        color_attachment=render_target,
        clear_on_load=True,
    ) as render_pass:
        renderer.render(quad_list=quad_list, render_pass=render_pass)

    graphics_cmd.submit().wait()
    print("  ✓ Rendering complete")

    # Read back the rendered image
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-dst"],
        meta=render_target.meta.into_buffer_meta(),
    )

    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=render_target, layout="copy-src")
    cmd.copy_image_to_buffer(src=render_target, dst=staging_buffer)
    cmd.submit().wait()

    image_data = staging_buffer.read()
    image_data = image_data.reshape((height, width, 4))
    image_array = image_data.numpy()

    # Save as PNG
    test_dir = Path(__file__).parent
    output_path = test_dir / "output_render2d_quads.png"
    img = Image.fromarray(image_array, mode="RGBA")
    img.save(output_path)
    print(f"\n✓ Output saved to: {output_path}")

    # Validate results
    print("\nValidating results...")

    # Check that we have non-black pixels (something was rendered)
    non_black_pixels = np.sum(image_array[:, :, :3].sum(axis=2) > 0)
    assert non_black_pixels > 0, "No pixels were drawn!"
    print(f"  ✓ Non-black pixels: {non_black_pixels}")

    # Check specific regions for expected colors

    # Check red quad region (batch 3, first quad at 20, 300, size 80x80)
    # Check the top-left corner which doesn't overlap with other quads
    red_region = image_array[300:340, 20:60, :]
    avg_red = red_region[:, :, 0].mean()
    avg_green = red_region[:, :, 1].mean()
    avg_blue = red_region[:, :, 2].mean()
    assert avg_red > 200, f"Expected red region to be red, got avg R={avg_red}"
    assert avg_green < 50, (
        f"Expected red region to have low green, got avg G={avg_green}"
    )
    assert avg_blue < 50, f"Expected red region to have low blue, got avg B={avg_blue}"
    print("  ✓ Red quad region verified")

    # Check blue quad region (batch 3, third quad at 100, 380, size 80x80)
    # Check the bottom-right corner which doesn't overlap with other quads
    blue_region = image_array[420:460, 140:180, :]
    avg_r = blue_region[:, :, 0].mean()
    avg_g = blue_region[:, :, 1].mean()
    avg_b = blue_region[:, :, 2].mean()
    assert avg_r < 50, f"Expected blue region to have low R, got {avg_r}"
    assert avg_g < 50, f"Expected blue region to have low G, got {avg_g}"
    assert avg_b > 200, f"Expected blue region to be blue, got avg B={avg_b}"
    print("  ✓ Blue quad region verified")

    # Check that cyan quad (last in batch 5) is visible in front
    # It's at (240, 240) with size 100x100, so center is around (290, 290)
    cyan_region = image_array[280:320, 280:320, :]
    avg_r = cyan_region[:, :, 0].mean()
    avg_g = cyan_region[:, :, 1].mean()
    avg_b = cyan_region[:, :, 2].mean()
    # Cyan should have low R, high G, high B
    assert avg_r < 50, f"Cyan region should have low R, got {avg_r}"
    assert avg_g > 200, f"Cyan region should have high G, got {avg_g}"
    assert avg_b > 200, f"Cyan region should have high B, got {avg_b}"
    print("  ✓ Z-ordering verified (cyan quad on top)")

    # Check gradient texture is rendering (batch 1, quad at 20,20 size 150x150)
    gradient_region = image_array[50:100, 50:100, :]
    # Should have some red and blue from the gradient
    assert gradient_region[:, :, :3].sum() > 0, "Gradient region should not be black"
    print("  ✓ Gradient texture region verified")

    # Check checkerboard with red border (batch 2, quad at 300,20 with 5px border)
    # The border should be red
    top_border = image_array[15:20, 300:420, :]  # Top border region
    avg_border_r = top_border[:, :, 0].mean()
    assert avg_border_r > 200, f"Top border should be red, got R={avg_border_r}"
    print("  ✓ Checkerboard border verified")

    print("\n✓ All tests passed!")

    # Cleanup
    ctx.dispose()


if __name__ == "__main__":
    test_render2d_quads()
