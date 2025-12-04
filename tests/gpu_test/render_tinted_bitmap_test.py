"""Test rendering a tinted bitmap using descriptor sets with texture and buffer bindings.

This test demonstrates:
- Creating and binding texture samplers
- Creating and binding uniform buffers
- Creating descriptor set layouts and descriptor sets
- Using descriptor sets with a pipeline

The test renders a full-screen quad with a texture, applying a color tint
from a uniform buffer. It tests multiple procedurally generated textures
(gradient, checkerboard) with multiple tint colors (white, red, green, blue, black).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero.gpu import (
    GpuBufferMeta,
    GpuContext,
    GpuDescriptorSetLayoutBinding,
    GpuDescriptorSetBinding,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
)


def make_context() -> tuple[GpuContext, GpuDevice]:
    """Create GPU context and device for testing."""
    ctx = GpuContext(
        app_name="gpu-tinted-bitmap-test",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = ctx.create_device(physical_device=phys, surface=None)
    return ctx, dev


def generate_gradient_texture(width: int, height: int) -> torch.Tensor:
    """Generate a horizontal gradient texture (RGBA8).

    Left side is red, right side is blue, with green gradient from top to bottom.
    """
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * (1.0 - x / (width - 1)))
            g = int(255 * (y / (height - 1)))
            b = int(255 * (x / (width - 1)))
            data[y, x] = [r, g, b, 255]
    return torch.from_numpy(data)


def generate_checkerboard_texture(
    width: int, height: int, tile_size: int = 32
) -> torch.Tensor:
    """Generate a checkerboard texture (RGBA8).

    Alternating white and magenta tiles.
    """
    data = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            tile_x = x // tile_size
            tile_y = y // tile_size
            if (tile_x + tile_y) % 2 == 0:
                data[y, x] = [255, 255, 255, 255]  # White
            else:
                data[y, x] = [255, 0, 255, 255]  # Magenta
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


def render_tinted_bitmap(
    dev: GpuDevice,
    texture: GpuImage,
    tint_color: tuple[float, float, float, float],
    output_width: int,
    output_height: int,
    shader_dir: Path,
) -> torch.Tensor:
    """Render a texture with a tint color and return the result."""
    # Create render target image (RGBA8)
    render_target = dev.create_image(
        usages=["color-attachment", "texture-binding"],
        meta=GpuImageMeta(shape=(output_height, output_width, 4), dtype=torch.uint8),
    )

    # Create sampler
    sampler = dev.create_sampler(
        mag_filter="linear",
        min_filter="linear",
        address_mode="clamp-to-edge",
    )

    # Create uniform buffer for tint color (4 floats for RGBA)
    tint_buffer = dev.create_buffer(
        usages=["uniform", "copy-dst"],
        meta=GpuBufferMeta(element_count=4, element_dtype=torch.float32),
    )

    # Upload tint color to uniform buffer via staging
    tint_staging = dev.create_buffer(
        usages=["staging", "copy-src"],
        meta=tint_buffer.meta,
    )
    tint_data = torch.tensor(tint_color, dtype=torch.float32)
    tint_staging.write(data=tint_data)

    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.copy_buffer_to_buffer(
        src=tint_staging, dst=tint_buffer, size=tint_buffer.meta.size
    )
    cmd.submit().wait()

    # Create descriptor set layout
    descriptor_set_layout = dev.create_descriptor_set_layout(
        bindings=[
            GpuDescriptorSetLayoutBinding(type="uniform-buffer", stages=["fragment"]),
            GpuDescriptorSetLayoutBinding(type="texture", stages=["fragment"]),
        ]
    )

    # Create descriptor set
    descriptor_set = dev.create_descriptor_set(
        layout=descriptor_set_layout,
        bindings=[tint_buffer, (texture, sampler)],
    )

    # Create pipeline layout
    pipeline_layout = dev.create_pipeline_layout(
        descriptor_set_layouts=[descriptor_set_layout],
    )

    # Load shaders
    vertex_shader = dev.create_shader(
        spirv_path=shader_dir / "tinted_bitmap.vert.spv",
        stage="vertex",
    )
    fragment_shader = dev.create_shader(
        spirv_path=shader_dir / "tinted_bitmap.frag.spv",
        stage="fragment",
    )

    # Create graphics pipeline
    pipeline = dev.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vk_color_format=render_target.vk_format,
        viewport_width=output_width,
        viewport_height=output_height,
        layout=pipeline_layout,
    )

    # Render the tinted quad
    cmd = dev.create_command_encoder(queue_type="graphics")
    cmd.transition_image_layout(image=render_target, layout="color-attachment-optimal")
    with cmd.render(
        color_attachment=render_target,
        clear_on_load=True,
    ) as render_pass:
        render_pass.bind_pipeline(pipeline=pipeline)
        render_pass.bind_descriptor_sets(
            layout=pipeline_layout,
            first_set=0,
            sets=[descriptor_set],
        )
        render_pass.draw(
            vertex_count=6,
            instance_count=1,
        )
    cmd.submit().wait()

    # Read back the rendered image
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-dst"],
        meta=render_target.meta.into_buffer_meta(),
    )

    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=render_target, layout="copy-src")
    cmd.copy_image_to_buffer(src=render_target, dst=staging_buffer)
    cmd.submit().wait()

    # Read data from staging buffer
    image_data = staging_buffer.read()
    image_data = image_data.reshape((output_height, output_width, 4))

    return image_data


def test_render_tinted_bitmap():
    """Render tinted bitmaps with various textures and tints, save as PNGs."""
    ctx, dev = make_context()

    # Test image dimensions
    width, height = 256, 256

    # Get shader directory relative to this test file
    test_dir = Path(__file__).parent
    shader_dir = test_dir / "data" / "shaders" / "tests"

    # Define textures
    textures = {
        "gradient": generate_gradient_texture(width, height),
        "checkerboard": generate_checkerboard_texture(width, height, tile_size=32),
    }

    # Define tint colors (RGBA)
    tints = {
        "white": (1.0, 1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0, 1.0),
    }

    # print("Testing tinted bitmap rendering with descriptor sets...")
    # print(f"  Textures: {list(textures.keys())}")
    # print(f"  Tints: {list(tints.keys())}")

    # Upload textures to GPU
    gpu_textures = {}
    for tex_name, tex_data in textures.items():
        gpu_textures[tex_name] = upload_texture(dev, tex_data)
        # print(f"  ✓ Uploaded texture: {tex_name}")

    # Render all combinations
    output_count = 0
    for tex_name, gpu_tex in gpu_textures.items():
        for tint_name, tint_color in tints.items():
            # Render
            result = render_tinted_bitmap(
                dev=dev,
                texture=gpu_tex,
                tint_color=tint_color,
                output_width=width,
                output_height=height,
                shader_dir=shader_dir,
            )

            # Convert to numpy for saving
            result_np = result.numpy()

            # Save as PNG
            output_path = test_dir / f"output_tinted_{tex_name}_{tint_name}.png"
            img = Image.fromarray(result_np, mode="RGBA")
            img.save(output_path)
            output_count += 1

            # Validate result
            # White tint should preserve original colors
            if tint_name == "white":
                # Check that the image has color variance (not all black/same)
                color_sum = result_np[:, :, :3].astype(np.int64).sum(axis=2)
                non_black_pixels = np.sum(color_sum > 0)
                assert non_black_pixels > 0, (
                    f"White tint on {tex_name} should have non-black pixels"
                )

            # Black tint should make everything black (except alpha)
            elif tint_name == "black":
                rgb_max = result_np[:, :, :3].max()
                assert rgb_max == 0, (
                    f"Black tint on {tex_name} should be all black, got max RGB={rgb_max}"
                )

            # print(f"  ✓ Rendered: {tex_name} + {tint_name} -> {output_path.name}")

    # print(f"\n✓ Successfully rendered {output_count} tinted bitmap combinations!")
    # print("✓ Descriptor sets with texture and buffer bindings work correctly!")

    # Cleanup
    ctx.dispose()


if __name__ == "__main__":
    test_render_tinted_bitmap()
