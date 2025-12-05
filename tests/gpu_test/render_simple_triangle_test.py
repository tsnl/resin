"""Test rendering a simple triangle to an offscreen image."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from zero.gpu import GpuContext, GpuDevice, GpuImageMeta


def make_context() -> tuple[GpuContext, GpuDevice]:
    """Create GPU context and device for testing."""
    ctx = GpuContext(
        app_name="gpu-triangle-test",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = ctx.create_device(physical_device=phys, surface=None)
    return ctx, dev


def test_render_simple_triangle():
    """Render a colored triangle to an offscreen image and save as PNG."""
    ctx, dev = make_context()

    # Test image dimensions
    width, height = 512, 512

    # Create render target image (RGBA8)
    render_target = dev.create_image(
        usages=["color-attachment", "texture-binding"],
        meta=GpuImageMeta(shape=(height, width, 4), dtype=torch.uint8),
    )

    # Get shader directory relative to this test file
    test_dir = Path(__file__).parent
    shader_dir = test_dir / "data" / "shaders" / "tests"

    # Load shaders
    vertex_shader = dev.create_shader(
        spirv_path=shader_dir / "triangle.vert.spv",
        stage="vertex",
    )
    fragment_shader = dev.create_shader(
        spirv_path=shader_dir / "triangle.frag.spv",
        stage="fragment",
    )

    # Create pipeline layout (empty for simple triangle - no descriptors)
    pipeline_layout = dev.create_pipeline_layout(descriptor_set_layouts=[])

    # Create graphics pipeline
    pipeline = dev.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vk_color_format=render_target.vk_format,
        viewport_width=width,
        viewport_height=height,
        layout=pipeline_layout,
    )

    # Render the triangle
    cmd = dev.create_command_encoder(queue_type="graphics")
    cmd.transition_image_layout(image=render_target, layout="color-attachment-optimal")
    with cmd.render(
        color_attachment=render_target,
        clear_on_load=True,
    ) as render_pass:
        render_pass.bind_pipeline(pipeline=pipeline)
        render_pass.draw(vertex_count=3, instance_count=1)
    cmd.submit().wait()

    # Read back the rendered image
    # Create staging buffer for readback
    staging_buffer = dev.create_buffer(
        usages=["staging", "copy-dst"],
        meta=render_target.meta.into_buffer_meta(),
    )

    # Copy image to staging buffer
    cmd = dev.create_command_encoder(queue_type="transfer")
    cmd.transition_image_layout(image=render_target, layout="copy-src")
    cmd.copy_image_to_buffer(src=render_target, dst=staging_buffer)
    cmd.submit().wait()

    # Read data from staging buffer
    image_data = staging_buffer.read()

    # Reshape to image dimensions (height, width, 4)
    image_data = image_data.reshape((height, width, 4))

    # Convert to numpy array
    image_array = image_data.numpy()

    # Verify we have non-zero pixels (triangle was rendered)
    non_black_pixels = np.sum(image_array[:, :, :3].sum(axis=2) > 0)
    assert non_black_pixels > 0, "No pixels were drawn - triangle not rendered"

    # Save as PNG for visual inspection
    output_path = test_dir / "output_triangle.png"
    img = Image.fromarray(image_array, mode="RGBA")
    img.save(output_path)

    # Basic validation: triangle should cover a reasonable number of pixels
    # A triangle covering roughly half the diagonal should have ~10-15k pixels
    min_expected_pixels = 5000
    assert non_black_pixels >= min_expected_pixels, (
        f"Triangle too small: {non_black_pixels} pixels "
        f"(expected at least {min_expected_pixels})"
    )

    # Cleanup
    ctx.dispose()
