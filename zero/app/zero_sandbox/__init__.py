import argparse
import sys
from pathlib import Path

import torch

import zero


def main():
    ap = argparse.ArgumentParser(description="Zero Sandbox Application")
    ap.add_argument(
        "--swapchain-image-count",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of swapchain images: 2 for double buffering, 3 for triple buffering",
    )
    ap.add_argument(
        "--enable-vulkan-debug-layers",
        action="store_true",
        help="Enable Vulkan debug layers for debugging purposes",
    )
    args = ap.parse_args()

    gpu_context = zero.GpuContext(
        app_name="zero-sandbox",
        enable_debug_layer_support=args.enable_vulkan_debug_layers,
        enable_present_support=True,
    )
    window_context = zero.WindowContext(
        gpu_context=gpu_context,
    )

    print("<gpu-context-debug-info>")
    gpu_context.print_debug_info(out=sys.stdout)
    print()
    print("</gpu-context-debug-info>")

    window = window_context.create_window(width=1024, height=1024, title="Zero Sandbox")
    surface = window.create_surface()

    physical_device = next(iter(gpu_context.enumerate_physical_devices()))
    device = gpu_context.create_device(physical_device=physical_device, surface=surface)
    swapchain = device.create_swapchain(
        surface=surface,
        image_count=args.swapchain_image_count,
    )

    # Create render target image (1024x1024 RGBA8)
    render_target_image = device.create_image(
        usages=["color-attachment"],
        meta=zero.GpuImageMeta(shape=(1024, 1024, 4), dtype=torch.uint8),
    )

    print(device, file=sys.stderr)
    print(render_target_image, file=sys.stderr)

    # Load shaders from compiled SPIR-V
    shader_dir = Path(__file__).parent.parent.parent / "data" / "shader"

    vertex_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.vert.spv",
        stage="vertex",
    )

    fragment_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.frag.spv",
        stage="fragment",
    )

    print(f"Loaded shaders: {vertex_shader}, {fragment_shader}")

    # Create graphics pipeline
    pipeline = device.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vk_color_format=render_target_image.meta.infer_vk_format(
            render_target_image.usages
        ),
        viewport_width=1024,
        viewport_height=1024,
    )

    print(f"Created pipeline: {pipeline}")

    # Render triangle
    cmd = device.create_command_encoder(queue_type="graphics")
    with cmd.render(
        color_attachment=render_target_image,
        clear_on_load=True,
    ) as render_pass:
        render_pass.bind_pipeline(pipeline=pipeline)
        render_pass.draw(vertex_count=3, instance_count=1)
    cmd.submit().wait()

    print("Triangle rendered successfully!")


if __name__ == "__main__":
    main()
