import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import zero

if TYPE_CHECKING:
    from _typeshed import SupportsWrite


def main():
    ap = argparse.ArgumentParser(description="Zero Sandbox Application")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run the application in headless mode without creating a window",
    )
    ap.add_argument(
        "--swapchain-image-count",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of swapchain images: 2 for double buffering, 3 for triple buffering",
    )
    ap.add_argument(
        "--disable-vulkan-debug-layers",
        dest="enable_vulkan_debug_layers",
        action="store_false",
        help="Disable Vulkan debug layers (enabled by default)",
    )
    args = ap.parse_args()

    if args.headless:
        main_headless(args)
    else:
        main_windowed(args)


def main_headless(args: argparse.Namespace):
    gpu_context = zero.GpuContext(
        app_name="zero-sandbox",
        enable_debug_layer_support=args.enable_vulkan_debug_layers,
        enable_present_support=True,
    )

    print_gpu_debug_info(gpu_context, file=sys.stdout)

    physical_device = next(iter(gpu_context.enumerate_physical_devices()))
    device = gpu_context.create_device(physical_device=physical_device, surface=None)

    # Create render target image (1024x1024 RGBA8)
    render_target_image = device.create_image(
        usages=["color-attachment"],
        meta=zero.GpuImageMeta(shape=(1024, 1024, 4), dtype=torch.uint8),
    )

    # Load shaders from compiled SPIR-V
    shader_dir = zero.BUNDLED_DATA_PATH / "shaders" / "tests"

    vertex_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.vert.spv",
        stage="vertex",
    )

    fragment_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.frag.spv",
        stage="fragment",
    )

    print(f"Loaded shaders: {vertex_shader}, {fragment_shader}")

    # Create pipeline layout (empty for simple triangle - no descriptors)
    pipeline_layout = device.create_pipeline_layout()

    # Create graphics pipeline
    pipeline = device.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vk_color_format=render_target_image.meta.infer_vk_format(
            render_target_image.usages
        ),
        viewport_width=1024,
        viewport_height=1024,
        layout=pipeline_layout,
    )

    print(f"Created pipeline: {pipeline}")

    # Render triangle
    cmd = device.create_command_encoder(queue_type="graphics")
    cmd.transition_image_layout(
        image=render_target_image,
        layout="color-attachment-optimal",
    )
    with cmd.render(
        color_attachment=render_target_image,
        clear_on_load=True,
    ) as render_pass:
        render_pass.bind_pipeline(pipeline=pipeline)
        render_pass.draw(vertex_count=3, instance_count=1)
    cmd.submit().wait()

    print("Triangle rendered successfully!")


def main_windowed(args: argparse.Namespace):
    gpu_context = zero.GpuContext(
        app_name="zero-sandbox",
        enable_debug_layer_support=args.enable_vulkan_debug_layers,
        enable_present_support=True,
    )
    window_context = zero.WindowContext(
        gpu_context=gpu_context,
    )

    print_gpu_debug_info(gpu_context, file=sys.stdout)

    # Create window, GPU surface:
    window = window_context.create_window(width=1024, height=1024, title="Zero Sandbox")
    surface = window.create_surface()

    # Create GPU device using the surface:
    physical_device = next(iter(gpu_context.enumerate_physical_devices()))
    device = gpu_context.create_device(physical_device=physical_device, surface=surface)

    # Create swapchain:
    swapchain = device.create_swapchain(
        surface=surface,
        image_count=args.swapchain_image_count,
    )

    # Create render pipeline using the swapchain:
    shader_dir = zero.BUNDLED_DATA_PATH / "shaders" / "tests"
    vertex_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.vert.spv",
        stage="vertex",
    )
    fragment_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.frag.spv",
        stage="fragment",
    )
    # Create pipeline layout (empty for simple triangle - no descriptors)
    pipeline_layout = device.create_pipeline_layout(descriptor_set_layouts=[])

    pipeline = device.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        vk_color_format=swapchain.vk_format,
        viewport_width=swapchain.width,
        viewport_height=swapchain.height,
        layout=pipeline_layout,
    )

    # Main window loop:
    window.show()
    while not window.should_close():
        window.poll_events()

        with swapchain.present() as present_target:
            cmd = device.create_command_encoder(
                queue_type="graphics",
                fence=present_target.render_done_fence,
                wait_semaphores=present_target.render_wait_semaphores,
                signal_semaphores=present_target.render_done_semaphores,
            )

            cmd.transition_image_layout(
                image=present_target.swapchain_image,
                layout="color-attachment-optimal",
            )

            with cmd.render(
                color_attachment=present_target.swapchain_image,
                clear_on_load=True,
            ) as render_pass:
                render_pass.bind_pipeline(pipeline=pipeline)
                render_pass.draw(vertex_count=3, instance_count=1)

            cmd.transition_image_layout(
                image=present_target.swapchain_image,
                layout="present-src",
            )

            cmd.submit()


def print_gpu_debug_info(
    gpu_context: zero.GpuContext,
    file: SupportsWrite[str] = sys.stdout,
):
    print("<gpu-debug-info>")
    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


if __name__ == "__main__":
    main()
