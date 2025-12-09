import argparse
import sys

import zfw_core

from .zfw_engine import ZfwEngine


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
        "--disable-vulkan-debug-layers",
        dest="enable_vulkan_debug_layers",
        action="store_false",
        help="Disable Vulkan debug layers (enabled by default)",
    )
    args = ap.parse_args()

    engine = ZfwEngine(
        app_name="Zero Sandbox",
        debug=args.enable_vulkan_debug_layers,
        swapchain_image_count=args.swapchain_image_count,
    )

    while not engine.window.should_close():
        engine.window.poll_events()
        # TODO: render


def print_gpu_debug_info(
    gpu_context: zfw_core.GpuContext,
    file: zfw_core.SupportsWrite[str] = sys.stdout,
):
    print("<gpu-debug-info>")
    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


if __name__ == "__main__":
    main()
