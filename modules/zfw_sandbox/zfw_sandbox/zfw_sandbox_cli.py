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
    ap.add_argument("--debug", action="store_true", help="Enable debug settings")
    args = ap.parse_args()

    engine = ZfwEngine(
        app_name="Zero Sandbox",
        debug=args.debug,
        swapchain_image_count=args.swapchain_image_count,
    )

    while not engine.window.should_close():
        engine.update()

        with engine.render() as canvas:
            canvas.draw(dst_xy=(32, 64), dst_wh=(256, 128), color=(1.0, 1.0, 1.0, 1.0))
            canvas.draw(dst_xy=(40, 72), dst_wh=(128, 64), color=(0.0, 0.5, 0.0, 1.0))


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
