import argparse
import sys

import zfw

from .bundled_data import BUNDLED_DATA_PATH


KENNEY_SOKOBAN_DATA_PATH = (
    BUNDLED_DATA_PATH
    / "data/KenneyGameAssetsAllInOne-3_3_0/2D assets"
    / "Sokoban Pack"
    / "PNG/Default size"
)

FONT_SIZE_PX = 18


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

    engine = zfw.Engine(
        app_name="Zero Sandbox",
        debug=args.debug,
        swapchain_image_count=args.swapchain_image_count,
    )

    block_01_image = zfw.RendererImage(
        renderer=engine.renderer,
        data=zfw.load_rgba_image(KENNEY_SOKOBAN_DATA_PATH / "Blocks/block_01.png"),
    )

    canvas = zfw.RendererCanvas(renderer=engine.renderer)

    while not engine.window.should_close():
        engine.update()

        canvas.clear()
        canvas.add_quad(
            dst_xy=(32, 64),
            dst_wh=(512, 256),
            color=(1.0, 1.0, 1.0, 1.0),
            border_thickness_px=(0, 0, 8, 0),
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
        canvas.add_quad(
            dst_xy=(184, 72),
            dst_wh=(64, 64),
            image=block_01_image,
        )
        canvas.add_text(
            text="Hello, world: weight=100",
            font="sans-serif",
            dst_xy=(40, 144),
            dst_wh=(400, 100),
            font_size_px=FONT_SIZE_PX,
            font_weight=100,
            color=(0.0, 0.0, 0.0, 1.0),
            wrap=False,
        )
        canvas.add_text(
            text="Hello, world: weight=400",
            font="sans-serif",
            dst_xy=(40, 174),
            dst_wh=(400, 100),
            font_size_px=FONT_SIZE_PX,
            font_weight=400,
            color=(0.0, 0.0, 0.0, 1.0),
            wrap=False,
        )
        canvas.add_text(
            text="Hello, world: weight=800",
            font="sans-serif",
            dst_xy=(40, 204),
            dst_wh=(400, 100),
            font_size_px=FONT_SIZE_PX,
            font_weight=800,
            color=(0.0, 0.0, 0.0, 1.0),
            wrap=False,
        )
        canvas.add_text(
            text="Hello, world: weight=1200",
            font="sans-serif",
            dst_xy=(40, 234),
            dst_wh=(400, 100),
            font_size_px=FONT_SIZE_PX,
            font_weight=1200,
            color=(0.0, 0.0, 0.0, 1.0),
            wrap=False,
        )
        engine.render(canvas=canvas)


def print_gpu_debug_info(
    gpu_context: zfw.GpuContext,
    file: zfw.SupportsWrite[str] = sys.stdout,
):
    print("<gpu-debug-info>")
    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


if __name__ == "__main__":
    main()
