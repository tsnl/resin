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
        enable_gui=True,
    )

    block_01_image = zfw.RendererImage(
        renderer=engine.renderer,
        data=zfw.load_rgba_image(KENNEY_SOKOBAN_DATA_PATH / "Blocks/block_01.png"),
    )

    gui_label = zfw.GuiLabel(
        parent_widget=engine.window,
        xywh_dip=(50, 50, 200, 54),
        text="Hello, GUI!",
        font_size_dip=18,
        padding=(12, 0, 12, 5),
        bg_color=(0.2, 0.2, 0.2, 1.0),
        bg_hover_color=(0.4, 0.4, 0.4, 1.0),
        fg_color=(1.0, 1.0, 1.0, 1.0),
        hover_border_color=(1.0, 1.0, 1.0, 1.0),
        hover_border_thickness=(2, 2, 2, 2),
    )

    _ = block_01_image
    _ = gui_label

    while not engine.window.should_close():
        engine.update()
        engine.render()


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
