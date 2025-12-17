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

    # Configure window grid
    engine.window.set_grid_config(
        num_rows=3,
        num_cols=2,
        row_sizes=(60, -1, 60),  # Header, Content, Footer
        col_sizes=(200, -1),  # Sidebar, Main
    )

    # Header
    zfw.GuiWidget(
        parent_node=engine.window,
        row=0,
        col=0,
        col_span=2,
        text="Header (Fixed 60px)",
        archetype="label",
    )

    # Sidebar
    zfw.GuiWidget(
        parent_node=engine.window,
        row=1,
        col=0,
        text="Sidebar (Fixed 200px)",
        archetype="button",
    )

    # Main Content
    zfw.GuiWidget(
        parent_node=engine.window,
        row=1,
        col=1,
        text="Main Content (Stretch)",
        archetype="label",
    )

    # Footer
    zfw.GuiWidget(
        parent_node=engine.window,
        row=2,
        col=0,
        col_span=2,
        text="Footer (Fixed 60px)",
        archetype="label",
    )

    _ = block_01_image

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
