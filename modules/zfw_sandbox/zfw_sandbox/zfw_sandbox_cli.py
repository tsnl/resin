import argparse
import sys
from pathlib import Path

import PIL.Image
import numpy as np

import zfw

from .bundled_data import BUNDLED_DATA_PATH


KENNEY_SOKOBAN_DATA_PATH = (
    BUNDLED_DATA_PATH
    / "data/KenneyGameAssetsAllInOne-3_3_0/2D assets"
    / "Sokoban Pack"
    / "PNG/Default size"
)


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
        data=load_image(KENNEY_SOKOBAN_DATA_PATH / "Blocks/block_01.png"),
    )

    while not engine.window.should_close():
        engine.update()

        engine.render(
            quads=[
                zfw.RendererQuad(
                    dst_xy=(32, 64),
                    dst_wh=(512, 256),
                    color=(1.0, 1.0, 1.0, 1.0),
                    border_thickness_px=(0, 0, 8, 0),
                    border_color=(0.0, 0.1, 0.8, 1.0),
                ),
                zfw.RendererQuad(
                    dst_xy=(40, 72),
                    dst_wh=(64, 64),
                    color=(0.0, 0.2, 0.0, 1.0),
                ),
                zfw.RendererQuad(
                    dst_xy=(112, 72),
                    dst_wh=(64, 64),
                    color=(0.0, 0.2, 0.0, 0.5),
                ),
                zfw.RendererQuad(
                    dst_xy=(184, 72),
                    dst_wh=(64, 64),
                    color=(0.0, 0.0, 0.0, 1.0),
                    image=block_01_image,
                ),
            ]
        )


def print_gpu_debug_info(
    gpu_context: zfw.GpuContext,
    file: zfw.SupportsWrite[str] = sys.stdout,
):
    print("<gpu-debug-info>")
    gpu_context.print_debug_info(out=file)
    print()
    print("</gpu-debug-info>")


def load_image(file_path: Path) -> np.ndarray:
    return np.array(PIL.Image.open(file_path).convert("RGBA"))


if __name__ == "__main__":
    main()
