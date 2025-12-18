"""
Universal Paperclips CLI - Phase 1 implementation.

TODO: background image: https://unsplash.com/photos/aerial-view-of-green-trees-and-road-during-daytime-ZeDw8ck4XEM
"""

import argparse
import sys

import zfw

from .bundled_data import BUNDLED_DATA_PATH


class MainMenuWidget(zfw.GuiWidget):
    def __init__(self, window: zfw.GuiWindow):
        super().__init__(
            window=window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
        )

        self._title = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["h1"],
            text="Main Menu",
        )


def main():
    engine = zfw.Engine(
        app_name="ZFW Sandbox",
        debug=True,
        swapchain_image_count=2,
        enable_gui=True,
    )

    main_menu_widget = MainMenuWidget(window=engine.window)
    engine.window.set_central_widget(main_menu_widget)

    while not engine.window.should_close():
        engine.update()
        engine.render()


if __name__ == "__main__":
    main()
