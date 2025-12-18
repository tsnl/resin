"""
Universal Paperclips CLI - Phase 1 implementation.

TODO: background image: https://unsplash.com/photos/aerial-view-of-green-trees-and-road-during-daytime-ZeDw8ck4XEM
"""

import argparse
import sys

import zfw

from .bundled_data import BUNDLED_DATA_PATH

from .universal_paperclips import UniversalPaperclipsWidget


class MainMenuWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
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

        self._universal_paperclips_button = zfw.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            style_classes=["button"],
            text="Universal Paperclips",
        )

        self._universal_paperclips_widget = None

        @self._universal_paperclips_button.click.subscribe()
        def universal_paperclips_button_click(button: zfw.MouseButton):
            if self._universal_paperclips_widget is None:
                self._universal_paperclips_widget = UniversalPaperclipsWidget(
                    engine=engine,
                )
            engine.window.set_central_widget(self._universal_paperclips_widget)


def main():
    engine = zfw.Engine(
        app_name="ZFW Sandbox",
        debug=True,
        swapchain_image_count=2,
        enable_gui=True,
    )

    main_menu_widget = MainMenuWidget(engine=engine)
    engine.window.set_central_widget(main_menu_widget)

    while not engine.window.should_close():
        engine.update()
        engine.render()


if __name__ == "__main__":
    main()
