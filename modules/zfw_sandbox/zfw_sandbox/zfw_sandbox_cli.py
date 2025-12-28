import argparse

import zfw


from .universal_paperclips import (
    UniversalPaperclipsMainMenuWidget,
)


class MainMenuWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
        )

        self._title_widget = zfw.GuiWidget(
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

        @self._universal_paperclips_button.click_event.subscribe()
        def universal_paperclips_button_click(button: zfw.MouseButton):
            if self._universal_paperclips_widget is None:
                self._universal_paperclips_widget = UniversalPaperclipsMainMenuWidget(
                    engine=engine,
                )
            engine.window.push_central_widget(self._universal_paperclips_widget)

    def _on_dispose(self) -> None:
        super()._on_dispose()
        self._title_widget.dispose()
        self._universal_paperclips_button.dispose()
        if self._universal_paperclips_widget is not None:
            self._universal_paperclips_widget.dispose()


def main():
    ap = argparse.ArgumentParser(description="ZFW Sandbox CLI")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (Vulkan validation layers, verbose debug logging, etc)",
    )
    args = ap.parse_args()

    engine = zfw.Engine(
        app_name="ZFW Sandbox",
        debug=args.debug,
        swapchain_image_count=3,
        enable_gui=True,
    )

    start_widget = MainMenuWidget(engine=engine)
    # start_widget = UniversalPaperclipsWidget(engine=engine)

    engine.window.set_central_widget(start_widget)

    while not engine.window.should_close():
        engine.update()
        engine.render()

    start_widget.dispose()
    engine.dispose()


if __name__ == "__main__":
    main()
