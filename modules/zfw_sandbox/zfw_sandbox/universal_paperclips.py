import zfw

from .bundled_data import BUNDLED_DATA_PATH


class UniversalPaperclipsMainMenuWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
            grid_rows=(-4, -1, -1, -1, -1, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme={
                "h1": {
                    "bg_color": (0.0, 0.0, 0.0, 1.0),
                    "fg_color": (1.0, 1.0, 1.0, 1.0),
                    "font_size_dip": 64,
                }
            },
        )

        self._title_widget = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["h1"],
            text="Universal Paperclips",
        )
        self._new_game_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=2,
            col=1,
            style_classes=["button"],
            text="New Game",
        )
        self._load_game_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=3,
            col=1,
            style_classes=["button"],
            text="Load Game",
        )
        self._settings_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=4,
            col=1,
            style_classes=["button"],
            text="Settings",
        )
        self._quit_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=5,
            col=1,
            style_classes=["button"],
            text="Quit",
        )

        @self._quit_button_widget.click.subscribe()
        def quit_button_click(button: zfw.MouseButton):
            engine.window.pop_central_widget()


class UniversalPaperclipsWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme={
                "label": {
                    "bg_color": (0.0, 0.0, 0.0, 0.8),
                },
                "h1": {
                    "bg_color": (0.0, 0.0, 0.0, 0.8),
                },
            },
        )

        self._title = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=1,
            col_span=3,
            style_classes=["label", "h1"],
            text="Universal Paperclips",
        )
