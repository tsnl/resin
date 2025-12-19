import zfw

from .bundled_data import BUNDLED_DATA_PATH


class UniversalPaperclipsMainMenuWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
            grid_rows=(-4, -1, -1, -1, -1, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme=_THEME,
        )
        self._engine = engine

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

        @self._new_game_button_widget.click.subscribe()
        def new_game_button_click(button: zfw.MouseButton):
            self._engine.window.push_central_widget(
                UniversalPaperclipsWidget(engine=self._engine),
            )

        @self._load_game_button_widget.click.subscribe()
        def load_game_button_click(button: zfw.MouseButton):
            raise NotImplementedError()

        @self._settings_button_widget.click.subscribe()
        def settings_button_click(button: zfw.MouseButton):
            raise NotImplementedError()

        @self._quit_button_widget.click.subscribe()
        def quit_button_click(button: zfw.MouseButton):
            engine.window.pop_central_widget()


class UniversalPaperclipsWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        super().__init__(
            window=engine.window,
            grid_rows=(200, 80, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme=_THEME,
        )

        self._console_widget = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["label", "console"],
            text="(Game console would go here)\nLine 2\nLine 3\n...",
        )
        self._paperclip_count_widget = zfw.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            col_span=3,
            style_classes=["label", "h1", "paperclip-counter"],
            text="Paperclips: 0",
        )

        self._business_module_widget = UniversalPaperclipsBusinessWidget(
            engine=engine,
            parent=self,
            row=2,
            col=0,
        )


class UniversalPaperclipsBusinessWidget(zfw.GuiWidget):
    def __init__(
        self,
        engine: zfw.Engine,
        parent: zfw.GuiWidget,
        row: int,
        col: int,
    ):
        super().__init__(
            parent_widget=parent,
            grid_rows=(40, -1),
            grid_cols=(-1,),
            row=row,
            col=col,
            style_classes=["module"],
        )
        self._engine = engine

        self._title_widget = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            style_classes=["label", "h2"],
            text="Business",
        )


_THEME = {
    "module": {
        "margin": (10, 10, 10, 10),
        "padding": (10, 10, 10, 10),
    },
    "label": {
        "bg_color": (0.1, 0.1, 0.1, 1.0),
        "fg_color": (0.9, 0.9, 0.9, 1.0),
        "font": "sans-serif",
        "padding": (10, 10, 10, 10),
        "text_horizontal_alignment": "left",
        "text_vertical_alignment": "top",
    },
    "h1": {
        "bg_color": (1.0, 1.0, 1.0, 1.0),
        "fg_color": (0.0, 0.0, 0.0, 1.0),
        "font_size_dip": 28,
        "font": "sans-serif",
        "text_horizontal_alignment": "center",
        "text_vertical_alignment": "middle",
    },
    "h2": {
        "bg_color": (1.0, 1.0, 1.0, 1.0),
        "fg_color": (0.0, 0.0, 0.0, 1.0),
        "font_size_dip": 22,
        "font": "sans-serif",
        "border_color": (0.0, 0.0, 0.0, 1.0),
        "border_thickness": (0, 0, 2, 0),
        "padding": (5, 5, 5, 5),
    },
    "paperclip-counter": {
        "text_horizontal_alignment": "left",
        "text_vertical_alignment": "top",
    },
    "console": {
        "font": "monospaced",
        "text_horizontal_alignment": "left",
        "text_vertical_alignment": "top",
        "wrap": True,
    },
}
