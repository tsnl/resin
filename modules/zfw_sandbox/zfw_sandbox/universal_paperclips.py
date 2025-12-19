import zfw

from .bundled_data import BUNDLED_DATA_PATH


#
# GUI
#


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
            grid_rows=(200, 40, 40, -1, -1),
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
        self._make_paperclip_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=2,
            col=0,
            style_classes=["button"],
            text="Make Paperclip",
        )

        self._business_module_widget = UniversalPaperclipsBusinessModuleWidget(
            engine=engine,
            parent=self,
            row=3,
            col=0,
        )

        # TODO: manufacturing module at (4,0)

        # TODO: computational resources module at (2, 1)

        # TODO: projects module at (4, 1)


class UniversalPaperclipsBusinessModuleWidget(zfw.GuiWidget):
    def __init__(
        self,
        engine: zfw.Engine,
        parent: zfw.GuiWidget,
        row: int,
        col: int,
    ):
        super().__init__(
            parent_widget=parent,
            grid_rows=(
                _MODULE_HEADER_SIZE_DIP,
                30,  # available funds
                30,  # unsold inventory
                30,  # price per paperclip
                30,  # public demand
                30,  # (spacer)
                30,  # marketing button + level
                30,  # marketing upgrade cost
            ),
            grid_cols=(50, 50, -1),
            row=row,
            col=col,
            style_classes=["module"],
        )
        self._engine = engine

        self._title_widget = zfw.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["label", "h2"],
            text="Business",
        )
        self._available_funds_widget = zfw.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Available Funds: $0.00",
        )
        self._unsold_inventory_widget = zfw.GuiWidget(
            parent_widget=self,
            row=2,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Unsold Inventory: 0 Paperclips",
        )
        self._price_per_paperclip_lower_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=3,
            col=0,
            style_classes=["button", "button-pair-left"],
            text="lower",
        )
        self._price_per_paperclip_higher_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=3,
            col=1,
            style_classes=["button", "button-pair-right"],
            text="raise",
        )
        self._price_per_paperclip_label_widget = zfw.GuiWidget(
            parent_widget=self,
            row=3,
            col=2,
            style_classes=["label"],
            text="Price per Paperclip: $0.00",
        )
        self._public_demand_widget = zfw.GuiWidget(
            parent_widget=self,
            row=4,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Public Demand: 0%",
        )
        self._marketing_button_widget = zfw.GuiWidget(
            parent_widget=self,
            row=6,
            col=0,
            col_span=2,
            style_classes=["button"],
            text="Marketing",
        )
        self._marketing_level_widget = zfw.GuiWidget(
            parent_widget=self,
            row=6,
            col=2,
            style_classes=["label"],
            text="Level: 1",
        )


_MODULE_HEADER_SIZE_DIP = 40
_MARGIN = 4

_THEME = {
    "central": {
        "bg_color": (1.0, 1.0, 1.0, 1.0),
        "bg_hover_color": (1.0, 1.0, 1.0, 1.0),
    },
    "module": {
        "margin": (10, 10, 10, 10),
        "padding": (10, 10, 10, 10),
    },
    "label": {
        "bg_color": (1.0, 1.0, 1.0, 0.0),
        "fg_color": (0.0, 0.0, 0.0, 1.0),
        "font": "serif",
        "font_size_dip": 18,
        "padding": (2, 5, 2, 5),
        "text_horizontal_alignment": "left",
        "text_vertical_alignment": "middle",
    },
    "button": {
        "padding": (0, 0, 0, 0),
        "margin": (_MARGIN,) * 4,
        "text_horizontal_alignment": "center",
        "text_vertical_alignment": "middle",
    },
    "h1": {
        "bg_color": (1.0,) * 4,
        "fg_color": (0.0, 0.0, 0.0, 1.0),
        "font_size_dip": 28,
        "font": "serif",
        "text_horizontal_alignment": "center",
        "text_vertical_alignment": "middle",
    },
    "h2": {
        "bg_color": (1.0,) * 4,
        "fg_color": (0.0, 0.0, 0.0, 1.0),
        "font_size_dip": 22,
        "font": "serif",
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
        "bg_color": (0.0, 0.0, 0.0, 1.0),
        "fg_color": (0.9, 0.9, 0.9, 1.0),
        "margin": (_MARGIN,) * 4,
        "border_color": (0.5, 0.5, 0.5, 1.0),
        "border_thickness": (2, 2, 2, 2),
    },
    "button-pair-left": {
        "margin": (_MARGIN, _MARGIN // 2, _MARGIN, _MARGIN),
    },
    "button-pair-right": {
        "margin": (_MARGIN, _MARGIN, _MARGIN, _MARGIN // 2),
    },
}
