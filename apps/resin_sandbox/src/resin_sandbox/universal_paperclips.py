import resin


LOG = resin.logger(__name__)


#
# GUI
#


class UniversalPaperclipsMainMenuWidget(resin.GuiWidget):
    def __init__(self, gui_window: resin.GuiWindow):
        super().__init__(
            gui_window=gui_window,
            grid_rows=(-4, -1, -1, -1, -1, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme=_THEME,
        )
        self._gui_window = gui_window

        self._title_widget = resin.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["h1", "title"],
            text="Universal Paperclips",
        )
        self._new_game_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=2,
            col=1,
            style_classes=["button"],
            text="New Game",
        )
        self._load_game_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=3,
            col=1,
            style_classes=["button"],
            text="Load Game",
        )
        self._settings_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=4,
            col=1,
            style_classes=["button"],
            text="Settings",
        )
        self._quit_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=5,
            col=1,
            style_classes=["button"],
            text="Quit",
        )

        @self._new_game_button_widget.click_event.subscribe()
        def new_game_button_click(button: resin.MouseButton):
            self._gui_window.push_central_widget(
                UniversalPaperclipsWidget(gui_window=self._gui_window),
            )

        @self._load_game_button_widget.click_event.subscribe()
        def load_game_button_click(button: resin.MouseButton):
            raise NotImplementedError()

        @self._settings_button_widget.click_event.subscribe()
        def settings_button_click(button: resin.MouseButton):
            raise NotImplementedError()

        @self._quit_button_widget.click_event.subscribe()
        def quit_button_click(button: resin.MouseButton):
            gui_window.pop_central_widget()


class UniversalPaperclipsWidget(resin.GuiWidget):
    def __init__(self, gui_window: resin.GuiWindow):
        super().__init__(
            gui_window=gui_window,
            grid_rows=(200, 40, 40, -1, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            theme=_THEME,
        )

        self._console_widget = resin.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["label", "console"],
            text="(Game console would go here)\nLine 2\nLine 3\n...",
        )
        self._paperclip_count_widget = resin.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            col_span=3,
            style_classes=["label", "h1", "paperclip-counter"],
            text="Paperclips: 0",
        )
        self._make_paperclip_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=2,
            col=0,
            style_classes=["button"],
            text="Make Paperclip",
        )

        self._business_module_widget = UniversalPaperclipsBusinessModuleWidget(
            parent=self,
            row=3,
            col=0,
        )

        self._manufacturing_module_widget = (
            UniversalPaperclipsManufacturingModuleWidget(
                parent=self,
                row=4,
                col=0,
            )
        )

        # TODO: computational resources module at (2, 1)

        # TODO: projects module at (4, 1)

        @self._make_paperclip_button_widget.click_event.subscribe()
        def make_paperclip_button_click(button: resin.MouseButton):
            LOG.info("Make Paperclip button clicked")


class UniversalPaperclipsBusinessModuleWidget(resin.GuiWidget):
    def __init__(
        self,
        parent: resin.GuiWidget,
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
                15,  # (spacer)
                30,  # marketing button + level
                30,  # marketing upgrade cost
                -1,  # (flexible spacer)
            ),
            grid_cols=(50, 50, -1),
            row=row,
            col=col,
            style_classes=["module"],
        )

        self._title_widget = resin.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            col_span=3,
            style_classes=["label", "h2"],
            text="Business",
        )
        self._available_funds_widget = resin.GuiWidget(
            parent_widget=self,
            row=1,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Available Funds: $ 0.00",
        )
        self._unsold_inventory_widget = resin.GuiWidget(
            parent_widget=self,
            row=2,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Unsold Inventory: 0 Paperclips",
        )
        self._price_per_paperclip_lower_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=3,
            col=0,
            style_classes=["button", "button-pair-left"],
            text="lower",
        )
        self._price_per_paperclip_higher_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=3,
            col=1,
            style_classes=["button", "button-pair-right"],
            text="raise",
        )
        self._price_per_paperclip_label_widget = resin.GuiWidget(
            parent_widget=self,
            row=3,
            col=2,
            style_classes=["label"],
            text="Price per Paperclip: $0.00",
        )
        self._public_demand_widget = resin.GuiWidget(
            parent_widget=self,
            row=4,
            col=0,
            col_span=3,
            style_classes=["label"],
            text="Public Demand: 0%",
        )
        self._marketing_button_widget = resin.GuiWidget(
            parent_widget=self,
            row=6,
            col=0,
            col_span=2,
            style_classes=["button"],
            text="Marketing",
            clickable=False,
        )
        self._marketing_level_widget = resin.GuiWidget(
            parent_widget=self,
            row=6,
            col=2,
            style_classes=["label"],
            text="Level: 1",
        )
        self._marketing_cost_label_widget = resin.GuiWidget(
            parent_widget=self,
            row=7,
            col=0,
            col_span=2,
            style_classes=["label"],
            text="Cost:        $ ",
        )
        self._marketing_cost_display_widget = resin.GuiWidget(
            parent_widget=self,
            row=7,
            col=2,
            style_classes=["label"],
            text="0.00",
        )


class UniversalPaperclipsManufacturingModuleWidget(resin.GuiWidget):
    def __init__(
        self,
        parent: resin.GuiWidget,
        row: int,
        col: int,
    ):
        super().__init__(
            parent_widget=parent,
            grid_rows=(_MODULE_HEADER_SIZE_DIP, -1),
            grid_cols=(-1,),
            row=row,
            col=col,
            style_classes=["module"],
        )

        self._title_widget = resin.GuiWidget(
            parent_widget=self,
            row=0,
            col=0,
            style_classes=["label", "h2"],
            text="Manufacturing",
        )


_MODULE_HEADER_SIZE_DIP = 40
_MARGIN = 4

_THEME = {
    "central": {
        "default": {
            "bg_color": (1.0, 1.0, 1.0, 1.0),
        },
        "hover": {
            "bg_color": (1.0, 1.0, 1.0, 1.0),
        },
    },
    "title": {
        "default": {
            "bg_color": (0.0, 0.0, 0.0, 1.0),
            "fg_color": (1.0, 1.0, 1.0, 1.0),
        }
    },
    "module": {
        "default": {
            "margin": (10, 10, 10, 10),
            "padding": (10, 10, 10, 10),
        },
    },
    "label": {
        "default": {
            "bg_color": (1.0, 1.0, 1.0, 0.0),
            "fg_color": (0.0, 0.0, 0.0, 1.0),
            "font": "serif",
            "font_size": "regular",
            "padding": (2, 5, 2, 5),
            "text_horizontal_alignment": "left",
            "text_vertical_alignment": "middle",
        },
    },
    "button": {
        "default": {
            "padding": (0, 0, 0, 0),
            "margin": (_MARGIN,) * 4,
            "text_horizontal_alignment": "center",
            "text_vertical_alignment": "middle",
        },
    },
    "h1": {
        "default": {
            "bg_color": (1.0,) * 4,
            "fg_color": (0.0, 0.0, 0.0, 1.0),
            "font_size": "extra-large",
            "font": "serif",
            "text_horizontal_alignment": "center",
            "text_vertical_alignment": "middle",
        },
    },
    "h2": {
        "default": {
            "bg_color": (1.0,) * 4,
            "fg_color": (0.0, 0.0, 0.0, 1.0),
            "font_size": "large",
            "font": "serif",
            "border_color": (0.0, 0.0, 0.0, 1.0),
            "border_thickness": (0, 0, 2, 0),
            "padding": (5, 5, 5, 5),
        },
    },
    "paperclip-counter": {
        "default": {
            "text_horizontal_alignment": "left",
            "text_vertical_alignment": "top",
        },
    },
    "console": {
        "default": {
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
    },
    "button-pair-left": {
        "default": {
            "margin": (_MARGIN, _MARGIN // 2, _MARGIN, _MARGIN),
        },
    },
    "button-pair-right": {
        "default": {
            "margin": (_MARGIN, _MARGIN, _MARGIN, _MARGIN // 2),
        },
    },
}
