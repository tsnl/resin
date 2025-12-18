import zfw


class UniversalPaperclipsWidget(zfw.GuiWidget):
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
            text="Universal Paperclips",
        )
