import zfw

from .bundled_data import BUNDLED_DATA_PATH


class UniversalPaperclipsWidget(zfw.GuiWidget):
    def __init__(self, engine: zfw.Engine):
        self._bg_image = zfw.RendererImage(
            renderer=engine.renderer,
            data=zfw.load_rgba_image(
                BUNDLED_DATA_PATH / "data/UniversalPaperclipsBackgrounds/hills.jpg"
            ),
        )

        super().__init__(
            window=engine.window,
            grid_rows=(100, -1, -1),
            grid_cols=(-1, -1, -1),
            style_classes=["central"],
            image=self._bg_image,
            image_layout="fit",
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
            col=0,
            col_span=3,
            style_classes=["label", "h1"],
            text="Universal Paperclips",
        )
