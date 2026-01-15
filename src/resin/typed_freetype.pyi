class Face:
    glyph: "GlyphSlot"
    size: "SizeMetrics"
    def __init__(self, filepath: str) -> None: ...
    def set_pixel_sizes(self, width: int, height: int) -> None: ...
    def load_glyph(self, glyph_index: int, flags: int) -> None: ...
    def get_var_design_coords(self) -> tuple[float, ...]: ...
    def set_var_design_coords(
        self, coords: tuple[float, ...] | list[float]
    ) -> None: ...
    def get_variation_info(self) -> VariationSpaceInfo: ...
    def get_char_index(self, charcode: int) -> int: ...

class GlyphSlot:
    bitmap: "Bitmap"
    bitmap_left: int
    bitmap_top: int
    metrics: "GlyphMetrics"
    advance: "Vector"
    def render(self, render_mode: int) -> None: ...

class Bitmap:
    rows: int
    width: int
    pitch: int
    buffer: bytes
    num_grays: int
    pixel_mode: int
    palette_mode: int
    palette: object | None

class VariationSpaceInfo:
    axes: tuple["VariationAxis"]

class VariationAxis:
    tag: str
    name: str
    minimum: float
    default: float
    maximum: float
    strid: str  # usually same as 'name'

class SizeMetrics:
    x_ppem: int
    y_ppem: int
    x_scale: int
    y_scale: int
    ascender: int
    descender: int
    height: int
    max_advance: int

class GlyphMetrics:
    width: int
    height: int
    horiBearingX: int
    horiBearingY: int
    horiAdvance: int
    vertBearingX: int
    vertBearingY: int
    vertAdvance: int

class Vector:
    x: int
    y: int

FT_LOAD_RENDER: int
FT_LOAD_TARGET_NORMAL: int
