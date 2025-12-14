class Face:
    def __init__(self, blob: bytes) -> None: ...

class Font:
    def __init__(self, face: Face) -> None: ...
    def set_scale(self, x_scale: int, y_scale: int) -> None: ...

class Buffer:
    glyph_infos: list["GlyphInfo"]
    glyph_positions: list["GlyphPosition"]

    def __init__(self) -> None: ...
    def add_str(self, text: str) -> None: ...
    def guess_segment_properties(self) -> None: ...

class GlyphInfo:
    codepoint: int
    cluster: int

class GlyphPosition:
    x_advance: int
    y_advance: int
    x_offset: int
    y_offset: int

def shape(font: Font, buffer: Buffer) -> None: ...
