class Face:
    def __init__(self, blob: bytes) -> None: ...

class Font:
    scale: tuple[int, int]
    def __init__(self, face: Face) -> None: ...
    def set_variations(self, variations: dict[str, float]) -> None: ...

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
