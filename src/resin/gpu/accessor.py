from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Accessor:
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]

    def __post_init__(self):
        assert self.offset >= 0
        assert len(self.shape) == len(self.pitch), "Inconsistent rank"
