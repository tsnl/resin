from typing import Protocol, TypeVar

TContra = TypeVar("TContra", contravariant=True)


class SupportsWrite(Protocol[TContra]):
    def write(self, s: TContra, /) -> object: ...
