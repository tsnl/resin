from typing import Protocol, TypeVar

TContra = TypeVar("TContra", contravariant=True)


class SupportsWrite(Protocol[TContra]):
    def write(self, s: TContra, /) -> object: ...


#
# ID manipulation:
#


def pascal_to_snake_case(s: str) -> str:
    return "_".join(parse_pascal_case_id(s)).lower()


def parse_pascal_case_id(s: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and (s[i - 1].islower()):
            result.append("".join(current))
            current = []
        current.append(c)
    if current:
        result.append("".join(current))
    return result
