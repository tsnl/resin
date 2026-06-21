from typing import Literal

type Scalar = float | int
_SCALAR_TYPES: tuple[type, ...] = (float, int)


def is_scalar(value: object) -> bool:
    return isinstance(value, _SCALAR_TYPES)


type ElementOperator = (
    UnaryElementOperator | BinaryElementOperator | BinaryCompareOperator
)
type UnaryElementOperator = Literal[
    "neg",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "not",
]
type BinaryAssocElementOperator = Literal["mul", "add", "max", "min"]
type BinaryElementOperator = Literal["pow", "div", "sub"] | BinaryAssocElementOperator
type BinaryCompareOperator = Literal["eq", "ne", "gt", "lt", "ge", "le"]

# Tile etypes (e.g. "mat4x4_f4") extend this union as they are added.
type ElementType = Literal["f4", "f2", "u4"]
type EKind = Literal["float", "uint"]

# Named element-type spellings for View[F4, (2, 3)] annotations. These exist to
# placate Python type-checkers and ruff: if we write the string literals in type
# annotations directly (View["f4", (2, 3)]), ruff F821 falsely treats the etype
# spelling inside the subscript as an undefined name.
F4: ElementType = "f4"
F2: ElementType = "f2"
U4: ElementType = "u4"


def etype_join(etype1: ElementType | str, etype2: ElementType | str) -> ElementType:
    kind = etype_join_kind(etype_kind(etype1), etype_kind(etype2))
    nbytes = max(etype_nbytes(etype1), etype_nbytes(etype2))
    return etype(kind, nbytes)


def etype(kind: EKind, nbytes: int) -> ElementType:
    match (kind, nbytes):
        case ("float", 4):
            return "f4"
        case ("float", 2):
            return "f2"
        case ("uint", 4):
            return "u4"
        case _:
            raise ValueError(f"Unsupported etype with kind={kind} and nbytes={nbytes}")


def etype_nbytes(etype: ElementType | str) -> int:
    return {"f4": 4, "f2": 2, "u4": 4}[etype]


def etype_kind(etype: ElementType | str) -> EKind:
    match etype:
        case "f4" | "f2":
            return "float"
        case "u4":
            return "uint"
        case _:
            raise ValueError(f"Unsupported etype {etype}")


def etype_join_kind(etype1: EKind, etype2: EKind) -> EKind:
    if etype1 != etype2:
        raise ValueError(f"Cannot join different kinds' etypes: {etype1} and {etype2}")
    return etype1


def spell_etype_in_pystruct(etype: ElementType | str) -> str:
    return {"f4": "f", "f2": "e", "u4": "I"}[etype]


def spell_etype_in_wgsl(etype: ElementType | str) -> str:
    return {"f4": "f32", "f2": "f16", "u4": "u32"}[etype]


def etype_needs_enable_f16(etype: ElementType | str) -> bool:
    return etype == "f2"
