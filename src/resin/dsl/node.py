"""
Computation graph node types for the Resin DSL.

Nodes represent some computational work that writes to its own output buffer.

Users interact with views on these nodes.
"""

__all__ = [
    "ConstNode",
    "ElementwiseNode",
    "MatmulNode",
    "Node",
    "ParamNode",
    "ReductionNode",
    "RemapNode",
]

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from resin.dsl.view import View

from resin.core.etype import (
    BinaryAssocElementOperator,
    ElementOperator,
    ElementType,
    etype_nbytes,
)
from resin.core.pytree import PyTensor


@dataclass(kw_only=True, frozen=True, eq=False)
class Node(ABC):
    shape: tuple[int, ...]
    etype: ElementType | str
    args: tuple[View, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * etype_nbytes(self.etype)


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: PyTensor


@dataclass(kw_only=True, frozen=True, eq=False)
class ParamNode(Node):
    name: str


@dataclass(kw_only=True, frozen=True, eq=False)
class ElementwiseNode(Node):
    operator: ElementOperator


@dataclass(kw_only=True, frozen=True, eq=False)
class ReductionNode(Node):
    operator: BinaryAssocElementOperator
    axes: tuple[int, ...]


@dataclass(kw_only=True, frozen=True, eq=False)
class MatmulNode(Node):
    pass


type RemapDirection = Literal["gather", "scatter"]
type RemapKeys = Literal["accessor", "indices"]


@dataclass(kw_only=True, frozen=True, eq=False)
class RemapNode(Node):
    direction: RemapDirection
    keys: RemapKeys
    operator: BinaryAssocElementOperator | None
    woffset: int
    wpitch: tuple[int, ...]
    source_shape: tuple[int, ...] | None = None

    @property
    def indices(self) -> "View | None":
        return self.args[1] if self.keys == "indices" else None
