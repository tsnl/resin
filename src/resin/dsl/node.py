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
    "ScatterNode",
]

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    args: tuple["View", ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * etype_nbytes(self.etype)


@dataclass(kw_only=True, frozen=True, eq=False)
class ConstNode(Node):
    value: PyTensor


@dataclass(kw_only=True, frozen=True, eq=False)
class ParamNode(Node):
    label: str | None


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


@dataclass(kw_only=True, frozen=True, eq=False)
class ScatterNode(Node):
    operator: BinaryAssocElementOperator | None
    woffset: int
    wpitch: tuple[int, ...]
