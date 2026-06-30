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
    "RemapGatherInfo",
    "RemapInfo",
    "RemapNode",
    "RemapScatterInfo",
]

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resin.dsl.view import View

from resin.core.accessor import Accessor
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
    etype: ElementType
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


@dataclass(frozen=True, kw_only=True)
class RemapScatterInfo:
    """Write each source element into an output slot.

    * ``accessor`` set — multi-index domain is ``accessor.shape``; map through
      ``accessor.offset`` / ``accessor.pitch`` to an output address (no indices
      operand). Typically ``accessor.shape == source.shape``.
    * ``accessor`` is ``None`` — ``args[1]`` supplies per-element output
      multi-indices; the output buffer is C-contiguous of ``node.shape``.

    ``operator``, when set, accumulates into the output (atomic on GPU);
    otherwise output slots are overwritten (buffer cleared first).
    """

    accessor: Accessor | None = None
    operator: BinaryAssocElementOperator | None = None


@dataclass(frozen=True, kw_only=True)
class RemapGatherInfo:
    """Read into a dense output from the source buffer.

    * ``accessor`` set — ``args[1]`` supplies multi-indices into the source,
      addressed via ``accessor``; ``source_shape`` is the logical source shape
      (usually ``accessor.shape``).
    * ``accessor`` is ``None`` — densify: one output element per source logical
      element, reading through the source view's own accessor (no indices arg).
    """

    accessor: Accessor | None = None
    source_shape: tuple[int, ...] | None = None


type RemapInfo = RemapScatterInfo | RemapGatherInfo


@dataclass(kw_only=True, frozen=True, eq=False)
class RemapNode(Node):
    info: RemapInfo
