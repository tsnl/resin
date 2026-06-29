"""
Computation graph node types for the Resin DSL.

Nodes represent computational work that writes dense buffers, one per output port.
Users interact with views on these ports.
"""

__all__ = [
    "DEFAULT_PORT",
    "ConstNode",
    "CustomNode",
    "ElementwiseNode",
    "MatmulNode",
    "Node",
    "ParamNode",
    "PrefixSumNode",
    "ReductionNode",
    "RemapGatherInfo",
    "RemapInfo",
    "RemapNode",
    "RemapScatterInfo",
    "SortNode",
]

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resin.dsl.view import View
    from resin.ir.ir import IrKernel

from resin.core.accessor import Accessor
from resin.core.etype import (
    U4,
    BinaryAssocElementOperator,
    ElementOperator,
    ElementType,
    etype_nbytes,
)
from resin.core.pytree import PyTensor

DEFAULT_PORT = "out"


@dataclass(kw_only=True, frozen=True, eq=False)
class Node(ABC):
    """Base node. Single-output nodes use ``shape``/``etype`` as port ``out``."""

    shape: tuple[int, ...]
    etype: ElementType
    args: tuple[View, ...]

    def output_ports(self) -> tuple[str, ...]:
        return (DEFAULT_PORT,)

    def port_shape(self, port: str) -> tuple[int, ...]:
        if port != DEFAULT_PORT:
            raise KeyError(f"unknown port {port!r} on {type(self).__name__}")
        return self.shape

    def port_etype(self, port: str) -> ElementType:
        if port != DEFAULT_PORT:
            raise KeyError(f"unknown port {port!r} on {type(self).__name__}")
        return self.etype

    def port_nbytes(self, port: str) -> int:
        return math.prod(self.port_shape(port)) * etype_nbytes(self.port_etype(port))

    @property
    def nbytes(self) -> int:
        return sum(self.port_nbytes(p) for p in self.output_ports())


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

    @property
    def indices(self) -> "View | None":
        return self.args[1] if len(self.args) > 1 else None


@dataclass(kw_only=True, frozen=True, eq=False)
class CustomNode(Node, ABC):
    """Multi-port node with custom IR emission and optional ``df_do`` adjoint."""

    @abstractmethod
    def build_kernel(self, *, used_ports: frozenset[str]) -> "IrKernel":
        """Build the forward IR kernel, optionally eliding unused output ports."""

    def df_do_ports(self, df_douts: dict[str, "View"]) -> tuple["View", ...]:
        """Return ∂f/∂operand for each arg given gradients keyed by output port."""
        raise NotImplementedError(f"{type(self).__name__} has no df_do_ports")


@dataclass(kw_only=True, frozen=True, eq=False)
class PrefixSumNode(CustomNode):
    """Exclusive prefix sum along the last axis of a 1D or ND tensor."""

    inclusive: bool = False

    def build_kernel(self, *, used_ports: frozenset[str]) -> "IrKernel":
        from resin.ir.ir import IrPrefixSumKernel

        _ = used_ports
        return IrPrefixSumKernel(
            arg_accessors=(self.args[0].accessor,),
            etype=self.etype,
            shape=self.shape,
            inclusive=self.inclusive,
            arg_etypes=(self.args[0].etype,),
        )

    def df_do_ports(self, df_douts: dict[str, "View"]) -> tuple["View", ...]:
        from resin.dsl.view import View

        df_dout = df_douts.get(DEFAULT_PORT)
        if df_dout is None:
            raise KeyError(DEFAULT_PORT)
        # Exclusive scan adjoint: grad_x[i] = sum_{j>i} grad_y[j]
        # Inclusive scan adjoint: grad_x[i] = sum_{j>=i} grad_y[j]
        # Implemented as reverse exclusive/inclusive prefix on the gradient.
        rev = View.reverse_prefix_sum(df_dout, inclusive=not self.inclusive)
        return (rev,)


@dataclass(kw_only=True, frozen=True, eq=False)
class SortNode(CustomNode):
    """Sort values ascending; ports ``values`` (sorted) and ``perm`` (u4 indices)."""

    perm_etype: ElementType = U4

    def output_ports(self) -> tuple[str, ...]:
        return ("values", "perm")

    def port_shape(self, port: str) -> tuple[int, ...]:
        if port not in ("values", "perm"):
            raise KeyError(f"unknown port {port!r} on SortNode")
        return self.shape

    def port_etype(self, port: str) -> ElementType:
        if port == "values":
            return self.etype
        if port == "perm":
            return self.perm_etype
        raise KeyError(f"unknown port {port!r} on SortNode")

    def build_kernel(self, *, used_ports: frozenset[str]) -> "IrKernel":
        from resin.ir.ir import IrSortKernel

        write_values = "values" in used_ports
        write_perm = "perm" in used_ports
        if not write_values and not write_perm:
            # Node was reached only via an unused path; still emit something valid.
            write_values = True
        return IrSortKernel(
            arg_accessors=(self.args[0].accessor,),
            etype=self.etype,
            shape=self.shape,
            arg_etypes=(self.args[0].etype,),
            write_values=write_values,
            write_perm=write_perm,
            perm_etype=self.perm_etype,
            num_outputs=(1 if write_values else 0) + (1 if write_perm else 0),
        )

    def output_port_order_for_kernel(self, *, used_ports: frozenset[str]) -> tuple[str, ...]:
        """Port binding order for the sort kernel (values then perm, if written)."""
        ports: list[str] = []
        if "values" in used_ports:
            ports.append("values")
        if "perm" in used_ports:
            ports.append("perm")
        if not ports:
            ports.append("values")
        return tuple(ports)

    def df_do_ports(self, df_douts: dict[str, "View"]) -> tuple["View", ...]:
        from resin.dsl.view import View

        # y[i] = x[p[i]] ⇒ scatter grad_y into grad_x at p.
        df_dvalues = df_douts.get("values")
        if df_dvalues is None:
            # No gradient through sorted values (e.g. only perm was consumed).
            return (View.zeros_like(self.args[0]),)
        perm = View.port(self, "perm")
        n = self.shape[0]
        # Build indices as (n, 1) u4 coords into a 1D source of length n.
        indices = perm.reshape((n, 1)) if perm.rank == 1 else perm
        return (
            View.remap(
                source=df_dvalues,
                info=RemapScatterInfo(operator="add"),
                indices=indices,
                out_shape=self.shape,
            ),
        )
