__all__ = [
    "NotDifferentiableException",
    "accessor_adjoint",
    "df_do",
    "grad",
    "grad_by_node",
]

from typing import assert_never, overload

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape
from resin.core.pytree import PyTree, map_pytree
from resin.dsl.node import (
    ConstNode,
    ElementwiseNode,
    MatmulNode,
    Node,
    ParamNode,
    ReductionNode,
    RemapGatherInfo,
    RemapNode,
    RemapScatterInfo,
)
from resin.dsl.view import View, ones, toposort


def _param_label(node: Node) -> str:
    if isinstance(node, ParamNode):
        return node.name
    return repr(node)


class NotDifferentiableException(Exception):
    pass


def accessor_adjoint(view: View, g: View) -> View:
    """
    Pushes a gradient `g` (w.r.t. `view`'s logical values) back into the backing
    node's dense output space.
    """
    broadcast_axes = tuple(i for i, p in enumerate(view.pitch) if p == 0)
    x = g.reduce(axes=broadcast_axes, operator="add") if broadcast_axes else g

    if view.is_identity():
        return x

    return View.remap(
        source=x,
        info=RemapScatterInfo(
            accessor=Accessor(
                offset=view.offset, shape=x.shape, pitch=view.pitch
            ),
            operator="add",
        ),
        out_shape=view.node.shape,
    )


def df_do(node: Node, df_dout: View) -> tuple[View | None, ...]:
    """Given ∂f/∂(node's dense output), return one entry per operand in ``node.args``.

    ``None`` means no adjoint propagates through that operand (e.g. remap indices).
    The returned tuple always has length ``len(node.args)``.
    """
    if not node.args:
        return ()

    match node:
        case ConstNode() | ParamNode():
            return ()
        case ElementwiseNode():
            return _df_do_elementwise(node, df_dout)
        case ReductionNode():
            return _df_do_reduction(node, df_dout)
        case MatmulNode():
            return (
                df_dout @ node.args[1].transpose(),
                node.args[0].transpose() @ df_dout,
            )
        case RemapNode():
            return _df_do_remap(node, df_dout)
        case _:
            raise NotDifferentiableException(node)


def _df_do_remap(node: RemapNode, df_dout: View) -> tuple[View | None, ...]:
    source = node.args[0]
    dense = df_dout.copy()

    match node.info:
        case RemapGatherInfo(accessor=None):
            return (
                View(
                    node=dense.node,
                    accessor=Accessor(
                        offset=0,
                        shape=source.shape,
                        pitch=c_contiguous_pitch_for_shape(source.shape),
                    ),
                ),
            )
        case RemapScatterInfo(accessor=accessor) if accessor is not None:
            return (
                View(
                    node=dense.node,
                    accessor=accessor,
                ),
            )
        case RemapScatterInfo(accessor=None):
            return (
                View.remap(
                    source=dense,
                    info=RemapGatherInfo(
                        accessor=Accessor.dense(node.shape),
                        source_shape=node.shape,
                    ),
                    indices=node.args[1],
                ),
                None,
            )
        case RemapGatherInfo(accessor=accessor, source_shape=source_shape) if (
            accessor is not None and source_shape is not None
        ):
            return (
                View.remap(
                    source=dense,
                    info=RemapScatterInfo(operator="add"),
                    indices=node.args[1],
                    out_shape=source_shape,
                ),
                None,
            )
        case _:
            raise NotDifferentiableException(node)



def _df_do_elementwise(node: ElementwiseNode, df_dout: View) -> tuple[View, ...]:
    n = View.identity(node)
    match node.operator:
        case "neg":
            return (-df_dout,)
        case "exp":
            return (df_dout * n,)
        case "log":
            return (df_dout / node.args[0],)
        case "sqrt":
            return (df_dout / (2 * n),)
        case "sin":
            return (df_dout * node.args[0].cos(),)
        case "cos":
            return (df_dout * -node.args[0].sin(),)
        case "pow":
            return (
                df_dout * node.args[1] * n / node.args[0],
                df_dout * node.args[0].log() * n,
            )
        case "mul":
            return (
                df_dout * node.args[1],
                df_dout * node.args[0],
            )
        case "div":
            return (
                df_dout / node.args[1],
                df_dout * -n / node.args[1],
            )
        case "add":
            return (df_dout, df_dout)
        case "sub":
            return (df_dout, -df_dout)
        case "max":
            return (
                df_dout * node.args[0].gt(node.args[1]),
                df_dout * node.args[1].gt(node.args[0]),
            )
        case "min":
            return (
                df_dout * node.args[0].lt(node.args[1]),
                df_dout * node.args[1].lt(node.args[0]),
            )
        case (
            "not"
            | "eq"
            | "ne"
            | "lt"
            | "gt"
            | "le"
            | "ge"
            | "floor"
            | "ceil"
            | "bitcast"
            | "band"
            | "bor"
            | "bxor"
            | "shl"
            | "shr"
        ):
            raise NotDifferentiableException(node)
        case _:
            assert_never(node.operator)


def _df_do_reduction(node: ReductionNode, df_dout: View) -> tuple[View, ...]:
    operand = node.args[0]

    g = View(
        node=df_dout.node,
        accessor=Accessor(
            offset=df_dout.offset,
            shape=operand.shape,
            pitch=tuple(
                (0 if i in node.axes else df_dout.pitch[i]) for i in range(operand.rank)
            ),
        ),
    )

    match node.operator:
        case "mul":
            return (g * (View.identity(node) / operand),)
        case "add":
            return (g,)
        case "max" | "min":
            cond = operand.eq(View.identity(node))
            return (g * cond,)
        case _:
            assert_never(node.operator)


def grad_by_node(f: View) -> dict[Node, View]:
    if f.shape != ():
        raise ValueError("Output graph must be a scalar (i.e. have shape=())")

    grad_node: dict[Node, View] = {}

    def accumulate(view: View, g: View) -> None:
        contrib = accessor_adjoint(view, g)
        existing = grad_node.get(view.node)
        grad_node[view.node] = (existing + contrib) if existing is not None else contrib

    accumulate(f, ones(f.shape, etype=f.etype))

    for node in reversed(toposort([f])):
        df_dn = grad_node.get(node)
        if df_dn is None:
            continue

        df_dos = df_do(node, df_dn)
        assert len(df_dos) == len(node.args)
        for operand, df_do_i in zip(node.args, df_dos, strict=True):
            if df_do_i is not None:
                accumulate(operand, df_do_i)

    return grad_node


@overload
def grad(f: View, *, wrt: None = None) -> dict[Node, View]: ...


@overload
def grad(f: View, *, wrt: View) -> View: ...


@overload
def grad[L: PyTree[View]](f: View, *, wrt: L) -> L: ...


def grad(
    f: View,
    *,
    wrt: View | PyTree[View] | None = None,
) -> dict[Node, View] | View | PyTree[View]:
    grad_node = grad_by_node(f)
    if wrt is None:
        return grad_node
    if isinstance(wrt, View):
        value = grad_node.get(wrt.node)
        if value is None:
            raise KeyError(f"no gradient for param {_param_label(wrt.node)!r}")
        return value

    def lookup(view: View) -> View:
        value = grad_node.get(view.node)
        if value is None:
            raise KeyError(f"no gradient for param {_param_label(view.node)!r}")
        return value

    return map_pytree(wrt, lookup)
