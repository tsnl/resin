__all__ = [
    "NotDifferentiableException",
    "accessor_adjoint",
    "df_do",
    "grad",
]

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape
from resin.dsl.dsl import (
    ConstNode,
    ElementwiseNode,
    GatherWithAccessorNode,
    GatherWithIndicesNode,
    MatmulNode,
    Node,
    ParamNode,
    ReductionNode,
    ScatterWithAccessorNode,
    ScatterWithIndicesNode,
    View,
    _gather_with_accessor,
    _gather_with_indices,
    _scatter_with_accessor,
    _scatter_with_indices,
    ones,
    toposort,
)


class NotDifferentiableException(Exception):
    pass


def accessor_adjoint(view: View, g: View) -> View:
    """
    Pushes a gradient `g` (w.r.t. `view`'s logical values) back into the backing
    node's dense output space.
    """
    broadcast_axes = tuple(i for i, p in enumerate(view.pitch) if p == 0)
    x = g.reduce(axes=broadcast_axes, operator="add") if broadcast_axes else g

    if view._is_identity():
        return x

    return _scatter_with_accessor(
        source=x,
        out_shape=view.node.shape,
        operator="add",
        woffset=view.offset,
        wpitch=view.pitch,
    )


def df_do(node: Node, df_dout: View) -> tuple[View, ...]:
    """
    Given ∂f/∂(node's dense output), returns (∂f/∂o₁, ∂f/∂o₂, ...) for each operand.
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
        case GatherWithAccessorNode():
            source = node.args[0]
            dense = (
                df_dout
                if df_dout._is_identity()
                else _gather_with_accessor(df_dout)
            )
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
        case ScatterWithAccessorNode():
            source = node.args[0]
            dense = (
                df_dout
                if df_dout._is_identity()
                else _gather_with_accessor(df_dout)
            )
            return (
                View(
                    node=dense.node,
                    accessor=Accessor(
                        offset=node.woffset,
                        shape=source.shape,
                        pitch=node.wpitch,
                    ),
                ),
            )
        case ScatterWithIndicesNode():
            source = node.args[0]
            scatter_indices = node.scatter_indices
            dense = (
                df_dout
                if df_dout._is_identity()
                else _gather_with_accessor(df_dout)
            )
            return (
                _gather_with_indices(
                    dense,
                    scatter_indices,
                    out_shape=node.shape,
                    woffset=node.woffset,
                ),
            )
        case GatherWithIndicesNode():
            source = node.args[0]
            scatter_indices = node.scatter_indices
            return (
                _scatter_with_indices(
                    source=df_dout,
                    out_shape=node.out_shape,
                    scatter_indices=scatter_indices,
                    operator="add",
                    woffset=node.woffset,
                ),
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
        case "not" | "eq" | "ne" | "lt" | "gt" | "le" | "ge":
            raise NotDifferentiableException(node)
        case _:
            raise NotImplementedError(f"{node.operator=}")


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
            raise NotImplementedError(f"{node.operator=}")


def grad(f: View) -> dict[Node, View]:
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

        for operand, df_do_i in zip(node.args, df_do(node, df_dn)):
            accumulate(operand, df_do_i)

    return grad_node