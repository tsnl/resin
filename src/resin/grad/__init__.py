__all__ = [
    "NotDifferentiableException",
    "accessor_adjoint",
    "df_do",
    "grad",
    "grad_by_port",
    "grad_by_node",
]

from typing import assert_never, overload

from resin.core.accessor import Accessor, c_contiguous_pitch_for_shape
from resin.core.pytree import PyTree, map_pytree
from resin.dsl.node import (
    DEFAULT_PORT,
    ConstNode,
    CustomNode,
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
    node's dense output space for ``view.port``.
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
        out_shape=view.node.port_shape(view.port),
        etype=view.etype,
    )


def df_do(
    node: Node,
    df_douts: View | dict[str, View],
) -> tuple[View, ...]:
    """
    Given ∂f/∂(node ports), returns (∂f/∂o₁, ∂f/∂o₂, ...) for each operand.

    ``df_douts`` may be a single ``View`` (default port) or a port→View map.
    """
    if not node.args:
        return ()

    ports: dict[str, View]
    if isinstance(df_douts, View):
        ports = {DEFAULT_PORT: df_douts}
    else:
        ports = df_douts

    match node:
        case ConstNode() | ParamNode():
            return ()
        case CustomNode():
            return node.df_do_ports(ports)
        case ElementwiseNode():
            return _df_do_elementwise(node, ports[DEFAULT_PORT])
        case ReductionNode():
            return _df_do_reduction(node, ports[DEFAULT_PORT])
        case MatmulNode():
            df_dout = ports[DEFAULT_PORT]
            return (
                df_dout @ node.args[1].transpose(),
                node.args[0].transpose() @ df_dout,
            )
        case RemapNode():
            return _df_do_remap(node, ports[DEFAULT_PORT])
        case _:
            raise NotDifferentiableException(node)


def _dense_remap_gradient(df_dout: View) -> View:
    return df_dout if df_dout.is_identity() else df_dout.copy()


def _df_do_remap(node: RemapNode, df_dout: View) -> tuple[View, ...]:
    source = node.args[0]
    dense = _dense_remap_gradient(df_dout)

    match node.info:
        case RemapGatherInfo(accessor=None):
            return (
                View(
                    node=dense.node,
                    port=dense.port,
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
                    port=dense.port,
                    accessor=accessor,
                ),
            )
        case RemapScatterInfo(accessor=None):
            indices = node.indices
            assert indices is not None
            return (
                View.remap(
                    source=dense,
                    info=RemapGatherInfo(
                        accessor=Accessor.dense(node.shape),
                        source_shape=node.shape,
                    ),
                    indices=indices,
                ),
            )
        case RemapGatherInfo(accessor=accessor, source_shape=source_shape) if (
            accessor is not None and source_shape is not None
        ):
            indices = node.indices
            assert indices is not None
            return (
                View.remap(
                    source=dense,
                    info=RemapScatterInfo(operator="add"),
                    indices=indices,
                    out_shape=source_shape,
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
        port=df_dout.port,
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


def grad_by_port(f: View) -> dict[tuple[Node, str], View]:
    if f.shape != ():
        raise ValueError("Output graph must be a scalar (i.e. have shape=())")

    grad_port: dict[tuple[Node, str], View] = {}

    def accumulate(view: View, g: View) -> None:
        contrib = accessor_adjoint(view, g)
        key = (view.node, view.port)
        existing = grad_port.get(key)
        grad_port[key] = (existing + contrib) if existing is not None else contrib

    accumulate(f, ones(f.shape, etype=f.etype))

    for node in reversed(toposort([f])):
        df_douts: dict[str, View] = {}
        for port in node.output_ports():
            g = grad_port.get((node, port))
            if g is not None:
                df_douts[port] = g
        if not df_douts:
            continue

        for operand, df_do_i in zip(node.args, df_do(node, df_douts)):
            accumulate(operand, df_do_i)

    return grad_port


def grad_by_node(f: View) -> dict[Node, View]:
    """Backward-compatible map of node → gradient on the default/primary port."""
    by_port = grad_by_port(f)
    result: dict[Node, View] = {}
    for (node, port), g in by_port.items():
        if port == DEFAULT_PORT or (
            port == node.output_ports()[0] and DEFAULT_PORT not in node.output_ports()
        ):
            result[node] = g
        elif node not in result and port == node.output_ports()[0]:
            result[node] = g
    # Prefer explicit default port when present.
    for (node, port), g in by_port.items():
        if port == DEFAULT_PORT:
            result[node] = g
    return result


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
    by_port = grad_by_port(f)
    if wrt is None:
        return grad_by_node(f)
    if isinstance(wrt, View):
        value = by_port.get((wrt.node, wrt.port))
        if value is None:
            raise KeyError(f"no gradient for param {_param_label(wrt.node)!r}")
        return value

    def lookup(view: View) -> View:
        value = by_port.get((view.node, view.port))
        if value is None:
            raise KeyError(f"no gradient for param {_param_label(view.node)!r}")
        return value

    return map_pytree(wrt, lookup)
