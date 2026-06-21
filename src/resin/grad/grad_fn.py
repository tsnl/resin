"""Per-call forward + gradient wrapper for typed DSL functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from resin.core.pytree import PyTree
from resin.dsl.dsl import Node, View, zeros
from resin.dsl.spec_walk import map_tensor_leaves, validate_against_spec, validate_kwargs
from resin.dsl.types import SignatureSpec, parse_signature


def grad_fn(
    f: Callable[..., PyTree[View]],
    *,
    objective: Callable[[PyTree[View]], View] = lambda output: output,
    grads_for: Sequence[str] | None = None,
) -> Callable[..., tuple[PyTree[View], dict[str, PyTree[View]]]]:
    spec = parse_signature(f)
    target_grads = tuple(grads_for or spec.args.keys())

    def run(**kwargs: PyTree[View]) -> tuple[PyTree[View], dict[str, PyTree[View]]]:
        from resin.grad import grad

        validate_kwargs(kwargs, spec)
        forward = f(**kwargs)
        validate_against_spec(forward, spec.return_spec)
        objective_value = objective(forward)
        grad_map = grad(objective_value)
        grad_args = _build_grad_args(kwargs, grad_map, target_grads, spec)
        return forward, grad_args

    return run


def _build_grad_args(
    kwargs: dict[str, PyTree[View]],
    grad_map: dict[Node, View],
    grads_for: Sequence[str],
    spec: SignatureSpec,
) -> dict[str, PyTree[View]]:
    grad_args: dict[str, PyTree[View]] = {}
    for name, arg_spec in spec.args.items():
        value = kwargs[name]
        if name in grads_for:
            grad_args[name] = map_tensor_leaves(
                value,
                lambda view: grad_map.get(view.node)
                or zeros(view.shape, etype=view.etype),
            )
        else:
            grad_args[name] = value
    return grad_args