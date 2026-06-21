"""Functional helpers for typed Resin DSL functions."""

from functools import wraps
from typing import Callable

from resin.core.pytree import PyTree
from resin.dsl.spec import (
    SignatureSpec,
    bind_call_args,
    materialize_spec,
    parse_signature,
    typecheck_against_spec,
    validate_kwargs,
)
from resin.dsl.view import View, param


def trace_from_specs[T](
    fn: Callable[..., T],
    spec: SignatureSpec,
    kwargs: dict[str, PyTree[View]],
) -> T:
    """Run a typed forward with pre-built kwargs and validate the return spec."""
    validate_kwargs(kwargs, spec)
    forward = fn(**kwargs)
    typecheck_against_spec(forward, spec.return_spec)
    return forward


def typecheck[T](
    fn: Callable[..., T],
) -> Callable[..., T]:
    """Validate call arguments and return values against a function's typed signature."""
    spec = parse_signature(fn)

    @wraps(fn)
    def wrapper(
        *args: PyTree[View],
        **kwargs: PyTree[View],
    ) -> T:
        bound = bind_call_args(spec, args, kwargs)
        return trace_from_specs(fn, spec, bound)

    return wrapper


def trace[T](
    fn: Callable[..., T],
) -> tuple[dict[str, PyTree[View]], T]:
    """Convert a typed function to fresh parameter bindings and an output graph."""
    spec = parse_signature(fn)
    bindings = {
        name: materialize_spec(
            arg_spec,
            lambda tensor_spec, label: param(
                shape=tensor_spec.shape,
                etype=tensor_spec.etype,
                label=label,
            ),
            path=name,
        )
        for name, arg_spec in spec.args.items()
    }
    output = trace_from_specs(fn, spec, bindings)
    return bindings, output