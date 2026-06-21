"""Functional helpers for typed Resin DSL functions."""

from __future__ import annotations

from typing import Callable

from resin.core.pytree import PyTree
from resin.dsl.dsl import View
from resin.dsl.spec_walk import materialize_spec, validate_against_spec
from resin.dsl.types import TensorSpec, parse_signature


def trace(
    fn: Callable[..., PyTree[View]],
) -> tuple[dict[str, PyTree[View]], PyTree[View]]:
    """Convert a typed function to fresh parameter bindings and an output graph."""
    spec = parse_signature(fn)
    bindings = {
        name: materialize_spec(
            arg_spec,
            lambda tensor_spec, label: tensor_spec.param(label),
            path=name,
        )
        for name, arg_spec in spec.args.items()
    }
    output = fn(**bindings)
    validate_against_spec(output, spec.return_spec)
    return bindings, output