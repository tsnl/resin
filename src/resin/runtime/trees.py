"""Host helpers for registering param trees and sink paths on an IR builder."""

__all__ = [
    "register_named_params",
    "sink_tree",
]

from collections.abc import Mapping

from resin.core.pytree import PyTree, flatten_pytree_paths
from resin.dsl.view import View
from resin.ir.ir import IrProgramBuilder


def register_named_params(
    builder: IrProgramBuilder,
    params: Mapping[str, PyTree[View]],
) -> dict[str, View]:
    flat = dict(flatten_pytree_paths(params))
    for name, view in flat.items():
        builder.register_param(name, view)
    return flat


def sink_tree(
    builder: IrProgramBuilder,
    prefix: str,
    tree: PyTree[View],
) -> None:
    for path, view in flatten_pytree_paths(tree, prefix=prefix):
        builder.build_sink(path, view)
