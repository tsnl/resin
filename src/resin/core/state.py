"""Host-side helpers for named param trees and sink registration."""

__all__ = [
    "flatten_named_views",
    "register_named_params",
    "sink_tree",
]

from collections.abc import Mapping
from typing import TYPE_CHECKING

from resin.core.pytree import PyTree, flatten_pytree_paths
from resin.dsl.view import View

if TYPE_CHECKING:
    from resin.ir.ir import IrProgramBuilder


def flatten_named_views(
    named: Mapping[str, View | PyTree[View]],
) -> dict[str, View]:
    flat: dict[str, View] = {}
    for key, value in named.items():
        if isinstance(value, View):
            flat[key] = value
            continue
        for path, view in flatten_pytree_paths(value, prefix=key):
            flat[path] = view
    return flat


def register_named_params(
    builder: IrProgramBuilder,
    params: Mapping[str, View | PyTree[View]],
) -> dict[str, View]:
    flat = flatten_named_views(params)
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