"""Recursive walkers over Spec PyTrees and runtime values."""

from __future__ import annotations

from typing import Callable, TypeVar

from resin.core.pytree import PyTree
from resin.dsl.dsl import View
from resin.dsl.types import PodSpec, SignatureSpec, Spec, TensorSpec

T = TypeVar("T")
U = TypeVar("U")


def validate_against_spec(value: object, spec: Spec) -> None:
    match spec:
        case TensorSpec(etype=etype, shape=shape):
            if not isinstance(value, View):
                raise TypeError(f"expected View, got {type(value).__name__}")
            if value.etype != etype:
                raise ValueError(
                    f"expected etype {etype!r}, got {value.etype!r}"
                )
            if value.shape != shape:
                raise ValueError(
                    f"expected shape {shape!r}, got {value.shape!r}"
                )
        case PodSpec(py_type=py_type):
            if not isinstance(value, py_type):
                raise TypeError(
                    f"expected {py_type.__name__}, got {type(value).__name__}"
                )
        case dict() as fields:
            if not isinstance(value, dict):
                raise TypeError(f"expected dict, got {type(value).__name__}")
            if set(value.keys()) != set(fields.keys()):
                raise ValueError(
                    f"expected keys {set(fields.keys())!r}, got {set(value.keys())!r}"
                )
            for key, field_spec in fields.items():
                validate_against_spec(value[key], field_spec)
        case tuple() as elements:
            if not isinstance(value, tuple):
                raise TypeError(f"expected tuple, got {type(value).__name__}")
            if len(value) != len(elements):
                raise ValueError(
                    f"expected tuple of length {len(elements)}, got {len(value)}"
                )
            for element_value, element_spec in zip(value, elements, strict=True):
                validate_against_spec(element_value, element_spec)
        case list() as elements:
            if not isinstance(value, list):
                raise TypeError(f"expected list, got {type(value).__name__}")
            element_spec = elements[0]
            for element_value in value:
                validate_against_spec(element_value, element_spec)
        case _:
            raise TypeError(f"unsupported spec: {spec!r}")


def validate_kwargs(kwargs: dict[str, object], spec: SignatureSpec) -> None:
    expected = set(spec.args.keys())
    actual = set(kwargs.keys())
    if actual != expected:
        raise ValueError(
            f"expected kwargs {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    for name, arg_spec in spec.args.items():
        validate_against_spec(kwargs[name], arg_spec)


def tree_map_leaves(
    value: PyTree[T],
    is_leaf: Callable[[T], bool],
    fn: Callable[[T], U],
) -> PyTree[U]:
    if is_leaf(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: tree_map_leaves(child, is_leaf, fn) for key, child in value.items()}
    if isinstance(value, list):
        return [tree_map_leaves(child, is_leaf, fn) for child in value]
    if isinstance(value, tuple):
        return tuple(tree_map_leaves(child, is_leaf, fn) for child in value)
    raise TypeError(f"unsupported PyTree node: {type(value).__name__}")


def map_tensor_leaves(
    value: PyTree[View],
    fn: Callable[[View], View],
) -> PyTree[View]:
    return tree_map_leaves(value, lambda node: isinstance(node, View), fn)


def materialize_spec(
    spec: Spec,
    make_tensor: Callable[[TensorSpec, str | None], View],
    *,
    path: str = "",
) -> PyTree[View]:
    match spec:
        case TensorSpec() as tensor_spec:
            return make_tensor(tensor_spec, path or None)
        case PodSpec():
            raise ValueError("cannot materialize PodSpec into a View")
        case dict() as fields:
            return {
                key: materialize_spec(
                    field_spec,
                    make_tensor,
                    path=f"{path}.{key}" if path else key,
                )
                for key, field_spec in fields.items()
            }
        case tuple() as elements:
            return tuple(
                materialize_spec(
                    element_spec,
                    make_tensor,
                    path=f"{path}[{index}]",
                )
                for index, element_spec in enumerate(elements)
            )
        case list() as elements:
            return [
                materialize_spec(
                    elements[0],
                    make_tensor,
                    path=f"{path}[0]",
                )
            ]
        case _:
            raise TypeError(f"unsupported spec: {spec!r}")