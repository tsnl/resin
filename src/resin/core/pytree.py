import struct
from collections.abc import Callable, Generator
from typing import Any, cast

from .etype import ElementType, Scalar, is_scalar, spell_etype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]
type PyTree[T] = dict[str, PyTree[T]] | list[PyTree[T]] | tuple[PyTree[T], ...] | T


def _is_pytree_container(value: object) -> bool:
    return isinstance(value, (dict, list, tuple))


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if isinstance(pytree, dict):
        for child in pytree.values():
            yield from flatten_pytree(child)
    elif isinstance(pytree, list):
        for child in pytree:
            yield from flatten_pytree(child)
    elif isinstance(pytree, tuple):
        for child in pytree:
            yield from flatten_pytree(child)
    else:
        yield pytree


def map_pytree[T, U](
    pytree: PyTree[T],
    f: Callable[[Any], U],
    *,
    is_leaf: Callable[[object], bool] | None = None,
    map_leaves_only: bool = False,
) -> PyTree[U]:
    if map_leaves_only:
        if is_leaf is None:
            raise TypeError("map_pytree() requires is_leaf when map_leaves_only=True")
        if is_leaf(pytree):
            return f(pytree)
    elif not _is_pytree_container(pytree):
        return f(pytree)

    if isinstance(pytree, dict):
        return {
            key: map_pytree(
                child,
                f,
                is_leaf=is_leaf,
                map_leaves_only=map_leaves_only,
            )
            for key, child in pytree.items()
        }
    if isinstance(pytree, list):
        return [
            map_pytree(
                child,
                f,
                is_leaf=is_leaf,
                map_leaves_only=map_leaves_only,
            )
            for child in pytree
        ]
    if isinstance(pytree, tuple):
        return tuple(
            map_pytree(
                child,
                f,
                is_leaf=is_leaf,
                map_leaves_only=map_leaves_only,
            )
            for child in pytree
        )

    if map_leaves_only:
        raise TypeError(f"unsupported PyTree node: {type(pytree).__name__}")
    return f(pytree)


def tree_map_leaves[T, U](
    value: PyTree[T],
    is_leaf: Callable[[object], bool],
    fn: Callable[[Any], U],
) -> PyTree[U]:
    return map_pytree(value, fn, is_leaf=is_leaf, map_leaves_only=True)


def infer_pytensor_shape(value: PyTensor) -> tuple[int, ...]:
    if is_scalar(value):
        return ()
    assert isinstance(value, list)
    if not value:
        return (0,)
    e0_shape = infer_pytensor_shape(value[0])
    for e1 in value[1:]:
        if infer_pytensor_shape(e1) != e0_shape:
            raise ValueError("Inconsistent shapes in nested list")
    return (len(value),) + e0_shape


def marshall_pytensor(value: PyTensor, etype: ElementType | str) -> bytes:
    values = flatten_pytensor(value)
    return struct.pack(f"<{len(values)}{spell_etype_in_pystruct(etype)}", *values)


def flatten_pytensor(value: PyTensor) -> list[Scalar]:
    if is_scalar(value):
        return [cast(Scalar, value)]
    assert isinstance(value, list)
    result = []
    for e in value:
        result.extend(flatten_pytensor(e))
    return result