import struct
from collections.abc import Callable, Generator
from typing import cast

from .etype import ElementType, Scalar, is_scalar, spell_etype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]

# Nested trees use plain dict, list, and tuple nodes only. Callers should treat
# these containers as immutable: walkers read structure without mutating it.
# Annotate bare leaves as PyTree[T], not T | PyTree[T].
type PyTree[T] = dict[str, PyTree[T]] | list[PyTree[T]] | tuple[PyTree[T], ...] | T


def _join_pytree_path(prefix: str, segment: str) -> str:
    return segment if not prefix else f"{prefix}.{segment}"


def flatten_pytree_paths[T](
    pytree: PyTree[T],
    *,
    prefix: str = "",
) -> Generator[tuple[str, T], None, None]:
    if isinstance(pytree, dict):
        for key, child in cast(dict[str, PyTree[T]], pytree).items():
            yield from flatten_pytree_paths(child, prefix=_join_pytree_path(prefix, key))
    elif isinstance(pytree, list):
        for index, child in enumerate(cast(list[PyTree[T]], pytree)):
            yield from flatten_pytree_paths(
                child, prefix=_join_pytree_path(prefix, str(index))
            )
    elif isinstance(pytree, tuple):
        for index, child in enumerate(cast(tuple[PyTree[T], ...], pytree)):
            yield from flatten_pytree_paths(
                child, prefix=_join_pytree_path(prefix, str(index))
            )
    else:
        yield prefix, pytree


def map_pytree_paths[T, U](
    pytree: PyTree[T],
    fn: Callable[[str, T], U],
    *,
    prefix: str = "",
) -> PyTree[U]:
    if isinstance(pytree, dict):
        return {
            key: map_pytree_paths(
                cast(dict[str, PyTree[T]], pytree)[key],
                fn,
                prefix=_join_pytree_path(prefix, key),
            )
            for key in cast(dict[str, PyTree[T]], pytree)
        }
    if isinstance(pytree, list):
        return [
            map_pytree_paths(
                child,
                fn,
                prefix=_join_pytree_path(prefix, str(index)),
            )
            for index, child in enumerate(cast(list[PyTree[T]], pytree))
        ]
    if isinstance(pytree, tuple):
        return tuple(
            map_pytree_paths(
                child,
                fn,
                prefix=_join_pytree_path(prefix, str(index)),
            )
            for index, child in enumerate(cast(tuple[PyTree[T], ...], pytree))
        )
    return fn(prefix, pytree)


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if isinstance(pytree, dict):
        for child in cast(dict[str, PyTree[T]], pytree).values():
            yield from flatten_pytree(child)
    elif isinstance(pytree, list):
        for child in cast(list[PyTree[T]], pytree):
            yield from flatten_pytree(child)
    elif isinstance(pytree, tuple):
        for child in cast(tuple[PyTree[T], ...], pytree):
            yield from flatten_pytree(child)
    else:
        yield pytree


def map_pytree[T, U](pytree: PyTree[T], f: Callable[[T], U]) -> PyTree[U]:
    if isinstance(pytree, dict):
        return {
            key: map_pytree(child, f)
            for key, child in cast(dict[str, PyTree[T]], pytree).items()
        }
    if isinstance(pytree, list):
        return [map_pytree(child, f) for child in cast(list[PyTree[T]], pytree)]
    if isinstance(pytree, tuple):
        return tuple(
            map_pytree(child, f) for child in cast(tuple[PyTree[T], ...], pytree)
        )
    return f(pytree)


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
    result: list[Scalar] = []
    for e in value:
        result.extend(flatten_pytensor(e))
    return result
