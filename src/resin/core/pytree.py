import struct
from collections.abc import Callable, Generator
from typing import cast

from .etype import ElementType, Scalar, is_scalar, spell_etype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]

# Nested trees use dict and list nodes only. Tuples (and every other non-container
# value) are leaves. Callers should treat containers as immutable: walkers read
# structure without mutating it. Annotate bare leaves as PyTree[T], not T | PyTree[T].
type PyTree[T] = dict[str, PyTree[T]] | list[PyTree[T]] | T


def _join_pytree_path(prefix: str, segment: str) -> str:
    return segment if not prefix else f"{prefix}.{segment}"


def flatten_pytree_paths[T](
    pytree: PyTree[T],
    *,
    prefix: str = "",
) -> Generator[tuple[str, T], None, None]:
    if isinstance(pytree, dict):
        for key, child in cast(dict[str, PyTree[T]], pytree).items():
            yield from flatten_pytree_paths(
                child, prefix=_join_pytree_path(prefix, key)
            )
    elif isinstance(pytree, list):
        for index, child in enumerate(cast(list[PyTree[T]], pytree)):
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
    return fn(prefix, pytree)


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if isinstance(pytree, dict):
        for child in cast(dict[str, PyTree[T]], pytree).values():
            yield from flatten_pytree(child)
    elif isinstance(pytree, list):
        for child in cast(list[PyTree[T]], pytree):
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
    return f(pytree)


def zip_pytree[T, U](a: PyTree[T], b: PyTree[U]) -> PyTree[tuple[T, U]]:
    if type(a) != type(b):
        raise TypeError("pytree shape mismatch")

    match (a, b):
        case dict(), dict():
            a_dict = cast(dict[str, PyTree[T]], a)
            b_dict = cast(dict[str, PyTree[U]], b)
            if set(a_dict.keys()) != set(b_dict.keys()):
                raise ValueError("pytree shape mismatch")
            return {key: zip_pytree(a_dict[key], b_dict[key]) for key in a_dict}
        case list(), list():
            a_list = cast(list[PyTree[T]], a)
            b_list = cast(list[PyTree[U]], b)
            if len(a_list) != len(b_list):
                raise ValueError("pytree shape mismatch")
            return [
                zip_pytree(a_child, b_child)
                for a_child, b_child in zip(a_list, b_list, strict=True)
            ]
        case _:
            return cast(tuple[T, U], (a, b))


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