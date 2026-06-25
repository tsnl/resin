import struct
from collections.abc import Callable, Generator, Mapping
from typing import NamedTuple, cast

from useful_types import SequenceNotStr

from .etype import ElementType, Scalar, is_scalar, spell_etype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]


class PyTreeZipped[T, U](NamedTuple):
    """
    Leaf pairing produced by :func:`zip_pytree`.

    A ``NamedTuple`` so zip results have a distinct static type from both plain
    ``tuple[T, U]`` leaves and ``tuple[PyTree[T], ...]`` containers. At runtime it
    is still a tuple: use ``pair.a`` / ``pair.b`` or unpack/index as usual.
    """

    a: T
    b: U


type PyTree[T] = (
    T
    | Mapping[str, PyTree[T]]
    | SequenceNotStr[PyTree[T]]
    | tuple[PyTree[T], ...]
)
"""
Recursive JSON-like trees of ``T`` leaves nested in dict/list containers.

**Runtime shape** (what walkers actually recurse on)::

    T | dict[str, PyTree[T]] | list[PyTree[T]]

**Static annotation** (wider on purpose; see ``tests/typing/test_pytree_containers.py``)::

    T
    | Mapping[str, PyTree[T]]
    | SequenceNotStr[PyTree[T]]
    | tuple[PyTree[T], ...]

Leaf type ``T`` should be the payload (e.g. ``View``, ``int``), not ``dict`` or
``list``. Nested structure belongs in the container arms, not in ``T``.

Why the static alias is wider than runtime:

- ``Mapping`` and ``SequenceNotStr`` (from ``useful_types``) are covariant, so
  module reprs such as ``list[Linear]`` subtype ``PyTree[View]`` without casts.
  Plain ``dict``/``list`` in the alias are invariant and break that subtyping.
  Runtime containers are plain ``dict`` and ``list``, but walkers treat them as
  immutable: they read structure without mutating nodes, so covariant annotations
  are sound (nothing is ever written through the wider static type).
- ``SequenceNotStr`` rejects ``str``, which otherwise satisfies ``Sequence``.
  Custom ``Protocol``s with a ``copy()`` return type break basedpyright's recursive
  leaf-``T`` inference in walkers.
- ``tuple[PyTree[T], ...]`` types homogeneous tuple containers alongside list
  nodes, but walkers still treat tuples as *leaves* at runtime (only ``dict`` and
  ``list`` are recursed into).
- Walkers call :func:`expect_pytree_leaf` after ruling out ``dict`` and ``list``
  because pyright cannot narrow the ``PyTree[T]`` union down to leaf ``T`` from
  runtime tests alone.

:func:`zip_pytree` returns :class:`PyTreeZipped` leaves instead of ``tuple[T, U]``
so zip pairings do not collide with tuple-container nodes in the type system.
"""


def expect_pytree_leaf[T](value: PyTree[T]) -> T:
    """
    Narrow a PyTree node to leaf ``T`` after ``dict``/``list`` branches are ruled out.

    Walkers recurse only on ``dict`` and ``list``. If ``value`` is neither, it must
    be a leaf. Pyright cannot infer that from ``isinstance`` alone.
    """
    assert not isinstance(value, (dict, list))
    return cast(T, value)


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
        yield prefix, expect_pytree_leaf(pytree)


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
    return fn(prefix, expect_pytree_leaf(pytree))


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if isinstance(pytree, dict):
        for child in cast(dict[str, PyTree[T]], pytree).values():
            yield from flatten_pytree(child)
    elif isinstance(pytree, list):
        for child in cast(list[PyTree[T]], pytree):
            yield from flatten_pytree(child)
    else:
        yield expect_pytree_leaf(pytree)


def map_pytree[T, U](pytree: PyTree[T], f: Callable[[T], U]) -> PyTree[U]:
    if isinstance(pytree, dict):
        return {
            key: map_pytree(child, f)
            for key, child in cast(dict[str, PyTree[T]], pytree).items()
        }
    if isinstance(pytree, list):
        return [map_pytree(child, f) for child in cast(list[PyTree[T]], pytree)]
    return f(expect_pytree_leaf(pytree))


def zip_pytree[T, U](a: PyTree[T], b: PyTree[U]) -> PyTree[PyTreeZipped[T, U]]:
    """Zip two PyTrees with matching structure; leaves become :class:`PyTreeZipped`."""
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
            return PyTreeZipped(expect_pytree_leaf(a), expect_pytree_leaf(b))


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