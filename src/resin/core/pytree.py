import struct
from collections.abc import Callable, Generator, ItemsView
from dataclasses import fields
from typing import Protocol, cast

from useful_types import SequenceNotStr

from .common import DataclassInstance
from .etype import ElementType, Scalar, is_scalar, spell_etype_in_pystruct

type PyTensor = Scalar | list[PyTensor] | tuple[PyTensor, ...]


class Mapping[K, V](Protocol):
    """
    Covariant stand-in for :class:`collections.abc.Mapping`, used as the dict arm
    of :data:`PyTree`. **Covariant in both key and value.**

    Exposes only ``items()``, so both ``K`` and ``V`` appear solely in covariant
    (output) positions. Stdlib :class:`collections.abc.Mapping` is invariant in its
    key because ``__getitem__(key)``/``get(key)`` place the key in an input
    position; dropping those frees the key to be covariant. Key covariance lets
    narrowed-key reprs such as ``dict[Literal["weight", "bias"], View]`` subtype
    ``Mapping[str, PyTree[View]]`` (hence ``PyTree[View]``) without a cast, which a
    stdlib ``Mapping[str, ...]`` arm would reject.

    Walkers dispatch on ``isinstance(x, dict)`` at runtime; this protocol only
    governs static assignability at PyTree boundaries. It deliberately shadows the
    name ``Mapping`` within this module, so ``collections.abc.Mapping`` is not
    imported here.
    """

    def items(self) -> ItemsView[K, V]: ...


class Module[T]:
    """
    Base for dataclass param trees — a first-class PyTree container node.

    Subclass as a *generic* frozen dataclass so the fields ARE the leaf type ``T``;
    then ``Linear[View]`` has all-View fields by construction (``Linear[View](weight=3)``
    is a static error), and walkers preserve the concrete type through grad/sgd::

        @dataclass(frozen=True)
        class Linear[T: View](Module[T]):
            weight: T
            bias: T | None = None

    Walkers recurse ``Module`` instances over :func:`dataclasses.fields` (field name =
    path segment), so ``list[Linear[View]]`` flattens to ``0.weight`` / ``0.bias``.
    ``T`` is unbounded here because ``core`` stays leaf-type-agnostic; bound the leaf
    on concrete modules (``[T: View]``).
    """

    def params(self) -> dict[str, T]:
        """Flatten this module's leaves to ``{dotted-path: leaf}``.

        The structural runtime check is the walker's :func:`expect_pytree_leaf`
        assertion (every collected value is a genuine leaf, not an un-flattened
        container); ``None`` fields are skipped.
        """
        return dict(flatten_pytree_paths(self))


type PyTree[T] = (
    T
    | Mapping[str, PyTree[T]]
    | SequenceNotStr[PyTree[T]]
    | tuple[PyTree[T], ...]
    | Module[T]
)
"""
Recursive JSON-like trees of ``T`` leaves nested in dict/list/Module containers.

**Runtime shape** (what walkers actually recurse on)::

    T | dict[str, PyTree[T]] | list[PyTree[T]] | Module[T]

**Static annotation** (wider on purpose; see ``tests/typing/test_pytree_containers.py``)::

    T
    | Mapping[str, PyTree[T]]
    | SequenceNotStr[PyTree[T]]
    | tuple[PyTree[T], ...]
    | Module[T]

Leaf type ``T`` should be the payload (e.g. ``View``, ``int``), not ``dict`` or
``list``. Nested structure belongs in the container arms, not in ``T``.
``dict[str, T]`` and narrowed-key ``dict[Literal[...], T]`` module reprs both
subtype the dict arm; ``TypedDict`` reprs do *not* (pyright sees their values as
``object``) — use a plain ``dict`` at PyTree boundaries, or register explicit
paths via :func:`resin.runtime.trees.register_named_params`.

Why the static alias is wider than runtime:

- ``Mapping`` (this module's covariant protocol, *not* ``collections.abc.Mapping``)
  and ``SequenceNotStr`` (from ``useful_types``) are covariant, so module reprs such
  as ``list[Linear]`` subtype ``PyTree[View]`` without casts. Plain ``dict``/``list``
  in the alias are invariant and break that subtyping. ``Mapping`` exposes only
  ``items()`` (key and value both in covariant output positions), so
  ``dict[Literal[...], T]`` keyed reprs subtype it too — ``collections.abc.Mapping``'s
  key-typed ``__getitem__`` forces key invariance and rejects narrowed (literal)
  keys. Runtime containers are plain ``dict`` and ``list``, but walkers treat them
  as immutable: they read structure without mutating nodes, so covariant
  annotations are sound (nothing is ever written through the wider static type).
- ``SequenceNotStr`` rejects ``str``, which otherwise satisfies ``Sequence``.
  Custom ``Protocol``s with a ``copy()`` return type break basedpyright's recursive
  leaf-``T`` inference in walkers.
- ``tuple[PyTree[T], ...]`` types homogeneous tuple containers alongside list
  nodes, but walkers still treat tuples as *leaves* at runtime (only ``dict``,
  ``list``, and ``Module`` are recursed into).
- ``Module[T]`` is a dataclass container node: walkers recurse its fields via
  :func:`dataclasses.fields`, using the field name as the path segment. ``None``
  fields are treated as empty subtrees (no leaves), so an optional ``bias: T | None``
  contributes nothing when unset.
- Walkers call :func:`expect_pytree_leaf` after ruling out ``dict``, ``list``, and
  ``Module`` because pyright cannot narrow the ``PyTree[T]`` union down to leaf ``T``
  from runtime tests alone.

:func:`tree_map` maps a callable over the leaves of N same-structured trees and
rebuilds the structure (the concrete type ``L`` is preserved), replacing the old
``zip_pytree``/``map_pytree`` pairing.
"""


def expect_pytree_leaf[T](value: PyTree[T]) -> T:
    """
    Narrow a PyTree node to leaf ``T`` after container branches are ruled out.

    Walkers recurse only on ``dict``, ``list``, and ``Module``. If ``value`` is none
    of those, it must be a leaf. Pyright cannot infer that from ``isinstance`` alone.
    """
    assert not isinstance(value, (dict, list, Module))
    return cast(T, value)


def _fields_of(node: object):
    # ``node`` is statically ``object`` here, so the view to DataclassInstance is a
    # widening cast (no reportInvalidCast); callers pass Module dataclass instances.
    return fields(cast("DataclassInstance", node))


def _join_pytree_path(prefix: str, segment: str) -> str:
    return segment if not prefix else f"{prefix}.{segment}"


def flatten_pytree_paths[T](
    pytree: PyTree[T],
    *,
    prefix: str = "",
) -> Generator[tuple[str, T], None, None]:
    if pytree is None:
        return
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
    elif isinstance(pytree, Module):
        node = cast("Module[T]", pytree)
        for field in _fields_of(node):
            child: PyTree[T] = getattr(node, field.name)
            yield from flatten_pytree_paths(
                child, prefix=_join_pytree_path(prefix, field.name)
            )
    else:
        yield prefix, expect_pytree_leaf(pytree)


def map_pytree_paths[T, U](
    pytree: PyTree[T],
    fn: Callable[[str, T], U],
    *,
    prefix: str = "",
) -> PyTree[U]:
    if pytree is None:
        return cast(PyTree[U], None)
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
    if isinstance(pytree, Module):
        node = cast("Module[T]", pytree)
        ctor = cast("Callable[..., PyTree[U]]", type(node))
        return ctor(
            **{
                field.name: map_pytree_paths(
                    cast(PyTree[T], getattr(node, field.name)),
                    fn,
                    prefix=_join_pytree_path(prefix, field.name),
                )
                for field in _fields_of(node)
            }
        )
    return fn(prefix, expect_pytree_leaf(pytree))


def flatten_pytree[T](pytree: PyTree[T]) -> Generator[T, None, None]:
    if pytree is None:
        return
    if isinstance(pytree, dict):
        for child in cast(dict[str, PyTree[T]], pytree).values():
            yield from flatten_pytree(child)
    elif isinstance(pytree, list):
        for child in cast(list[PyTree[T]], pytree):
            yield from flatten_pytree(child)
    elif isinstance(pytree, Module):
        node = cast("Module[T]", pytree)
        for field in _fields_of(node):
            child: PyTree[T] = getattr(node, field.name)
            yield from flatten_pytree(child)
    else:
        yield expect_pytree_leaf(pytree)


def map_pytree[T, U](pytree: PyTree[T], f: Callable[[T], U]) -> PyTree[U]:
    if pytree is None:
        return cast(PyTree[U], None)
    if isinstance(pytree, dict):
        return {
            key: map_pytree(child, f)
            for key, child in cast(dict[str, PyTree[T]], pytree).items()
        }
    if isinstance(pytree, list):
        return [map_pytree(child, f) for child in cast(list[PyTree[T]], pytree)]
    if isinstance(pytree, Module):
        node = cast("Module[T]", pytree)
        ctor = cast("Callable[..., PyTree[U]]", type(node))
        return ctor(
            **{
                field.name: map_pytree(cast(PyTree[T], getattr(node, field.name)), f)
                for field in _fields_of(node)
            }
        )
    return f(expect_pytree_leaf(pytree))


def tree_map[L](fn: Callable[..., object], *trees: L) -> L:
    """Map ``fn`` over the leaves of N same-structured trees, rebuilding the structure.

    Homomorphic and structure-preserving: the concrete tree type ``L`` (e.g.
    ``list[Linear[View]]``) is preserved end-to-end. ``fn`` receives one leaf from
    each tree, positionally, and returns the new leaf. ``None`` subtrees stay
    ``None``. Replaces the old ``zip_pytree`` + ``map_pytree`` pairing.
    """
    head = trees[0]
    if head is None:
        return cast(L, None)
    if isinstance(head, dict):
        head_dict = cast(dict[str, PyTree[object]], head)
        dicts = [cast(dict[str, PyTree[object]], t) for t in trees]
        return cast(
            L, {key: tree_map(fn, *[d[key] for d in dicts]) for key in head_dict}
        )
    if isinstance(head, list):
        lists = [cast(list[PyTree[object]], t) for t in trees]
        return cast(
            L, [tree_map(fn, *children) for children in zip(*lists, strict=True)]
        )
    if isinstance(head, Module):
        node = cast("Module[object]", head)
        ctor = cast("Callable[..., L]", type(node))
        return ctor(
            **{
                field.name: tree_map(
                    fn, *[cast(PyTree[object], getattr(t, field.name)) for t in trees]
                )
                for field in _fields_of(node)
            }
        )
    return cast(L, fn(*trees))


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
