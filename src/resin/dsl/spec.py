"""Typed-function Spec PyTrees, walkers, and signature parsing."""

import builtins
import inspect
import types
import typing
from dataclasses import dataclass
from typing import (
    Callable,
    cast,
    overload,
    assert_never,
    get_args,
    get_origin,
    get_type_hints,
)

from resin.core.etype import ElementType, Scalar, is_scalar
from resin.core.pytree import PyTree, map_pytree
from resin.dsl.view import TensorMeta, View


@dataclass(frozen=True)
class TensorSpec:
    etype: ElementType | str
    shape: tuple[int, ...]


type Spec = TensorSpec | dict[str, Spec] | tuple[Spec, ...] | list[Spec]


def _is_view_scalar_union(hint: object) -> bool:
    origin = get_origin(hint)
    if origin is not types.UnionType:
        return False
    args = get_args(hint)
    return View in args and Scalar in args


def annotation_to_spec(hint: object) -> Spec:
    """Convert any param/return annotation PyTree to a Spec PyTree."""
    if isinstance(hint, TensorMeta):
        return TensorSpec(hint.etype, hint.shape)

    origin = get_origin(hint)
    if origin is typing.Annotated:
        args = get_args(hint)
        if len(args) != 2:
            raise ValueError(f"unsupported Annotated annotation: {hint!r}")
        meta = cast(object, args[1])
        if isinstance(meta, TensorMeta):
            return TensorSpec(meta.etype, meta.shape)
        raise ValueError(f"unsupported Annotated metadata: {meta!r}")

    if _is_view_scalar_union(hint):
        raise ValueError(
            "View | Scalar carries no etype/shape; use View[F4, ()] with F4 from resin.core.etype"
        )

    if typing.is_typeddict(hint):
        typed_hints = cast(
            dict[str, object],
            get_type_hints(hint, include_extras=True),
        )
        return {
            key: annotation_to_spec(type_hint) for key, type_hint in typed_hints.items()
        }

    if hint is dict or hint is builtins.dict:
        raise ValueError("bare dict annotations unsupported — use TypedDict")

    if origin is tuple:
        el_args = get_args(hint)
        if not el_args:
            raise ValueError(f"unsupported tuple annotation: {hint!r}")
        return tuple(
            annotation_to_spec(element) for element in cast(tuple[object, ...], el_args)
        )

    if origin is list:
        el_args = get_args(hint)
        if len(el_args) != 1:
            raise ValueError(f"unsupported list annotation: {hint!r}")
        return [annotation_to_spec(cast(object, el_args[0]))]

    raise ValueError(f"unsupported annotation: {hint!r}")


def _param_names(fn: Callable[..., object]) -> tuple[str, ...]:
    sig = inspect.signature(fn)
    names: list[str] = []
    for name, parameter in sig.parameters.items():
        match parameter.kind:
            case inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(f"unsupported *args parameter: {name!r}")
            case inspect.Parameter.VAR_KEYWORD:
                raise ValueError(f"unsupported **kwargs parameter: {name!r}")
            case inspect.Parameter.POSITIONAL_ONLY:
                raise ValueError(f"unsupported positional-only parameter: {name!r}")
            case _:
                annotation = cast(object, parameter.annotation)
                if annotation is inspect.Parameter.empty:
                    raise ValueError(f"unannotated parameter: {name!r}")
                names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class SignatureSpec:
    args: dict[str, Spec]
    return_spec: Spec


def parse_signature(fn: Callable[..., object]) -> SignatureSpec:
    hints = cast(dict[str, object], get_type_hints(fn, include_extras=True))
    if "return" not in hints:
        raise ValueError(f"unannotated return for {fn.__qualname__!r}")
    return SignatureSpec(
        args={name: annotation_to_spec(hints[name]) for name in _param_names(fn)},
        return_spec=annotation_to_spec(hints["return"]),
    )


def typecheck_against_spec(value: object, spec: Spec) -> None:
    match spec:
        case TensorSpec(etype=etype, shape=shape):
            if is_scalar(value):
                if shape != ():
                    raise TypeError(
                        f"expected View with shape {shape!r}, got scalar {value!r}"
                    )
                return
            if not isinstance(value, View):
                raise TypeError(f"expected View, got {type(value).__name__}")
            if value.etype != etype:
                raise ValueError(f"expected etype {etype!r}, got {value.etype!r}")
            if value.shape != shape:
                raise ValueError(f"expected shape {shape!r}, got {value.shape!r}")
        case dict() as fields:
            if not isinstance(value, dict):
                raise TypeError(f"expected dict, got {type(value).__name__}")
            value_dict = cast(dict[str, object], value)
            if set(value_dict.keys()) != set(fields.keys()):
                raise ValueError(
                    f"expected keys {set(fields.keys())!r}, "
                    + f"got {set(value_dict.keys())!r}"
                )
            for key, field_spec in fields.items():
                typecheck_against_spec(value_dict[key], field_spec)
        case tuple() as elements:
            if not isinstance(value, tuple):
                raise TypeError(f"expected tuple, got {type(value).__name__}")
            value_tuple = cast(tuple[object, ...], value)
            if len(value_tuple) != len(elements):
                raise ValueError(
                    f"expected tuple of length {len(elements)}, "
                    + f"got {len(value_tuple)}"
                )
            for element_value, element_spec in zip(value_tuple, elements, strict=True):
                typecheck_against_spec(element_value, element_spec)
        case list() as elements:
            if not isinstance(value, list):
                raise TypeError(f"expected list, got {type(value).__name__}")
            value_list = cast(list[object], value)
            element_spec = elements[0]
            for element_value in value_list:
                typecheck_against_spec(element_value, element_spec)
        case _:
            assert_never(spec)


def bind_call_args[T](
    spec: SignatureSpec,
    args: tuple[T, ...],
    kwargs: dict[str, T],
) -> dict[str, T]:
    names = tuple(spec.args.keys())
    if len(args) > len(names):
        raise TypeError(
            f"too many positional arguments: got {len(args)}, expected at most {len(names)}"
        )
    bound = dict(kwargs)
    for index, value in enumerate(args):
        name = names[index]
        if name in bound:
            raise TypeError(f"got multiple values for argument {name!r}")
        bound[name] = value
    missing = set(names) - set(bound.keys())
    if missing:
        raise TypeError(f"missing required argument(s): {', '.join(sorted(missing))}")
    extra = set(bound.keys()) - set(names)
    if extra:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(extra))}")
    return bound


def validate_kwargs[T](kwargs: dict[str, T], spec: SignatureSpec) -> None:
    expected = set(spec.args.keys())
    actual = set(kwargs.keys())
    if actual != expected:
        raise ValueError(
            f"expected kwargs {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    for name, arg_spec in spec.args.items():
        typecheck_against_spec(kwargs[name], arg_spec)


def map_tensor_leaves(
    value: PyTree[View],
    fn: Callable[[View], View],
) -> PyTree[View]:
    return map_pytree(value, fn)


@overload
def materialize_spec(
    spec: TensorSpec,
    make_tensor: Callable[[TensorSpec, str | None], View],
    *,
    path: str = "",
) -> View: ...


@overload
def materialize_spec(
    spec: Spec,
    make_tensor: Callable[[TensorSpec, str | None], View],
    *,
    path: str = "",
) -> PyTree[View]: ...


def materialize_spec(
    spec: Spec,
    make_tensor: Callable[[TensorSpec, str | None], View],
    *,
    path: str = "",
) -> PyTree[View]:
    # list[...] specs are intentionally unsupported: variadic list materialization
    # needs a separate design before trace() can build bindings for them.
    match spec:
        case TensorSpec() as tensor_spec:
            return make_tensor(tensor_spec, path or None)
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
        case _:
            raise TypeError(f"unsupported spec: {spec!r}")
