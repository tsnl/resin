"""Typed-function Spec PyTrees, walkers, and signature parsing."""

import builtins
import inspect
import types
import typing
from dataclasses import dataclass
from typing import Callable, get_args, get_origin, get_type_hints

from resin.core.etype import ElementType, Scalar, is_scalar
from resin.core.pytree import PyTree, tree_map_leaves
from resin.dsl.view import TensorMeta, View


@dataclass(frozen=True)
class TensorSpec:
    etype: ElementType
    shape: tuple[int, ...]


type Spec = TensorSpec | dict[str, Spec] | tuple[Spec, ...] | list[Spec]


def _is_view_scalar_union(hint: object) -> bool:
    origin = get_origin(hint)
    if origin not in (types.UnionType, typing.Union):
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
        meta = args[1]
        if isinstance(meta, TensorMeta):
            return TensorSpec(meta.etype, meta.shape)
        raise ValueError(f"unsupported Annotated metadata: {meta!r}")

    if _is_view_scalar_union(hint):
        raise ValueError(
            "View | Scalar carries no etype/shape; use View[F4, ()] with F4 from resin.core.etype"
        )

    if typing.is_typeddict(hint):
        return {
            key: annotation_to_spec(type_hint)
            for key, type_hint in get_type_hints(hint, include_extras=True).items()
        }

    if hint is dict or hint is builtins.dict:
        raise ValueError("bare dict annotations unsupported — use TypedDict")

    if origin in (tuple, typing.Tuple):
        el_args = get_args(hint)
        if not el_args:
            raise ValueError(f"unsupported tuple annotation: {hint!r}")
        return tuple(annotation_to_spec(element) for element in el_args)

    if origin in (list, typing.List):
        el_args = get_args(hint)
        if len(el_args) != 1:
            raise ValueError(f"unsupported list annotation: {hint!r}")
        return [annotation_to_spec(el_args[0])]

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
                if parameter.annotation is inspect.Parameter.empty:
                    raise ValueError(f"unannotated parameter: {name!r}")
                names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class SignatureSpec:
    args: dict[str, Spec]
    return_spec: Spec


def parse_signature(fn: Callable[..., object]) -> SignatureSpec:
    hints = get_type_hints(fn, include_extras=True)
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
            if set(value.keys()) != set(fields.keys()):
                raise ValueError(
                    f"expected keys {set(fields.keys())!r}, got {set(value.keys())!r}"
                )
            for key, field_spec in fields.items():
                typecheck_against_spec(value[key], field_spec)
        case tuple() as elements:
            if not isinstance(value, tuple):
                raise TypeError(f"expected tuple, got {type(value).__name__}")
            if len(value) != len(elements):
                raise ValueError(
                    f"expected tuple of length {len(elements)}, got {len(value)}"
                )
            for element_value, element_spec in zip(value, elements, strict=True):
                typecheck_against_spec(element_value, element_spec)
        case list() as elements:
            if not isinstance(value, list):
                raise TypeError(f"expected list, got {type(value).__name__}")
            element_spec = elements[0]
            for element_value in value:
                typecheck_against_spec(element_value, element_spec)
        case _:
            raise TypeError(f"unsupported spec: {spec!r}")


def bind_call_args(
    spec: SignatureSpec,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
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
        raise TypeError(
            f"missing required argument(s): {', '.join(sorted(missing))}"
        )
    extra = set(bound.keys()) - set(names)
    if extra:
        raise TypeError(
            f"unexpected keyword argument(s): {', '.join(sorted(extra))}"
        )
    return bound


def validate_kwargs(kwargs: dict[str, object], spec: SignatureSpec) -> None:
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
    return tree_map_leaves(value, lambda node: isinstance(node, View), fn)


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