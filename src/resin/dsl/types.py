"""Typed-function annotations and signature parsing for the Resin DSL."""

from __future__ import annotations

import builtins
import inspect
import typing
from dataclasses import dataclass
from typing import Annotated, Callable, get_args, get_origin, get_type_hints

from resin.core.etype import ElementType
from resin.dsl.dsl import View, param

ALLOWED_POD: frozenset[type] = frozenset({int, float, str})


@dataclass(frozen=True)
class TensorMeta:
    etype: ElementType
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    etype: ElementType
    shape: tuple[int, ...]

    def param(self, label: str | None = None) -> View:
        return param(shape=self.shape, etype=self.etype, label=label)


@dataclass(frozen=True)
class PodSpec:
    py_type: type


type Spec = TensorSpec | PodSpec | dict[str, Spec] | tuple[Spec, ...] | list[Spec]


class _TensorAlias:
    def __class_getitem__(
        cls, params: tuple[ElementType, tuple[int, ...]]
    ) -> Annotated[View, TensorMeta]:
        etype, shape = params
        if not isinstance(etype, str):
            raise TypeError(f"Tensor etype must be a string literal, got {etype!r}")
        return Annotated[View, TensorMeta(etype, shape)]


Tensor = _TensorAlias


@dataclass(frozen=True)
class SignatureSpec:
    args: dict[str, Spec]
    return_spec: Spec


def _tensor_spec_from_args(params: tuple[object, ...]) -> TensorSpec:
    if len(params) != 2:
        raise ValueError(f"Tensor expects (etype, shape), got {params!r}")
    etype, shape = params
    if not isinstance(etype, str):
        raise ValueError(f"Tensor etype must be a string literal, got {etype!r}")
    if not isinstance(shape, tuple):
        raise ValueError(f"Tensor shape must be a tuple, got {shape!r}")
    return TensorSpec(etype, shape)


def annotation_to_spec(hint: object) -> Spec:
    """Convert any param/return annotation PyTree to a Spec PyTree."""
    origin = get_origin(hint)
    if origin is _TensorAlias:
        return _tensor_spec_from_args(get_args(hint))
    if origin is typing.Annotated:
        args = get_args(hint)
        if len(args) != 2:
            raise ValueError(f"unsupported Annotated annotation: {hint!r}")
        meta = args[1]
        if isinstance(meta, TensorMeta):
            return TensorSpec(meta.etype, meta.shape)
        raise ValueError(f"unsupported Annotated metadata: {meta!r}")

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

    if isinstance(hint, type) and hint in ALLOWED_POD:
        return PodSpec(hint)

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


def parse_signature(fn: Callable[..., object]) -> SignatureSpec:
    hints = get_type_hints(fn, include_extras=True)
    if "return" not in hints:
        raise ValueError(f"unannotated return for {fn.__qualname__!r}")
    return SignatureSpec(
        args={
            name: annotation_to_spec(hints[name])
            for name in _param_names(fn)
        },
        return_spec=annotation_to_spec(hints["return"]),
    )