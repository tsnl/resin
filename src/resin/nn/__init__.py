import math
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

from resin.core.etype import F4
from resin.core.pytree import PyTree
from resin.dsl.view import View, const, param


@dataclass
class Module:
    def __post_init__(self):
        assert is_dataclass(self), "Module must be a dataclass"

    def params(self) -> PyTree[View]:
        return Module._dataclass_params(self)

    @staticmethod
    def _dataclass_params(dc: object) -> dict[str, PyTree[View]]:
        if not is_dataclass(dc) or isinstance(dc, type):
            raise TypeError(f"Unsupported type in param tree: {type(dc)}")
        return {
            field.name: Module._parse_param_value(cast(object, getattr(dc, field.name)))
            for field in fields(dc)
        }

    @staticmethod
    def _parse_param_value(value: object) -> PyTree[View]:
        if isinstance(value, View):
            return value
        if isinstance(value, dict):
            value_dict = cast(dict[object, object], value)
            return {str(k): Module._parse_param_value(v) for k, v in value_dict.items()}
        if isinstance(value, list):
            value_list = cast(list[object], value)
            return [Module._parse_param_value(v) for v in value_list]
        if isinstance(value, tuple):
            value_tuple = cast(tuple[object, ...], value)
            return tuple(Module._parse_param_value(v) for v in value_tuple)
        if is_dataclass(value):
            return Module._dataclass_params(value)
        raise TypeError(f"Unsupported type in param tree: {type(value)}")


@dataclass
class Linear:
    weight: View
    bias: View | None = None

    @staticmethod
    def new(in_features: int, out_features: int, bias: bool = True) -> "Linear":
        weight = param(shape=(out_features, in_features), etype=F4)
        bias_node = param(shape=(out_features,), etype=F4) if bias else None
        return Linear(weight=weight, bias=bias_node)

    def __call__(self, x: View) -> View:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


def relu(x: View) -> View:
    return x.max(const(0, etype=x.etype))


def softmax(x: View, axes: tuple[int, ...] = (0,)) -> View:
    exp_x = x.exp()
    sum_exp_x = exp_x.reduce(axes=axes, operator="add")
    return exp_x / sum_exp_x


def cross_entropy(y_hat: View, y: View) -> View:
    assert y_hat.shape == y.shape
    axis = len(y_hat.shape) - 1
    return -(y * y_hat.log()).sum(axes=(axis,)).squeeze(axes=(axis,))


def mean(n: View) -> View:
    count = math.prod(n.shape)
    reduced = n.sum(axes=tuple(range(n.rank)))
    for axis in reversed(range(reduced.rank)):
        reduced = reduced.squeeze(axes=(axis,))
    return reduced / count
