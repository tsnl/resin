import math
from dataclasses import dataclass, fields, is_dataclass

import resin.dsl as dsl
from resin.core.etype import F4


@dataclass
class Module:
    def __post_init__(self):
        assert is_dataclass(self), "Module must be a dataclass"

    def params(self) -> dsl.PyTree[dsl.View]:
        return Module._parse_param_tree(self)

    @staticmethod
    def _parse_param_tree(
        it: Module | dsl.PyTree[dsl.View],
    ) -> dsl.PyTree[dsl.View]:
        if isinstance(it, dsl.View):
            return it
        elif isinstance(it, dict):
            return {k: Module._parse_param_tree(v) for k, v in it.items()}
        elif isinstance(it, list):
            return [Module._parse_param_tree(v) for v in it]
        elif is_dataclass(it):
            res = {}
            for field in fields(it):
                value = getattr(it, field.name)
                res[field.name] = Module._parse_param_tree(value)
            return res
        else:
            raise TypeError(f"Unsupported type in param tree: {type(it)}")


@dataclass
class Linear:
    weight: dsl.View
    bias: dsl.View | None = None

    @staticmethod
    def new(in_features: int, out_features: int, bias: bool = True) -> "Linear":
        weight = dsl.param(shape=(out_features, in_features), etype=F4)
        bias_node = dsl.param(shape=(out_features,), etype=F4) if bias else None
        return Linear(weight=weight, bias=bias_node)

    def __call__(self, x: dsl.View) -> dsl.View:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


def relu(x: dsl.View) -> dsl.View:
    return x.max(dsl.const(0, etype=x.etype))


def softmax(x: dsl.View, axes: tuple[int, ...] = (0,)) -> dsl.View:
    exp_x = x.exp()
    sum_exp_x = exp_x.reduce(axes=axes, operator="add")
    return exp_x / sum_exp_x


def cross_entropy(y_hat: dsl.View, y: dsl.View) -> dsl.View:
    assert y_hat.shape == y.shape
    axis = len(y_hat.shape) - 1
    return -(y * y_hat.log()).sum(axes=(axis,)).squeeze(axes=(axis,))


def mean(n: dsl.View) -> dsl.View:
    count = math.prod(n.shape)
    reduced = n.sum(axes=tuple(range(n.rank)))
    for axis in reversed(range(reduced.rank)):
        reduced = reduced.squeeze(axes=(axis,))
    return reduced / count
