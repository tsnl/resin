import math
from dataclasses import dataclass, fields, is_dataclass

from . import graph as rg


@dataclass
class Module:
    def __post_init__(self):
        assert is_dataclass(self), "Module must be a dataclass"

    def params(self) -> rg.PyTree[rg.ParamNode]:
        return Module._parse_param_tree(self)

    @staticmethod
    def _parse_param_tree(
        it: Module | rg.PyTree[rg.ParamNode],
    ) -> rg.PyTree[rg.ParamNode]:
        if isinstance(it, rg.ParamNode):
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
    weight: rg.ParamNode
    bias: rg.ParamNode | None = None

    @staticmethod
    def new(in_features: int, out_features: int, bias: bool = True) -> "Linear":
        weight = rg.ParamNode.new(shape=(out_features, in_features), stype="fp32")
        bias_node = (
            rg.ParamNode.new(shape=(out_features,), stype="fp32") if bias else None
        )
        return Linear(weight=weight, bias=bias_node)

    def __call__(self, x: rg.Node) -> rg.Node:
        out = self.weight @ x
        if self.bias is not None:
            out = out + self.bias
        return out


def relu(x: rg.Node) -> rg.Node:
    return x.max(rg.ConstNode.new(value=0, stype=x.stype))


def softmax(x: rg.Node, axes: tuple[int, ...] = (0,)) -> rg.Node:
    exp_x = x.exp()
    sum_exp_x = exp_x.reduce(axes=axes, operator="add")
    return exp_x / sum_exp_x


def cross_entropy(y_hat: rg.Node, y: rg.Node) -> rg.Node:
    """
    Computes the cross-entropy loss between predicted probabilities `y_hat` and true
    labels `y`.
    """
    assert y_hat.shape == y.shape
    axis = len(y_hat.shape) - 1
    return -(y * y_hat.log()).sum(axes=(axis,)).squeeze(axes=(axis,)) / y.shape[axis]


def mean(n: rg.Node) -> rg.Node:
    count = math.prod(n.shape)
    return n.sum() / count
