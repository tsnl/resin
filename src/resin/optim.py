from dataclasses import dataclass

from . import graph as rg


@dataclass
class Linear:
    weight: rg.ParamNode
    bias: rg.ParamNode | None = None

    @staticmethod
    def new(in_features: int, out_features: int, bias: bool = True) -> "Linear":
        weight = rg.Node.param((out_features, in_features), dtype="fp32")
        bias_node = rg.Node.param((out_features,), dtype="fp32") if bias else None
        return Linear(weight=weight, bias=bias_node)

    def __call__(self, x: rg.Node) -> rg.Node:
        out = self.weight @ x
        if self.bias is not None:
            out = out + self.bias
        return out


def relu(x: rg.Node) -> rg.Node:
    return x.max(rg.Node.const(value=0, dtype=x.dtype))


def softmax(x: rg.Node, axes: tuple[int, ...] = (0,)) -> rg.Node:
    exp_x = x.exp()
    sum_exp_x = exp_x.reduce(axes=axes, operator="add")
    return exp_x / sum_exp_x
