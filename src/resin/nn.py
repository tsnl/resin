from dataclasses import dataclass, fields, is_dataclass

from . import graph as rg


type Tree[T] = "dict[str, Tree[T]] | list[Tree[T]] | T"


@dataclass
class Module:
    def __post_init__(self):
        assert is_dataclass(self), "Module must be a dataclass"

    def params(self) -> Tree[rg.ParamNode]:
        return Module._parse_param_tree(self)

    @staticmethod
    def _parse_param_tree(it: Module | Tree[rg.ParamNode]) -> Tree[rg.ParamNode]:
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
