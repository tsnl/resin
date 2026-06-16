import math
from dataclasses import dataclass, fields, is_dataclass

from resin import gpu


@dataclass
class Module:
    def __post_init__(self):
        assert is_dataclass(self), "Module must be a dataclass"

    def params(self) -> gpu.front.PyTree[gpu.front.View]:
        return Module._parse_param_tree(self)

    @staticmethod
    def _parse_param_tree(
        it: Module | gpu.front.PyTree[gpu.front.View],
    ) -> gpu.front.PyTree[gpu.front.View]:
        if isinstance(it, gpu.front.View):
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
    weight: gpu.front.View
    bias: gpu.front.View | None = None

    @staticmethod
    def new(in_features: int, out_features: int, bias: bool = True) -> "Linear":
        weight = gpu.front.param(shape=(out_features, in_features), stype="f4")
        bias_node = gpu.front.param(shape=(out_features,), stype="f4") if bias else None
        return Linear(weight=weight, bias=bias_node)

    def __call__(self, x: gpu.front.View) -> gpu.front.View:
        out = self.weight @ x
        if self.bias is not None:
            out = out + self.bias
        return out


def relu(x: gpu.front.View) -> gpu.front.View:
    return x.max(gpu.front.const(0, stype=x.stype))


def softmax(x: gpu.front.View, axes: tuple[int, ...] = (0,)) -> gpu.front.View:
    exp_x = x.exp()
    sum_exp_x = exp_x.reduce(axes=axes, operator="add")
    return exp_x / sum_exp_x


def cross_entropy(y_hat: gpu.front.View, y: gpu.front.View) -> gpu.front.View:
    """
    Computes the cross-entropy loss between predicted probabilities `y_hat` and true
    labels `y`.
    """
    assert y_hat.shape == y.shape
    axis = len(y_hat.shape) - 1
    return -(y * y_hat.log()).sum(axes=(axis,)).squeeze(axes=(axis,)) / y.shape[axis]


def mean(n: gpu.front.View) -> gpu.front.View:
    count = math.prod(n.shape)
    return n.sum() / count
