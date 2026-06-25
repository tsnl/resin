"""Neural-network param trees and ops for the Resin DSL."""

__all__ = [
    "Linear",
    "cross_entropy",
    "linear",
    "linear_new",
    "mean",
    "relu",
    "softmax",
]

import math

from resin.core.etype import F4
from resin.dsl.view import View, const, param

# ``dict[str, View]`` subtypes ``Mapping[str, PyTree[View]]``; ``TypedDict`` does not in
# pyright (value types are ``object``), so module reprs use a plain dict alias here.
type Linear = dict[str, View]


def linear_new(m: int, n: int, *, bias: bool = True) -> Linear:
    weight = param(shape=(n, m), etype=F4, name="weight")
    if bias:
        return {
            "weight": weight,
            "bias": param(shape=(n,), etype=F4, name="bias"),
        }
    return {"weight": weight}


def linear(layer: Linear, x: View) -> View:
    out = x @ layer["weight"].transpose()
    if "bias" in layer:
        out = out + layer["bias"]
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
