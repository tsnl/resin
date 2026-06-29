"""Neural-network param trees and ops for the Resin DSL."""

__all__ = [
    "Linear",
    "Module",
    "cross_entropy",
    "mean",
    "relu",
    "softmax",
]

import math
from dataclasses import dataclass

from resin.core.etype import F4
from resin.core.pytree import Module
from resin.dsl.view import View, const, param


@dataclass(frozen=True)
class Linear[T: View](Module[T]):
    """A dense layer as a generic dataclass :class:`Module`.

    Parameterizing by the leaf type ``T`` makes the fields *be* ``T``: ``Linear[View]``
    has all-View params by construction, and walkers preserve ``list[Linear[View]]``
    through grad/sgd. ``bias`` is ``None`` when absent (an empty subtree).

    Construct with :meth:`new`; run the forward pass via ``__call__`` (``layer(x)``).
    """

    weight: T
    bias: T | None = None

    @staticmethod
    def new(m: int, n: int, *, bias: bool = True) -> Linear[View]:
        return Linear(
            weight=param(shape=(n, m), etype=F4, name="weight"),
            bias=param(shape=(n,), etype=F4, name="bias") if bias else None,
        )

    def __call__(self: Linear[View], x: View) -> View:
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