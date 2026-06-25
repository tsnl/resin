"""Optimizer helpers that build in-graph update expressions."""

__all__ = [
    "sgd",
]

from resin.core.pytree import PyTree, map_pytree, zip_pytree
from resin.dsl.view import View


def sgd(params: PyTree[View], grads: PyTree[View], *, lr: float) -> PyTree[View]:
    return map_pytree(
        zip_pytree(params, grads),
        lambda pair: pair.a - lr * pair.b,
    )