"""Optimizer helpers that build in-graph update expressions."""

__all__ = [
    "sgd",
]

from resin.core.pytree import PyTree, tree_map
from resin.dsl.view import View


def sgd[L: PyTree[View]](params: L, grads: L, *, lr: float) -> L:
    def step(p: View, g: View) -> View:
        return p - lr * g

    return tree_map(step, params, grads)
