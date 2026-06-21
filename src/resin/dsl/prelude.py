"""Convenience re-exports for typed Resin DSL programs."""

from resin.dsl.dsl import View, const, param, zeros
from resin.dsl.functional import trace
from resin.dsl.types import Tensor
from resin.grad import grad_fn

__all__ = [
    "View",
    "const",
    "grad_fn",
    "param",
    "Tensor",
    "trace",
    "zeros",
]