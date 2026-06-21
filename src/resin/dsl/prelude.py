"""Convenience re-exports for typed Resin DSL programs."""

from resin.core.etype import F2, F4, Scalar, U4
from resin.dsl.functional import trace, typecheck
from resin.dsl import TensorOperand, View, const, param, zeros
from resin.grad import grad_fn

__all__ = [
    "F2",
    "F4",
    "Scalar",
    "TensorOperand",
    "U4",
    "View",
    "const",
    "grad_fn",
    "param",
    "trace",
    "typecheck",
    "zeros",
]