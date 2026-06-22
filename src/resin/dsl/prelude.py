"""Convenience re-exports for typed Resin DSL programs."""

from resin.core.etype import F2, F4, Scalar, U4
from resin.dsl import TensorOperand, View, const, param, zeros

__all__ = [
    "F2",
    "F4",
    "Scalar",
    "TensorOperand",
    "U4",
    "View",
    "const",
    "param",
    "zeros",
]
