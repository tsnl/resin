"""Reverse-Polish notation expressions for elementwise IR kernels."""

from dataclasses import dataclass

from resin.core.etype import ElementOperator


@dataclass
class RpnExpr[Operator]:
    string: tuple[Operator | int, ...]

    def __getitem__(self, key: int) -> Operator | int:
        return self.string[key]


@dataclass
class ElementRpnExpr(RpnExpr[ElementOperator]):
    """Per-element RPN over buffer slots.

    Uses ``ElementOperator`` today (one scalar op per buffer element). Tile
    element types may require a richer operator set or a different representation.
    """
