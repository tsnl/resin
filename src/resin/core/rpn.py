from dataclasses import dataclass

from .scalar import ScalarOperator


@dataclass
class RpnExpr[Operator]:
    string: tuple[Operator | int, ...]

    def __getitem__(self, key: int) -> Operator | int:
        return self.string[key]


@dataclass
class ScalarRpnExpr(RpnExpr[ScalarOperator]): ...
