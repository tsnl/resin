from resin.core.etype import F4
from resin.dsl import View


def accepts_tensor(x: View[F4, (2,)]) -> None:
    _ = x


accepts_tensor(42)