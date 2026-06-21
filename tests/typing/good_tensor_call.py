from resin.core.etype import F4
from resin.dsl import View, param


def accepts_tensor(x: View[F4, (2,)]) -> None:
    _ = x


accepts_tensor(param(shape=(2,), etype="f4", label="x"))