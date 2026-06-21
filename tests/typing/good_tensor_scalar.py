from resin.core.etype import Scalar, F4
from resin.dsl import View, const


def accepts_scalar_tensor(x: View | Scalar) -> None:
    _ = x


accepts_scalar_tensor(1.0)
accepts_scalar_tensor(42)
accepts_scalar_tensor(const(1.0, etype=F4))