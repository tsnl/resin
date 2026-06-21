from resin.core.etype import F4
from resin.dsl import View, const, param


def reduce_scalar(x: View[F4, (2,)]) -> View[F4, ()]:
    return x.sum().squeeze(axes=(0,))


_ = reduce_scalar(param(shape=(2,), etype="f4", label="x"))
_ = reduce_scalar(const(1.0, etype="f4").broadcast((2,)))