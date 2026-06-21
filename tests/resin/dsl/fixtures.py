from typing import TypedDict

from resin.core.etype import F4
from resin.dsl import View, const


class LinearParams(TypedDict):
    weight: View[F4, (10, 784)]
    bias: View[F4, (10,)]


def mlp_step(
    x: View[F4, (64, 784)],
    y: View[F4, (64, 10)],
    params: LinearParams,
) -> View[F4, ()]:
    logits = (x @ params["weight"].transpose() + params["bias"]).max(
        const(0, etype="f4")
    )
    probs = logits.exp() / logits.exp().reduce(axes=(1,), operator="add")
    per_ex = -(y * probs.log()).reduce(axes=(1,), operator="add")
    mean_loss = per_ex.reduce(axes=(0,), operator="add") / const(64, etype="f4")
    return mean_loss.squeeze(axes=tuple(range(mean_loss.rank)))