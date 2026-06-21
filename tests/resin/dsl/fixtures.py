from typing import TypedDict

from resin.dsl.dsl import const
from resin.dsl.types import Tensor


class LinearParams(TypedDict):
    weight: Tensor["f4", (10, 784)]
    bias: Tensor["f4", (10,)]


def mlp_step(
    x: Tensor["f4", (64, 784)],
    y: Tensor["f4", (64, 10)],
    params: LinearParams,
) -> Tensor["f4", ()]:
    logits = (x @ params["weight"].transpose() + params["bias"]).max(
        const(0, etype="f4")
    )
    probs = logits.exp() / logits.exp().reduce(axes=(1,), operator="add")
    per_ex = -(y * probs.log()).reduce(axes=(1,), operator="add")
    mean_loss = per_ex.reduce(axes=(0,), operator="add") / const(64, etype="f4")
    return mean_loss.squeeze(axes=tuple(range(mean_loss.rank)))