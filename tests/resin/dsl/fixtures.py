from resin.core.etype import F4
from resin.dsl import View, const


def mlp_step(
    x: View,
    y: View,
    params: dict[str, View],
) -> View:
    logits = (x @ params["weight"].transpose() + params["bias"]).max(const(0, etype=F4))
    probs = logits.exp() / logits.exp().reduce(axes=(1,), operator="add")
    per_ex = -(y * probs.log()).reduce(axes=(1,), operator="add")
    mean_loss = per_ex.reduce(axes=(0,), operator="add") / const(64, etype=F4)
    return mean_loss.squeeze(axes=tuple(range(mean_loss.rank)))
