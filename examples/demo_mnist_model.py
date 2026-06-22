"""MNIST MLP param tree and forward pass for the Resin demo."""

from typing import TypedDict

import resin.nn as nn
from resin.dsl.view import View
from resin.nn import Linear


class Mlp(TypedDict):
    l1: Linear
    l2: Linear
    l3: Linear


def mlp_new(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> Mlp:
    return {
        "l1": nn.linear_new(input_size, hidden_size),
        "l2": nn.linear_new(hidden_size, hidden_size),
        "l3": nn.linear_new(hidden_size, output_size),
    }


def mlp(model: Mlp, x: View) -> View:
    x = nn.relu(nn.linear(model["l1"], x))
    x = nn.relu(nn.linear(model["l2"], x))
    return nn.softmax(nn.linear(model["l3"], x), axes=(len(x.shape) - 1,))
