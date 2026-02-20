from dataclasses import dataclass

import resin.graph as rg
import resin.interp as ri
import resin.nn as nn


@dataclass
class MnistMlpConfig:
    input_size: int
    hidden_size: int
    output_size: int


@dataclass
class MnistMlp:
    l1: nn.Linear
    l2: nn.Linear
    l3: nn.Linear

    @staticmethod
    def new(config: MnistMlpConfig) -> "MnistMlp":
        return MnistMlp(
            l1=nn.Linear.new(config.input_size, config.hidden_size),
            l2=nn.Linear.new(config.hidden_size, config.hidden_size),
            l3=nn.Linear.new(config.hidden_size, config.output_size),
        )

    def __call__(self, x: rg.Node) -> rg.Node:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        x = nn.softmax(self.l3(x))
        return x


def main():
    config = MnistMlpConfig(input_size=784, hidden_size=128, output_size=10)

    batch_size = 64
    img_size = 48

    input = rg.Node.param(shape=(batch_size, img_size * img_size), dtype="fp32")
    model = MnistMlp.new(config)
    probs = model(input)

    interp = ri.NumpyInterp({"probs": probs})

    # TODO: randomly initialize model parameters

    # TODO: training loop
    # - write to input buffer
    # - run the interpreter
    # - plug output buffer into optimizer to compute new parameter values, ideally
    #   in-place?

    interp.run()


if __name__ == "__main__":
    main()
