from dataclasses import dataclass

import resin.nn as nn
from resin import gpu


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

    def __call__(self, x: gpu.front.View) -> gpu.front.View:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        x = nn.softmax(self.l3(x))
        return x


def main():
    batch_size = 64
    img_size = 48
    num_classes = 10

    config = MnistMlpConfig(
        input_size=img_size**2,
        hidden_size=128,
        output_size=num_classes,
    )

    image = gpu.front.param(shape=(batch_size, img_size * img_size), stype="fp32")
    label = gpu.front.param(shape=(batch_size, num_classes), stype="fp32")
    model = MnistMlp.new(config)
    probs = model(image)
    error = nn.mean(nn.cross_entropy(probs, label))
    grads = gpu.front.grad(error)
    # TODO: update model parameters with gradients, e.g. using SGD or Adam

    # TODO: run interpreter

    # TODO: randomly initialize model parameters

    # TODO: training loop
    # - write to input buffer
    # - run the interpreter
    # - plug output buffer into optimizer to compute new parameter values, ideally
    #   in-place?

    # interp.run()


if __name__ == "__main__":
    main()
