import argparse
import math
import random
import struct
import sys
from dataclasses import dataclass

import resin_rt_pybind

import resin
from resin import dsl, nn
from resin.dataset import MnistDataLoader, MnistDataset
from resin.dsl.prelude import F4

IMG_W = 28
IMG_H = 28
IMG_WH = IMG_W * IMG_H
NUM_CLS = 10
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
SEED = 0


@dataclass(frozen=True)
class Mlp(nn.Object[dsl.View]):
    layers: list[nn.Linear[dsl.View]]

    @staticmethod
    def new(
        in_dim: int,
        out_dim: int,
        n_hidden: int,
        hidden_dim: int,
        bias: bool,
    ) -> Mlp:
        layers: list[nn.Linear[dsl.View]] = []
        layers.append(nn.Linear.new(in_dim, hidden_dim, bias=bias))
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear.new(hidden_dim, hidden_dim, bias=bias))
        layers.append(nn.Linear.new(hidden_dim, out_dim, bias=bias))
        return Mlp(layers=layers)

    def __call__(self, x: dsl.View) -> dsl.View:
        assert len(x.shape) == 2
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.layers[-1](x)
        return nn.softmax(x, axes=(1,))


def random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--epochs", type=int, default=EPOCHS)
    _ = ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    mlp = Mlp.new(
        in_dim=IMG_WH,
        out_dim=NUM_CLS,
        n_hidden=2,
        hidden_dim=128,
        bias=True,
    )

    xs = dsl.param(shape=(BATCH_SIZE, IMG_WH), etype=F4, name="xs")
    ys = dsl.param(shape=(BATCH_SIZE, NUM_CLS), etype=F4, name="ys")

    y_hats = mlp(xs)
    losses = nn.cross_entropy(y_hats, ys)
    loss = nn.mean(losses)

    grads = resin.grad.grad(loss, wrt=mlp)
    new_model = resin.opt.sgd(mlp, grads, lr=LR)

    compiled = resin.runtime.compile_program(
        params={"xs": xs, "ys": ys, "model": mlp},
        sinks={"loss": loss, "new_model": new_model},
    )

    random.seed(args.seed)
    train_dataset = MnistDataset.load("train")
    data_loader = MnistDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    interp = resin_rt_pybind.Interp("wgpu")
    program_id = compiled.admit(interp)
    binding = compiled.binding(interp, program_id)

    for param_view in mlp.values():
        binding.write_view(param_view, random_param_bytes(param_view.shape))

    for epoch in range(args.epochs):
        for image_bytes, label_bytes in data_loader.iter_batches(epoch=epoch):
            binding.write({"xs": image_bytes, "ys": label_bytes})
            interp.run(program_id)
            binding.commit(from_prefix="new_model", to_prefix="model", tree=mlp)

        loss_bytes = binding.read_sink("loss")
        loss_value = struct.unpack("<f", loss_bytes)[0]
        print(f"epoch {epoch}: loss={loss_value:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
