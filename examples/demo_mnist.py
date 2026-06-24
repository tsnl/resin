import argparse
import math
import random
import struct
import sys
from typing import cast

import resin_rt_pybind

import resin
from resin.dataset import MnistDataLoader, MnistDataset
from resin.dsl.prelude import F4

IMG_W = 28
IMG_H = 28
IMG_WH = IMG_W * IMG_H
NUM_CLS = 10
BATCH_SIZE = 64
LR = 1e-3
STEPS = 1_000
EVAL_INTERVAL = 100
SEED = 0

type Mlp = list[resin.nn.Linear]


def mlp_new(
    in_dim: int,
    out_dim: int,
    n_hidden: int,
    hidden_dim: int,
    bias: bool,
) -> Mlp:
    res: list[resin.nn.Linear] = []
    res.append(resin.nn.linear_new(in_dim, hidden_dim, bias=bias))
    for _ in range(n_hidden - 1):
        res.append(resin.nn.linear_new(hidden_dim, hidden_dim, bias=bias))
    res.append(resin.nn.linear_new(hidden_dim, out_dim, bias=bias))
    return res


def mlp(model: Mlp, x: resin.dsl.View) -> resin.dsl.View:
    assert len(x.shape) == 2
    for layer in model[:-1]:
        x = resin.nn.linear(layer, x)
        x = resin.nn.relu(x)
    x = resin.nn.linear(model[-1], x)
    return resin.nn.softmax(x, axes=(1,))


def random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--steps", type=int, default=STEPS)
    _ = ap.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    _ = ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    model = mlp_new(
        in_dim=IMG_WH,
        out_dim=NUM_CLS,
        n_hidden=2,
        hidden_dim=128,
        bias=True,
    )

    xs = resin.dsl.param(shape=(BATCH_SIZE, IMG_WH), etype=F4, name="xs")
    ys = resin.dsl.param(shape=(BATCH_SIZE, NUM_CLS), etype=F4, name="ys")

    y_hats = mlp(model, xs)
    losses = resin.nn.cross_entropy(y_hats, ys)
    loss = resin.nn.mean(losses)

    model_tree = cast(resin.core.pytree.PyTree[resin.dsl.View], model)
    grads = resin.grad.grad(loss, wrt=model_tree)
    new_model = resin.opt.sgd(model_tree, grads, lr=LR)

    compiled = resin.runtime.compile_program(
        params={"xs": xs, "ys": ys, "model": model_tree},
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
    batch_iter = data_loader.iter_batches()

    interp = resin_rt_pybind.Interp("wgpu")
    program_id = compiled.admit(interp)
    binding = compiled.binding(interp, program_id)

    for param_view in resin.core.pytree.flatten_pytree(
        cast(resin.core.pytree.PyTree[resin.dsl.View], model)
    ):
        binding.write_view(param_view, random_param_bytes(param_view.shape))

    for step in range(args.steps):
        try:
            image_bytes, label_bytes = next(batch_iter)
        except StopIteration:
            batch_iter = data_loader.iter_batches(epoch=step)
            image_bytes, label_bytes = next(batch_iter)

        binding.write({"xs": image_bytes, "ys": label_bytes})
        interp.run(program_id)
        binding.commit(from_prefix="new_model", to_prefix="model", tree=model_tree)

        if step % args.eval_interval == 0 or step == args.steps - 1:
            loss_bytes = binding.read_sink("loss")
            loss_value = struct.unpack("<f", loss_bytes)[0]
            print(f"step {step}: loss={loss_value:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()