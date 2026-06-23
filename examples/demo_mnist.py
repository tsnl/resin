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
from resin.wgpu import WgpuProgram, build_wgpu_program, param_buffer_index

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


def arrange_grads(
    model: Mlp,
    grads: dict[resin.dsl.Node, resin.dsl.View],
) -> resin.core.pytree.PyTree[resin.dsl.View]:
    return resin.core.pytree.map_pytree(
        cast(resin.core.pytree.PyTree[resin.dsl.View], model),
        lambda view: grads[view.node],
    )


def sink_buffer_index(program: WgpuProgram, sink_name: str) -> int:
    view_index = program.sinks[sink_name]
    return program.buffer_views[view_index]["buffer_index"]


def random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def write_param(
    interp: resin_rt_pybind.Interp,
    program_id: int,
    program: WgpuProgram,
    param_view: resin.dsl.View,
    data: bytes,
) -> None:
    node = param_view.node
    assert isinstance(node, resin.dsl.ParamNode)
    interp.write_buffer(program_id, param_buffer_index(program, node), data)


def commit_model(
    interp: resin_rt_pybind.Interp,
    program_id: int,
    program: WgpuProgram,
    model: resin.core.pytree.PyTree[resin.dsl.View],
) -> None:
    for path, model_view in resin.core.pytree.flatten_pytree_paths(model):
        sink_name = f"new_model.{path}"
        src_buffer = sink_buffer_index(program, sink_name)
        node = model_view.node
        assert isinstance(node, resin.dsl.ParamNode)
        dst_buffer = param_buffer_index(program, node)
        interp.copy_buffer_to_buffer(
            program_id,
            src_buffer,
            program_id,
            dst_buffer,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    _ = ap.add_argument("--steps", type=int, default=STEPS)
    _ = ap.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    _ = ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    #
    # Record the training program:
    # - xs, ys, model: param buffers that can be written
    # - new_model: output sink buffers, should be copied to model for next iter
    #

    model = mlp_new(
        in_dim=IMG_WH,
        out_dim=NUM_CLS,
        n_hidden=2,
        hidden_dim=128,
        bias=True,
    )

    xs = resin.dsl.param(shape=(BATCH_SIZE, IMG_WH), etype=F4, label="xs")
    ys = resin.dsl.param(shape=(BATCH_SIZE, NUM_CLS), etype=F4, label="ys")

    y_hats = mlp(model, xs)
    losses = resin.nn.cross_entropy(y_hats, ys)
    loss = resin.nn.mean(losses)

    grads_dict = resin.grad.grad(loss)
    grads = arrange_grads(model, grads_dict)

    new_model = resin.core.pytree.map_pytree(
        resin.core.pytree.zip_pytree(
            cast(resin.core.pytree.PyTree[resin.dsl.View], model),
            grads,
        ),
        lambda pair: pair[0] - LR * pair[1],
    )

    #
    # Compile the program
    #

    ir_program_builder = resin.ir.IrProgramBuilder()
    ir_program_builder.build_sink("loss", loss)
    for path, view in resin.core.pytree.flatten_pytree_paths(new_model):
        ir_program_builder.build_sink(f"new_model.{path}", view)
    program = build_wgpu_program(ir_program_builder.finish())

    #
    # Run the program
    #

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
    program_id = interp.admit(program.to_msgpack())

    model_params: list[resin.dsl.View] = list(
        resin.core.pytree.flatten_pytree(
            cast(resin.core.pytree.PyTree[resin.dsl.View], model)
        )
    )
    for param_view in model_params:
        write_param(
            interp,
            program_id,
            program,
            param_view,
            random_param_bytes(param_view.shape),
        )

    loss_buffer = sink_buffer_index(program, "loss")

    for step in range(args.steps):
        try:
            image_bytes, label_bytes = next(batch_iter)
        except StopIteration:
            batch_iter = data_loader.iter_batches(epoch=step)
            image_bytes, label_bytes = next(batch_iter)

        write_param(interp, program_id, program, xs, image_bytes)
        write_param(interp, program_id, program, ys, label_bytes)
        interp.run(program_id)
        commit_model(
            interp,
            program_id,
            program,
            cast(resin.core.pytree.PyTree[resin.dsl.View], model),
        )

        if step % args.eval_interval == 0 or step == args.steps - 1:
            loss_bytes = interp.read_buffer(program_id, loss_buffer)
            loss_value = struct.unpack("<f", loss_bytes)[0]
            print(f"step {step}: loss={loss_value:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
