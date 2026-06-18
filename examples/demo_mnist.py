import math
import random
import struct
import sys
from dataclasses import dataclass

import resin_rt_pybind

import resin.nn as nn
from resin import dsl
from resin.dataset import MnistDataLoader, MnistDataset
from resin.train import build_param_update_program, commit_param_updates
from resin.wgpu import WgpuProgram, param_buffer_index


@dataclass
class MnistMlpConfig:
    input_size: int
    hidden_size: int
    output_size: int


@dataclass
class MnistMlp(nn.Module):
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

    def __call__(self, x: dsl.View) -> dsl.View:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return nn.softmax(self.l3(x), axes=(len(x.shape) - 1,))


def _random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def _write_param(
    interp: resin_rt_pybind.WgpuInterp,
    program: WgpuProgram,
    param_view: dsl.View,
    data: bytes,
) -> None:
    node = param_view.node
    assert isinstance(node, dsl.ParamNode)
    interp.write_buffer(param_buffer_index(program, node), data)


def main() -> None:
    batch_size = 64
    img_size = 28
    num_classes = 10
    learning_rate = 5e-3
    steps = 100000

    dataset = MnistDataset.load("train")
    data_loader = MnistDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    epoch_batches = data_loader.iter_epochs()

    config = MnistMlpConfig(
        input_size=img_size**2,
        hidden_size=128,
        output_size=num_classes,
    )

    image = dsl.param(shape=(batch_size, img_size * img_size), stype="f4")
    label = dsl.param(shape=(batch_size, num_classes), stype="f4")
    model = MnistMlp.new(config)
    probs = model(image)
    error = nn.mean(nn.cross_entropy(probs, label))

    trainable_params = list(dsl.flatten_pytree(model.params()))
    program, updated_param_sinks = build_param_update_program(
        error,
        trainable_params=trainable_params,
        learning_rate=learning_rate,
    )
    interp = resin_rt_pybind.WgpuInterp(program.to_msgpack())

    for param_view in trainable_params:
        _write_param(interp, program, param_view, _random_param_bytes(param_view.shape))

    loss_view_index = program.sinks["loss"]
    loss_buffer_index = program.buffer_views[loss_view_index].buffer_index

    step = 0
    batch_iter = next(epoch_batches)
    while step < steps:
        try:
            image_bytes, label_bytes = next(batch_iter)
        except StopIteration:
            batch_iter = next(epoch_batches)
            image_bytes, label_bytes = next(batch_iter)
        _write_param(interp, program, image, image_bytes)
        _write_param(interp, program, label, label_bytes)

        interp.run()
        commit_param_updates(interp, program, updated_param_sinks)

        loss_bytes = interp.read_buffer(loss_buffer_index)
        loss = struct.unpack("<f", loss_bytes)[0]
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step}: loss={loss:.6f}", file=sys.stderr)
        step += 1


if __name__ == "__main__":
    main()
