"""Resin (WGPU) backend for the MNIST MLP demo."""

import math
import random
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import resin_rt_pybind

import resin.nn as nn
from demo_mnist_common import (
    ClassificationMetrics,
    TrainConfig,
    argmax,
    iter_test_batches,
    metrics_from_confusion,
    next_batch,
    print_eval_metrics,
    update_confusion,
)
from resin import dsl
from resin.core.etype import F4
from resin.core.pytree import flatten_pytree
from resin.dataset import MnistDataLoader, MnistDataset
from resin.dsl.view import View
from resin.ir import IrProgramBuilder
from resin.train import (
    ParamUpdateInterpreter,
    build_param_update_program,
    commit_param_updates,
)
from resin.wgpu import WgpuProgram, build_wgpu_program, param_buffer_index


class ResinInterp(ParamUpdateInterpreter, Protocol):
    def write_buffer(self, buffer_index: int, data: bytes) -> None: ...

    def read_buffer(self, buffer_index: int) -> bytes: ...

    def run(self) -> None: ...


@dataclass
class ResinMnistMlpConfig:
    input_size: int
    hidden_size: int
    output_size: int


@dataclass
class ResinMnistMlp(nn.Module):
    l1: nn.Linear
    l2: nn.Linear
    l3: nn.Linear

    @staticmethod
    def new(mlp_config: ResinMnistMlpConfig) -> ResinMnistMlp:
        return ResinMnistMlp(
            l1=nn.Linear.new(mlp_config.input_size, mlp_config.hidden_size),
            l2=nn.Linear.new(mlp_config.hidden_size, mlp_config.hidden_size),
            l3=nn.Linear.new(mlp_config.hidden_size, mlp_config.output_size),
        )

    def __call__(self, x: View) -> View:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return nn.softmax(self.l3(x), axes=(len(x.shape) - 1,))


def random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def write_param(
    interp: ResinInterp,
    program: WgpuProgram,
    param_view: View,
    data: bytes,
) -> None:
    node = param_view.node
    assert isinstance(node, dsl.ParamNode)
    interp.write_buffer(param_buffer_index(program, node), data)


def sync_model_params(
    *,
    src_interp: ResinInterp,
    src_program: WgpuProgram,
    dst_interp: ResinInterp,
    dst_program: WgpuProgram,
    model_params: Sequence[View],
) -> None:
    for param_view in model_params:
        node = param_view.node
        assert isinstance(node, dsl.ParamNode)
        src_buffer = param_buffer_index(src_program, node)
        dst_buffer = param_buffer_index(dst_program, node)
        data = src_interp.read_buffer(src_buffer)
        dst_interp.write_buffer(dst_buffer, data)


def read_probs(
    interp: ResinInterp,
    program: WgpuProgram,
    *,
    batch_size: int,
    num_classes: int,
) -> tuple[float, ...]:
    view_index = program.sinks["probs"]
    buffer_index = program.buffer_views[view_index]["buffer_index"]
    raw = interp.read_buffer(buffer_index)
    expected = batch_size * num_classes
    values = struct.unpack(f"<{expected}f", raw)
    if len(values) != expected:
        raise ValueError(f"expected {expected} prob values, got {len(values)}")
    return values


def evaluate_test_set(
    *,
    eval_interp: ResinInterp,
    eval_program: WgpuProgram,
    train_interp: ResinInterp,
    train_program: WgpuProgram,
    model_params: Sequence[View],
    image_param: View,
    test_dataset: MnistDataset,
    batch_size: int,
    num_classes: int,
) -> ClassificationMetrics:
    sync_model_params(
        src_interp=train_interp,
        src_program=train_program,
        dst_interp=eval_interp,
        dst_program=eval_program,
        model_params=model_params,
    )

    confusion = [[0] * num_classes for _ in range(num_classes)]
    for image_bytes, labels in iter_test_batches(test_dataset, batch_size):
        write_param(eval_interp, eval_program, image_param, image_bytes)
        eval_interp.run()
        probs = read_probs(
            eval_interp,
            eval_program,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        for sample_index, true_label in enumerate(labels):
            offset = sample_index * num_classes
            predicted_label = argmax(
                probs[offset : offset + num_classes],
                num_classes,
            )
            update_confusion(confusion, true_label, predicted_label)

    return metrics_from_confusion(confusion)


def run_resin(
    config: TrainConfig,
    *,
    benchmark_steps: int,
    benchmark_warmup: int,
) -> None:
    random.seed(config.seed)
    train_dataset = MnistDataset.load("train")
    test_dataset = MnistDataset.load("test")
    data_loader = MnistDataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        drop_last=True,
    )
    epoch_batches = data_loader.iter_epochs()

    image = dsl.param(
        shape=(config.batch_size, config.img_size * config.img_size),
        etype=F4,
    )
    label = dsl.param(shape=(config.batch_size, config.num_classes), etype=F4)
    model = ResinMnistMlp.new(
        ResinMnistMlpConfig(
            input_size=config.img_size**2,
            hidden_size=config.hidden_size,
            output_size=config.num_classes,
        )
    )
    error = nn.mean(nn.cross_entropy(model(image), label))
    trainable_params = list(flatten_pytree(model.params()))
    program, updated_param_sinks = build_param_update_program(
        error,
        trainable_params=trainable_params,
        learning_rate=config.learning_rate,
    )
    train_interp = resin_rt_pybind.WgpuInterp(program.to_msgpack())

    eval_builder = IrProgramBuilder()
    eval_builder.build_sink("probs", model(image))
    eval_program = build_wgpu_program(eval_builder.finish())
    eval_interp = resin_rt_pybind.WgpuInterp(eval_program.to_msgpack())

    for param_view in trainable_params:
        write_param(
            train_interp,
            program,
            param_view,
            random_param_bytes(param_view.shape),
        )

    loss_buffer_index = program.buffer_views[program.sinks["loss"]]["buffer_index"]

    def train_step(image_bytes: bytes, label_bytes: bytes) -> float:
        write_param(train_interp, program, image, image_bytes)
        write_param(train_interp, program, label, label_bytes)
        train_interp.run()
        commit_param_updates(train_interp, program, updated_param_sinks)
        loss_bytes = train_interp.read_buffer(loss_buffer_index)
        loss_value = cast(float, struct.unpack("<f", loss_bytes)[0])
        return loss_value

    if benchmark_steps > 0:
        batch_iter = next(epoch_batches)
        timings: list[float] = []
        for step in range(benchmark_warmup + benchmark_steps):
            (image_bytes, label_bytes), batch_iter = next_batch(
                batch_iter, epoch_batches
            )
            start = time.perf_counter()
            _ = train_step(image_bytes, label_bytes)
            elapsed = time.perf_counter() - start
            if step >= benchmark_warmup:
                timings.append(elapsed)
        mean_step_s = sum(timings) / len(timings)
        print(
            "resin benchmark (sgd, wgpu): "
            + f"mean_train_step={mean_step_s * 1e3:.3f} ms "
            + f"({1.0 / mean_step_s:.1f} steps/s)",
            file=sys.stderr,
        )
        return

    step = 0
    batch_iter = next(epoch_batches)
    while step < config.steps:
        (image_bytes, label_bytes), batch_iter = next_batch(batch_iter, epoch_batches)
        loss = train_step(image_bytes, label_bytes)

        if step % config.eval_interval == 0 or step == config.steps - 1:
            print(f"step {step}: loss={loss:.6f}", file=sys.stderr)
            metrics = evaluate_test_set(
                eval_interp=eval_interp,
                eval_program=eval_program,
                train_interp=train_interp,
                train_program=program,
                model_params=trainable_params,
                image_param=image,
                test_dataset=test_dataset,
                batch_size=config.batch_size,
                num_classes=config.num_classes,
            )
            print_eval_metrics(step, metrics)
        step += 1
