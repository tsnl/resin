"""Resin (WGPU) backend for the MNIST MLP demo."""

import struct
import sys
import time

import resin_rt_pybind
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
from demo_mnist_model import mlp, mlp_new
from demo_mnist_train import (
    param_view_buffer_index,
    random_params,
    run_sgd_step,
    sgd_step,
    sgd_update,
    sink_buffer_index,
    write_params,
)
from resin_rt_pybind import ProgramId

import resin.nn as nn
from resin.core.etype import F4
from resin.dataset import MnistDataLoader, MnistDataset
from resin.dsl import param
from resin.ir import IrProgramBuilder
from resin.wgpu import build_wgpu_program


def evaluate_test_set(
    *,
    interp: resin_rt_pybind.Interp,
    eval_program_id: ProgramId,
    eval_x_buffer_index: int,
    eval_output_buffer_index: int,
    test_dataset: MnistDataset,
    batch_size: int,
    num_classes: int,
) -> ClassificationMetrics:
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for image_bytes, labels in iter_test_batches(test_dataset, batch_size):
        interp.write_buffer(eval_program_id, eval_x_buffer_index, image_bytes)
        interp.run(eval_program_id)
        raw = interp.read_buffer(eval_program_id, eval_output_buffer_index)
        probs = struct.unpack(f"<{batch_size * num_classes}f", raw)
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

    model = mlp_new(
        input_size=config.img_size**2,
        hidden_size=config.hidden_size,
        output_size=config.num_classes,
    )
    input_shape = (config.batch_size, config.img_size * config.img_size)
    label_shape = (config.batch_size, config.num_classes)
    xs_param = param(shape=input_shape, etype=F4)
    ys_param = param(shape=label_shape, etype=F4)

    pred = mlp(model, xs_param)
    loss = nn.mean(nn.cross_entropy(pred, ys_param))
    train_program = sgd_step(loss, model)

    eval_builder = IrProgramBuilder()
    eval_builder.build_sink("output", mlp(model, xs_param))
    eval_program = build_wgpu_program(eval_builder.finish())

    interp = resin_rt_pybind.Interp("wgpu")
    train_program_id = interp.admit(train_program.to_msgpack())
    eval_program_id = interp.admit(eval_program.to_msgpack())

    x_buffer_index = param_view_buffer_index(train_program, xs_param)
    y_buffer_index = param_view_buffer_index(train_program, ys_param)
    params = random_params(model, seed=config.seed)
    eval_x_buffer_index = param_view_buffer_index(eval_program, xs_param)
    eval_output_buffer_index = sink_buffer_index(eval_program, "output")

    if benchmark_steps > 0:
        batch_iter = next(epoch_batches)
        timings: list[float] = []
        for step_index in range(benchmark_warmup + benchmark_steps):
            (image_bytes, label_bytes), batch_iter = next_batch(
                batch_iter, epoch_batches
            )
            start = time.perf_counter()
            write_params(
                interp, train_program, model, params, program_id=train_program_id
            )
            interp.write_buffer(train_program_id, x_buffer_index, image_bytes)
            interp.write_buffer(train_program_id, y_buffer_index, label_bytes)
            _, grads = run_sgd_step(
                interp,
                train_program,
                model,
                program_id=train_program_id,
            )
            params = sgd_update(params, grads, config.learning_rate)
            elapsed = time.perf_counter() - start
            if step_index >= benchmark_warmup:
                timings.append(elapsed)
        mean_step_s = sum(timings) / len(timings)
        print(
            "resin benchmark (sgd, wgpu): "
            + f"mean_train_step={mean_step_s * 1e3:.3f} ms "
            + f"({1.0 / mean_step_s:.1f} steps/s)",
            file=sys.stderr,
        )
        return

    step_index = 0
    batch_iter = next(epoch_batches)
    while step_index < config.steps:
        (image_bytes, label_bytes), batch_iter = next_batch(batch_iter, epoch_batches)
        write_params(interp, train_program, model, params, program_id=train_program_id)
        interp.write_buffer(train_program_id, x_buffer_index, image_bytes)
        interp.write_buffer(train_program_id, y_buffer_index, label_bytes)
        loss_value, grads = run_sgd_step(
            interp,
            train_program,
            model,
            program_id=train_program_id,
        )
        params = sgd_update(params, grads, config.learning_rate)

        if step_index % config.eval_interval == 0 or step_index == config.steps - 1:
            print(f"step {step_index}: loss={loss_value:.6f}", file=sys.stderr)
            write_params(
                interp,
                eval_program,
                model,
                params,
                program_id=eval_program_id,
            )
            metrics = evaluate_test_set(
                interp=interp,
                eval_program_id=eval_program_id,
                eval_x_buffer_index=eval_x_buffer_index,
                eval_output_buffer_index=eval_output_buffer_index,
                test_dataset=test_dataset,
                batch_size=config.batch_size,
                num_classes=config.num_classes,
            )
            print_eval_metrics(step_index, metrics)
        step_index += 1
