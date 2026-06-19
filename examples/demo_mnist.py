"""MNIST MLP demo with Resin (WGPU) or PyTorch backends."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
#   "resin-rt-pybind",
#   "numpy",
#   "torch",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# resin-rt-pybind = { path = "../crates/resin-rt-pybind", editable = true }
# ///

from __future__ import annotations

import argparse
import array
import math
import random
import struct
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from resin.dataset import MnistDataLoader, MnistDataset

Backend = Literal["resin", "pytorch"]


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    img_size: int = 28
    num_classes: int = 10
    hidden_size: int = 128
    learning_rate: float = 5e-3
    steps: int = 100_000
    eval_interval: int = 100
    seed: int = 0


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]


def _images_batch_bytes(
    images: bytes, indices: Sequence[int], image_size: int
) -> bytes:
    out = array.array("f")
    for index in indices:
        offset = index * image_size
        for pixel in images[offset : offset + image_size]:
            out.append(pixel / 255.0)
    return out.tobytes()


def _iter_test_batches(
    dataset: MnistDataset, batch_size: int
) -> Iterator[tuple[bytes, list[int]]]:
    indices = list(range(dataset.num_samples))
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if len(batch) < batch_size:
            break
        images = _images_batch_bytes(
            dataset.images,
            batch,
            dataset.image_size,
        )
        labels = [dataset.labels[index] for index in batch]
        yield images, labels


def _argmax(logits: Sequence[float], num_classes: int) -> int:
    best_class = 0
    best_value = logits[0]
    for class_index in range(1, num_classes):
        value = logits[class_index]
        if value > best_value:
            best_value = value
            best_class = class_index
    return best_class


def _update_confusion(
    confusion: list[list[int]],
    true_label: int,
    predicted_label: int,
) -> None:
    confusion[true_label][predicted_label] += 1


def _metrics_from_confusion(confusion: list[list[int]]) -> ClassificationMetrics:
    num_classes = len(confusion)
    total = sum(sum(row) for row in confusion)
    correct = sum(
        confusion[class_index][class_index] for class_index in range(num_classes)
    )
    accuracy = correct / total if total else 0.0

    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for class_index in range(num_classes):
        true_positive = confusion[class_index][class_index]
        false_positive = (
            sum(confusion[row][class_index] for row in range(num_classes))
            - true_positive
        )
        false_negative = (
            sum(confusion[class_index][col] for col in range(num_classes))
            - true_positive
        )
        class_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        class_recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=sum(precision) / num_classes,
        macro_recall=sum(recall) / num_classes,
        macro_f1=sum(f1) / num_classes,
        per_class_precision=tuple(precision),
        per_class_recall=tuple(recall),
        per_class_f1=tuple(f1),
    )


def _print_eval_metrics(step: int, metrics: ClassificationMetrics) -> None:
    print(
        f"step {step}: eval "
        f"accuracy={metrics.accuracy:.4f} "
        f"macro_precision={metrics.macro_precision:.4f} "
        f"macro_recall={metrics.macro_recall:.4f} "
        f"macro_f1={metrics.macro_f1:.4f}",
        file=sys.stderr,
    )
    for class_index, (precision, recall, f1) in enumerate(
        zip(
            metrics.per_class_precision,
            metrics.per_class_recall,
            metrics.per_class_f1,
            strict=True,
        )
    ):
        print(
            f"  class {class_index}: "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}",
            file=sys.stderr,
        )


def _next_batch(
    batch_iter: Iterator[tuple[bytes, bytes]],
    epoch_batches: Iterator[Iterator[tuple[bytes, bytes]]],
) -> tuple[tuple[bytes, bytes], Iterator[tuple[bytes, bytes]]]:
    try:
        return next(batch_iter), batch_iter
    except StopIteration:
        batch_iter = next(epoch_batches)
        return next(batch_iter), batch_iter


# ---------------------------------------------------------------------------
# Resin backend
# ---------------------------------------------------------------------------


@dataclass
class ResinMnistMlpConfig:
    input_size: int
    hidden_size: int
    output_size: int


def _resin_random_param_bytes(shape: tuple[int, ...]) -> bytes:
    count = math.prod(shape)
    values = [random.uniform(-0.1, 0.1) for _ in range(count)]
    return struct.pack(f"<{count}f", *values)


def _resin_write_param(
    interp: Any,
    program: Any,
    param_view: Any,
    data: bytes,
) -> None:
    from resin import dsl
    from resin.wgpu import param_buffer_index

    node = param_view.node
    assert isinstance(node, dsl.ParamNode)
    interp.write_buffer(param_buffer_index(program, node), data)


def _resin_sync_model_params(
    *,
    src_interp: Any,
    src_program: Any,
    dst_interp: Any,
    dst_program: Any,
    model_params: Sequence[Any],
) -> None:
    from resin import dsl
    from resin.wgpu import param_buffer_index

    for param_view in model_params:
        node = param_view.node
        assert isinstance(node, dsl.ParamNode)
        src_buffer = param_buffer_index(src_program, node)
        dst_buffer = param_buffer_index(dst_program, node)
        data = src_interp.read_buffer(src_buffer)
        dst_interp.write_buffer(dst_buffer, data)


def _resin_read_probs(
    interp: Any,
    program: Any,
    *,
    batch_size: int,
    num_classes: int,
) -> tuple[float, ...]:
    view_index = program.sinks["probs"]
    buffer_index = program.buffer_views[view_index].buffer_index
    raw = interp.read_buffer(buffer_index)
    expected = batch_size * num_classes
    values = struct.unpack(f"<{expected}f", raw)
    if len(values) != expected:
        raise ValueError(f"expected {expected} prob values, got {len(values)}")
    return values


def _resin_evaluate_test_set(
    *,
    eval_interp: Any,
    eval_program: Any,
    train_interp: Any,
    train_program: Any,
    model_params: Sequence[Any],
    image_param: Any,
    test_dataset: MnistDataset,
    batch_size: int,
    num_classes: int,
) -> ClassificationMetrics:
    _resin_sync_model_params(
        src_interp=train_interp,
        src_program=train_program,
        dst_interp=eval_interp,
        dst_program=eval_program,
        model_params=model_params,
    )

    confusion = [[0] * num_classes for _ in range(num_classes)]
    for image_bytes, labels in _iter_test_batches(test_dataset, batch_size):
        _resin_write_param(eval_interp, eval_program, image_param, image_bytes)
        eval_interp.run()
        probs = _resin_read_probs(
            eval_interp,
            eval_program,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        for sample_index, true_label in enumerate(labels):
            offset = sample_index * num_classes
            predicted_label = _argmax(
                probs[offset : offset + num_classes],
                num_classes,
            )
            _update_confusion(confusion, true_label, predicted_label)

    return _metrics_from_confusion(confusion)


def _run_resin(
    config: TrainConfig,
    *,
    benchmark_steps: int,
    benchmark_warmup: int,
) -> None:
    import resin_rt_pybind

    import resin.nn as nn
    from resin import dsl
    from resin.ir import IrProgramBuilder
    from resin.train import build_param_update_program, commit_param_updates
    from resin.wgpu import build_wgpu_program

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

        def __call__(self, x: dsl.View) -> dsl.View:
            x = nn.relu(self.l1(x))
            x = nn.relu(self.l2(x))
            return nn.softmax(self.l3(x), axes=(len(x.shape) - 1,))

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
        dtype="f4",
    )
    label = dsl.param(shape=(config.batch_size, config.num_classes), dtype="f4")
    model = ResinMnistMlp.new(
        ResinMnistMlpConfig(
            input_size=config.img_size**2,
            hidden_size=config.hidden_size,
            output_size=config.num_classes,
        )
    )
    error = nn.mean(nn.cross_entropy(model(image), label))
    trainable_params = list(dsl.flatten_pytree(model.params()))
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
        _resin_write_param(
            train_interp,
            program,
            param_view,
            _resin_random_param_bytes(param_view.shape),
        )

    loss_buffer_index = program.buffer_views[program.sinks["loss"]].buffer_index

    def train_step(image_bytes: bytes, label_bytes: bytes) -> float:
        _resin_write_param(train_interp, program, image, image_bytes)
        _resin_write_param(train_interp, program, label, label_bytes)
        train_interp.run()
        commit_param_updates(train_interp, program, updated_param_sinks)
        loss_bytes = train_interp.read_buffer(loss_buffer_index)
        return struct.unpack("<f", loss_bytes)[0]

    if benchmark_steps > 0:
        batch_iter = next(epoch_batches)
        timings: list[float] = []
        for step in range(benchmark_warmup + benchmark_steps):
            (image_bytes, label_bytes), batch_iter = _next_batch(
                batch_iter, epoch_batches
            )
            start = time.perf_counter()
            train_step(image_bytes, label_bytes)
            elapsed = time.perf_counter() - start
            if step >= benchmark_warmup:
                timings.append(elapsed)
        mean_step_s = sum(timings) / len(timings)
        print(
            f"resin benchmark (sgd, wgpu): mean_train_step={mean_step_s * 1e3:.3f} ms "
            f"({1.0 / mean_step_s:.1f} steps/s)",
            file=sys.stderr,
        )
        return

    step = 0
    batch_iter = next(epoch_batches)
    while step < config.steps:
        (image_bytes, label_bytes), batch_iter = _next_batch(batch_iter, epoch_batches)
        loss = train_step(image_bytes, label_bytes)

        if step % config.eval_interval == 0 or step == config.steps - 1:
            print(f"step {step}: loss={loss:.6f}", file=sys.stderr)
            metrics = _resin_evaluate_test_set(
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
            _print_eval_metrics(step, metrics)
        step += 1


# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------


def _run_pytorch(
    config: TrainConfig,
    *,
    optimizer_name: str,
    device_name: str,
    match_resin_loss: bool,
    benchmark_steps: int,
    benchmark_warmup: int,
) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def bytes_to_image_tensor(
        image_bytes: bytes,
        *,
        batch_size: int,
        image_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = struct.unpack(f"<{batch_size * image_size}f", image_bytes)
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(
            batch_size,
            image_size,
        )

    def bytes_to_label_tensor(
        label_bytes: bytes,
        *,
        batch_size: int,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = struct.unpack(f"<{batch_size * num_classes}f", label_bytes)
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(
            batch_size,
            num_classes,
        )

    def resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_name)

    def sync_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    class TorchMnistMlp(nn.Module):
        def __init__(
            self, *, input_size: int, hidden_size: int, output_size: int
        ) -> None:
            super().__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, output_size)
            self._init_params()

        def _init_params(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.1, 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            return F.softmax(self.l3(x), dim=-1)

        def logits(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            return self.l3(x)

    def cross_entropy_mean(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -(labels * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

    def cross_entropy_mean_from_logits(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return -(labels * log_probs).sum(dim=-1).mean()

    def build_optimizer(
        model: TorchMnistMlp,
    ) -> torch.optim.Optimizer:
        params = model.parameters()
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=config.learning_rate)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=config.learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=config.learning_rate)
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    @torch.no_grad()
    def evaluate_test_set(
        model: TorchMnistMlp,
        *,
        test_dataset: MnistDataset,
        device: torch.device,
    ) -> ClassificationMetrics:
        model.eval()
        confusion = [[0] * config.num_classes for _ in range(config.num_classes)]
        for image_bytes, labels in _iter_test_batches(test_dataset, config.batch_size):
            images = bytes_to_image_tensor(
                image_bytes,
                batch_size=config.batch_size,
                image_size=test_dataset.image_size,
                device=device,
            )
            probs_flat = model(images).detach().cpu().reshape(-1).tolist()
            for sample_index, true_label in enumerate(labels):
                offset = sample_index * config.num_classes
                predicted_label = _argmax(
                    probs_flat[offset : offset + config.num_classes],
                    config.num_classes,
                )
                _update_confusion(confusion, true_label, predicted_label)
        model.train()
        return _metrics_from_confusion(confusion)

    use_logits_loss = not match_resin_loss

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(device_name)

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

    model = TorchMnistMlp(
        input_size=config.img_size**2,
        hidden_size=config.hidden_size,
        output_size=config.num_classes,
    ).to(device)
    optimizer = build_optimizer(model)

    def train_step(image_bytes: bytes, label_bytes: bytes) -> float:
        images = bytes_to_image_tensor(
            image_bytes,
            batch_size=config.batch_size,
            image_size=config.img_size * config.img_size,
            device=device,
        )
        labels = bytes_to_label_tensor(
            label_bytes,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        if use_logits_loss:
            loss = cross_entropy_mean_from_logits(model.logits(images), labels)
        else:
            loss = cross_entropy_mean(model(images), labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    if benchmark_steps > 0:
        batch_iter = next(epoch_batches)
        timings: list[float] = []
        for step in range(benchmark_warmup + benchmark_steps):
            (image_bytes, label_bytes), batch_iter = _next_batch(
                batch_iter, epoch_batches
            )
            sync_device(device)
            start = time.perf_counter()
            train_step(image_bytes, label_bytes)
            sync_device(device)
            elapsed = time.perf_counter() - start
            if step >= benchmark_warmup:
                timings.append(elapsed)
        mean_step_s = sum(timings) / len(timings)
        print(
            f"pytorch benchmark ({optimizer_name}, {device.type}): "
            f"mean_train_step={mean_step_s * 1e3:.3f} ms "
            f"({1.0 / mean_step_s:.1f} steps/s)",
            file=sys.stderr,
        )
        return

    step = 0
    batch_iter = next(epoch_batches)
    while step < config.steps:
        (image_bytes, label_bytes), batch_iter = _next_batch(batch_iter, epoch_batches)
        loss = train_step(image_bytes, label_bytes)

        if step % config.eval_interval == 0 or step == config.steps - 1:
            print(f"step {step}: loss={loss:.6f}", file=sys.stderr)
            metrics = evaluate_test_set(
                model,
                test_dataset=test_dataset,
                device=device,
            )
            _print_eval_metrics(step, metrics)
        step += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("resin", "pytorch"),
        default="resin",
        help="Training runtime (default: resin).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw", "sgd"),
        default="adam",
        help="PyTorch only (default: adam). Resin always uses SGD.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch only: auto, cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--match-resin-loss",
        action="store_true",
        help="PyTorch only: softmax in forward, then log(probs). "
        "Default uses log-softmax on logits.",
    )
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"backend={args.backend}", file=sys.stderr)
    config = TrainConfig(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        steps=args.steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )

    match args.backend:
        case "resin":
            _run_resin(
                config,
                benchmark_steps=args.benchmark_steps,
                benchmark_warmup=args.benchmark_warmup,
            )
        case "pytorch":
            _run_pytorch(
                config,
                optimizer_name=args.optimizer,
                device_name=args.device,
                match_resin_loss=args.match_resin_loss,
                benchmark_steps=args.benchmark_steps,
                benchmark_warmup=args.benchmark_warmup,
            )
        case _:
            raise NotImplementedError()


if __name__ == "__main__":
    main()
