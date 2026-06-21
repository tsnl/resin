"""Shared MNIST demo utilities for Resin and PyTorch backends."""

import array
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from resin.dataset import MnistDataset


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


def images_batch_bytes(
    images: bytes, indices: Sequence[int], image_size: int
) -> bytes:
    out = array.array("f")
    for index in indices:
        offset = index * image_size
        for pixel in images[offset : offset + image_size]:
            out.append(pixel / 255.0)
    return out.tobytes()


def iter_test_batches(
    dataset: MnistDataset, batch_size: int
) -> Iterator[tuple[bytes, list[int]]]:
    indices = list(range(dataset.num_samples))
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        if len(batch) < batch_size:
            break
        images = images_batch_bytes(
            dataset.images,
            batch,
            dataset.image_size,
        )
        labels = [dataset.labels[index] for index in batch]
        yield images, labels


def argmax(logits: Sequence[float], num_classes: int) -> int:
    best_class = 0
    best_value = logits[0]
    for class_index in range(1, num_classes):
        value = logits[class_index]
        if value > best_value:
            best_value = value
            best_class = class_index
    return best_class


def update_confusion(
    confusion: list[list[int]],
    true_label: int,
    predicted_label: int,
) -> None:
    confusion[true_label][predicted_label] += 1


def metrics_from_confusion(confusion: list[list[int]]) -> ClassificationMetrics:
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


def print_eval_metrics(step: int, metrics: ClassificationMetrics) -> None:
    print(
        "step "
        + f"{step}: eval "
        + f"accuracy={metrics.accuracy:.4f} "
        + f"macro_precision={metrics.macro_precision:.4f} "
        + f"macro_recall={metrics.macro_recall:.4f} "
        + f"macro_f1={metrics.macro_f1:.4f}",
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
            + f"precision={precision:.4f} "
            + f"recall={recall:.4f} "
            + f"f1={f1:.4f}",
            file=sys.stderr,
        )


def next_batch(
    batch_iter: Iterator[tuple[bytes, bytes]],
    epoch_batches: Iterator[Iterator[tuple[bytes, bytes]]],
) -> tuple[tuple[bytes, bytes], Iterator[tuple[bytes, bytes]]]:
    try:
        return next(batch_iter), batch_iter
    except StopIteration:
        batch_iter = next(epoch_batches)
        return next(batch_iter), batch_iter