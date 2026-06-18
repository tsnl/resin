"""Dataset download and loading utilities."""

import array
import gzip
import os
import random
import struct
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MnistSplit = Literal["train", "test"]

_MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
_MNIST_FILES: dict[MnistSplit, tuple[str, str]] = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}
_MNIST_IMAGE_MAGIC = 2051
_MNIST_LABEL_MAGIC = 2049


def mnist_cache_dir() -> Path:
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / "resin" / "mnist"
    return Path.home() / ".cache" / "resin" / "mnist"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        out.write(response.read())


def ensure_mnist_cached(cache_dir: Path | None = None) -> Path:
    root = cache_dir or mnist_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    for images_name, labels_name in _MNIST_FILES.values():
        for filename in (images_name, labels_name):
            dest = root / filename
            if not dest.exists():
                _download_file(f"{_MNIST_BASE_URL}/{filename}", dest)
    return root


def _read_idx_images(path: Path) -> tuple[int, int, int, bytes]:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
    if magic != _MNIST_IMAGE_MAGIC:
        raise ValueError(f"unexpected MNIST image magic: {magic:#x}")
    expected_bytes = count * rows * cols
    with gzip.open(path, "rb") as f:
        f.seek(16)
        data = f.read(expected_bytes)
    if len(data) != expected_bytes:
        raise ValueError(
            f"MNIST image file {path} has {len(data)} bytes, expected {expected_bytes}"
        )
    return count, rows, cols, data


def _read_idx_labels(path: Path) -> tuple[int, bytes]:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
    if magic != _MNIST_LABEL_MAGIC:
        raise ValueError(f"unexpected MNIST label magic: {magic:#x}")
    with gzip.open(path, "rb") as f:
        f.seek(8)
        data = f.read(count)
    if len(data) != count:
        raise ValueError(
            f"MNIST label file {path} has {len(data)} bytes, expected {count}"
        )
    return count, data


@dataclass(frozen=True)
class MnistDataset:
    images: bytes
    labels: bytes
    rows: int
    cols: int

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    @property
    def image_size(self) -> int:
        return self.rows * self.cols

    @classmethod
    def load(
        cls,
        split: MnistSplit = "train",
        *,
        cache_dir: Path | None = None,
        download: bool = True,
    ) -> MnistDataset:
        root = (
            ensure_mnist_cached(cache_dir)
            if download
            else (cache_dir or mnist_cache_dir())
        )
        images_name, labels_name = _MNIST_FILES[split]
        image_count, rows, cols, images = _read_idx_images(root / images_name)
        label_count, labels = _read_idx_labels(root / labels_name)
        if image_count != label_count:
            raise ValueError(
                f"MNIST {split} split has {image_count} images and {label_count} labels"
            )
        return cls(images=images, labels=labels, rows=rows, cols=cols)


def _batch_indices(
    dataset_size: int,
    batch_size: int,
    *,
    shuffle: bool,
    rng: random.Random,
) -> Iterator[list[int]]:
    indices = list(range(dataset_size))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, dataset_size, batch_size):
        yield indices[start : start + batch_size]


def _images_batch_bytes(
    images: bytes, indices: Sequence[int], image_size: int
) -> bytes:
    out = array.array("f")
    for index in indices:
        offset = index * image_size
        for pixel in images[offset : offset + image_size]:
            out.append(pixel / 255.0)
    return out.tobytes()


def _labels_batch_one_hot_bytes(
    labels: bytes,
    indices: Sequence[int],
    *,
    num_classes: int,
) -> bytes:
    batch_size = len(indices)
    out = array.array("f", [0.0] * (batch_size * num_classes))
    for batch_index, sample_index in enumerate(indices):
        label = labels[sample_index]
        if not 0 <= label < num_classes:
            raise ValueError(f"label {label} is outside [0, {num_classes})")
        out[batch_index * num_classes + label] = 1.0
    return out.tobytes()


@dataclass
class MnistDataLoader:
    dataset: MnistDataset
    batch_size: int
    shuffle: bool = True
    seed: int | None = None
    num_classes: int = 10
    drop_last: bool = False

    def __iter__(self) -> Iterator[tuple[bytes, bytes]]:
        yield from self.iter_batches()

    def iter_batches(self, *, epoch: int = 0) -> Iterator[tuple[bytes, bytes]]:
        rng = random.Random(None if self.seed is None else self.seed + epoch)
        for indices in _batch_indices(
            self.dataset.num_samples,
            self.batch_size,
            shuffle=self.shuffle,
            rng=rng,
        ):
            if self.drop_last and len(indices) < self.batch_size:
                continue
            images = _images_batch_bytes(
                self.dataset.images,
                indices,
                self.dataset.image_size,
            )
            labels = _labels_batch_one_hot_bytes(
                self.dataset.labels,
                indices,
                num_classes=self.num_classes,
            )
            yield images, labels

    def iter_epochs(self) -> Iterator[Iterator[tuple[bytes, bytes]]]:
        epoch = 0
        while True:
            yield self.iter_batches(epoch=epoch)
            epoch += 1
