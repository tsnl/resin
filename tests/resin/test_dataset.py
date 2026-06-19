import gzip
import struct
from pathlib import Path

import pytest

from resin.dataset import MnistDataLoader, MnistDataset


def _write_idx_images(path: Path, rows: int, cols: int, pixels: list[int]) -> None:
    header = struct.pack(">IIII", 2051, len(pixels) // (rows * cols), rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(bytes(pixels))


def _write_idx_labels(path: Path, labels: list[int]) -> None:
    header = struct.pack(">II", 2049, len(labels))
    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(bytes(labels))


@pytest.fixture
def tiny_mnist_cache(tmp_path: Path) -> Path:
    images = [0, 127, 255, 10, 20, 30, 40, 50, 60, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = [3, 7]
    _write_idx_images(
        tmp_path / "train-images-idx3-ubyte.gz", rows=3, cols=3, pixels=images
    )
    _write_idx_labels(tmp_path / "train-labels-idx1-ubyte.gz", labels=labels)
    _write_idx_images(
        tmp_path / "t10k-images-idx3-ubyte.gz", rows=3, cols=3, pixels=images
    )
    _write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte.gz", labels=labels)
    return tmp_path


class TestMnistDataset:
    def test_load_train_split(self, tiny_mnist_cache: Path) -> None:
        dataset = MnistDataset.load("train", cache_dir=tiny_mnist_cache, download=False)
        assert dataset.num_samples == 2
        assert dataset.rows == 3
        assert dataset.cols == 3
        assert dataset.image_size == 9

    def test_dataloader_batch_bytes(self, tiny_mnist_cache: Path) -> None:
        dataset = MnistDataset.load("train", cache_dir=tiny_mnist_cache, download=False)
        loader = MnistDataLoader(dataset, batch_size=2, shuffle=False)
        images, labels = next(iter(loader))

        assert len(images) == 2 * 9 * 4
        assert images[0:4] == struct.pack("<f", 0.0)
        assert images[4:8] == struct.pack("<f", 127 / 255.0)
        assert images[8:12] == struct.pack("<f", 1.0)

        expected_labels = [0.0] * 20
        expected_labels[3] = 1.0
        expected_labels[10 + 7] = 1.0
        assert struct.unpack("<20f", labels) == tuple(expected_labels)
