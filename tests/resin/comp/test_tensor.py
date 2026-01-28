import pytest
import sys

import resin.comp.tensor as tensor


def test_basic():
    t = tensor.new(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype="float32",
    )
    print(f"{str(t)}")


if __name__ == "__main__":
    test_basic()
