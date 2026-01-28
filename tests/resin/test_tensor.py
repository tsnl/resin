import pytest
import sys

import resin.tensor as tensor


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
    print(t.to_sexp())


if __name__ == "__main__":
    test_basic()
