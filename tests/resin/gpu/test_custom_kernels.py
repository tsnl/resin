"""GPU tests for PrefixSumNode."""

from resin.dsl import param
from tests.resin.gpu.interp_helpers import run_graph


def test_prefix_sum_exclusive_gpu() -> None:
    x = param(shape=(4,), etype="f4", name="x")
    y = x.prefix_sum(inclusive=False)
    out = run_graph(y, params={x: [1.0, 2.0, 3.0, 4.0]})
    assert out == [0.0, 1.0, 3.0, 6.0]


def test_prefix_sum_inclusive_gpu() -> None:
    x = param(shape=(4,), etype="f4", name="x")
    y = x.prefix_sum(inclusive=True)
    out = run_graph(y, params={x: [1.0, 2.0, 3.0, 4.0]})
    assert out == [1.0, 3.0, 6.0, 10.0]
