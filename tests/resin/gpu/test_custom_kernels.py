"""GPU tests for PrefixSumNode and SortNode."""

from resin.dsl import param
from tests.resin.gpu.interp_helpers import run_graph


def test_prefix_sum_exclusive_gpu() -> None:
    x = param(shape=(4,), etype=F4, name="x")
    y = x.prefix_sum(inclusive=False)
    out = run_graph(y, params={x: [1.0, 2.0, 3.0, 4.0]})
    assert out == [0.0, 1.0, 3.0, 6.0]


def test_prefix_sum_inclusive_gpu() -> None:
    x = param(shape=(4,), etype=F4, name="x")
    y = x.prefix_sum(inclusive=True)
    out = run_graph(y, params={x: [1.0, 2.0, 3.0, 4.0]})
    assert out == [1.0, 3.0, 6.0, 10.0]


def test_sort_values_and_perm_gpu() -> None:
    x = param(shape=(4,), etype=F4, name="x")
    values, perm = x.sort()
    vals = run_graph(values, params={x: [3.0, 1.0, 4.0, 2.0]})
    idxs = run_graph(perm, params={x: [3.0, 1.0, 4.0, 2.0]})
    assert vals == [1.0, 2.0, 3.0, 4.0]
    assert [int(i) for i in idxs] == [1, 3, 0, 2]


def test_gather_via_sort_perm() -> None:
    x = param(shape=(4,), etype=F4, name="x")
    values, perm = x.sort()
    data = [9.0, 1.0, 5.0, 3.0]
    vals = run_graph(values, params={x: data})
    idxs = [int(i) for i in run_graph(perm, params={x: data})]
    assert vals == sorted(data)
    assert [data[i] for i in idxs] == vals
