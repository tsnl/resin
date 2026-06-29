"""GPU tests for PrefixSumNode, SortNode, multi-output ports, and DCE."""

from resin.dsl import View, param
from resin.ir.ir import IrProgramBuilder, IrSortKernel, reachable_ports
from resin.runtime import compile_program
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


def test_sort_builds_radix_subgraph() -> None:
    """``View.sort()`` inlines hist / prefix-sum / scatter digit passes (not one SortNode)."""
    x = param(shape=(3,), etype=F4, name="x")
    _values, perm = x.sort()
    used = reachable_ports([perm])
    builder = IrProgramBuilder(used_ports=used)
    builder.build_sink("perm", perm)
    program = builder.finish()
    # 1 key-encode + 4×(hist + prefix + scatter) = 13 dispatches when only perm is live
    # (final gather for values is DCE'd).
    assert len(program.queue) >= 4
    assert not any(isinstance(d.kernel, IrSortKernel) for d in program.queue)


def test_sort_values_uses_gather_not_monolithic_sort_kernel() -> None:
    x = param(shape=(3,), etype=F4, name="x")
    values, _perm = x.sort()
    used = reachable_ports([values])
    builder = IrProgramBuilder(used_ports=used)
    builder.build_sink("values", values)
    program = builder.finish()
    assert len(program.queue) >= 5
    assert not any(isinstance(d.kernel, IrSortKernel) for d in program.queue)


def test_compile_program_propagates_port_dce() -> None:
    x = param(shape=(3,), etype=F4, name="x")
    _values, perm = x.sort()
    compiled = compile_program(params={"x": x}, sinks={"perm": perm})
    # Program admits and runs with only perm consumed.
    import resin_rt_pybind
    from resin.core.pytree import marshall_pytensor

    interp = resin_rt_pybind.Interp("wgpu")
    pid = compiled.admit(interp)
    binding = compiled.binding(interp, pid)
    binding.write({"x": marshall_pytensor([2.0, 0.0, 1.0], etype=F4)})
    interp.run(pid)
    raw = binding.read_sink("perm")
    out = run_graph(perm, params={x: [2.0, 0.0, 1.0]})
    assert [int(i) for i in out] == [1, 2, 0]
    assert len(raw) == 12  # 3 * u4


def test_gather_via_sort_perm() -> None:
    """Sorted values equal gather(x, perm) when both ports are live."""
    x = param(shape=(4,), etype=F4, name="x")
    values, perm = x.sort()
    data = [9.0, 1.0, 5.0, 3.0]
    vals = run_graph(values, params={x: data})
    idxs = [int(i) for i in run_graph(perm, params={x: data})]
    assert vals == sorted(data)
    assert [data[i] for i in idxs] == vals
