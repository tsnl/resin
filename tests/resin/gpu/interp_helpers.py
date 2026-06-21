"""Helpers for end-to-end WgpuInterp graph execution tests."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping

import resin_rt_pybind

from resin import dsl
from resin.ir import IrProgramBuilder
from resin.core.etype import ElementType, spell_etype_in_pystruct
from resin.core.pytree import (
    PyTensor,
    flatten_pytensor,
    infer_pytensor_shape,
    marshall_pytensor,
)
from resin.wgpu import WgpuProgram, build_wgpu_program, param_buffer_index

def sink_buffer_index(program: WgpuProgram, sink_name: str) -> int:
    view_index = program.sinks[sink_name]
    return program.buffer_views[view_index].buffer_index


def unmarshall_buffer(
    data: bytes, shape: tuple[int, ...], etype: ElementType | str
) -> list[float]:
    count = math.prod(shape)
    fmt = spell_etype_in_pystruct(etype)
    return list(struct.unpack(f"<{count}{fmt}", data))


def run_graph(
    sink: dsl.View,
    *,
    sink_name: str = "out",
    params: Mapping[dsl.View, PyTensor] | None = None,
) -> list[float]:
    builder = IrProgramBuilder()
    param_items = list((params or {}).items())
    for index, (view, _) in enumerate(param_items):
        builder.build_sink(f"__param_{index}", view)
    builder.build_sink(sink_name, sink)
    program = build_wgpu_program(builder.finish())
    interp = resin_rt_pybind.WgpuInterp(program.to_msgpack())

    for view, value in param_items:
        node = view.node
        assert isinstance(node, dsl.ParamNode)
        expected_shape = infer_pytensor_shape(value)
        if expected_shape != view.shape:
            raise ValueError(
                f"param shape mismatch: view {view.shape}, value {expected_shape}"
            )
        data = marshall_pytensor(value, etype=view.etype)
        interp.write_buffer(param_buffer_index(program, node), data)

    interp.run()
    raw = interp.read_buffer(sink_buffer_index(program, sink_name))
    return unmarshall_buffer(raw, sink.shape, sink.etype)


def run_scalar(
    sink: dsl.View,
    *,
    params: Mapping[dsl.View, PyTensor] | None = None,
) -> float:
    scalar = sink
    while scalar.rank > 0:
        scalar = scalar.squeeze(axes=(0,))
    values = run_graph(scalar, params=params)
    assert len(values) == 1
    return values[0]


def as_list(value: PyTensor) -> list[float]:
    return flatten_pytensor(value)