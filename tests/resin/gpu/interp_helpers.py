"""Helpers for end-to-end Interp graph execution tests."""

import math
import struct
from collections.abc import Mapping

import resin_rt_pybind

from resin import dsl
from resin.core.etype import ElementType, spell_etype_in_pystruct
from resin.core.pytree import (
    PyTensor,
    flatten_pytensor,
    infer_pytensor_shape,
    marshall_pytensor,
)
from resin.runtime import compile_program


def unmarshall_buffer(
    data: bytes, shape: tuple[int, ...], etype: ElementType
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
    param_items = list((params or {}).items())
    named_params = {
        f"__param_{index}": view for index, (view, _) in enumerate(param_items)
    }
    compiled = compile_program(
        params=named_params,
        sinks={sink_name: sink},
    )
    interp = resin_rt_pybind.Interp("wgpu")
    program_id = compiled.admit(interp)
    binding = compiled.binding(interp, program_id)

    for index, (view, value) in enumerate(param_items):
        expected_shape = infer_pytensor_shape(value)
        if expected_shape != view.shape:
            raise ValueError(
                f"param shape mismatch: view {view.shape}, value {expected_shape}"
            )
        data = marshall_pytensor(value, etype=view.etype)
        binding.write({f"__param_{index}": data})

    interp.run(program_id)
    raw = binding.read_sink(sink_name)
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
