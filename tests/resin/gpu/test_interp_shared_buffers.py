"""Buffers are scoped to a program; cross-program copies use explicit program/buffer ids."""

import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.ir import IrProgramBuilder
from resin.wgpu import build_wgpu_program, param_buffer_index


def test_copy_buffer_to_buffer_across_programs() -> None:
    weights = dsl.param(shape=(2,), etype=F4, name="weights")

    train_builder = IrProgramBuilder()
    train_builder.build_sink("out", weights)
    train_program = build_wgpu_program(train_builder.finish())

    eval_builder = IrProgramBuilder()
    eval_builder.build_sink("out", weights)
    eval_program = build_wgpu_program(eval_builder.finish())

    interp = resin_rt_pybind.Interp("wgpu")
    train_program_id = interp.admit(train_program.to_msgpack())
    eval_program_id = interp.admit(eval_program.to_msgpack())

    train_weights_id = param_buffer_index(train_program, "weights")
    eval_weights_id = param_buffer_index(eval_program, "weights")

    payload = struct.pack("<2f", 1.0, 2.0)
    interp.write_buffer(train_program_id, train_weights_id, payload)
    interp.copy_buffer_to_buffer(
        train_program_id,
        train_weights_id,
        eval_program_id,
        eval_weights_id,
    )
    raw = interp.read_buffer(eval_program_id, eval_weights_id)
    assert struct.unpack("<2f", raw) == (1.0, 2.0)