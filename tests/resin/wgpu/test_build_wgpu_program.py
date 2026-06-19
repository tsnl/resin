import math
import struct

import resin_rt_pybind

from resin import dsl
from resin.ir import IrProgramBuilder
from resin.wgpu import (
    WgpuProgram,
    build_wgpu_program,
    spell_etype_in_pystruct,
    etype_nbytes,
)


class TestBuildWgpuProgram:
    def test_const_graph_round_trips_msgpack(self) -> None:
        t = dsl.const([1.0, 2.0], etype="f4")
        builder = IrProgramBuilder()
        builder.build_sink("out", t)
        ir_program = builder.finish()

        wgpu_program = build_wgpu_program(ir_program)
        blob = wgpu_program.to_msgpack()
        restored = WgpuProgram.from_msgpack(blob)

        assert len(restored.buffers) == 1
        buffer = restored.buffers[0]
        assert buffer.etype == "f4"
        assert buffer.init is not None
        assert buffer.init == wgpu_program.buffers[0].init
        assert len(buffer.init) == math.prod(buffer.shape) * etype_nbytes(
            buffer.etype
        )
        fmt = spell_etype_in_pystruct(buffer.etype)
        assert struct.unpack(f"<{math.prod(buffer.shape)}{fmt}", buffer.init) == (
            1.0,
            2.0,
        )
        assert restored.sinks["out"] == 0
        assert restored.queue == ()

    def test_msgpack_loadable_by_runtime(self) -> None:
        t = dsl.const([1.0, 2.0], etype="f4")
        builder = IrProgramBuilder()
        builder.build_sink("out", t)
        ir_program = builder.finish()

        wgpu_program = build_wgpu_program(ir_program)
        resin_rt_pybind.WgpuInterp(wgpu_program.to_msgpack())

    def test_elementwise_graph_has_pipeline_and_dispatch(self) -> None:
        t1 = dsl.const([1.0, 2.0], etype="f4")
        t2 = dsl.const([3.0, 4.0], etype="f4")
        out = t1 + t2

        builder = IrProgramBuilder()
        builder.build_sink("out", out)
        ir_program = builder.finish()

        wgpu_program = build_wgpu_program(ir_program)
        payload = wgpu_program.to_dict()

        assert len(payload["pipelines"]) == 1
        assert len(payload["queue"]) == 1
        assert payload["queue"][0]["kind"] == "dispatch"
        assert "fn main" in payload["pipelines"][0]["wgsl"]

    def test_dense_gather_uses_copy_queue_op(self) -> None:
        x = dsl.param(shape=(3,), etype="f4")
        builder = IrProgramBuilder()
        builder.build_sink("out", x.copy())
        wgpu_program = build_wgpu_program(builder.finish())

        assert wgpu_program.pipelines == ()
        assert len(wgpu_program.queue) == 1
        assert wgpu_program.to_dict()["queue"][0] == {
            "kind": "copy",
            "source_buffer_view_index": 0,
            "output_buffer_index": 1,
        }

    def test_sparse_gather_uses_compute_pipeline(self) -> None:
        t = dsl.const([1, 2, 3, 4, 5, 6], etype="f4")
        builder = IrProgramBuilder()
        builder.build_sink("out", t[::2].copy())
        wgpu_program = build_wgpu_program(builder.finish())

        assert len(wgpu_program.pipelines) == 1
        pipeline = wgpu_program.pipelines[0]
        assert wgpu_program.to_dict()["queue"][0]["kind"] == "dispatch"
        assert "fn main" in pipeline.wgsl