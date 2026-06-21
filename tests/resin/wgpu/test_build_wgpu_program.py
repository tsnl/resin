import math
import struct

import resin_rt_pybind

from resin import dsl
from resin.core.etype import F4
from resin.ir import IrProgramBuilder
from resin.wgpu import (
    WgpuProgram,
    build_wgpu_program,
    spell_etype_in_pystruct,
    etype_nbytes,
)


class TestBuildWgpuProgram:
    def test_const_graph_round_trips_msgpack(self) -> None:
        t = dsl.const([1.0, 2.0], etype=F4)
        builder = IrProgramBuilder()
        builder.build_sink("out", t)
        ir_program = builder.finish()

        wgpu_program = build_wgpu_program(ir_program)
        blob = wgpu_program.to_msgpack()
        restored = WgpuProgram.from_msgpack(blob)

        assert len(restored.buffers) == 1
        buffer = restored.buffers[0]
        assert buffer["etype"] == F4
        init = buffer.get("init")
        assert isinstance(init, bytes)
        original_init = wgpu_program.buffers[0].get("init")
        assert isinstance(original_init, bytes)
        assert init == original_init
        assert len(init) == math.prod(buffer["shape"]) * etype_nbytes(buffer["etype"])
        fmt = spell_etype_in_pystruct(buffer["etype"])
        assert struct.unpack(f"<{math.prod(buffer['shape'])}{fmt}", init) == (
            1.0,
            2.0,
        )
        assert restored.sinks["out"] == 0
        assert restored.queue == ()

    def test_msgpack_loadable_by_runtime(self) -> None:
        t = dsl.const([1.0, 2.0], etype=F4)
        builder = IrProgramBuilder()
        builder.build_sink("out", t)
        ir_program = builder.finish()

        wgpu_program = build_wgpu_program(ir_program)
        _ = resin_rt_pybind.WgpuInterp(wgpu_program.to_msgpack())

    def test_elementwise_graph_has_pipeline_and_dispatch(self) -> None:
        t1 = dsl.const([1.0, 2.0], etype=F4)
        t2 = dsl.const([3.0, 4.0], etype=F4)
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
