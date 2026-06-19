"""E2E: compose a simple const graph and interpret it on the GPU."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
#   "resin-rt-pybind",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# resin-rt-pybind = { path = "../crates/resin-rt-pybind", editable = true }
# ///

import struct
import sys

import resin_rt_pybind

from resin import dsl
from resin.ir import IrProgramBuilder
from resin.wgpu import build_wgpu_program


def main() -> None:
    t = dsl.const([1.0, 2.0, 3.0], etype="f4")

    builder = IrProgramBuilder()
    builder.build_sink("out", t)
    ir_program = builder.finish()

    wgpu_program = build_wgpu_program(ir_program)
    interp = resin_rt_pybind.WgpuInterp(wgpu_program.to_msgpack())
    interp.run()

    sink_view_index = wgpu_program.sinks["out"]
    buffer_view = wgpu_program.buffer_views[sink_view_index]
    raw = interp.read_buffer(buffer_view.buffer_index)
    values = struct.unpack("<3f", raw)

    print(f"sink values: {values}", file=sys.stderr)
    assert values == (1.0, 2.0, 3.0)


if __name__ == "__main__":
    main()
