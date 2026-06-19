"""E2E: compose a simple graph and export a WgpuProgram MessagePack artifact."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# ///

import sys
from pathlib import Path

from resin import dsl
from resin.ir import IrProgramBuilder
from resin.wgpu import WgpuProgram, build_wgpu_program

# TODO: load the exported blob in a Rust game engine via WgpuProgram::from_msgpack
# and run WgpuInterp.


def main() -> None:
    t = dsl.const([1.0, 2.0, 3.0], etype="f4")

    builder = IrProgramBuilder()
    builder.build_sink("out", t)
    ir_program = builder.finish()

    wgpu_program = build_wgpu_program(ir_program)
    blob = wgpu_program.to_msgpack()

    out_path = Path.cwd() / "demo_export.wgpu.msgpack"
    out_path.write_bytes(blob)

    restored = WgpuProgram.from_msgpack(blob)
    assert restored.sinks["out"] == 0
    assert len(restored.buffers) == 1
    assert restored.buffers[0].etype == "f4"

    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
