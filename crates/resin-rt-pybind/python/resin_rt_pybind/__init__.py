from .resin_rt_pybind import (
    Interp,
    decode_wgpu_program_msgpack,
    encode_wgpu_program_msgpack,
)

type ProgramId = int
type BufferId = int

__all__ = [
    "BufferId",
    "Interp",
    "ProgramId",
    "decode_wgpu_program_msgpack",
    "encode_wgpu_program_msgpack",
]