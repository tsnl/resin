"""Interpreter protocols for executing compiled Resin programs."""

__all__ = [
    "BufferId",
    "Interp",
    "ProgramId",
]

from typing import Protocol

type ProgramId = int
type BufferId = int


class Interp(Protocol):
    """Structural interface implemented by concrete runtimes such as ``resin_rt_pybind.Interp``."""

    def admit(self, program_msgpack: bytes) -> ProgramId: ...

    def program_count(self) -> int: ...

    def run(self, program_id: ProgramId) -> None: ...

    def write_buffer(
        self,
        program_id: ProgramId,
        buffer_id: BufferId,
        data: bytes,
    ) -> None: ...

    def read_buffer(self, program_id: ProgramId, buffer_id: BufferId) -> bytes: ...

    def copy_buffer_to_buffer(
        self,
        src_program_id: ProgramId,
        src_buffer_id: BufferId,
        dst_program_id: ProgramId,
        dst_buffer_id: BufferId,
    ) -> None: ...
