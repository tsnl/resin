"""
TODO: Compiler should emit C code that then depends on the `wgpu-native` library in `deps/`.
"""

from dataclasses import dataclass

from . import graph as rg
from . import tree as rt

#
# Compiler
#


class Compiler:
    pass


#
# Program
#


@dataclass(frozen=True)
class Program:
    routines: dict[str, Routine]
    buffers: dict[str, Buffer]


@dataclass(frozen=True)
class Routine:
    sinks: rt.Tree[rg.Node]
    binds: dict[rg.ParamNode, str]


@dataclass(frozen=True)
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: rg.DType


#
# ProgramBuilder
#


class ProgramBuilder:
    def __init__(self) -> None:
        self._routines: dict[str, Routine] = {}
        self._buffers: dict[str, Buffer] = {}

    def add_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: rg.DType,
    ) -> Buffer:
        buffer = Buffer(name=name, shape=shape, dtype=dtype)
        self._buffers[name] = buffer
        return buffer

    def add_routine(
        self,
        name: str,
        sinks: rt.Tree[rg.Node],
        binds: dict[rg.ParamNode, str],
    ) -> Routine:
        routine = Routine(sinks=sinks, binds=binds)
        self._routines[name] = routine
        return routine

    def build(self) -> Program:
        return Program(routines=dict(self._routines), buffers=dict(self._buffers))
