"""Compile DSL graphs into runtime artifacts with stable param binding."""

__all__ = [
    "CompiledProgram",
    "ParamBinding",
    "ParamEntry",
    "ParamManifest",
    "compile_program",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from frozendict import frozendict

import resin.dsl as dsl
from resin.core.pytree import PyTree, flatten_pytree_paths
from resin.core.state import register_named_params, sink_tree
from resin.interp import Interp, ProgramId
from resin.ir.ir import IrProgramBuilder
from resin.wgpu import WgpuProgram, build_wgpu_program, param_buffer_index
from resin.wgpu.codegen import WgslKernelConfig


@dataclass(frozen=True, kw_only=True)
class ParamEntry:
    name: str
    shape: tuple[int, ...]
    etype: str
    buffer_index: int


@dataclass(frozen=True, kw_only=True)
class ParamManifest:
    entries: tuple[ParamEntry, ...]

    def buffer_index(self, name: str) -> int:
        for entry in self.entries:
            if entry.name == name:
                return entry.buffer_index
        raise KeyError(name)

    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)


@dataclass(frozen=True, kw_only=True)
class CompiledProgram:
    artifact: WgpuProgram
    manifest: ParamManifest
    view_to_name: frozendict[dsl.View, str]

    def admit(self, interp: Interp) -> ProgramId:
        return interp.admit(self.artifact.to_msgpack())

    def binding(self, interp: Interp, program_id: ProgramId) -> ParamBinding:
        return ParamBinding(
            interp=interp,
            program_id=program_id,
            artifact=self.artifact,
            manifest=self.manifest,
            view_to_name=self.view_to_name,
        )


@dataclass(frozen=True, kw_only=True)
class ParamBinding:
    interp: Interp
    program_id: ProgramId
    artifact: WgpuProgram
    manifest: ParamManifest
    view_to_name: frozendict[dsl.View, str]

    def write(self, data: Mapping[str, bytes]) -> None:
        for name, payload in data.items():
            buffer_id = param_buffer_index(self.artifact, name)
            self.interp.write_buffer(self.program_id, buffer_id, payload)

    def write_view(self, view: dsl.View, data: bytes) -> None:
        name = self.view_to_name[view]
        self.write({name: data})

    def read_sink(self, name: str) -> bytes:
        view_index = self.artifact.sinks[name]
        buffer_view = self.artifact.buffer_views[view_index]
        buffer_id = buffer_view["buffer_index"]
        return self.interp.read_buffer(self.program_id, buffer_id)

    def commit(
        self,
        *,
        from_prefix: str,
        to_prefix: str,
        tree: PyTree[dsl.View],
    ) -> None:
        for rel_path, _view in flatten_pytree_paths(tree):
            sink_name = (
                f"{from_prefix}.{rel_path}" if from_prefix and rel_path else from_prefix or rel_path
            )
            param_name = (
                f"{to_prefix}.{rel_path}" if to_prefix and rel_path else to_prefix or rel_path
            )
            src_buffer = self._sink_buffer_index(sink_name)
            dst_buffer = param_buffer_index(self.artifact, param_name)
            self.interp.copy_buffer_to_buffer(
                self.program_id,
                src_buffer,
                self.program_id,
                dst_buffer,
            )

    def _sink_buffer_index(self, sink_name: str) -> int:
        view_index = self.artifact.sinks[sink_name]
        return self.artifact.buffer_views[view_index]["buffer_index"]


def _build_sinks(
    builder: IrProgramBuilder,
    sinks: Mapping[str, dsl.View | PyTree[dsl.View]],
) -> None:
    for key, value in sinks.items():
        if isinstance(value, dsl.View):
            builder.build_sink(key, value)
        else:
            sink_tree(builder, key, cast(PyTree[dsl.View], value))


def _manifest_for(
    artifact: WgpuProgram,
    flat_params: Mapping[str, dsl.View],
) -> ParamManifest:
    entries = tuple(
        ParamEntry(
            name=name,
            shape=view.shape,
            etype=view.etype,
            buffer_index=param_buffer_index(artifact, name),
        )
        for name, view in flat_params.items()
    )
    return ParamManifest(entries=entries)


def compile_program(
    *,
    sinks: Mapping[str, dsl.View | PyTree[dsl.View]],
    params: Mapping[str, dsl.View | PyTree[dsl.View]] | None = None,
    wgsl_kernel_config: WgslKernelConfig | None = None,
) -> CompiledProgram:
    builder = IrProgramBuilder()
    flat_params = (
        register_named_params(builder, params) if params is not None else {}
    )
    _build_sinks(builder, sinks)
    artifact = build_wgpu_program(builder.finish(), wgsl_kernel_config=wgsl_kernel_config)
    manifest = _manifest_for(artifact, flat_params)
    view_to_name = frozendict[dsl.View, str](
        {view: name for name, view in flat_params.items()}
    )
    return CompiledProgram(
        artifact=artifact,
        manifest=manifest,
        view_to_name=view_to_name,
    )