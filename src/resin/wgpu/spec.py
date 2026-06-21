from __future__ import annotations

import math
import msgpack
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from frozendict import frozendict
from resin_rt_pybind import decode_wgpu_program_msgpack

from resin.core.etype import ElementType, etype_nbytes

SCHEMA_VERSION = 1

__all__ = [
    "SCHEMA_VERSION",
    "WgpuAccessorSpec",
    "WgpuBufferSpec",
    "WgpuBufferViewSpec",
    "WgpuComputePipelineSpec",
    "WgpuCopy",
    "WgpuDispatch",
    "WgpuProgram",
    "WgpuQueueOp",
]

#
# WgpuProgram types (mirrors resin_rt::program)
#


@dataclass(frozen=True)
class WgpuProgram:
    buffers: tuple[WgpuBufferSpec, ...]
    buffer_views: tuple[WgpuBufferViewSpec, ...]
    pipelines: tuple[WgpuComputePipelineSpec, ...]
    queue: tuple[WgpuQueueOp, ...]
    sinks: frozendict[str, int]
    param_buffer_ids: frozendict[int, int] = field(
        default_factory=lambda: frozendict[int, int]()
    )
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "param_buffer_ids": dict(self.param_buffer_ids),
            "sinks": dict(self.sinks),
            "queue": [_queue_op_to_dict(op) for op in self.queue],
            "buffers": [_buffer_to_dict(b) for b in self.buffers],
            "buffer_views": [_buffer_view_to_dict(v) for v in self.buffer_views],
            "pipelines": [_pipeline_to_dict(p) for p in self.pipelines],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WgpuProgram:
        return cls(
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            param_buffer_ids=frozendict(payload.get("param_buffer_ids", {})),
            sinks=frozendict(payload["sinks"]),
            queue=tuple(_queue_op_from_dict(op) for op in payload["queue"]),
            buffers=tuple(_buffer_from_dict(b) for b in payload["buffers"]),
            buffer_views=tuple(
                _buffer_view_from_dict(v) for v in payload["buffer_views"]
            ),
            pipelines=tuple(_pipeline_from_dict(p) for p in payload["pipelines"]),
        )

    def to_msgpack(self) -> bytes:
        blob = cast(bytes, msgpack.packb(self.to_dict(), use_bin_type=True))
        decode_wgpu_program_msgpack(blob)
        return blob

    @classmethod
    def from_msgpack(cls, data: bytes) -> WgpuProgram:
        payload = msgpack.unpackb(data, raw=False)
        return cls.from_dict(payload)


@dataclass(frozen=True)
class WgpuBufferSpec:
    """Buffer metadata in a :class:`WgpuProgram`."""

    shape: tuple[int, ...]
    etype: ElementType | str
    init: bytes | None = None
    readonly: bool = False

    def __post_init__(self) -> None:
        if self.init is None:
            return
        expected = math.prod(self.shape) * etype_nbytes(self.etype)
        if len(self.init) != expected:
            raise ValueError(
                f"buffer init size {len(self.init)} != expected {expected} "
                f"for shape {self.shape} and etype {self.etype!r}"
            )


@dataclass(frozen=True)
class WgpuBufferViewSpec:
    buffer_index: int
    accessor: WgpuAccessorSpec


@dataclass(frozen=True)
class WgpuAccessorSpec:
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]


@dataclass(frozen=True)
class WgpuComputePipelineSpec:
    wgsl: str
    dispatch_size: tuple[int, int, int]
    num_arg_bindings: int
    clear_output_before_dispatch: bool = False
    entry_point: str = "main"


@dataclass(frozen=True)
class WgpuDispatch:
    pipeline_index: int
    arg_buffer_view_indices: tuple[int, ...]
    output_buffer_index: int


@dataclass(frozen=True)
class WgpuCopy:
    source_buffer_view_index: int
    output_buffer_index: int


type WgpuQueueOp = WgpuDispatch | WgpuCopy


def _buffer_to_dict(spec: WgpuBufferSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "shape": list(spec.shape),
        "etype": spec.etype,
        "readonly": spec.readonly,
    }
    if spec.init is not None:
        payload["init"] = spec.init
    return payload


def _buffer_from_dict(payload: dict[str, Any]) -> WgpuBufferSpec:
    init = payload.get("init")
    if isinstance(init, list):
        init = bytes(init)
    return WgpuBufferSpec(
        shape=tuple(payload["shape"]),
        etype=payload["etype"],
        init=init,
        readonly=payload.get("readonly", False),
    )


def _accessor_to_dict(spec: WgpuAccessorSpec) -> dict[str, Any]:
    return {
        "offset": spec.offset,
        "shape": list(spec.shape),
        "pitch": list(spec.pitch),
    }


def _accessor_from_dict(payload: dict[str, Any]) -> WgpuAccessorSpec:
    return WgpuAccessorSpec(
        offset=payload["offset"],
        shape=tuple(payload["shape"]),
        pitch=tuple(payload["pitch"]),
    )


def _buffer_view_to_dict(spec: WgpuBufferViewSpec) -> dict[str, Any]:
    return {
        "buffer_index": spec.buffer_index,
        "accessor": _accessor_to_dict(spec.accessor),
    }


def _buffer_view_from_dict(payload: dict[str, Any]) -> WgpuBufferViewSpec:
    return WgpuBufferViewSpec(
        buffer_index=payload["buffer_index"],
        accessor=_accessor_from_dict(payload["accessor"]),
    )


def _pipeline_to_dict(spec: WgpuComputePipelineSpec) -> dict[str, Any]:
    return {
        "wgsl": spec.wgsl,
        "entry_point": spec.entry_point,
        "dispatch_size": list(spec.dispatch_size),
        "num_arg_bindings": spec.num_arg_bindings,
        "clear_output_before_dispatch": spec.clear_output_before_dispatch,
    }


def _pipeline_from_dict(payload: dict[str, Any]) -> WgpuComputePipelineSpec:
    return WgpuComputePipelineSpec(
        wgsl=payload["wgsl"],
        entry_point=payload.get("entry_point", "main"),
        dispatch_size=tuple(payload["dispatch_size"]),
        num_arg_bindings=payload["num_arg_bindings"],
        clear_output_before_dispatch=payload.get("clear_output_before_dispatch", False),
    )


def _queue_op_to_dict(op: WgpuQueueOp) -> dict[str, Any]:
    match op:
        case WgpuDispatch():
            return {
                "kind": "dispatch",
                "pipeline_index": op.pipeline_index,
                "arg_buffer_view_indices": list(op.arg_buffer_view_indices),
                "output_buffer_index": op.output_buffer_index,
            }
        case WgpuCopy():
            return {
                "kind": "copy",
                "source_buffer_view_index": op.source_buffer_view_index,
                "output_buffer_index": op.output_buffer_index,
            }
        case _:
            raise ValueError(f"unsupported queue op: {op!r}")


def _queue_op_from_dict(payload: dict[str, Any]) -> WgpuQueueOp:
    kind: Literal["dispatch", "copy"] = payload["kind"]
    match kind:
        case "dispatch":
            return WgpuDispatch(
                pipeline_index=payload["pipeline_index"],
                arg_buffer_view_indices=tuple(payload["arg_buffer_view_indices"]),
                output_buffer_index=payload["output_buffer_index"],
            )
        case "copy":
            return WgpuCopy(
                source_buffer_view_index=payload["source_buffer_view_index"],
                output_buffer_index=payload["output_buffer_index"],
            )
        case _:
            raise ValueError(f"unsupported queue op kind: {kind!r}")