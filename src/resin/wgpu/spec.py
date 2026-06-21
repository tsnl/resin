import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, cast

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


class WgpuAccessorSpec(TypedDict):
    offset: int
    shape: list[int]
    pitch: list[int]


class WgpuBufferSpec(TypedDict):
    shape: list[int]
    etype: ElementType | str
    readonly: NotRequired[bool]
    init: NotRequired[bytes | list[int]]


class WgpuBufferViewSpec(TypedDict):
    buffer_index: int
    accessor: WgpuAccessorSpec


class WgpuComputePipelineSpec(TypedDict):
    wgsl: str
    dispatch_size: list[int]
    num_arg_bindings: int
    entry_point: NotRequired[str]
    clear_output_before_dispatch: NotRequired[bool]


class WgpuDispatch(TypedDict):
    kind: Literal["dispatch"]
    pipeline_index: int
    arg_buffer_view_indices: list[int]
    output_buffer_index: int


class WgpuCopy(TypedDict):
    kind: Literal["copy"]
    source_buffer_view_index: int
    output_buffer_index: int


type WgpuQueueOp = WgpuDispatch | WgpuCopy


class WgpuProgramSpec(TypedDict):
    sinks: dict[str, int]
    queue: list[WgpuQueueOp]
    buffers: list[WgpuBufferSpec]
    buffer_views: list[WgpuBufferViewSpec]
    pipelines: list[WgpuComputePipelineSpec]
    schema_version: NotRequired[int]
    param_buffer_ids: NotRequired[dict[int, int]]


_msgpack = cast(object, importlib.import_module("msgpack"))


def _msgpack_packb(value: object, *, use_bin_type: bool) -> bytes:
    packb = cast(Callable[..., bytes], getattr(_msgpack, "packb"))
    return packb(value, use_bin_type=use_bin_type)


def _msgpack_unpackb(data: bytes, *, raw: bool) -> object:
    unpackb = cast(Callable[..., object], getattr(_msgpack, "unpackb"))
    return unpackb(data, raw=raw)


def _validate_buffer_spec(spec: WgpuBufferSpec) -> None:
    init = spec.get("init")
    if init is None:
        return
    if isinstance(init, list):
        init = bytes(init)
    expected = math.prod(spec["shape"]) * etype_nbytes(spec["etype"])
    if len(init) != expected:
        raise ValueError(
            f"buffer init size {len(init)} != expected {expected} "
            + f"for shape {spec['shape']} and etype {spec['etype']!r}"
        )


def _normalize_buffer_spec(spec: WgpuBufferSpec) -> WgpuBufferSpec:
    init = spec.get("init")
    if isinstance(init, list):
        return {**spec, "init": bytes(init)}
    return spec


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

    def __post_init__(self) -> None:
        for buffer in self.buffers:
            _validate_buffer_spec(buffer)

    def to_dict(self) -> WgpuProgramSpec:
        return {
            "schema_version": self.schema_version,
            "param_buffer_ids": dict(self.param_buffer_ids),
            "sinks": dict(self.sinks),
            "queue": list(self.queue),
            "buffers": list(self.buffers),
            "buffer_views": list(self.buffer_views),
            "pipelines": list(self.pipelines),
        }

    @classmethod
    def from_dict(cls, payload: WgpuProgramSpec) -> WgpuProgram:
        return cls(
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            param_buffer_ids=frozendict(payload.get("param_buffer_ids", {})),
            sinks=frozendict(payload["sinks"]),
            queue=tuple(payload["queue"]),
            buffers=tuple(_normalize_buffer_spec(b) for b in payload["buffers"]),
            buffer_views=tuple(payload["buffer_views"]),
            pipelines=tuple(payload["pipelines"]),
        )

    def to_msgpack(self) -> bytes:
        blob = _msgpack_packb(self.to_dict(), use_bin_type=True)
        decode_wgpu_program_msgpack(blob)
        return blob

    @classmethod
    def from_msgpack(cls, data: bytes) -> WgpuProgram:
        payload = cast(WgpuProgramSpec, _msgpack_unpackb(data, raw=False))
        return cls.from_dict(payload)
