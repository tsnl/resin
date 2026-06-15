import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jinja2

from resin.gpu.scalar import ScalarType


class Program:
    buffers: list[Buffer]
    buffer_views: list[BufferView]
    tape: list[Kernel]


@dataclass(frozen=True, kw_only=True)
class Buffer:
    shape: tuple[int, ...]
    stype: ScalarType


@dataclass(frozen=True, kw_only=True)
class BufferView:
    buffer: Buffer
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class Kernel:
    template_name: ShaderName
    vars: dict[str, str]
    args: tuple[BufferView, ...]
    output: Buffer


def render_template(template_name: str, template_vars: dict[str, str]) -> str:
    pass
