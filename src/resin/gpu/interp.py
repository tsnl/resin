import math

import wgpu
from wgpu.structs import dataclass

from .kernel import Kernel
from .program import Buffer, Program
from .scalar import stype_nbytes


@dataclass(frozen=True, kw_only=True)
class Interp:
    device: wgpu.GPUDevice
    program: Program
    buffers: tuple[wgpu.GPUBuffer, ...]
    kernels: tuple[wgpu.GPUComputePipeline, ...]


class InterpBuilder:
    device: wgpu.GPUDevice
    buffer_memo: dict[Buffer, wgpu.GPUBuffer]
    kernel_memo: dict[Kernel, wgpu.GPUComputePipeline]

    def __init__(self):
        super().__init__()

    def _build_buffer(self, buffer: Buffer) -> wgpu.GPUBuffer:
        if wgpu_buffer := self.buffer_memo.get(buffer):
            return wgpu_buffer

        wgpu_buffer = self.device.create_buffer(
            size=math.prod(buffer.shape) * stype_nbytes(buffer.stype),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            mapped_at_creation=bool(buffer.init),
        )

        if buffer.init is not None:
            self.device.queue.write_buffer(wgpu_buffer, 0, buffer.init)
            wgpu_buffer.unmap()

        self.buffer_memo[buffer] = wgpu_buffer
        return wgpu_buffer

    def _build_kernel(self, kernel: Kernel) -> wgpu.GPUComputePipeline:
        if wgpu_kernel := self.kernel_memo.get(kernel):
            return wgpu_kernel

        # TODO: pick up from here.
        raise NotImplementedError()
