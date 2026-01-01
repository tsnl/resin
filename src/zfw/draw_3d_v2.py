import numpy as np
from .basic import BaseResource

from .gpu import GpuCommandEncoder, GpuDevice, GpuImage, GpuImageMeta


class Draw3dRenderer(BaseResource):
    _gpu_device: "GpuDevice"
    _target_width_px: int
    _target_height_px: int

    def __init__(self, gpu_device: "GpuDevice") -> None:
        super().__init__(parent_resource=None)
        self._gpu_device = gpu_device

    def resize(self, target_width_px: int, target_height_px: int) -> None:
        self._target_width_px = target_width_px
        self._target_height_px = target_height_px


class Draw3dTarget(BaseResource):
    _renderer: "Draw3dRenderer"

    _output: "GpuImage"

    def __init__(self, renderer: "Draw3dRenderer") -> None:
        super().__init__(parent_resource=renderer)

        self._renderer = renderer

        self._output = self._new_output_image()

    @property
    def output(self) -> "GpuImage":
        return self._output

    def resize(self) -> None:
        self._output.dispose()
        self._output = self._new_output_image()

    def _new_output_image(self) -> GpuImage:
        return GpuImage(
            device=self._renderer._gpu_device,
            usages=["color-attachment", "transfer-src", "texture-binding"],
            meta=GpuImageMeta(
                shape=(
                    self._renderer._target_height_px,
                    self._renderer._target_width_px,
                    4,
                ),
                dtype=np.float32,
            ),
        )

    def _record(self, *, command_encoder: "GpuCommandEncoder", todo: object) -> None:
        raise NotImplementedError()
