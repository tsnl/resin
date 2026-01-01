from abc import ABC, abstractmethod
from dataclasses import dataclass

from .basic import BaseResource
from .gpu import GpuContext, GpuImage, GpuGraphicsPipeline, GpuDevice


##--------------------------------------------------------------------------------------
## Renderer
##--------------------------------------------------------------------------------------


class Renderer(BaseResource):
    """
    Renderer holds immutable GPU resources for drawing 2D and 3D scenes.
    """

    def __init__(self, *, gpu_device: GpuDevice):
        super().__init__()
        self.gpu_device = gpu_device


class RendererFrame(BaseResource):
    """
    RendererFrame holds mutable resources for a single frame in flight, including a GPU
    image used as a render target.

    Each Renderer may have multiple RendererFrame instances running concurrently to
    maximize GPU utilization, e.g. one per frame in flight for double or triple
    buffering.

    RendererFrame instances can and should be reused across frames, with proper
    synchronization to avoid resource contention.
    """

    renderer: Renderer

    def __init__(self, *, renderer: Renderer):
        super().__init__()
        self.renderer = renderer


@dataclass
class RenderQuad:
    """
    A RenderQuad represents a single textured quad to be drawn on screen.
    """

    image: GpuImage | None
    dst_xywh_px: tuple[int, int, int, int]
    src_xywh_px: tuple[int, int, int, int]
    border_thickness_px: tuple[int, int, int, int]  # TRBL
    color: tuple[float, float, float, float]  # linear RGBA
    border_color: tuple[float, float, float, float]  # linear RGBA
