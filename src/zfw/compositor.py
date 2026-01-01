"""Minimal fullscreen compositor used for presenting render targets."""

from collections import OrderedDict

from .basic import BaseResource
from .bundled_data import BUNDLED_DATA_PATH
from .gpu import (
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuGraphicsPipeline,
    GpuImage,
    GpuPipelineLayout,
    GpuSampler,
    GpuShader,
    GpuCommandEncoder,
)


class Compositor(BaseResource):
    """Simple pass-through compositor that blits a texture to an output image."""

    _gpu_device: GpuDevice
    _viewport_width: int
    _viewport_height: int
    _vk_color_format: int

    _descriptor_set_layout: GpuDescriptorSetLayout
    _pipeline_layout: GpuPipelineLayout
    _vertex_shader: GpuShader
    _fragment_shader: GpuShader
    _pipeline: GpuGraphicsPipeline
    _sampler: GpuSampler

    def __init__(
        self,
        *,
        gpu_device: GpuDevice,
        viewport_width: int,
        viewport_height: int,
        vk_color_format: int,
        parent_resource: BaseResource | None = None,
    ) -> None:
        super().__init__(parent_resource=parent_resource)

        self._gpu_device = gpu_device
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._vk_color_format = vk_color_format

        self._descriptor_set_layout = GpuDescriptorSetLayout(
            device=self._gpu_device,
            bindings=OrderedDict(
                {
                    "sourceTexture": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "sourceSampler": GpuDescriptorSetLayoutBinding(
                        type="sampler",
                        stages=["fragment"],
                    ),
                }.items()
            ),
        )
        self._pipeline_layout = GpuPipelineLayout(
            device=self._gpu_device,
            descriptor_set_layouts=[self._descriptor_set_layout],
        )
        self._vertex_shader = GpuShader(
            device=self._gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/compositor.vert.spv"),
            stage="vertex",
        )
        self._fragment_shader = GpuShader(
            device=self._gpu_device,
            spirv_path=(BUNDLED_DATA_PATH / "shaders/compositor.frag.spv"),
            stage="fragment",
        )

        self._pipeline = self._build_pipeline()
        self._sampler = GpuSampler(
            device=self._gpu_device,
            min_filter="nearest",
            mag_filter="nearest",
        )

    @property
    def descriptor_set_layout(self) -> GpuDescriptorSetLayout:
        return self._descriptor_set_layout

    @property
    def sampler(self) -> GpuSampler:
        return self._sampler

    def resize(
        self,
        *,
        viewport_width: int,
        viewport_height: int,
        color_format: str | None = None,
    ) -> None:
        """Recreate the pipeline when output size or format changes."""

        color_format = color_format or self._vk_color_format
        if (
            viewport_width == self._viewport_width
            and viewport_height == self._viewport_height
            and color_format == self._vk_color_format
        ):
            return

        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._vk_color_format = color_format

        self._gpu_device.wait_idle()
        self._pipeline.dispose()
        self._pipeline = self._build_pipeline()

    def record(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        output_image: GpuImage,
        input: "CompositorInput",
    ) -> None:
        """Record the fullscreen blit for a given input into the output image."""

        command_encoder.transition_image_layout(
            image=output_image,
            layout="color-attachment-optimal",
        )
        command_encoder.transition_image_layout(
            image=input.image,
            layout="texture-binding",
        )

        with command_encoder.render(
            color_attachment=output_image,
            clear_color="black",
        ) as rp:
            rp.bind_pipeline(pipeline=self._pipeline)
            rp.bind_descriptor_set(set_=input.descriptor_set, set_index=0)
            rp.draw(vertex_count=3, first_instance=0, instance_count=1)

    def _build_pipeline(self) -> GpuGraphicsPipeline:
        return GpuGraphicsPipeline(
            device=self._gpu_device,
            vertex_shader=self._vertex_shader,
            fragment_shader=self._fragment_shader,
            enable_depth_test=False,
            enable_alpha_blending=False,
            viewport_width=self._viewport_width,
            viewport_height=self._viewport_height,
            layout=self._pipeline_layout,
            vk_color_format=self._vk_color_format,
        )

    def _on_dispose(self) -> None:
        self._sampler.dispose()
        self._pipeline.dispose()
        self._fragment_shader.dispose()
        self._vertex_shader.dispose()
        self._pipeline_layout.dispose()
        self._descriptor_set_layout.dispose()


class CompositorInput(BaseResource):
    """Tracks a single sampled image bound for compositing."""

    _compositor: Compositor
    _image: GpuImage
    _descriptor_set: GpuDescriptorSet

    def __init__(
        self,
        *,
        compositor: Compositor,
        image: GpuImage,
    ) -> None:
        super().__init__(parent_resource=compositor)
        self._compositor = compositor
        self._image = image
        self._descriptor_set = self._build_descriptor_set()

    @property
    def image(self) -> GpuImage:
        return self._image

    @property
    def descriptor_set(self) -> GpuDescriptorSet:
        return self._descriptor_set

    def set_image(self, image: GpuImage) -> None:
        if image is self._image:
            return
        self._image = image
        self._descriptor_set.dispose()
        self._descriptor_set = self._build_descriptor_set()

    def _build_descriptor_set(self) -> GpuDescriptorSet:
        return GpuDescriptorSet(
            device=self._compositor._gpu_device,
            bindings={
                "sourceTexture": self._image,
                "sourceSampler": self._compositor.sampler,
            },
            layout=self._compositor.descriptor_set_layout,
        )

    def _on_dispose(self) -> None:
        self._descriptor_set.dispose()
