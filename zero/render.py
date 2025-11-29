"""
2D Renderer for GPU-accelerated quad rendering.

This module provides a GPU-accelerated 2D renderer that renders quads and textured
quads with an orthographic projection. It uses instanced rendering to efficiently
draw many quads in a single draw call per batch.

Quad corner ordering: top-left, top-right, bottom-right, bottom-left (like CSS)
Border side ordering: top, right, bottom, left (like CSS)
"""

__all__ = [
    "RenderContext",
    "Renderer",
    "DrawQuadBatch",
    "DrawQuadList",
]

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .core import BaseResource
from .gpu import (
    GpuBuffer,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuDescriptorBinding,
    GpuDescriptorSet,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
    GpuRenderPassCommandEncoder,
)
from .bundled_data import BUNDLED_DATA_PATH


@dataclass
class DrawQuadBatch:
    """A batch of quads to render with a single draw call.

    Each batch has an optional atlas texture. All quads in the batch share
    the same atlas. If no atlas is provided, a default 1x1 white texture is used.

    Attributes:
        atlas: Optional texture atlas for this batch. If None, uses a default white texture.
        offset_px: Quad corner positions in pixels, shape (N, 4, 2), dtype=uint32.
                   Corner order: top-left, top-right, bottom-right, bottom-left.
        texcoord_px: Texture coordinates in pixels, shape (N, 4, 2), dtype=uint32.
                     Corner order: top-left, top-right, bottom-right, bottom-left.
        tint_color: RGBA tint color per quad, shape (N, 4), dtype=uint32.
                    Each component is in range [0, 255].
        border_color: RGBA border color per quad, shape (N, 4), dtype=uint32.
                      Each component is in range [0, 255].
        border_thickness_px: Border thickness per quad, shape (N, 4), dtype=uint32.
                             Order: top, right, bottom, left (like CSS).
    """

    atlas: GpuImage | None = None
    offset_px: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4, 2), dtype=torch.uint32)
    )
    texcoord_px: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4, 2), dtype=torch.uint32)
    )
    tint_color: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4), dtype=torch.uint32)
    )
    border_color: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4), dtype=torch.uint32)
    )
    border_thickness_px: torch.Tensor = field(
        default_factory=lambda: torch.zeros((0, 4), dtype=torch.uint32)
    )

    @property
    def quad_count(self) -> int:
        """Return the number of quads in this batch."""
        return self.offset_px.shape[0]

    def validate(self) -> None:
        """Validate tensor shapes and types."""
        n = self.quad_count
        assert self.offset_px.shape == (n, 4, 2), (
            f"offset_px shape mismatch: {self.offset_px.shape}"
        )
        assert self.texcoord_px.shape == (n, 4, 2), (
            f"texcoord_px shape mismatch: {self.texcoord_px.shape}"
        )
        assert self.tint_color.shape == (n, 4), (
            f"tint_color shape mismatch: {self.tint_color.shape}"
        )
        assert self.border_color.shape == (n, 4), (
            f"border_color shape mismatch: {self.border_color.shape}"
        )
        assert self.border_thickness_px.shape == (n, 4), (
            f"border_thickness_px shape mismatch: {self.border_thickness_px.shape}"
        )
        assert self.offset_px.dtype == torch.uint32
        assert self.texcoord_px.dtype == torch.uint32
        assert self.tint_color.dtype == torch.uint32
        assert self.border_color.dtype == torch.uint32
        assert self.border_thickness_px.dtype == torch.uint32


class R2dBatchGpuResources(BaseResource):
    """GPU resources allocated for a single batch."""

    # Device-local buffers
    offset_buffer: GpuBuffer
    texcoord_buffer: GpuBuffer
    tint_buffer: GpuBuffer
    border_color_buffer: GpuBuffer
    border_thickness_buffer: GpuBuffer
    uniform_buffer: GpuBuffer
    descriptor_set: GpuDescriptorSet

    # Staging buffers (host-visible, for upload)
    offset_staging: GpuBuffer
    texcoord_staging: GpuBuffer
    tint_staging: GpuBuffer
    border_color_staging: GpuBuffer
    border_thickness_staging: GpuBuffer
    uniform_staging: GpuBuffer

    def __init__(
        self,
        *,
        parent: BaseResource | None,
        offset_buffer: GpuBuffer,
        texcoord_buffer: GpuBuffer,
        tint_buffer: GpuBuffer,
        border_color_buffer: GpuBuffer,
        border_thickness_buffer: GpuBuffer,
        uniform_buffer: GpuBuffer,
        descriptor_set: GpuDescriptorSet,
        offset_staging: GpuBuffer,
        texcoord_staging: GpuBuffer,
        tint_staging: GpuBuffer,
        border_color_staging: GpuBuffer,
        border_thickness_staging: GpuBuffer,
        uniform_staging: GpuBuffer,
    ) -> None:
        super().__init__(parent=parent)

        self.offset_buffer = offset_buffer
        self.texcoord_buffer = texcoord_buffer
        self.tint_buffer = tint_buffer
        self.border_color_buffer = border_color_buffer
        self.border_thickness_buffer = border_thickness_buffer
        self.uniform_buffer = uniform_buffer
        self.descriptor_set = descriptor_set

        self.offset_staging = offset_staging
        self.texcoord_staging = texcoord_staging
        self.tint_staging = tint_staging
        self.border_color_staging = border_color_staging
        self.border_thickness_staging = border_thickness_staging
        self.uniform_staging = uniform_staging

    def _on_dispose(self) -> None:
        self.offset_buffer.dispose()
        self.texcoord_buffer.dispose()
        self.tint_buffer.dispose()
        self.border_color_buffer.dispose()
        self.border_thickness_buffer.dispose()
        self.uniform_buffer.dispose()
        self.descriptor_set.dispose()

        self.offset_staging.dispose()
        self.texcoord_staging.dispose()
        self.tint_staging.dispose()
        self.border_color_staging.dispose()
        self.border_thickness_staging.dispose()
        self.uniform_staging.dispose()


@dataclass
class DrawQuadList:
    """A list of quad batches to render.

    Each batch corresponds to one Vulkan draw call. No two batches in a QuadList
    should share the same atlas (though this is not strictly enforced).

    Attributes:
        batches: List of R2dQuadBatch instances to render in order.
    """

    batches: list[DrawQuadBatch] = field(default_factory=list)

    def add_batch(self, batch: DrawQuadBatch) -> None:
        """Add a batch to the list."""
        self.batches.append(batch)

    @property
    def total_quads(self) -> int:
        """Return the total number of quads across all batches."""
        return sum(batch.quad_count for batch in self.batches)


class RenderContext(BaseResource):
    """Context for managing 2D renderer resources.

    This context holds references to the GPU device and provides factory
    methods for creating Renderer2d instances.
    """

    def __init__(self, *, device: GpuDevice) -> None:
        super().__init__()
        self.device = device

    def _on_dispose(self) -> None:
        pass

    def create_renderer(
        self,
        *,
        framebuffer_width: int,
        framebuffer_height: int,
        max_quads_per_batch: int = 65536,
    ) -> "Renderer":
        """Create a new Renderer2d instance.

        Args:
            framebuffer_width: Width of the target framebuffer in pixels.
            framebuffer_height: Height of the target framebuffer in pixels.
            max_quads_per_batch: Maximum number of quads per batch (for buffer sizing).

        Returns:
            A new Renderer2d instance.
        """
        return Renderer(
            context=self,
            framebuffer_width=framebuffer_width,
            framebuffer_height=framebuffer_height,
            max_quads_per_batch=max_quads_per_batch,
        )


Renderer2dResource = BaseResource


class Renderer(Renderer2dResource):
    """
    GPU-accelerated 2D quad renderer.

    This renderer uses instanced rendering to efficiently draw many quads.
    It supports:
    -   Textured quads with atlas sampling
    -   Per-quad tint colors
    -   Per-quad borders with configurable thickness and color
    -   Orthographic projection based on framebuffer size

    The renderer uses a single graphics pipeline with the following bindings:
    -   Binding 0: Combined image sampler (atlas texture)
    -   Binding 1: Uniform buffer (framebuffer size, atlas size)
    -   Binding 2-6: Storage buffers (per-instance quad data)
    """

    def __init__(
        self,
        *,
        context: RenderContext,
        framebuffer_width: int,
        framebuffer_height: int,
        max_quads_per_batch: int,
    ) -> None:
        super().__init__(parent=context)

        self.device = context.device
        self.framebuffer_width = framebuffer_width
        self.framebuffer_height = framebuffer_height
        self.max_quads_per_batch = max_quads_per_batch

        # Find shader directory
        self._shader_dir = BUNDLED_DATA_PATH / "shaders" / "zero"

        # Per-batch GPU resources (created during prepare())
        self._batch_resources: list[R2dBatchGpuResources] = []

        # Create resources
        self._create_default_texture()
        self._create_sampler()
        self._create_descriptor_resources()
        self._create_pipeline()

    def _on_dispose(self) -> None:
        pass

    def _create_default_texture(self) -> None:
        """Create a 1x1 white texture for use when no atlas is provided."""
        self.default_texture = self.device.create_image(
            usages=["texture-binding"],
            meta=GpuImageMeta(shape=(1, 1, 4), dtype=torch.uint8),
        )

        # Upload white pixel - create a (1, 1, 4) tensor for a 1x1 RGBA image
        white_pixel = torch.zeros((1, 1, 4), dtype=torch.uint8)
        white_pixel[0, 0, :] = torch.tensor([255, 255, 255, 255], dtype=torch.uint8)

        staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=self.default_texture.meta.into_buffer_meta(),
        )
        staging.write(data=white_pixel.contiguous())

        cmd = self.device.create_command_encoder(queue_type="transfer")
        cmd.transition_image_layout(image=self.default_texture, layout="copy-dst")
        cmd.copy_buffer_to_image(src=staging, dst=self.default_texture)
        cmd.transition_image_layout(
            image=self.default_texture, layout="texture-binding"
        )
        cmd.submit().wait()

    def _create_sampler(self) -> None:
        """Create a sampler for texture sampling."""
        self.sampler = self.device.create_sampler(
            mag_filter="nearest",
            min_filter="nearest",
            address_mode="clamp-to-edge",
        )

    def _create_descriptor_resources(self) -> None:
        """Create descriptor set layout and pool."""
        # Descriptor set layout
        self.descriptor_set_layout = self.device.create_descriptor_set_layout(
            bindings=[
                GpuDescriptorBinding(
                    binding=0,
                    descriptor_type="combined-image-sampler",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=1,
                    descriptor_type="uniform-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=2,
                    descriptor_type="storage-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=3,
                    descriptor_type="storage-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=4,
                    descriptor_type="storage-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=5,
                    descriptor_type="storage-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
                GpuDescriptorBinding(
                    binding=6,
                    descriptor_type="storage-buffer",
                    count=1,
                    stages=["vertex", "fragment"],
                ),
            ]
        )

    def _create_pipeline(self) -> None:
        """Create the graphics pipeline."""
        # Load shaders
        self.vertex_shader = self.device.create_shader(
            spirv_path=self._shader_dir / "render2d.vert.spv",
            stage="vertex",
        )
        self.fragment_shader = self.device.create_shader(
            spirv_path=self._shader_dir / "render2d.frag.spv",
            stage="fragment",
        )

        # Create pipeline layout
        self.pipeline_layout = self.device.create_pipeline_layout(
            descriptor_set_layouts=[self.descriptor_set_layout],
        )

        # Create pipeline
        self.pipeline = self.device.create_pipeline(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader,
            vk_color_format=37,  # VK_FORMAT_R8G8B8A8_UNORM
            viewport_width=self.framebuffer_width,
            viewport_height=self.framebuffer_height,
            layout=self.pipeline_layout,
        )

    def _create_batch_resources(
        self, batch: DrawQuadBatch, atlas: GpuImage
    ) -> R2dBatchGpuResources:
        """Create GPU resources for a single batch."""
        n = max(batch.quad_count, 1)  # At least 1 element for valid buffers

        # Create device-local storage buffers for this batch
        offset_buffer = self.device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(element_count=n * 4 * 2, element_dtype=torch.uint32),
        )
        texcoord_buffer = self.device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(element_count=n * 4 * 2, element_dtype=torch.uint32),
        )
        tint_buffer = self.device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        border_color_buffer = self.device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        border_thickness_buffer = self.device.create_buffer(
            usages=["storage", "copy-dst"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        uniform_buffer = self.device.create_buffer(
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(element_count=4, element_dtype=torch.uint32),
        )

        # Create staging buffers for this batch
        offset_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=n * 4 * 2, element_dtype=torch.uint32),
        )
        texcoord_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=n * 4 * 2, element_dtype=torch.uint32),
        )
        tint_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        border_color_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        border_thickness_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=n * 4, element_dtype=torch.uint32),
        )
        uniform_staging = self.device.create_buffer(
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=4, element_dtype=torch.uint32),
        )

        # Create descriptor set for this batch
        descriptor_set = self.device.create_descriptor_set(
            layout=self.descriptor_set_layout,
            bindings=[
                GpuDescriptorBinding(
                    binding=0,
                    descriptor_type="combined-image-sampler",
                    count=1,
                    image=atlas,
                    sampler=self.sampler,
                ),
                GpuDescriptorBinding(
                    binding=1,
                    descriptor_type="uniform-buffer",
                    count=1,
                    buffer=uniform_buffer,
                ),
                GpuDescriptorBinding(
                    binding=2,
                    descriptor_type="storage-buffer",
                    count=1,
                    buffer=offset_buffer,
                ),
                GpuDescriptorBinding(
                    binding=3,
                    descriptor_type="storage-buffer",
                    count=1,
                    buffer=texcoord_buffer,
                ),
                GpuDescriptorBinding(
                    binding=4,
                    descriptor_type="storage-buffer",
                    count=1,
                    buffer=tint_buffer,
                ),
                GpuDescriptorBinding(
                    binding=5,
                    descriptor_type="storage-buffer",
                    count=1,
                    buffer=border_color_buffer,
                ),
                GpuDescriptorBinding(
                    binding=6,
                    descriptor_type="storage-buffer",
                    count=1,
                    buffer=border_thickness_buffer,
                ),
            ],
        )

        return R2dBatchGpuResources(
            parent=self,
            offset_buffer=offset_buffer,
            texcoord_buffer=texcoord_buffer,
            tint_buffer=tint_buffer,
            border_color_buffer=border_color_buffer,
            border_thickness_buffer=border_thickness_buffer,
            uniform_buffer=uniform_buffer,
            descriptor_set=descriptor_set,
            offset_staging=offset_staging,
            texcoord_staging=texcoord_staging,
            tint_staging=tint_staging,
            border_color_staging=border_color_staging,
            border_thickness_staging=border_thickness_staging,
            uniform_staging=uniform_staging,
        )

    def _upload_batch_data(
        self,
        batch: DrawQuadBatch,
        resources: R2dBatchGpuResources,
        atlas: GpuImage,
        cmd: GpuCommandEncoder,
    ) -> None:
        """Upload batch data to GPU buffers.

        Args:
            batch: The quad batch to upload.
            resources: The GPU resources for this batch.
            atlas: The atlas image for this batch.
            cmd: A transfer command encoder to record copy commands into.
        """
        n = batch.quad_count
        if n == 0:
            return

        # Flatten tensors for upload
        offset_flat = batch.offset_px.contiguous().view(-1)
        texcoord_flat = batch.texcoord_px.contiguous().view(-1)
        tint_flat = batch.tint_color.contiguous().view(-1)
        border_color_flat = batch.border_color.contiguous().view(-1)
        border_thickness_flat = batch.border_thickness_px.contiguous().view(-1)

        # Write to per-batch staging buffers
        resources.offset_staging.write(data=offset_flat)
        resources.texcoord_staging.write(data=texcoord_flat)
        resources.tint_staging.write(data=tint_flat)
        resources.border_color_staging.write(data=border_color_flat)
        resources.border_thickness_staging.write(data=border_thickness_flat)

        # Copy to device buffers for this batch
        cmd.copy_buffer_to_buffer(
            src=resources.offset_staging,
            dst=resources.offset_buffer,
            size=n * 4 * 2 * 4,  # N * 4 corners * 2 coords * 4 bytes
        )
        cmd.copy_buffer_to_buffer(
            src=resources.texcoord_staging,
            dst=resources.texcoord_buffer,
            size=n * 4 * 2 * 4,
        )
        cmd.copy_buffer_to_buffer(
            src=resources.tint_staging,
            dst=resources.tint_buffer,
            size=n * 4 * 4,  # N * 4 components * 4 bytes
        )
        cmd.copy_buffer_to_buffer(
            src=resources.border_color_staging,
            dst=resources.border_color_buffer,
            size=n * 4 * 4,
        )
        cmd.copy_buffer_to_buffer(
            src=resources.border_thickness_staging,
            dst=resources.border_thickness_buffer,
            size=n * 4 * 4,
        )

        # Upload uniform data
        atlas_height, atlas_width = atlas.meta.shape[0], atlas.meta.shape[1]
        uniform_data = torch.tensor(
            [
                self.framebuffer_width,
                self.framebuffer_height,
                atlas_width,
                atlas_height,
            ],
            dtype=torch.uint32,
        )
        resources.uniform_staging.write(data=uniform_data)
        cmd.copy_buffer_to_buffer(
            src=resources.uniform_staging,
            dst=resources.uniform_buffer,
            size=16,  # 4 uints * 4 bytes
        )

    def upload(
        self,
        *,
        quad_list: DrawQuadList,
        cmd: GpuCommandEncoder,
    ) -> None:
        """Upload quad data to GPU buffers.

        This should be called before render() with a transfer command encoder.
        The caller is responsible for submitting the command encoder and
        synchronizing before calling render().

        This method creates per-batch GPU resources and uploads data to them.

        Args:
            quad_list: The list of quad batches to upload.
            cmd: A transfer command encoder to record copy commands into.
        """
        # Clear old batch resources
        self._batch_resources.clear()

        for batch in quad_list.batches:
            if batch.quad_count == 0:
                continue

            batch.validate()

            # Use default texture if no atlas provided
            atlas = batch.atlas if batch.atlas is not None else self.default_texture

            # Create resources for this batch
            resources = self._create_batch_resources(batch, atlas)
            self._batch_resources.append(resources)

            # Upload batch data
            self._upload_batch_data(batch, resources, atlas, cmd)

    def render(
        self,
        *,
        quad_list: DrawQuadList,
        render_pass: GpuRenderPassCommandEncoder,
    ) -> None:
        """Render a list of quad batches using the provided render pass.

        The caller is responsible for:
        1. Uploading quad data via upload() and synchronizing
        2. Transitioning the target image to color-attachment-optimal layout
        3. Beginning a render pass with cmd.render()
        4. Calling this method with the render pass
        5. Ending the render pass and submitting the command encoder

        Args:
            quad_list: The list of quad batches to render.
            render_pass: An active render pass command encoder.
        """
        render_pass.bind_pipeline(pipeline=self.pipeline)

        # Filter to non-empty batches (matching what was uploaded)
        non_empty_batches = [b for b in quad_list.batches if b.quad_count > 0]

        for batch, resources in zip(non_empty_batches, self._batch_resources):
            # Bind descriptor set and draw
            render_pass.bind_descriptor_sets(
                layout=self.pipeline_layout,
                first_set=0,
                sets=[resources.descriptor_set],
            )

            # Draw 6 vertices per quad (2 triangles), instanced
            render_pass.draw(
                vertex_count=6,
                instance_count=batch.quad_count,
                first_vertex=0,
                first_instance=0,
            )
