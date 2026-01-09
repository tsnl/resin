__all__ = [
    "Draw2dFrame",
    "Draw2dQuad",
    "Draw2dRenderer",
]

from dataclasses import dataclass
from typing import Literal

import wgpu
import numpy as np
import numpy.typing as npt


type Draw2dTextureFormat = Literal["rgba8unorm", "rgba8unorm-srgb"]


class Draw2dRenderer:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh: tuple[int, int],
        target_format: Draw2dTextureFormat = "rgba8unorm",
    ):
        self.device = device
        self.target_size_wh = target_size_wh
        self.target_format = target_format

        self.bind_group_layout = device.create_bind_group_layout(
            label="Draw2dRenderer.QuadBatch.BindGroupLayout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.VERTEX,
                    buffer=wgpu.BufferBindingLayout(
                        type=wgpu.BufferBindingType.read_only_storage,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    sampler=wgpu.SamplerBindingLayout(
                        type=wgpu.SamplerBindingType.non_filtering,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=2,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.unfilterable_float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                        multisampled=False,
                    ),
                ),
            ],
        )

        pipeline_layout = device.create_pipeline_layout(
            label="Draw2dRenderer.PipelineLayout",
            bind_group_layouts=[self.bind_group_layout],
        )

        with open(__file__.replace(".py", ".wgsl"), "r") as f:
            shader_source = f.read()

        shader_module = device.create_shader_module(code=shader_source)

        self.pipeline = device.create_render_pipeline(
            label="Draw2dRenderer.Pipeline",
            layout=pipeline_layout,
            vertex=wgpu.VertexState(
                module=shader_module,
                entry_point="vs_main",
            ),
            fragment=wgpu.FragmentState(
                module=shader_module,
                entry_point="fs_main",
                targets=[
                    wgpu.ColorTargetState(
                        format=self.target_format,
                        blend=wgpu.BlendState(
                            color=wgpu.BlendComponent(
                                src_factor=wgpu.BlendFactor.src_alpha,
                                dst_factor=wgpu.BlendFactor.one_minus_src_alpha,
                                operation=wgpu.BlendOperation.add,
                            ),
                            alpha=wgpu.BlendComponent(
                                src_factor=wgpu.BlendFactor.one,
                                dst_factor=wgpu.BlendFactor.one_minus_src_alpha,
                                operation=wgpu.BlendOperation.add,
                            ),
                        ),
                        write_mask=wgpu.ColorWrite.ALL,
                    )
                ],
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
                front_face=wgpu.FrontFace.cw,
                cull_mode=wgpu.CullMode.back,
            ),
        )

        self.default_white_texture = self.device.create_texture(
            label="Draw2dRenderer.DefaultWhiteTexture",
            size=(32, 32, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        queue.write_texture(
            destination=wgpu.TexelCopyTextureInfo(
                texture=self.default_white_texture,
                mip_level=0,
                origin=(0, 0, 0),
            ),
            data=bytes([0xFF] * (4 * 32 * 32)),
            data_layout=wgpu.TexelCopyBufferLayout(
                offset=0,
                bytes_per_row=4 * 32,
                rows_per_image=32,
            ),
            size=(32, 32, 1),
        )

        self.sampler = device.create_sampler(
            label="Draw2dRenderer.QuadBatch.Sampler",
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            mipmap_filter=wgpu.MipmapFilterMode.nearest,
        )

    def record(
        self,
        quads: list["Draw2dQuad"],
        frame: "Draw2dFrame",
        command_encoder: wgpu.GPUCommandEncoder,
    ):
        frame.record(
            device=self.device,
            bind_group_layout=self.bind_group_layout,
            sampler=self.sampler,
            default_white_texture=self.default_white_texture,
            pipeline=self.pipeline,
            target_size_wh=self.target_size_wh,
            command_encoder=command_encoder,
            quads=quads,
        )


class Draw2dFrame:
    def __init__(self, *, renderer: Draw2dRenderer):
        self.output_image = renderer.device.create_texture(
            label="Draw2dFrame.OutputImage",
            size=(renderer.target_size_wh[0], renderer.target_size_wh[1], 1),
            dimension=wgpu.TextureDimension.d2,
            format=renderer.target_format,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.TEXTURE_BINDING
            ),
        )
        self.quad_group_cache: dict[wgpu.GPUTexture, "QuadGroup"] = {}

    def get_output_image(self) -> wgpu.GPUTexture:
        return self.output_image

    def record(
        self,
        *,
        device: wgpu.GPUDevice,
        bind_group_layout: wgpu.GPUBindGroupLayout,
        sampler: wgpu.GPUSampler,
        default_white_texture: wgpu.GPUTexture,
        pipeline: wgpu.GPURenderPipeline,
        target_size_wh: tuple[int, int],
        command_encoder: wgpu.GPUCommandEncoder,
        quads: list["Draw2dQuad"],
    ):
        quad_batch_list = QuadBatchList(quads, target_size_wh)

        group_bind_groups: dict[wgpu.GPUTexture, wgpu.GPUBindGroup] = {}

        for texture, group_quads in quad_batch_list.bind_groups.items():
            actual_texture = texture if texture is not None else default_white_texture
            bind_group = self.acquire_quad_group(
                device,
                command_encoder,
                actual_texture,
                group_quads,
                bind_group_layout,
                sampler,
            )
            group_bind_groups[actual_texture] = bind_group

        render_pass = command_encoder.begin_render_pass(
            label="Draw2dFrame.RenderPass",
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=self.output_image.create_view(),
                    resolve_target=None,
                    load_op="clear",
                    store_op="store",
                    clear_value=(0.0, 0.0, 0.0, 0.0),
                )
            ],
        )

        render_pass.set_pipeline(pipeline)

        for texture, draw_range in quad_batch_list.draw_ranges:
            actual_texture = texture or default_white_texture
            bind_group = group_bind_groups[actual_texture]

            render_pass.set_bind_group(0, bind_group, [], 0, 0)
            render_pass.draw(6, draw_range.stop - draw_range.start, 0, draw_range.start)

        render_pass.end()

    def acquire_quad_group(
        self,
        device: wgpu.GPUDevice,
        command_encoder: wgpu.GPUCommandEncoder,
        key: wgpu.GPUTexture,
        quads: npt.NDArray,
        bind_group_layout: wgpu.GPUBindGroupLayout,
        sampler: wgpu.GPUSampler,
    ) -> wgpu.GPUBindGroup:
        if key not in self.quad_group_cache:
            self.quad_group_cache[key] = QuadGroup(
                device, key, len(quads), bind_group_layout, sampler
            )

        return self.quad_group_cache[key].update(
            device=device,
            encoder=command_encoder,
            quad_batch_bind_group_layout=bind_group_layout,
            quad_batch_sampler=sampler,
            data=quads,
        )


@dataclass
class Draw2dQuad:
    dst_xy_px: tuple[int, int] = (0, 0)
    dst_wh_px: tuple[int, int] | None = None
    src_xy_px: tuple[int, int] | None = None
    src_wh_px: tuple[int, int] | None = None
    fill_texture: wgpu.GPUTexture | None = None
    fill_color_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_thickness_px: tuple[int, int, int, int] = (0, 0, 0, 0)
    border_color_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


#
# Implementation:
#

POD_QUAD_DTYPE = np.dtype(
    [
        ("dst_xy_ndc", np.float32, (2,)),
        ("dst_wh_ndc", np.float32, (2,)),
        ("src_xy_uv", np.float32, (2,)),
        ("src_wh_uv", np.float32, (2,)),
        ("fill_color_rgba", np.float32, (4,)),
        ("border_thickness_ndc", np.float32, (4,)),
        ("border_color_rgba", np.float32, (4,)),
        ("_rsv", np.float32, (4,)),
    ]
)


def pod_quad_from_draw2d_quad(
    original: Draw2dQuad, framebuffer_size_wh: tuple[int, int]
) -> np.ndarray:
    """Convert Draw2dQuad to PodQuad numpy record."""

    def texture_size():
        if original.fill_texture:
            return (original.fill_texture.width, original.fill_texture.height)
        return (1, 1)

    def ndc2_xy(a):
        return (
            (a[0] / framebuffer_size_wh[0]) * 2.0 - 1.0,
            -(a[1] / framebuffer_size_wh[1]) * 2.0 + 1.0,
        )

    def ndc2_wh(a):
        return (
            (a[0] / framebuffer_size_wh[0]) * 2.0,
            -(a[1] / framebuffer_size_wh[1]) * 2.0,
        )

    def ndc_trbl(a):
        return (
            -(a[0] / framebuffer_size_wh[1]) * 2.0,
            (a[1] / framebuffer_size_wh[0]) * 2.0,
            -(a[2] / framebuffer_size_wh[1]) * 2.0,
            (a[3] / framebuffer_size_wh[0]) * 2.0,
        )

    def uv(a):
        tex_wh = texture_size()
        return (
            a[0] / tex_wh[0],
            a[1] / tex_wh[1],
        )

    dst_wh = original.dst_wh_px if original.dst_wh_px else framebuffer_size_wh
    src_xy = original.src_xy_px if original.src_xy_px else (0, 0)
    src_wh = original.src_wh_px if original.src_wh_px else texture_size()

    dst_xy_ndc = ndc2_xy(original.dst_xy_px)
    dst_wh_ndc = ndc2_wh(dst_wh)
    src_xy_uv = uv(src_xy)
    src_wh_uv = uv(src_wh)
    border_thickness_ndc = ndc_trbl(original.border_thickness_px)

    record = np.array(
        (
            dst_xy_ndc,
            dst_wh_ndc,
            src_xy_uv,
            src_wh_uv,
            original.fill_color_rgba,
            border_thickness_ndc,
            original.border_color_rgba,
            [0.0, 0.0, 0.0, 0.0],
        ),
        dtype=POD_QUAD_DTYPE,
    )
    return record


class QuadGroup:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        atlas: wgpu.GPUTexture,
        min_capacity: int,
        quad_batch_bind_group_layout: wgpu.GPUBindGroupLayout,
        quad_batch_sampler: wgpu.GPUSampler,
    ):
        self.capacity = 1 << (min_capacity - 1).bit_length()  # next_power_of_two
        self.atlas = atlas
        self.device_buffer = device.create_buffer(
            label="Draw2dFrame.QuadBatch.DeviceBuffer",
            size=self.capacity * POD_QUAD_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.staging_buffer = device.create_buffer(
            label="Draw2dFrame.QuadBatch.StagingBuffer",
            size=self.capacity * POD_QUAD_DTYPE.itemsize,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

        self.bind_group = device.create_bind_group(
            label="Draw2dFrame.QuadBatch.BindGroup",
            layout=quad_batch_bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=wgpu.BufferBinding(
                        buffer=self.device_buffer,
                        offset=0,
                        size=self.device_buffer.size,
                    ),
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=quad_batch_sampler,
                ),
                wgpu.BindGroupEntry(
                    binding=2,
                    resource=atlas.create_view(),
                ),
            ],
        )

    def update(
        self,
        *,
        device: wgpu.GPUDevice,
        encoder: wgpu.GPUCommandEncoder,
        quad_batch_bind_group_layout: wgpu.GPUBindGroupLayout,
        quad_batch_sampler: wgpu.GPUSampler,
        data: npt.NDArray,
    ) -> wgpu.GPUBindGroup:
        self.realloc_if_needed(
            device=device,
            target_capacity=len(data),
            quad_batch_bind_group_layout=quad_batch_bind_group_layout,
            quad_batch_sampler=quad_batch_sampler,
        )

        data_array = np.array(data, dtype=POD_QUAD_DTYPE)

        self.staging_buffer.map_sync(wgpu.MapMode.WRITE, 0, data_array.nbytes)
        self.staging_buffer.write_mapped(data=data_array)
        self.staging_buffer.unmap()

        encoder.copy_buffer_to_buffer(
            source=self.staging_buffer,
            source_offset=0,
            destination=self.device_buffer,
            destination_offset=0,
            size=data_array.nbytes,
        )

        return self.bind_group

    def realloc_if_needed(
        self,
        *,
        device: wgpu.GPUDevice,
        target_capacity: int,
        quad_batch_bind_group_layout: wgpu.GPUBindGroupLayout,
        quad_batch_sampler: wgpu.GPUSampler,
    ):
        if target_capacity > self.capacity:
            self.__init__(
                device,
                self.atlas,
                target_capacity,
                quad_batch_bind_group_layout,
                quad_batch_sampler,
            )
        elif target_capacity < self.capacity // 4:
            self.__init__(
                device,
                self.atlas,
                max(1, self.capacity // 2),
                quad_batch_bind_group_layout,
                quad_batch_sampler,
            )


class QuadBatchList:
    def __init__(self, quads: list[Draw2dQuad], framebuffer_size_wh: tuple[int, int]):
        self.draw_ranges: list[tuple[wgpu.GPUTexture | None, range]] = []
        self.bind_groups: dict[wgpu.GPUTexture | None, npt.NDArray] = {}

        if not quads:
            return

        runs = [range(0, 1)]
        for i in range(1, len(quads)):
            quad = quads[i]
            prev_quad = quads[i - 1]
            # Compare texture identity
            tex1 = quad.fill_texture
            tex2 = prev_quad.fill_texture

            # In Python, None == None is True.
            if tex1 is tex2:
                runs[-1] = range(runs[-1].start, runs[-1].stop + 1)
            else:
                runs.append(range(i, i + 1))

        group_quad_dict = {}
        for run in runs:
            image = quads[run.start].fill_texture

            if image not in group_quad_dict:
                group_quad_dict[image] = []

            group_quads = group_quad_dict[image]
            group_offset = len(group_quads)
            group_length = len(run)

            for i in run:
                quad = pod_quad_from_draw2d_quad(quads[i], framebuffer_size_wh)
                group_quads.append(quad)

            self.bind_groups[image] = np.array(group_quads, dtype=POD_QUAD_DTYPE)

            self.draw_ranges.append(
                (image, range(group_offset, group_offset + group_length))
            )
