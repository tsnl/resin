import wgpu
import ctypes
import numpy as np
from dataclasses import dataclass
from .gpu_util import Rgba8UnormTexture, StorageBuffer, StagingBuffer

__all__ = [
    "Draw2dFrame",
    "Draw2dQuad",
    "Draw2dRenderer",
]


class Draw2dRenderer:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh: tuple[int, int],
    ):
        self.device = device
        self.target_size_wh = target_size_wh

        self.bind_group_layout = device.create_bind_group_layout(
            label="Draw2dRenderer.QuadBatch.BindGroupLayout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.read_only_storage,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {
                        "type": wgpu.SamplerBindingType.filtering,
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                },
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
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.cw,
                "cull_mode": wgpu.CullMode.back,
            },
        )

        self.default_white_texture = Rgba8UnormTexture(
            device=device,
            size_wh=(32, 32),
            label="Draw2dRenderer.DefaultWhiteTexture",
        )
        queue.write_texture(
            self.default_white_texture.texel_copy_texture_info(),
            bytes([0xFF] * (4 * 32 * 32)),
            self.default_white_texture.texel_copy_buffer_layout(),
            self.default_white_texture.size(),
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

    @classmethod
    def create(
        cls,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
        target_size_wh: tuple[int, int],
    ) -> "Draw2dRenderer":
        return cls(device, queue, target_size_wh)

    def record(
        self,
        quads: list[Draw2dQuad],
        frame: Draw2dFrame,
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
    def __init__(self, *, device: wgpu.GPUDevice, target_size_wh: tuple[int, int]):
        self.output_image = Rgba8UnormTexture(
            device=device,
            size_wh=target_size_wh,
            label="Draw2dFrame.OutputImage",
        )
        self.quad_group_cache: dict[wgpu.GPUTexture, "QuadGroup"] = {}

    def get_output_image(self) -> Rgba8UnormTexture:
        return self.output_image

    def record(
        self,
        *,
        device: wgpu.GPUDevice,
        bind_group_layout: wgpu.GPUBindGroupLayout,
        sampler: wgpu.GPUSampler,
        default_white_texture: Rgba8UnormTexture,
        pipeline: wgpu.GPURenderPipeline,
        target_size_wh: tuple[int, int],
        command_encoder: wgpu.GPUCommandEncoder,
        quads: list[Draw2dQuad],
    ):
        quad_batch_list = QuadBatchList(quads, target_size_wh)

        group_bind_groups: dict[wgpu.GPUTexture, wgpu.GPUBindGroup] = {}

        for texture, group_quads in quad_batch_list.bind_groups.items():
            actual_texture = (
                texture if texture is not None else default_white_texture.wgpu_texture()
            )
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
                {
                    "view": self.output_image.wgpu_texture().create_view(),
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 0),
                }
            ],
        )

        render_pass.set_pipeline(pipeline)

        for texture, draw_range in quad_batch_list.draw_ranges:
            actual_texture = (
                texture if texture is not None else default_white_texture.wgpu_texture()
            )
            bind_group = group_bind_groups[actual_texture]

            render_pass.set_bind_group(0, bind_group, [], 0, 0)
            render_pass.draw(6, draw_range.stop - draw_range.start, 0, draw_range.start)

        render_pass.end()

    def acquire_quad_group(
        self,
        device: wgpu.GPUDevice,
        command_encoder: wgpu.GPUCommandEncoder,
        key: wgpu.GPUTexture,
        quads: list[PodQuad],
        bind_group_layout: wgpu.GPUBindGroupLayout,
        sampler: wgpu.GPUSampler,
    ) -> wgpu.GPUBindGroup:
        if key not in self.quad_group_cache:
            self.quad_group_cache[key] = QuadGroup(
                device, key, len(quads), bind_group_layout, sampler
            )

        return self.quad_group_cache[key].update(
            device=device,
            command_encoder=command_encoder,
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


class PodQuad(ctypes.Structure):
    _fields_ = [
        ("dst_xy_ndc", ctypes.c_float * 2),
        ("dst_wh_ndc", ctypes.c_float * 2),
        ("src_xy_uv", ctypes.c_float * 2),
        ("src_wh_uv", ctypes.c_float * 2),
        ("fill_color_rgba", ctypes.c_float * 4),
        ("border_thickness_ndc", ctypes.c_float * 4),
        ("border_color_rgba", ctypes.c_float * 4),
        ("_rsv", ctypes.c_float * 4),
    ]

    @staticmethod
    def from_quad(
        original: Draw2dQuad, framebuffer_size_wh: tuple[int, int]
    ) -> "PodQuad":
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

        return PodQuad(
            dst_xy_ndc=(ctypes.c_float * 2)(*dst_xy_ndc),
            dst_wh_ndc=(ctypes.c_float * 2)(*dst_wh_ndc),
            src_xy_uv=(ctypes.c_float * 2)(*src_xy_uv),
            src_wh_uv=(ctypes.c_float * 2)(*src_wh_uv),
            fill_color_rgba=(ctypes.c_float * 4)(*original.fill_color_rgba),
            border_thickness_ndc=(ctypes.c_float * 4)(*border_thickness_ndc),
            border_color_rgba=(ctypes.c_float * 4)(*original.border_color_rgba),
            _rsv=(ctypes.c_float * 4)(0, 0, 0, 0),
        )


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
        self.device_buffer = StorageBuffer(
            device=device,
            count=self.capacity,
            dtype=PodQuad,
            label="Draw2dFrame.QuadBatch.DeviceBuffer",
        )
        self.staging_buffer = StagingBuffer(
            device=device,
            count=self.capacity,
            dtype=PodQuad,
            label="Draw2dFrame.QuadBatch.StagingBuffer",
        )

        self.bind_group = device.create_bind_group(
            label="Draw2dFrame.QuadBatch.BindGroup",
            layout=quad_batch_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.device_buffer.wgpu_buffer(),
                        "offset": 0,
                        "size": self.device_buffer.size_in_bytes,
                    },
                },
                {
                    "binding": 1,
                    "resource": quad_batch_sampler,
                },
                {
                    "binding": 2,
                    "resource": atlas.create_view(),
                },
            ],
        )

    def update(
        self,
        *,
        device: wgpu.GPUDevice,
        command_encoder: wgpu.GPUCommandEncoder,
        quad_batch_bind_group_layout: wgpu.GPUBindGroupLayout,
        quad_batch_sampler: wgpu.GPUSampler,
        data: list[PodQuad],
    ) -> wgpu.GPUBindGroup:
        self.realloc_if_needed(
            device=device,
            target_capacity=len(data),
            quad_batch_bind_group_layout=quad_batch_bind_group_layout,
            quad_batch_sampler=quad_batch_sampler,
        )

        self.staging_buffer.write(data=np.asarray(data))
        self.staging_buffer.copy_to_buffer(
            dst=self.device_buffer,
            command_encoder=command_encoder,
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
        self.bind_groups: dict[wgpu.GPUTexture | None, list[PodQuad]] = {}

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

        for run in runs:
            image = quads[run.start].fill_texture

            if image not in self.bind_groups:
                self.bind_groups[image] = []

            group_quads = self.bind_groups[image]
            group_offset = len(group_quads)
            group_length = len(run)

            for i in run:
                group_quads.append(PodQuad.from_quad(quads[i], framebuffer_size_wh))

            self.draw_ranges.append(
                (image, range(group_offset, group_offset + group_length))
            )
