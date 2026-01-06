from pathlib import Path
import numpy as np
import wgpu

import zfw

from conftest import GpuFixture


def help_render_texture_to_framebuffer(
    device: wgpu.GPUDevice,
    queue: wgpu.GPUQueue,
    attachment_texture_format: zfw.ImageFormat,
    compressed_texture_data: np.ndarray,
    input_w: int,
    input_h: int,
    bytes_per_row: int,
    output_path: Path,
) -> np.ndarray:
    with open(Path(__file__).parent / "test_image_bc.wgsl") as f:
        shader_code = f.read()

    shader_module = device.create_shader_module(code=shader_code)

    render_pipeline = device.create_render_pipeline(
        label="TestImageBc.TestImageBc4Pipeline",
        layout="auto",
        vertex=wgpu.VertexState(
            module=shader_module,
            entry_point="vs_main",
        ),
        fragment=wgpu.FragmentState(
            module=shader_module,
            entry_point="fs_main",
            targets=[wgpu.ColorTargetState(format="rgba32float")],
        ),
        primitive=wgpu.PrimitiveState(topology="triangle-list"),
        depth_stencil=None,
        multisample=None,
    )
    render_target = device.create_texture(
        label="TestImageBc.HelpRenderTextureToFramebuffer.RenderTexture",
        size=(input_w, input_h, 1),
        dimension="2d",
        format="rgba32float",
        usage=(wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC),
    )

    texture = device.create_texture(
        label="TestImageBc.HelpRenderTextureToFramebuffer.Texture",
        size=(input_w, input_h, 1),
        dimension="2d",
        format=attachment_texture_format,
        usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
    )
    queue.write_texture(
        destination=wgpu.TexelCopyTextureInfo(
            texture=texture,
            mip_level=0,
            origin=(0, 0, 0),
        ),
        data=compressed_texture_data.tobytes(),
        data_layout=wgpu.TexelCopyBufferLayout(
            offset=0,
            bytes_per_row=bytes_per_row,
            rows_per_image=input_h,
        ),
        size=(input_w, input_h, 1),
    )

    bind_group = device.create_bind_group(
        label="TestImageBc.HelpRenderTextureToFramebuffer.BindGroup",
        layout=render_pipeline.get_bind_group_layout(0),
        entries=[
            wgpu.BindGroupEntry(
                binding=0,
                resource=device.create_texture(
                    label="TestImageBc.HelpRenderTextureToFramebuffer.TextureView",
                    size=(input_w, input_h, 1),
                    dimension="2d",
                    format=attachment_texture_format,
                    usage=(
                        wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                    ),
                ).create_view(),
            ),
            wgpu.BindGroupEntry(
                binding=1,
                resource=device.create_sampler(
                    label="TestImageBc.HelpRenderTextureToFramebuffer.Sampler",
                    mag_filter="nearest",
                    min_filter="nearest",
                ),
            ),
        ],
    )

    command_encoder = device.create_command_encoder()

    rp = command_encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=render_target.create_view(),
                clear_value=(0.0, 0.0, 0.0, 1.0),
                load_op="clear",
                store_op="store",
            )
        ]
    )
    rp.set_pipeline(render_pipeline)
    rp.set_bind_group(0, bind_group)
    rp.draw(vertex_count=3)
    rp.end()

    queue.submit([command_encoder.finish()])

    # Read back rendered data
    readback_buffer_size = input_w * input_h * 4 * 4  # rgba32float
    readback_buffer = device.create_buffer(
        label="TestImageBc.HelpRenderTextureToFramebuffer.ReadbackBuffer",
        size=readback_buffer_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        source=wgpu.TexelCopyTextureInfo(
            texture=render_target,
            mip_level=0,
            origin=(0, 0, 0),
        ),
        destination=wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer,
            offset=0,
            bytes_per_row=input_w * 4 * 4,
            rows_per_image=input_h,
        ),
        copy_size=(input_w, input_h, 1),
    )
    queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    buffer_data = np.asarray(readback_buffer.read_mapped(), copy=True)
    readback_buffer.unmap()
    readback_buffer.destroy()

    texture.destroy()
    render_target.destroy()

    res = buffer_data.view(np.float32).reshape((input_h, input_w, 4))

    zfw.debug_save_rgba_image(file_path=output_path, data=res)

    return res


def test_help_render_texture_to_framebuffer(
    gpu: GpuFixture,
    rainbow_512x512_image_grayscale: zfw.ImageResource,
):
    assert rainbow_512x512_image_grayscale.data.dtype == np.float32
    assert rainbow_512x512_image_grayscale.data.shape == (512, 512, 1)
    input_h, input_w, _ = rainbow_512x512_image_grayscale.data.shape

    image_data = rainbow_512x512_image_grayscale.data.astype(np.float16)

    image = help_render_texture_to_framebuffer(
        device=gpu.device,
        queue=gpu.queue,
        attachment_texture_format="r16float",
        compressed_texture_data=image_data.view(np.uint8),
        input_w=input_w,
        input_h=input_h,
        bytes_per_row=input_w * 2,
        output_path=Path(
            "output/zfw/test_image_bc/test_help_render_texture_to_framebuffer.png"
        ),
    )

    assert np.allclose(image[:input_h, :input_w, :1], image_data, atol=1e-6)


def test_image_bc4(
    gpu: GpuFixture,
    rainbow_512x512_image_grayscale: zfw.ImageResource,
):
    assert rainbow_512x512_image_grayscale.data.dtype == np.float32
    assert rainbow_512x512_image_grayscale.data.shape == (512, 512, 1)
    input_h, input_w, _ = rainbow_512x512_image_grayscale.data.shape

    bc4_data = zfw.compress_bc4(rainbow_512x512_image_grayscale.data)
    assert bc4_data.dtype == np.uint8
    assert bc4_data.shape == (input_h // 4, input_w // 4, 8)

    image = help_render_texture_to_framebuffer(
        device=gpu.device,
        queue=gpu.queue,
        attachment_texture_format="bc4-r-unorm",
        compressed_texture_data=bc4_data,
        input_w=input_w,
        input_h=input_h,
        bytes_per_row=(input_w // 4) * 8,
        output_path=Path("output/zfw/test_image_bc/test_image_bc4.png"),
    )

    psnr = zfw.compute_psnr(
        img1=rainbow_512x512_image_grayscale.data,
        img2=image[:input_h, :input_w, :1],
    )
    assert psnr > 50.0, f"BC4 PSNR too low: {psnr:.2f} dB"


LOG = zfw.logger(__name__)
