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
            targets=[wgpu.ColorTargetState(format="rgba8unorm")],
        ),
        primitive=wgpu.PrimitiveState(topology="triangle-list"),
        depth_stencil=None,
        multisample=None,
    )
    render_target = device.create_texture(
        label="TestImageBc.TestImageBc4RenderTexture",
        size=(input_w, input_h, 1),
        dimension="2d",
        format="rgba32float",
        usage=(wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC),
    )

    texture = device.create_texture(
        label="TestImageBc.TestImageBc4Texture",
        size=(input_w, input_h, 1),
        dimension="2d",
        format=attachment_texture_format,
        usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
    )
    queue.write_texture(
        destination={
            "texture": texture,
            "mipLevel": 0,
            "origin": (0, 0, 0),
        },
        data=compressed_texture_data.tobytes(),
        data_layout={
            "offset": 0,
            "bytesPerRow": bytes_per_row,
            "rowsPerImage": input_h,
        },
        size=(input_w, input_h, 1),
    )

    bind_group = device.create_bind_group(
        label="TestImageBc.TestImageBc4BindGroup",
        layout=render_pipeline.get_bind_group_layout(0),
        entries=[
            wgpu.BindGroupEntry(
                binding=0,
                resource=device.create_texture(
                    label="TestImageBc.TestImageBc4Texture",
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
                    label="TestImageBc.TestImageBc4Sampler",
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
        label="TestImageBc.TestImageBc4ReadbackBuffer",
        size=readback_buffer_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        source={
            "texture": render_target,
            "mipLevel": 0,
            "origin": (0, 0, 0),
        },
        destination={
            "buffer": readback_buffer,
            "offset": 0,
            "bytesPerRow": input_w * 4 * 4,
            "rowsPerImage": input_h,
        },
        copy_size=(input_w, input_h, 1),
    )
    queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    buffer_data = readback_buffer.read_mapped()
    readback_buffer.unmap()
    readback_buffer.destroy()

    texture.destroy()
    render_target.destroy()

    return np.array(buffer_data, dtype=np.float32).reshape(input_h, input_w, 4)


def test_image_bc4(
    gpu: GpuFixture,
    rainbow_512x512_image_greyscale: zfw.ImageResource,
):
    assert rainbow_512x512_image_greyscale.data.dtype == np.float32
    assert rainbow_512x512_image_greyscale.data.shape == (512, 512, 1)
    input_h, input_w, _ = rainbow_512x512_image_greyscale.data.shape

    bc4_data = zfw.compress_bc4(rainbow_512x512_image_greyscale.data)
    assert bc4_data.dtype == np.uint8
    assert bc4_data.shape == (input_h // 4, input_w // 4, 8)

    shader_module = gpu.device.create_shader_module(code=_TEST_WGSL)

    render_pipeline = gpu.device.create_render_pipeline(
        label="TestImageBc.TestImageBc4Pipeline",
        layout="auto",
        vertex=wgpu.VertexState(module=shader_module, entry_point="vs_main"),
        fragment=wgpu.FragmentState(
            module=shader_module,
            entry_point="fs_main",
            targets=[wgpu.ColorTargetState(format="rgba8unorm")],
        ),
        primitive=wgpu.PrimitiveState(topology="triangle-list"),
        depth_stencil=None,
        multisample=None,
    )

    bind_group = gpu.device.create_bind_group(
        label="TestImageBc.TestImageBc4BindGroup",
        layout=render_pipeline.get_bind_group_layout(0),
        entries=[
            wgpu.BindGroupEntry(
                binding=0,
                resource=gpu.device.create_texture(
                    label="TestImageBc.TestImageBc4Texture",
                    size=(input_w, input_h, 1),
                    dimension="2d",
                    format="bc4-r-unorm",
                    usage=(
                        wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                    ),
                ).create_view(),
            ),
            wgpu.BindGroupEntry(
                binding=1,
                resource=gpu.device.create_sampler(
                    label="TestImageBc.TestImageBc4Sampler",
                    mag_filter="nearest",
                    min_filter="nearest",
                ),
            ),
        ],
    )


LOG = zfw.logger(__name__)
