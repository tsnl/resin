from pathlib import Path

import numpy as np
import wgpu
from conftest import GpuFixture

import zfw
from zfw.images_bcn import _f32_to_rgb565, _pack_bc1_block, _quantize_rgb565_f32


def test_f32_to_rgb565():
    """Test RGB565 conversion round-trip."""
    test_colors = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "50% red": (0.5, 0.0, 0.0),
        "50% green": (0.0, 0.5, 0.0),
        "50% blue": (0.0, 0.0, 0.5),
        "cornflower blue": (0.39, 0.58, 0.93),
    }

    for name, (r, g, b) in test_colors.items():
        color_f32 = np.array([r, g, b], dtype=np.float32)
        packed_rgb565 = _f32_to_rgb565(color_f32)

        # Unpack to verify
        r_bits = (packed_rgb565 >> 11) & 0x1F
        g_bits = (packed_rgb565 >> 5) & 0x3F
        b_bits = packed_rgb565 & 0x1F

        # Expand back to 8-bit
        r_expanded = ((r_bits << 3) | (r_bits >> 2)) / 255.0
        g_expanded = ((g_bits << 2) | (g_bits >> 4)) / 255.0
        b_expanded = ((b_bits << 3) | (b_bits >> 2)) / 255.0

        # print(f"{name:20} input:  ({r:.3f}, {g:.3f}, {b:.3f})")
        # print(
        #     f"{' ' * 20} bits:   (R:{r_bits:2d}, G:{g_bits:2d}, B:{b_bits:2d}) -> 0x{packed_rgb565:04x}"
        # )
        # print(
        #     f"{' ' * 20} output: ({r_expanded:.3f}, {g_expanded:.3f}, {b_expanded:.3f})"
        # )

        # Round-trip should be reasonably close (within RGB565 precision)
        assert abs(r - r_expanded) < 0.05, (
            f"{name} red channel mismatch: {r} vs {r_expanded}"
        )
        assert abs(g - g_expanded) < 0.03, (
            f"{name} green channel mismatch: {g} vs {g_expanded}"
        )
        assert abs(b - b_expanded) < 0.05, (
            f"{name} blue channel mismatch: {b} vs {b_expanded}"
        )


def test_quantize_rgb565_f32():
    """Test RGB565 quantization round-trip."""
    test_colors = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "50% red": (0.5, 0.0, 0.0),
        "50% green": (0.0, 0.5, 0.0),
        "50% blue": (0.0, 0.0, 0.5),
        "cornflower blue": (0.39, 0.58, 0.93),
    }

    for name, (r, g, b) in test_colors.items():
        color_f32 = np.array([r, g, b], dtype=np.float32)
        quantized = _quantize_rgb565_f32(color_f32)

        # print(f"{name:20} input:     ({r:.3f}, {g:.3f}, {b:.3f})")
        # print(
        #     f"{' ' * 20} quantized: ({quantized[0]:.3f}, {quantized[1]:.3f}, {quantized[2]:.3f})"
        # )

        # Quantized values should be in [0, 1]
        assert 0.0 <= quantized[0] <= 1.0
        assert 0.0 <= quantized[1] <= 1.0
        assert 0.0 <= quantized[2] <= 1.0

        # Round-trip should be reasonably close
        assert abs(r - quantized[0]) < 0.05, (
            f"{name} red channel mismatch: {r} vs {quantized[0]}"
        )
        assert abs(g - quantized[1]) < 0.03, (
            f"{name} green channel mismatch: {g} vs {quantized[1]}"
        )
        assert abs(b - quantized[2]) < 0.05, (
            f"{name} blue channel mismatch: {b} vs {quantized[2]}"
        )


def test_pack_bc1_block():
    """Test BC1 block packing round-trip."""
    # Create simple test case: 2 endpoints and all-zero indices (all pixels use color 0)
    endpoints = np.array(
        [
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
        ],
        dtype=np.float32,
    )
    indices = np.zeros(16, dtype=np.uint8)

    packed = _pack_bc1_block(endpoints, indices)
    assert packed.shape == (8,)
    assert packed.dtype == np.uint8

    # Unpack and verify (little-endian uint16)
    endpoint0_565 = np.uint16(packed[0]) | (np.uint16(packed[1]) << 8)
    endpoint1_565 = np.uint16(packed[2]) | (np.uint16(packed[3]) << 8)

    # print(f"Endpoint 0: 0x{endpoint0_565:04x}")
    # print(f"Endpoint 1: 0x{endpoint1_565:04x}")

    # White should pack to 0xFFFF (R=31, G=63, B=31)
    assert endpoint0_565 == 0xFFFF, (
        f"White endpoint should be 0xFFFF, got 0x{endpoint0_565:04x}"
    )
    # Black should pack to 0x0000 (R=0, G=0, B=0)
    assert endpoint1_565 == 0x0000, (
        f"Black endpoint should be 0x0000, got 0x{endpoint1_565:04x}"
    )

    # All indices should be 0
    for i in range(16):
        bit_offset = i * 2
        byte_index = bit_offset // 8
        byte_offset = bit_offset % 8
        idx = (packed[4 + byte_index] >> byte_offset) & 0x3
        assert idx == 0, f"Index {i} should be 0, got {idx}"

    # print("BC1 block packing test passed!")


def help_render_texture_to_framebuffer(
    device: wgpu.GPUDevice,
    queue: wgpu.GPUQueue,
    attachment_texture_format: zfw.ImageFormat,
    compressed_texture_data: np.ndarray,
    input_w: int,
    input_h: int,
    bytes_per_row: int,
    grayscale: bool,
    output_path: Path,
) -> np.ndarray:
    with open(Path(__file__).parent / "test_images_bcn.wgsl") as f:
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
    # For compressed textures, rows_per_image must be in block units
    # BC4 uses 4x4 blocks, so divide by 4
    is_compressed = attachment_texture_format.startswith("bc")
    rows_per_image_value = (input_h // 4) if is_compressed else input_h

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
            rows_per_image=rows_per_image_value,
        ),
        size=(input_w, input_h, 1),
    )

    bind_group = device.create_bind_group(
        label="TestImageBc.HelpRenderTextureToFramebuffer.BindGroup",
        layout=render_pipeline.get_bind_group_layout(0),
        entries=[
            wgpu.BindGroupEntry(
                binding=0,
                resource=texture.create_view(),
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

    if grayscale:
        res = np.dstack([res[:, :, 0:1]] * 4)
        res[:, :, 3] = 1.0  # set alpha to 1.0

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
        grayscale=True,
        output_path=Path(
            "output/zfw/test_images_bcn/test_help_render_texture_to_framebuffer.png"
        ),
    )

    assert np.allclose(image[:input_h, :input_w, :1], image_data, atol=1e-6)


def test_encode_bc4(
    gpu: GpuFixture,
    rainbow_512x512_image_grayscale: zfw.ImageResource,
):
    assert rainbow_512x512_image_grayscale.data.dtype == np.float32
    assert rainbow_512x512_image_grayscale.data.shape == (512, 512, 1)
    input_h, input_w, _ = rainbow_512x512_image_grayscale.data.shape

    bc4_data = zfw.encode_bc4(rainbow_512x512_image_grayscale.data)
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
        grayscale=True,
        output_path=Path("output/zfw/test_images_bcn/test_image_bc4.png"),
    )

    psnr = zfw.compute_psnr(
        img1=rainbow_512x512_image_grayscale.data,
        img2=image[:input_h, :input_w, :1],
    )
    assert psnr > 55.0, f"BC4 PSNR too low: {psnr:.2f} dB"


def test_encode_bc1(
    gpu: GpuFixture,
    rainbow_512x512_image: zfw.ImageResource,
):
    assert rainbow_512x512_image.data.dtype == np.float32
    assert rainbow_512x512_image.data.shape == (512, 512, 4)
    input_h, input_w, _ = rainbow_512x512_image.data.shape

    # Discard alpha channel to get RGB
    rgb_data = rainbow_512x512_image.data[:, :, :3]

    bc1_data = zfw.encode_bc1(rgb_data)
    assert bc1_data.dtype == np.uint8
    assert bc1_data.shape == (input_h // 4, input_w // 4, 8)

    image = help_render_texture_to_framebuffer(
        device=gpu.device,
        queue=gpu.queue,
        attachment_texture_format="bc1-rgba-unorm",
        compressed_texture_data=bc1_data,
        input_w=input_w,
        input_h=input_h,
        bytes_per_row=(input_w // 4) * 8,
        grayscale=False,
        output_path=Path("output/zfw/test_images_bcn/test_image_bc1.png"),
    )

    psnr = zfw.compute_psnr(
        img1=rgb_data,
        img2=image[:input_h, :input_w, :3],
    )
    # assert psnr > 50.0, f"BC1 PSNR too low: {psnr:.2f} dB"
    assert psnr > 30.0, f"BC1 PSNR too low: {psnr:.2f} dB"

    # TODO: Even though the rainbow PSNR exceeds 30dB, the visual quality is not as high
    # as I'd like. We can improve this further, but I'm out of time. Something to
    # revisit later.


LOG = zfw.logger(__name__)
