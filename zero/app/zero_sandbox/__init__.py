import json
import sys
from pathlib import Path

import torch

import zero


def main():
    gpu_context = zero.GpuContext(
        app_name="zero-replay",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    print_debug_info(gpu_context)

    physical_device = next(iter(gpu_context.enumerate_physical_devices()))
    device = gpu_context.create_device(physical_device=physical_device, surface=None)

    # Create render target image (1024x1024 RGBA8)
    render_target_image = device.create_image(
        usages=["color-attachment"],
        meta=zero.GpuImageMeta(shape=(1024, 1024, 4), dtype=torch.uint8),
    )

    print(device)
    print(render_target_image)

    # Load shaders from compiled SPIR-V
    shader_dir = Path(__file__).parent.parent.parent / "data" / "shader"

    vertex_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.vert.spv",
        stage="vertex",
    )

    fragment_shader = device.create_shader(
        spirv_path=shader_dir / "triangle.frag.spv",
        stage="fragment",
    )

    print(f"Loaded shaders: {vertex_shader}, {fragment_shader}")

    # Create graphics pipeline
    pipeline = device.create_pipeline(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader,
        color_format=render_target_image.meta.infer_vk_format(
            render_target_image.usages
        ),
        viewport_width=1024,
        viewport_height=1024,
    )

    print(f"Created pipeline: {pipeline}")

    # Render triangle
    cmd = device.create_command_encoder(queue_type="graphics")
    with cmd.render(
        color_attachment=render_target_image,
        clear_on_load=True,
    ) as render_pass:
        render_pass.bind_pipeline(pipeline=pipeline)
        render_pass.draw(vertex_count=3, instance_count=1)
    cmd.submit().wait()

    print("Triangle rendered successfully!")


def print_debug_info(gpu_context: zero.GpuContext) -> None:
    print("<gpu-info>")
    json.dump(
        {
            "physical-devices": [
                {
                    "name": physical_device.vk_properties.deviceName,
                    "vendor-id": f"0x{physical_device.vk_properties.vendorID:08x}",
                    "device-id": f"0x{physical_device.vk_properties.deviceID:08x}",
                    "api-version": f"0x{physical_device.vk_properties.apiVersion:08x}",
                    "device-type": physical_device.spell_device_type(),
                }
                for physical_device in gpu_context.enumerate_physical_devices()
            ]
        },
        sys.stdout,
        indent=4,
    )
    print()
    print("</gpu-info>")


if __name__ == "__main__":
    main()
