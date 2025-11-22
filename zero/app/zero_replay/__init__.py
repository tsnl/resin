import sys
import json

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

    render_target_image = device.create_image(
        usages=("color-attachment",),
        meta=zero.GpuImageMeta(shape=(1024, 1024, 4), dtype=torch.uint8),
    )

    print(device)
    print(render_target_image)

    render_target_image.write(torch.ones((1024, 1024, 4), dtype=torch.uint8))


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
