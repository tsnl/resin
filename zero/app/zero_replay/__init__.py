import sys
import json

import zero


def main():
    gpu_context = zero.GpuContext(
        app_name="zero-replay",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )

    print_debug_info(gpu_context)


def print_debug_info(gpu_context: zero.GpuContext) -> None:
    print("<gpu-info>")
    json.dump(
        {
            "physical-devices": [
                {
                    "name": physical_device.properties.deviceName,
                    "vendor-id": f"0x{physical_device.properties.vendorID:08x}",
                    "device-id": f"0x{physical_device.properties.deviceID:08x}",
                    "api-version": f"0x{physical_device.properties.apiVersion:08x}",
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
