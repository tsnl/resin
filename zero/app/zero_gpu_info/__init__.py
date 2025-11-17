import sys
import json

import zero


def main():
    gpu_context = zero.GpuContext(
        app_name="zero-gpu-info",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    json.dump(
        {
            "physical-devices": [
                {
                    "name": physical_device.properties.device_name,
                    "vendor-id": f"0x{physical_device.properties.vendor_id:08x}",
                    "device-id": f"0x{physical_device.properties.device_id:08x}",
                    "api-version": f"0x{physical_device.properties.api_version:08x}",
                    "device-type": physical_device.properties.device_type.name,
                }
                for physical_device in gpu_context.enumerate_physical_devices()
            ]
        },
        sys.stdout,
        indent=4,
    )
    print()


if __name__ == "__main__":
    main()
