import wgpu


def request_wgpu_device(
    adapter: wgpu.GPUAdapter,
    *,
    label: str | None = None,
    required_features: list[str] | None = None,
    required_limits: dict[str, int | None] | None = None,
) -> wgpu.GPUDevice:
    return adapter.request_device_sync(
        label=(label or "ZfwDevice"),
        required_features=(
            [
                "shader-f16",
                "texture-compression-bc",
            ]
            + (required_features or [])
        ),
        required_limits=(
            (required_limits or {})
            | {
                "maxBufferSize": 1 << 30,  # 1 GiB
                "maxStorageBufferBindingSize": 1 << 30,  # 1 GiB
            }
        ),
    )
