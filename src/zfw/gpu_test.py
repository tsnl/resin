import numpy as np
import pytest

from .gpu import (
    GpuBuffer,
    GpuBufferImageCopyRegion,
    GpuBufferMeta,
    GpuBufferUsage,
    GpuCommandEncoder,
    GpuContext,
    GpuDevice,
    GpuImage,
    GpuImageMeta,
)
from .excepts import LogicError


def make_context() -> tuple[GpuContext, GpuDevice]:
    ctx = GpuContext(
        app_name="gpu-tests",
        enable_debug_layer_support=True,
        enable_present_support=False,
    )
    phys = ctx.enumerate_physical_devices()[0]
    dev = GpuDevice(context=ctx, physical_device=phys, surface=None)
    return ctx, dev


def test_buffer_roundtrip():
    _, dev = make_context()

    data0 = np.random.randn(1024).astype(np.float32)
    meta = GpuBufferMeta.from_array(data0)

    device_buf_usage: list[GpuBufferUsage] = ["copy-src", "copy-dst", "storage"]
    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    device_buf = GpuBuffer(device=dev, usages=device_buf_usage, meta=meta)
    host_buf_1 = GpuBuffer(device=dev, usages=host_buf_usage, meta=meta)
    host_buf_2 = GpuBuffer(device=dev, usages=host_buf_usage, meta=meta)

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to device_buf
    cmd = GpuCommandEncoder(device=dev, queue_type="transfer")
    cmd.copy_buffer_to_buffer(src=host_buf_1, dst=device_buf, size=meta.size)
    cmd.submit().wait()

    # copy device_buf to host_buf_2
    cmd = GpuCommandEncoder(device=dev, queue_type="transfer")
    cmd.copy_buffer_to_buffer(src=device_buf, dst=host_buf_2, size=meta.size)
    cmd.submit().wait()

    # read host_buf_2
    data1 = host_buf_2.read()

    # Test:
    assert np.allclose(data0, data1)


def test_image_roundtrip():
    _, dev = make_context()

    data0 = np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
    meta = GpuImageMeta.from_array(data0)

    host_buf_usage: list[GpuBufferUsage] = ["staging", "copy-src", "copy-dst"]
    image = GpuImage(device=dev, usages=["texture-binding"], meta=meta)
    host_buf_1 = GpuBuffer(
        device=dev, usages=host_buf_usage, meta=meta.into_buffer_meta()
    )
    host_buf_2 = GpuBuffer(
        device=dev, usages=host_buf_usage, meta=meta.into_buffer_meta()
    )

    # fill host_buf_1
    host_buf_1.write(data=data0)

    # copy host_buf_1 to image
    cmd = GpuCommandEncoder(device=dev, queue_type="transfer")
    cmd.transition_image_layout(image=image, layout="transfer-dst-optimal")
    cmd.copy_buffer_to_image(
        dst=image,
        src=host_buf_1,
        regions=[
            GpuBufferImageCopyRegion(
                buffer_offset=0,
                image_offset=(0, 0, 0),
                image_extent=(meta.shape[1], meta.shape[0], 1),
            )
        ],
    )
    cmd.submit().wait()

    # copy image to host_buf_2
    cmd = GpuCommandEncoder(device=dev, queue_type="transfer")
    cmd.transition_image_layout(image=image, layout="transfer-src-optimal")
    cmd.copy_image_to_buffer(src=image, dst=host_buf_2)
    cmd.submit().wait()

    # read host_buf_2
    data1 = host_buf_2.read()

    # reshape data1
    data1 = data1.reshape(data0.shape)

    # Test:
    assert np.array_equal(data0, data1)


# =============================================================================
# GpuBufferMeta tests
# =============================================================================


def test_buffer_meta_trivial_dtype():
    """
    Test GpuBufferMeta with trivial (scalar) dtypes like float32, int32.
    """
    # float32: 4 bytes per element
    meta_f32 = GpuBufferMeta(element_count=10, element_dtype=np.float32)
    assert meta_f32.element_count == 10
    assert meta_f32.element_dtype == np.dtype(np.float32)
    assert meta_f32.element_size == 4
    assert meta_f32.size == 40

    # int32: 4 bytes per element
    meta_i32 = GpuBufferMeta(element_count=5, element_dtype=np.int32)
    assert meta_i32.element_count == 5
    assert meta_i32.element_dtype == np.dtype(np.int32)
    assert meta_i32.element_size == 4
    assert meta_i32.size == 20

    # float64: 8 bytes per element
    meta_f64 = GpuBufferMeta(element_count=3, element_dtype=np.float64)
    assert meta_f64.element_count == 3
    assert meta_f64.element_dtype == np.dtype(np.float64)
    assert meta_f64.element_size == 8
    assert meta_f64.size == 24


def test_buffer_meta_structured_dtype():
    """
    Test GpuBufferMeta with structured dtypes (named fields).

    Structured dtypes are common for vertex buffers with multiple attributes
    like position, normal, texcoord packed together.
    """
    # Vertex dtype with position (3 floats), normal (3 floats), texcoord (2 floats)
    # Total: 8 floats = 32 bytes per vertex
    vertex_dtype = np.dtype(
        [
            ("position", np.float32, 3),
            ("normal", np.float32, 3),
            ("texcoord", np.float32, 2),
        ]
    )

    meta = GpuBufferMeta(element_count=100, element_dtype=vertex_dtype)
    assert meta.element_count == 100
    assert meta.element_dtype == vertex_dtype
    assert meta.element_size == 32
    assert meta.size == 3200

    # Verify from_array works correctly with structured dtype
    vertices = np.zeros(100, dtype=vertex_dtype)
    meta_from_arr = GpuBufferMeta.from_array(vertices)
    assert meta_from_arr.element_count == 100
    assert meta_from_arr.element_dtype == vertex_dtype
    assert meta_from_arr.element_size == 32
    assert meta_from_arr.size == 3200


def test_buffer_meta_subarray_dtype():
    """
    Test that GpuBufferMeta rejects subarray dtypes.

    Subarray dtypes like ('<f4', (4, 4)) have surprising behavior in numpy:
    - When you create an array with `np.zeros(n, dtype=subarray_dtype)`, numpy
      expands the shape to include the subarray dimensions.
    - The resulting array's .dtype is the BASE dtype (e.g., float32), not the
      subarray dtype.

    Because of this, GpuBufferMeta rejects subarray dtypes. Use flat dtypes
    (e.g., float32) and handle shaping at a higher level.
    """

    # 4x4 float32 matrix dtype: 16 floats = 64 bytes per matrix
    mat4x4_dtype = np.dtype(("<f4", (4, 4)))

    # Verify the dtype properties that make it problematic
    assert mat4x4_dtype.ndim == 2  # This is a subarray dtype
    assert mat4x4_dtype.itemsize == 64
    assert mat4x4_dtype.base == np.dtype(np.float32)
    assert mat4x4_dtype.shape == (4, 4)

    # GpuBufferMeta should reject this dtype
    with pytest.raises(LogicError, match="does not support subarray dtypes"):
        GpuBufferMeta(element_count=10, element_dtype=mat4x4_dtype)

    # Demonstrate the correct alternative: use flat float32
    # For 10 matrices of 4x4 floats, use 160 float32 elements
    meta = GpuBufferMeta(element_count=160, element_dtype=np.float32)
    assert meta.element_count == 160
    assert meta.element_size == 4
    assert meta.size == 640  # Same total size as 10 matrices


if __name__ == "__main__":
    pytest.main([__file__])
