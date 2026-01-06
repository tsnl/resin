"""
BC4 and BC6H image compression and decompression.
"""

__all__ = [
    "encode_bc1",
    "encode_bc4",
]

import numpy as np
import numpy.typing as npt
import numba

from .basic import NUMBA_CACHE_ENABLED

#
# BC1 encode: compress RGB images
#


def encode_bc1(input_: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    Compress a 3-channel RGB image to BC1 format.

    We do not support BC1 with alpha.

    :param data: input image data: shape (h, w, 3), dtype float32 with h%4 == w%4 == 0.
    :returns: Compressed BC1 data: shape (height/4, width/4, 8), dtype uint8.
    """

    # Ref:
    # https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc1
    # https://www.ludicon.com/castano/blog/2022/11/bc1-compression-revisited/
    # https://fgiesen.wordpress.com/2022/11/08/whats-that-magic-computation-in-stb__refineblock/

    # BC1 encoding is similar to BC4 encoding, but for RGB data.
    # Each 4x4 block is compressed to 8 bytes:
    # - 2 bytes: endpoint0 (RGB565)
    # - 2 bytes: endpoint1 (RGB565)
    # - 4 bytes: 16 2-bit indices, each selecting one of 4 colors in the palette.
    #   - Palette: endpoint0, endpoint1, and two interpolated colors with 1/3, 2/3
    #     weights for endpoint0 respectively.

    # From Ignacio Castaño's blog:
    # > A simple BC1 encoding strategy is to compute the initial indices using a simple
    # > heuristic, to then recompute the endpoints solving the above equation, and
    # > finally to update the indices based on the most recent endpoints. This process
    # > can be repeated multiple times until the error does not go down anymore. This is
    # > the strategy employed by the stb_dxt.h encoder, and just as I was writing this
    # > article Fabien Giesen wrote another blog post describing that implementation in
    # > more detail.

    # From Fabien Giesen's blog:
    # > The basic algorithm uses the same primitives most BC1 encoders use (I’ll assume
    # > in the following you know how BC1 works): compute the average and covariance
    # > matrix of the block of pixels, compute the principal component of the covariance
    # > to get an initial guess for what direction the vector between the two endpoints
    # > should point in. Then we project all pixel values onto that vector to find the
    # > min/max support points in that direction as the initial seed endpoints (which
    # > determines the initial palette), assign each pixel the palette entry closest to
    # > it, and do some iterative refinement of the whole thing.

    # Fabien's power iteration method iteratively computes the principal component of
    # the covariance matrix. Here, we simply use `np.linalg.eig` to compute the
    # eigenvectors directly.

    h, w, d = input_.shape
    if h % 4 != 0 or w % 4 != 0:
        raise ValueError("Input image dimensions must be multiples of 4")
    if d != 3:
        raise ValueError("Input image must have three channels")
    if input_.dtype != np.float32:
        raise ValueError("Input image must have dtype float32")

    # Convert float32 [0,1] to uint8 [0,255]
    input_u8 = np.round(input_ * 255.0).astype(np.uint8)

    return _encode_bc1_impl(input_u8)


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _encode_bc1_impl(input_: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w, d = input_.shape
    assert d == 3

    output = np.zeros((h // 4, w // 4, 8), dtype=np.uint8)

    for by in range(h // 4):
        for bx in range(w // 4):
            y_begin = by * 4
            x_begin = bx * 4
            input_block = input_[y_begin : y_begin + 4, x_begin : x_begin + 4, :]

            endpoints, indices = _encode_bc1_block(block=input_block)
            output[by, bx] = _pack_bc1_block(endpoints, indices)

    return output


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _encode_bc1_block(
    block: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """
    Encode a single 4x4 block to BC1 format.

    :param block: input block: shape (4, 4, 3), dtype uint8.
    :returns: A 2-tuple of:
        - endpoints: palette endpoints, aka color0 and color1: shape (2, 3), dtype uint8.
        - indices: shape (16,) (row-major), dtype uint8.
    """

    # Flatten the block into a list of colors, normalized to `f32`:
    block_colors = np.empty((16, 3), dtype=np.float32)
    for y in range(4):
        for x in range(4):
            block_colors[y * 4 + x] = block[y, x].astype(np.float32) / 255.0

    # Next, compute mean and 3x3 covariance matrix of the block, characterizing a
    # 3D Gaussian distribution of colors in the block:
    mean = np.average(block_colors, axis=0)
    cov = np.cov(block_colors, rowvar=True)

    # Compute principal component of covariance matrix to find the largest axis of an
    # "ellipsoid" fitting the color distribution.
    # Note that the eigenvectors returned by `np.linalg.eig` are column-major.
    # Helpful resource:
    # https://users.cs.utah.edu/~tch/CS4640F2019/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf
    cov_eigenvalues, cov_eigenvectors_t = np.linalg.eig(cov)
    cov_eigenvectors = cov_eigenvectors_t.T
    principal_component = cov_eigenvectors[np.argmax(cov_eigenvalues), :]

    # Identify extreme points along the principal component axis:
    projections = np.dot(block_colors - mean[np.newaxis, :], principal_component)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    endpoint0_f = block_colors[min_idx]
    endpoint1_f = block_colors[max_idx]

    # Quantize endpoints to RGB565: these would fit in a uint8[3], but we use int32 for
    # easier calculations later.
    endpoints = np.array(
        [
            _rgbf_to_rgb565_unpacked(endpoint0_f),
            _rgbf_to_rgb565_unpacked(endpoint1_f),
        ],
        dtype=np.int32,
    )
    assert endpoints.shape == (2, 3)

    # Build palette with 4 colors: endpoint0, endpoint1, and two interpolated colors.
    palette = np.empty((4, 3), dtype=np.int32)
    palette[0, :] = endpoints[0]
    palette[1, :] = endpoints[1]
    palette[2, :] = (2.0 * endpoints[0] + 1.0 * endpoints[1]) / 3.0
    palette[3, :] = (1.0 * endpoints[0] + 2.0 * endpoints[1]) / 3.0

    # Assign indices based on closest palette color, using quantized palette for best
    # results:
    indices = np.empty((4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            a = np.int32(block_colors[y * 4 + x])

            i_min = 0
            b_min = np.int32(palette[i_min])
            e_min = np.linalg.norm(a - b_min)

            for i in range(1, 4):
                b = np.int32(palette[i])
                e = np.linalg.norm(a - b)
                if e < e_min:
                    i_min = i
                    b_min = b
                    e_min = e

            indices[y, x] = np.uint8(i_min)

    # Return:
    return endpoints.astype(np.uint8), indices.ravel()


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _rgbf_to_rgb565_unpacked(
    color_f32: npt.NDArray[np.float32],
) -> npt.NDArray[np.int32]:
    """
    Convert an RGB color from float32 [0,1] to RGB565 format.
    :param color_f32: input color: shape (3,), dtype float32.
    :returns: RGB565 color: shape (3,), dtype int32.
    """

    res = np.empty(3, dtype=np.int32)
    res[0] = (color_f32[0] * 31.0).astype(np.int32) & 0x1F
    res[1] = (color_f32[1] * 63.0).astype(np.int32) & 0x3F
    res[2] = (color_f32[2] * 31.0).astype(np.int32) & 0x1F
    return res


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _pack_bc1_block(
    endpoints: npt.NDArray[np.uint8],
    indices: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    """
    Packs BC1 block data into 8 bytes.

    :param endpoints: palette endpoints, aka color0 and color1: shape (2, 3), dtype uint8.
    :param indices: shape (16,) (row-major), dtype uint8.
    :returns: Packed BC1 block: shape (8,), dtype uint8.
    """

    packed = np.empty(8, dtype=np.uint8)
    packed[0:2] = _pack_rgb565(endpoints[0]).view(np.uint8)
    packed[2:4] = _pack_rgb565(endpoints[1]).view(np.uint8)
    for index_offset, index in enumerate(indices):
        bit_offset = index_offset * 2
        byte_index = bit_offset // 8
        byte_offset = bit_offset % 8
        packed[4 + byte_index] |= (index & 0x3) << byte_offset
    return packed


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _pack_rgb565(color: npt.NDArray[np.uint8]) -> np.uint16:
    r = np.uint16(color[0] & 0x1F)
    g = np.uint16(color[1] & 0x3F)
    b = np.uint16(color[2] & 0x1F)
    return (r << 11) | (g << 5) | b


#
# BC4 encode: compress mono images
#


def encode_bc4(input_: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    Compress a single-channel image to BC4 format.

    :param data: input image data: shape (h, w, 1), dtype float32 with h%4 == w%4 == 0.
    :returns: Compressed BC4 data: shape (height/4, width/4, 8), dtype uint8.
    """

    # Tutorial on BC4:
    # https://acefanatic02.github.io/posts/intro_bcn_part1/

    # Resources:
    # https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression#bc4

    # i | 8-color mode (endpoint0 > endpoint1) | 6-color mode (endpoint0 <= endpoint1)
    # ---------------------------------------------------------------------------------
    # 0 | endpoint0                            | endpoint0
    # 1 | endpoint1                            | endpoint1
    # 2 | (endpoint0 * 6 + endpoint1 * 1) / 7  | (endpoint0 * 4 + endpoint1 * 1) / 5
    # 3 | (endpoint0 * 5 + endpoint1 * 2) / 7  | (endpoint0 * 3 + endpoint1 * 2) / 5
    # 4 | (endpoint0 * 4 + endpoint1 * 3) / 7  | (endpoint0 * 2 + endpoint1 * 3) / 5
    # 5 | (endpoint0 * 3 + endpoint1 * 4) / 7  | (endpoint0 * 1 + endpoint1 * 4) / 5
    # 6 | (endpoint0 * 2 + endpoint1 * 5) / 7  | 0
    # 7 | (endpoint0 * 1 + endpoint1 * 6) / 7  | 255

    h, w, d = input_.shape
    if h % 4 != 0 or w % 4 != 0:
        raise ValueError("Input image dimensions must be multiples of 4")
    if d != 1:
        raise ValueError("Input image must have a single channel")
    if input_.dtype != np.float32:
        raise ValueError("Input image must have dtype float32")

    # Convert float32 [0,1] to uint8 [0,255]
    input_u8 = np.round(input_ * 255.0).astype(np.uint8)

    return _encode_bc4_impl(input_u8)


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _encode_bc4_impl(input_: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    h, w, d = input_.shape
    assert d == 1

    output = np.empty((h // 4, w // 4, 8), dtype=np.uint8)

    for by in range(h // 4):
        for bx in range(w // 4):
            y_begin = by * 4
            x_begin = bx * 4
            input_block = input_[y_begin : y_begin + 4, x_begin : x_begin + 4, 0]

            endpoint_min = np.int32(input_block.min())
            endpoint_max = np.int32(input_block.max())

            idx_8c, err_8c = _encode_bc4_block(
                block=input_block,
                palette=np.array(
                    [
                        endpoint_max,
                        endpoint_min,
                        (endpoint_max * 6 + endpoint_min * 1) / 7,
                        (endpoint_max * 5 + endpoint_min * 2) / 7,
                        (endpoint_max * 4 + endpoint_min * 3) / 7,
                        (endpoint_max * 3 + endpoint_min * 4) / 7,
                        (endpoint_max * 2 + endpoint_min * 5) / 7,
                        (endpoint_max * 1 + endpoint_min * 6) / 7,
                    ],
                    dtype=np.uint8,
                ),
            )
            idx_6c, err_6c = _encode_bc4_block(
                input_block,
                palette=np.array(
                    [
                        endpoint_min,
                        endpoint_max,
                        (endpoint_min * 4 + endpoint_max * 1) / 5,
                        (endpoint_min * 3 + endpoint_max * 2) / 5,
                        (endpoint_min * 2 + endpoint_max * 3) / 5,
                        (endpoint_min * 1 + endpoint_max * 4) / 5,
                        0x00,
                        0xFF,
                    ],
                    dtype=np.uint8,
                ),
            )

            use_8c = err_8c < err_6c

            output[by, bx] = _pack_bc4_block(
                endpoint_min=np.uint8(endpoint_min),
                endpoint_max=np.uint8(endpoint_max),
                indices=idx_8c if use_8c else idx_6c,
                mode_is_8c_not_6c=use_8c,
            )

    return output


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _encode_bc4_block(
    block: npt.NDArray[np.uint8],
    palette: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], float]:
    indices = np.empty((4, 4), dtype=np.uint8)
    error_sq_sum = 0.0

    for y in range(4):
        for x in range(4):
            a = np.int32(block[y, x])

            i_min = 0
            b_min = np.int32(palette[i_min])

            for i in range(1, 8):
                b = np.int32(palette[i])
                if abs(a - b) < abs(a - b_min):
                    i_min = i
                    b_min = b

            indices[y, x] = i_min
            error_sq_sum += (a - b_min) ** 2

    return indices.flatten(), error_sq_sum


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def _pack_bc4_block(
    endpoint_min: np.uint8,
    endpoint_max: np.uint8,
    indices: npt.NDArray[np.uint8],
    mode_is_8c_not_6c: bool,
) -> npt.NDArray[np.uint8]:
    packed_indices_u64 = np.zeros(1, dtype=np.uint64)
    for i in range(16):
        # Cast to uint64 BEFORE shifting
        packed_indices_u64[0] |= np.int32(indices[i] & 0x7) << (3 * i)

    packed = np.empty(8, dtype=np.uint8)
    packed[0], packed[1] = (
        (endpoint_max, endpoint_min)
        if mode_is_8c_not_6c
        else (endpoint_min, endpoint_max)
    )
    packed[2:] = packed_indices_u64.view(np.uint8)[:6]

    return packed
