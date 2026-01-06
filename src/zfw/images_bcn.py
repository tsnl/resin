"""
BC4 and BC6H image compression and decompression.
"""

import numpy as np
import numpy.typing as npt
import numba

from .basic import NUMBA_CACHE_ENABLED


def compress_bc4(input_: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    Compress a single-channel image to BC4 format.

    :param data: input image data: shape (h, w, 1), dtype float32 with h%4 == w%4 == 0.
    :returns: Compressed BC4 data: shape (height/4, width/4, 8), dtype uint8.
    """

    # Tutorial on BC4:
    # https://acefanatic02.github.io/posts/intro_bcn_part1/

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

    return _compress_bc4_impl(input_)


# @numba.njit(cache=False)  # Disabled for debugging
def _compress_bc4_impl(input_: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    h, w, d = input_.shape
    assert d == 1

    output = np.empty((h // 4, w // 4, 8), dtype=np.uint8)

    for by in range(h // 4):
        for bx in range(w // 4):
            y_begin = by * 4
            x_begin = bx * 4
            input_block = input_[y_begin : y_begin + 4, x_begin : x_begin + 4, 0]

            endpoint_min = np.min(input_block)
            endpoint_max = np.max(input_block)

            idx_8c, err_8c = _compress_bc4_block(
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
                    dtype=np.float32,
                ),
            )
            idx_6c, err_6c = _compress_bc4_block(
                input_block,
                palette=np.array(
                    [
                        endpoint_min,
                        endpoint_max,
                        (endpoint_min * 4 + endpoint_max * 1) / 5,
                        (endpoint_min * 3 + endpoint_max * 2) / 5,
                        (endpoint_min * 2 + endpoint_max * 3) / 5,
                        (endpoint_min * 1 + endpoint_max * 4) / 5,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float32,
                ),
            )

            use_8c = err_8c < err_6c

            output[by, bx] = _pack_bc4_block(
                endpoint_min=np.uint8(endpoint_min * 255.0),
                endpoint_max=np.uint8(endpoint_max * 255.0),
                indices=idx_8c if use_8c else idx_6c,
                mode_is_8c_not_6c=use_8c,
            )

    return output


# @numba.njit(cache=NUMBA_CACHE_ENABLED)  # Disabled for debugging
def _compress_bc4_block(
    block: npt.NDArray[np.float32],
    palette: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], float]:
    indices = np.empty((4, 4), dtype=np.uint8)
    error_sq_sum = 0.0

    for y in range(4):
        for x in range(4):
            a = block[y, x]

            i_min = 0
            b_min = palette[i_min]

            for i in range(1, 8):
                b = palette[i]
                if np.abs(a - b) < np.abs(a - b_min):
                    i_min = i
                    b_min = b

            indices[y, x] = i_min
            error_sq_sum += (a - b_min) ** 2

    return indices.flatten(), error_sq_sum


# @numba.njit(cache=NUMBA_CACHE_ENABLED)  # Disabled for debugging
def _pack_bc4_block(
    endpoint_min: np.uint8,
    endpoint_max: np.uint8,
    indices: npt.NDArray[np.uint8],
    mode_is_8c_not_6c: bool,
) -> npt.NDArray[np.uint8]:
    packed_indices_u64 = np.zeros(1, dtype=np.uint64)
    for i in range(16):
        # Cast to uint64 BEFORE shifting
        packed_indices_u64[0] |= int(indices[i] & 0x7) << (3 * i)

    packed = np.empty(8, dtype=np.uint8)
    packed[0], packed[1] = (
        (endpoint_max, endpoint_min)
        if mode_is_8c_not_6c
        else (endpoint_min, endpoint_max)
    )
    packed[2:] = packed_indices_u64.view(np.uint8)[:6]

    return packed
