from pathlib import Path

import numpy as np

from .images import (
    load_rgba_image,
    save_rgba_image,
    convert_srgb_to_linear,
    convert_linear_to_srgb,
)


def test_image_roundtrip():
    INPUT_PATH = Path("test_data/rainbow-512x512.png")
    OUTPUT_PATH = Path("output/zfw/images_test/test_image_roundtrip.png")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    im0 = load_rgba_image(file_path=INPUT_PATH)
    assert im0.shape == (512, 512, 4)

    save_rgba_image(file_path=OUTPUT_PATH, data=im0)

    im1 = load_rgba_image(file_path=OUTPUT_PATH)
    assert im1.shape == im0.shape
    np.testing.assert_allclose(im0, im1, rtol=1e-4, atol=1e-2)

    # FIXME: Why is atol so high here? Colorspace conversion is not lossy, see below
    # test atol: test_srgb_to_linear_to_srgb_roundtrip


def test_srgb_to_linear_to_srgb_roundtrip():
    ITER_COUNT = 1024

    np.random.seed(42)
    for _ in range(ITER_COUNT):
        srgb = np.random.uniform(low=0.0, high=1.0, size=(4,)).astype(np.float32)
        linear = convert_srgb_to_linear(srgb)
        srgb_roundtrip = convert_linear_to_srgb(linear)
        np.testing.assert_allclose(srgb, srgb_roundtrip, atol=1e-7)
