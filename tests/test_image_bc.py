import numpy as np
import zfw

from conftest import GpuFixture


def test_image_bc(
    gpu: GpuFixture,
    rainbow_512x512_image: zfw.ImageResource,
):
    assert rainbow_512x512_image.data.dtype == np.float32
    assert rainbow_512x512_image.data.shape == (512, 512, 4)

    # zfw.compress_bc6h(rainbow_512x512_image.data)

    # TODO: compress rainbow_512x512_image.data to BC6H, use as a texture for draw2d
    renderer = zfw.Draw2dRenderer(
        device=gpu.device,
        queue=gpu.queue,
        target_size_wh=(512, 512),
        target_format="rgba8unorm",
    )
    frame = zfw.Draw2dFrame(renderer=renderer)

    _ = frame

    LOG.warning("test_image_bc is not implemented yet")


LOG = zfw.logger(__name__)
