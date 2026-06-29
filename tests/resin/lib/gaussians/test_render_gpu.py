"""GPU smoke tests for untiled 3DGS forward blend and color gradients."""

from typing import cast

import pytest

from resin import dsl
from resin.core.etype import F4, U4
from resin.core.pytree import PyTensor
from resin.grad import grad
from resin.lib.gaussians.blend import gaussian_blend
from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import blend_gaussians_cpu, sort_by_depth_cpu
from tests.resin.gpu.interp_helpers import run_graph


def _pack_means2d(means2d: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[mx, my] for mx, my in means2d]


def _pack_triplets(
    items: tuple[tuple[float, float, float], ...],
) -> list[list[float]]:
    return [[a, b, c] for a, b, c in items]


def test_sort_depths_gpu() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=32, height=32)
    depths = dsl.param(shape=(len(pre["depths"]),), etype=F4)
    _values, perm = depths.sort()
    cpu_order = sort_by_depth_cpu(pre["depths"])
    gpu_order = [int(x) for x in run_graph(perm, params={depths: list(pre["depths"])})]
    assert tuple(gpu_order) == cpu_order


def test_blend_matches_cpu_reference() -> None:
    width = height = 32
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    order = sort_by_depth_cpu(pre["depths"])

    means_p = dsl.param(shape=(len(pre["means2d"]), 2), etype=F4)
    conics_p = dsl.param(shape=(len(pre["conics"]), 3), etype=F4)
    colors_p = dsl.param(shape=(len(pre["colors"]), 3), etype=F4)
    opacities_p = dsl.param(shape=(len(pre["opacities"]),), etype=F4)
    order_p = dsl.const(list(order), etype=U4)

    image = gaussian_blend(
        width=width,
        height=height,
        means2d=means_p,
        conics=conics_p,
        colors=colors_p,
        opacities=opacities_p,
        order=order_p,
    )

    cpu_image = blend_gaussians_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )

    gpu_image = run_graph(
        image,
        params={
            means_p: cast(PyTensor, _pack_means2d(pre["means2d"])),
            conics_p: cast(PyTensor, _pack_triplets(pre["conics"])),
            colors_p: cast(PyTensor, _pack_triplets(pre["colors"])),
            opacities_p: cast(PyTensor, list(pre["opacities"])),
        },
    )

    assert len(gpu_image) == len(cpu_image)
    for a, b in zip(gpu_image, cpu_image):
        assert a == pytest.approx(b, abs=1e-4)


def test_color_grad_smoke() -> None:
    """Reconstruction loss pulls colors toward a target image."""
    width = height = 16
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    order = sort_by_depth_cpu(pre["depths"])
    n = len(pre["depths"])

    means_p = dsl.param(shape=(n, 2), etype=F4, name="means2d")
    conics_p = dsl.param(shape=(n, 3), etype=F4, name="conics")
    colors_p = dsl.param(shape=(n, 3), etype=F4, name="colors")
    opacities_p = dsl.param(shape=(n,), etype=F4, name="opacities")
    order_p = dsl.const(list(order), etype=U4)
    target = dsl.param(shape=(height, width, 3), etype=F4, name="target")

    image = gaussian_blend(
        width=width,
        height=height,
        means2d=means_p,
        conics=conics_p,
        colors=colors_p,
        opacities=opacities_p,
        order=order_p,
    )
    # Scalar MSE loss.
    loss = ((image - target) * (image - target)).sum().squeeze(axes=(0, 1, 2))
    d_colors = grad(loss, wrt=colors_p)

    target_img = blend_gaussians_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )
    # Perturb colors so loss is nonzero.
    colors_bad = [[c * 0.5 for c in rgb] for rgb in pre["colors"]]
    grads = run_graph(
        d_colors,
        params={
            means_p: cast(PyTensor, _pack_means2d(pre["means2d"])),
            conics_p: cast(PyTensor, _pack_triplets(pre["conics"])),
            colors_p: cast(PyTensor, colors_bad),
            opacities_p: cast(PyTensor, list(pre["opacities"])),
            target: cast(PyTensor, [
                [list(target_img[i : i + 3]) for i in range(r * width * 3, (r + 1) * width * 3, 3)]
                for r in range(height)
            ]),
        },
    )
    # At least one color channel receives a nonzero gradient.
    assert any(abs(g) > 1e-8 for g in grads)
