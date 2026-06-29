"""GPU tests for 3DGS sort and blend kernels."""

from typing import cast

import pytest

from resin import dsl
from resin.core.pytree import PyTensor
from resin.core.etype import F4, U4
from resin.gs.gnomen import make_gnomen_cloud
from resin.gs.reference import (
    blend_gaussians_cpu,
    preprocess_gaussians_cpu,
    sort_by_depth_cpu,
)
from resin.gs.render import argsort_depths, gaussian_blend, pack_means2d, pack_triplets

from tests.resin.gpu.interp_helpers import run_graph


class TestArgsortGpu:
    def test_sort_gnomen_depths(self, capsys: pytest.CaptureFixture[str]) -> None:
        cloud = make_gnomen_cloud()
        pre = preprocess_gaussians_cpu(cloud, width=32, height=32)
        depths = dsl.param(shape=(len(pre["depths"]),), etype=F4)
        order_view = argsort_depths(depths)
        cpu_order = sort_by_depth_cpu(pre["depths"])
        gpu_order_raw = run_graph(
            order_view,
            params={depths: list(pre["depths"])},
        )
        gpu_order = [int(x) for x in gpu_order_raw]
        print(f"cpu order: {cpu_order}")
        print(f"gpu order: {tuple(gpu_order)}")
        captured = capsys.readouterr()
        assert "cpu order" in captured.out
        assert tuple(gpu_order) == cpu_order


class TestBlendGpu:
    def test_blend_matches_cpu_reference(self, capsys: pytest.CaptureFixture[str]) -> None:
        width = height = 32
        cloud = make_gnomen_cloud()
        pre = preprocess_gaussians_cpu(cloud, width=width, height=height)
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
                means_p: cast(PyTensor, pack_means2d(pre["means2d"])),
                conics_p: cast(PyTensor, pack_triplets(pre["conics"])),
                colors_p: cast(PyTensor, pack_triplets(pre["colors"])),
                opacities_p: cast(PyTensor, list(pre["opacities"])),
            },
        )

        cx, cy = 16, 16
        off = (cy * width + cx) * 3
        cpu_center = cpu_image[off : off + 3]
        gpu_center = gpu_image[off : off + 3]
        print(f"cpu center: {cpu_center}")
        print(f"gpu center: {gpu_center}")
        captured = capsys.readouterr()
        assert "cpu center" in captured.out

        assert len(gpu_image) == len(cpu_image)
        assert gpu_image == pytest.approx(list(cpu_image), rel=1e-4, abs=1e-4)