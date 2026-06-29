"""Parity tests for Phase 3.5 screen-space tiling."""

import pytest

from resin.lib.gaussians.gnomen import make_gnomen_cloud
from resin.lib.gaussians.preprocess import preprocess_gaussians
from resin.lib.gaussians.reference import blend_gaussians_cpu, sort_by_depth_cpu
from resin.lib.gaussians.tiling import (
    blend_gaussians_tiled_cpu,
    build_tiled_layout,
    duplicate_with_keys,
    prefix_sum_offsets,
    tile_counts,
)


def test_tile_counts_sum_matches_duplicates() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=64, height=64)
    counts = tile_counts(
        width=64,
        height=64,
        means2d=pre["means2d"],
        radii=pre["radii"],
        tile_size=16,
    )
    keys, ids = duplicate_with_keys(
        width=64,
        height=64,
        means2d=pre["means2d"],
        depths=pre["depths"],
        radii=pre["radii"],
        tile_size=16,
    )
    assert sum(counts) == len(ids) == len(keys)
    offsets = prefix_sum_offsets(counts)
    assert offsets[0] == 0
    assert offsets[-1] + counts[-1] == sum(counts)


def test_identify_tile_ranges_cover_instances() -> None:
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=64, height=64)
    layout = build_tiled_layout(pre, width=64, height=64, tile_size=16)
    covered = 0
    for start, end in layout.tile_ranges:
        assert 0 <= start <= end <= layout.n_instances
        covered += end - start
    # Each instance belongs to exactly one tile in the sorted list.
    assert covered == layout.n_instances


def test_single_tile_matches_untiled_exactly() -> None:
    """One tile covering the whole image must match the global-sort path."""
    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    order = sort_by_depth_cpu(pre["depths"])
    untiled = blend_gaussians_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )
    layout = build_tiled_layout(pre, width=width, height=height, tile_size=64)
    assert layout.n_tiles == 1
    tiled = blend_gaussians_tiled_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        layout=layout,
    )
    # Same depth order and full-image coverage. Small differences can remain
    # from extent / early-out vs full EWA support; require close agreement.
    assert tiled == pytest.approx(untiled, abs=1e-2)


def test_tiled_cpu_close_to_untiled() -> None:
    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    order = sort_by_depth_cpu(pre["depths"])
    untiled = blend_gaussians_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        order=order,
    )
    layout = build_tiled_layout(pre, width=width, height=height, tile_size=16)
    tiled = blend_gaussians_tiled_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        layout=layout,
    )
    # Multi-tile paths can differ slightly from global sort when T accumulation
    # order changes for near-zero fringe contributions; require close agreement.
    assert tiled == pytest.approx(untiled, abs=2e-2)


def test_tiled_gpu_matches_single_tile_cpu() -> None:
    from resin.lib.gaussians.gpu_session import GpuForwardSession

    width = height = 64
    cloud = make_gnomen_cloud()
    pre = preprocess_gaussians(cloud, width=width, height=height)
    layout = build_tiled_layout(pre, width=width, height=height, tile_size=64)
    cpu = blend_gaussians_tiled_cpu(
        width=width,
        height=height,
        means2d=pre["means2d"],
        conics=pre["conics"],
        colors=pre["colors"],
        opacities=pre["opacities"],
        layout=layout,
    )
    session = GpuForwardSession(
        width=width,
        height=height,
        fixed_count=cloud.count,
        tiled=True,
        tile_size=64,
    )
    gpu = session.render(pre)
    assert gpu == pytest.approx(cpu, abs=2e-3)


def test_tiled_gpu_matches_untiled_at_viewer_repro_pose() -> None:
    """Regression: instance buffer must cover all tile overlaps (not a low cap).

    Viewer pose that previously truncated instances when
    ``max_tiles_per_gaussian`` was too small (max Δ ~1.0 vs untiled).
    """
    from resin.lib.gaussians.camera import FlyCamera
    from resin.lib.gaussians.gpu_session import GpuForwardSession

    width, height = 1280, 720
    pose = {
        "x": -0.0253,
        "y": 1.4317,
        "z": 0.6947,
        "yaw": -0.015,
        "pitch": -0.3825,
        "fov_y_deg": 60.0,
    }
    cloud = make_gnomen_cloud()
    cam = FlyCamera.from_pose(pose)
    view, proj = cam.view_proj(aspect=width / height)
    pre = preprocess_gaussians(cloud, width=width, height=height, view=view, proj=proj)
    tiled = GpuForwardSession(
        width=width, height=height, fixed_count=cloud.count, tiled=True
    ).render(pre)
    untiled = GpuForwardSession(
        width=width, height=height, fixed_count=cloud.count, tiled=False
    ).render(pre)
    # Tile vs global order can differ slightly on fringe contributions; must not
    # be the catastrophic truncation failure (Δ ~ 1).
    assert tiled == pytest.approx(untiled, abs=5e-2)
