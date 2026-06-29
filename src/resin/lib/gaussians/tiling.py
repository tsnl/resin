"""Screen-space tiling for 3DGS (Phase 3.5).

Pipeline (host-side binning + GPU tiled blend):
  1. tile_counts — how many gaussians overlap each tile
  2. prefix-sum offsets — exclusive scan of counts (start index per tile)
  3. duplicate-with-keys — one instance per (gaussian × overlapping tile)
  4. sort by (tile_id, depth)
  5. identify_tile_ranges — [start, end) into the sorted instance list
  6. tiled blend — each pixel only walks its tile's range
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from resin.core.etype import F4, U4, ElementType
from resin.dsl.node import CustomNode
from resin.dsl.view import View, zeros
from resin.lib.gaussians.kernels import tiled_gaussian_blend_wgsl
from resin.lib.gaussians.reference import PreprocessResult

if TYPE_CHECKING:
    from resin.ir.ir import IrKernel

DEFAULT_TILE_SIZE = 16


@dataclass(frozen=True)
class TiledLayout:
    """Sorted gaussian-tile instances and per-tile ranges into that list."""

    tile_size: int
    n_tiles_x: int
    n_tiles_y: int
    # Length = n_tiles_x * n_tiles_y; each entry is (start, end) into instance_ids.
    tile_ranges: tuple[tuple[int, int], ...]
    # Gaussian indices, sorted by (tile_id, depth) for front-to-back within a tile.
    instance_ids: tuple[int, ...]

    @property
    def n_tiles(self) -> int:
        return self.n_tiles_x * self.n_tiles_y

    @property
    def n_instances(self) -> int:
        return len(self.instance_ids)


def tile_counts(
    *,
    width: int,
    height: int,
    means2d: tuple[tuple[float, float], ...],
    radii: tuple[float, ...],
    tile_size: int = DEFAULT_TILE_SIZE,
) -> tuple[int, ...]:
    """Count gaussians overlapping each tile (row-major tile order)."""
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size
    counts = [0] * (ntx * nty)
    for (mx, my), radius in zip(means2d, radii):
        # Inflate slightly so binning is a superset of the EWA support.
        for tid in _overlapping_tiles(mx, my, radius * 1.1 + 1.0, ntx, nty, tile_size):
            counts[tid] += 1
    return tuple(counts)


def prefix_sum_offsets(counts: tuple[int, ...]) -> tuple[int, ...]:
    """Exclusive prefix sum of tile counts → write offsets per tile."""
    offsets: list[int] = []
    acc = 0
    for c in counts:
        offsets.append(acc)
        acc += c
    return tuple(offsets)


def duplicate_with_keys(
    *,
    width: int,
    height: int,
    means2d: tuple[tuple[float, float], ...],
    depths: tuple[float, ...],
    radii: tuple[float, ...],
    tile_size: int = DEFAULT_TILE_SIZE,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Emit (sort_key, gaussian_id) for each gaussian–tile overlap.

    ``sort_key`` packs ``tile_id`` in the high bits and a depth rank in the low
    bits so sorting by key yields tile-major, front-to-back order within a tile.
    """
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size
    # Quantize depth to 16 bits (relative to min/max in the batch).
    if depths:
        dmin = min(depths)
        dmax = max(depths)
        span = max(dmax - dmin, 1e-6)
    else:
        dmin, span = 0.0, 1.0

    # Return sort keys as (tile_id, depth) pairs encoded for stable ordering:
    # we keep a parallel float depth list and sort with Python tuples in
    # ``build_tiled_layout`` (avoids 16-bit depth quantization reordering).
    keys: list[int] = []
    ids: list[int] = []
    for gi, ((mx, my), depth, radius) in enumerate(zip(means2d, depths, radii)):
        _ = depth, dmin, span
        for tid in _overlapping_tiles(mx, my, radius * 1.1 + 1.0, ntx, nty, tile_size):
            keys.append(tid)
            ids.append(gi)
    return tuple(keys), tuple(ids)


def identify_tile_ranges(
    sorted_keys: tuple[int, ...],
    *,
    n_tiles: int,
) -> tuple[tuple[int, int], ...]:
    """Scan sorted keys and build [start, end) ranges per tile_id."""
    ranges = [(0, 0)] * n_tiles
    if not sorted_keys:
        return tuple(ranges)

    start = 0
    cur_tile = sorted_keys[0] >> 16
    for i in range(1, len(sorted_keys) + 1):
        tile = sorted_keys[i] >> 16 if i < len(sorted_keys) else -1
        if tile != cur_tile:
            if 0 <= cur_tile < n_tiles:
                ranges[cur_tile] = (start, i)
            start = i
            cur_tile = tile
    return tuple(ranges)


def build_tiled_layout(
    pre: PreprocessResult,
    *,
    width: int,
    height: int,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> TiledLayout:
    """Full host tiling pipeline: counts → duplicate → sort → ranges."""
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size
    n_tiles = ntx * nty

    counts = tile_counts(
        width=width,
        height=height,
        means2d=pre["means2d"],
        radii=pre["radii"],
        tile_size=tile_size,
    )
    _offsets = prefix_sum_offsets(counts)  # available for GPU scatter fill later
    _ = _offsets

    tile_ids, ids = duplicate_with_keys(
        width=width,
        height=height,
        means2d=pre["means2d"],
        depths=pre["depths"],
        radii=pre["radii"],
        tile_size=tile_size,
    )
    # Sort by tile, then true depth (front-to-back), matching untiled order
    # restricted to each tile's overlapping set.
    order = sorted(
        range(len(tile_ids)),
        key=lambda i: (tile_ids[i], pre["depths"][ids[i]]),
    )
    sorted_tile_ids = tuple(tile_ids[i] for i in order)
    sorted_ids = tuple(ids[i] for i in order)
    # identify_tile_ranges expects keys with tile in high bits; pass tile_id << 16.
    ranges = identify_tile_ranges(
        tuple(t << 16 for t in sorted_tile_ids), n_tiles=n_tiles
    )
    return TiledLayout(
        tile_size=tile_size,
        n_tiles_x=ntx,
        n_tiles_y=nty,
        tile_ranges=ranges,
        instance_ids=sorted_ids,
    )


def blend_gaussians_tiled_cpu(
    *,
    width: int,
    height: int,
    means2d: tuple[tuple[float, float], ...],
    conics: tuple[tuple[float, float, float], ...],
    colors: tuple[tuple[float, float, float], ...],
    opacities: tuple[float, ...],
    layout: TiledLayout,
) -> tuple[float, ...]:
    """CPU tiled blend for parity checks against the untiled path."""
    image = [0.0] * (width * height * 3)
    ts = layout.tile_size
    for py in range(height):
        for px in range(width):
            tid = (py // ts) * layout.n_tiles_x + (px // ts)
            start, end = layout.tile_ranges[tid]
            fx = px + 0.5
            fy = py + 0.5
            r = g = b = 0.0
            T = 1.0
            for k in range(start, end):
                gi = layout.instance_ids[k]
                mx, my = means2d[gi]
                c0, c1, c2 = conics[gi]
                cr, cg, cb = colors[gi]
                opacity = opacities[gi]
                dx = fx - mx
                dy = fy - my
                power = -0.5 * (c0 * dx * dx + c2 * dy * dy) - c1 * dx * dy
                if power > 0.0:
                    continue
                alpha = min(0.99, opacity * math.exp(power))
                if alpha < 1.0 / 255.0:
                    continue
                weight = alpha * T
                r += cr * weight
                g += cg * weight
                b += cb * weight
                T *= 1.0 - alpha
                if T < 1e-4:
                    break
            off = (py * width + px) * 3
            image[off] = r
            image[off + 1] = g
            image[off + 2] = b
    return tuple(image)


def _overlapping_tiles(
    mx: float,
    my: float,
    radius: float,
    ntx: int,
    nty: int,
    tile_size: int,
) -> list[int]:
    if radius < 1.0:
        radius = 1.0
    ftx0 = (mx - radius) / tile_size
    ftx1 = (mx + radius) / tile_size
    fty0 = (my - radius) / tile_size
    fty1 = (my + radius) / tile_size
    # Fully outside the image → no tiles (do not clamp into edge tiles).
    if ftx1 < 0.0 or fty1 < 0.0 or ftx0 >= ntx or fty0 >= nty:
        return []
    tx0 = max(0, min(ntx - 1, int(math.floor(ftx0))))
    tx1 = max(0, min(ntx - 1, int(math.floor(ftx1))))
    ty0 = max(0, min(nty - 1, int(math.floor(fty0))))
    ty1 = max(0, min(nty - 1, int(math.floor(fty1))))
    out: list[int] = []
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            out.append(ty * ntx + tx)
    return out


@dataclass(kw_only=True, frozen=True, eq=False)
class TiledGaussianBlendNode(CustomNode):
    """Tiled alpha blend (single image output to stay within WebGPU 8-buffer limit).

    Bindings: output image + means2d, conics, colors, opacities, instances, ranges
    = 7 storage buffers (device limit is often 8).
    """

    width: int
    height: int
    tile_size: int
    n_tiles_x: int
    n_tiles_y: int
    n_instances: int

    def output_ports(self) -> tuple[str, ...]:
        return ("image",)

    def port_shape(self, port: str) -> tuple[int, ...]:
        if port != "image":
            raise KeyError(port)
        return (self.height, self.width, 3)

    def port_etype(self, port: str) -> ElementType:
        if port != "image":
            raise KeyError(port)
        return F4

    def build_kernel(self, *, used_ports: frozenset[str]) -> "IrKernel":
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports
        wg = (self.width * self.height + 63) // 64
        return IrWgslMultiOutputKernel(
            arg_accessors=tuple(a.accessor for a in self.args),
            etype=F4,
            shape=(self.height, self.width, 3),
            wgsl=tiled_gaussian_blend_wgsl(
                width=self.width,
                height=self.height,
                tile_size=self.tile_size,
                n_tiles_x=self.n_tiles_x,
                n_instances=self.n_instances,
            ),
            entry_point="main",
            dispatch_size=(wg, 1, 1),
            arg_etypes=tuple(a.etype for a in self.args),
            output_etypes=(F4,),
            num_outputs=1,
            clear_output_before_dispatch=True,
        )

    def df_do_ports(self, df_douts: dict[str, View]) -> tuple[View, ...]:
        _ = df_douts
        means2d, conics, colors, opacities, instances, ranges = self.args
        return (
            zeros(means2d.shape, etype=F4),
            zeros(conics.shape, etype=F4),
            zeros(colors.shape, etype=F4),
            zeros(opacities.shape, etype=F4),
            zeros(instances.shape, etype=U4),
            zeros(ranges.shape, etype=U4),
        )


def gaussian_blend_tiled(
    *,
    width: int,
    height: int,
    means2d: View,
    conics: View,
    colors: View,
    opacities: View,
    instances: View,
    tile_ranges: View,
    tile_size: int,
    n_tiles_x: int,
    n_tiles_y: int,
) -> View:
    """Return the ``image`` port of a TiledGaussianBlendNode."""
    n_instances = instances.shape[0]
    node = TiledGaussianBlendNode(
        shape=(height, width, 3),
        etype=F4,
        args=(means2d, conics, colors, opacities, instances, tile_ranges),
        width=width,
        height=height,
        tile_size=tile_size,
        n_tiles_x=n_tiles_x,
        n_tiles_y=n_tiles_y,
        n_instances=n_instances,
    )
    return View.identity(node, port="image")
