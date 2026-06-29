"""GPU tiling binning composed mainly from builtin PrefixSum + radix Sort + Remap.

Library-level CustomNodes (not something app authors write) only for irregular
expand / segment-range steps. End users of the viewer/session never touch them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from resin.core.accessor import Accessor
from resin.core.etype import F4, U4, ElementType
from resin.dsl.node import CustomNode, DEFAULT_PORT, RemapGatherInfo
from resin.dsl.view import View, const
from resin.lib.gaussians.tiling import DEFAULT_TILE_SIZE, gaussian_blend_tiled

if TYPE_CHECKING:
    from resin.ir.ir import IrKernel


@dataclass(kw_only=True, frozen=True, eq=False)
class TileCountNode(CustomNode):
    """Atomic tile histogram: how many gaussians overlap each tile."""

    width: int
    height: int
    tile_size: int
    n_gaussians: int

    def build_kernel(self, *, used_ports: frozenset[str]) -> IrKernel:
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports
        n = self.n_gaussians
        ts = self.tile_size
        ntx = (self.width + ts - 1) // ts
        nty = (self.height + ts - 1) // ts
        wgsl = f"""
@group(0) @binding(0)
var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(1)
var<storage, read> means2d: array<f32>;
@group(0) @binding(2)
var<storage, read> radii: array<f32>;

const N: u32 = {n}u;
const TS: u32 = {ts}u;
const NTX: u32 = {ntx}u;
const NTY: u32 = {nty}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let g = gid.x;
    if (g >= N) {{ return; }}
    let mx = means2d[g * 2u];
    let my = means2d[g * 2u + 1u];
    var rad = radii[g] * 1.1 + 1.0;
    if (rad < 1.0) {{ rad = 1.0; }}
    var tx0 = i32(floor((mx - rad) / f32(TS)));
    var tx1 = i32(floor((mx + rad) / f32(TS)));
    var ty0 = i32(floor((my - rad) / f32(TS)));
    var ty1 = i32(floor((my + rad) / f32(TS)));
    tx0 = clamp(tx0, 0, i32(NTX) - 1);
    tx1 = clamp(tx1, 0, i32(NTX) - 1);
    ty0 = clamp(ty0, 0, i32(NTY) - 1);
    ty1 = clamp(ty1, 0, i32(NTY) - 1);
    for (var ty = ty0; ty <= ty1; ty++) {{
        for (var tx = tx0; tx <= tx1; tx++) {{
            let tid = u32(ty) * NTX + u32(tx);
            atomicAdd(&output[tid], 1u);
        }}
    }}
}}
"""
        threads = (n + 255) // 256
        return IrWgslMultiOutputKernel(
            arg_accessors=(self.args[0].accessor, self.args[1].accessor),
            etype=U4,
            shape=self.shape,
            wgsl=wgsl,
            entry_point="main",
            dispatch_size=(max(threads, 1), 1, 1),
            arg_etypes=(F4, F4),
            output_etypes=(U4,),
            num_outputs=1,
            clear_output_before_dispatch=True,
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class TileFillNode(CustomNode):
    """Write (key, gaussian_id) instances using exclusive tile offsets (atomics)."""

    width: int
    height: int
    tile_size: int
    n_gaussians: int
    max_instances: int

    def output_ports(self) -> tuple[str, ...]:
        return ("keys", "ids")

    def port_shape(self, port: str) -> tuple[int, ...]:
        if port not in ("keys", "ids"):
            raise KeyError(port)
        return (self.max_instances,)

    def port_etype(self, port: str) -> ElementType:
        return U4

    def build_kernel(self, *, used_ports: frozenset[str]) -> IrKernel:
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports
        n = self.n_gaussians
        ts = self.tile_size
        ntx = (self.width + ts - 1) // ts
        nty = (self.height + ts - 1) // ts
        max_i = self.max_instances
        # offsets arg is exclusive prefix of counts; we atomicAdd on a cursor
        # buffer initialized by copying offsets in the shader from offsets_in.
        n_tiles = ntx * nty
        # Single-thread fill with private cursors copied from exclusive offsets
        # (PrefixSumNode output). Avoids atomic/init races and stays ≤8 bindings.
        wgsl = f"""
@group(0) @binding(0)
var<storage, read_write> keys: array<u32>;
@group(0) @binding(1)
var<storage, read_write> ids: array<u32>;
@group(0) @binding(2)
var<storage, read> means2d: array<f32>;
@group(0) @binding(3)
var<storage, read> depths: array<f32>;
@group(0) @binding(4)
var<storage, read> radii: array<f32>;
@group(0) @binding(5)
var<storage, read> offsets: array<u32>;

const N: u32 = {n}u;
const TS: u32 = {ts}u;
const NTX: u32 = {ntx}u;
const NTY: u32 = {nty}u;
const MAX_I: u32 = {max_i}u;
const N_TILES: u32 = {n_tiles}u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    var cursor: array<u32, {n_tiles}>;
    for (var t: u32 = 0u; t < N_TILES; t++) {{
        cursor[t] = offsets[t];
    }}
    // Unused instance slots sort last.
    for (var s: u32 = 0u; s < MAX_I; s++) {{
        keys[s] = 0xffffffffu;
        ids[s] = 0u;
    }}
    for (var g: u32 = 0u; g < N; g++) {{
        let mx = means2d[g * 2u];
        let my = means2d[g * 2u + 1u];
        let depth_bits = bitcast<u32>(depths[g]);
        let depth_q = depth_bits >> 16u;
        var rad = radii[g] * 1.1 + 1.0;
        if (rad < 1.0) {{ rad = 1.0; }}
        var tx0 = i32(floor((mx - rad) / f32(TS)));
        var tx1 = i32(floor((mx + rad) / f32(TS)));
        var ty0 = i32(floor((my - rad) / f32(TS)));
        var ty1 = i32(floor((my + rad) / f32(TS)));
        tx0 = clamp(tx0, 0, i32(NTX) - 1);
        tx1 = clamp(tx1, 0, i32(NTX) - 1);
        ty0 = clamp(ty0, 0, i32(NTY) - 1);
        ty1 = clamp(ty1, 0, i32(NTY) - 1);
        for (var ty = ty0; ty <= ty1; ty++) {{
            for (var tx = tx0; tx <= tx1; tx++) {{
                let tid = u32(ty) * NTX + u32(tx);
                let slot = cursor[tid];
                cursor[tid] = slot + 1u;
                if (slot < MAX_I) {{
                    keys[slot] = (tid << 16u) | (depth_q & 0xffffu);
                    ids[slot] = g;
                }}
            }}
        }}
    }}
}}
"""
        return IrWgslMultiOutputKernel(
            arg_accessors=tuple(a.accessor for a in self.args),
            etype=U4,
            shape=(self.max_instances,),
            wgsl=wgsl,
            entry_point="main",
            dispatch_size=(1, 1, 1),
            arg_etypes=(F4, F4, F4, U4),
            output_etypes=(U4, U4),
            num_outputs=2,
            clear_output_before_dispatch=True,
        )


@dataclass(kw_only=True, frozen=True, eq=False)
class TileRangesNode(CustomNode):
    """Scan sorted keys and write [start,end) pairs per tile."""

    n_tiles: int
    n_instances: int

    def build_kernel(self, *, used_ports: frozenset[str]) -> IrKernel:
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports
        n_tiles = self.n_tiles
        n_inst = self.n_instances
        wgsl = f"""
@group(0) @binding(0)
var<storage, read_write> output: array<u32>;
@group(0) @binding(1)
var<storage, read> keys: array<u32>;

const N_TILES: u32 = {n_tiles}u;
const N_INST: u32 = {n_inst}u;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x != 0u) {{ return; }}
    for (var t: u32 = 0u; t < N_TILES; t++) {{
        output[t * 2u] = 0u;
        output[t * 2u + 1u] = 0u;
    }}
    if (N_INST == 0u) {{ return; }}
    var start: u32 = 0u;
    var cur = keys[0] >> 16u;
    for (var i: u32 = 1u; i <= N_INST; i++) {{
        let tile = select(0xffffffffu, keys[i] >> 16u, i < N_INST);
        if (tile != cur) {{
            if (cur < N_TILES) {{
                output[cur * 2u] = start;
                output[cur * 2u + 1u] = i;
            }}
            start = i;
            cur = tile;
        }}
    }}
}}
"""
        return IrWgslMultiOutputKernel(
            arg_accessors=(self.args[0].accessor,),
            etype=U4,
            shape=self.shape,
            wgsl=wgsl,
            entry_point="main",
            dispatch_size=(1, 1, 1),
            arg_etypes=(U4,),
            output_etypes=(U4,),
            num_outputs=1,
            clear_output_before_dispatch=True,
        )


def build_gpu_tiled_blend_graph(
    *,
    width: int,
    height: int,
    means2d: View,
    depths: View,
    radii: View,
    conics: View,
    colors: View,
    opacities: View,
    tile_size: int = DEFAULT_TILE_SIZE,
    max_tiles_per_gaussian: int = 32,
) -> View:
    """Compose GPU binning (counts → prefix-sum → fill → radix sort → ranges) + tiled blend.

    Uses **PrefixSumNode** and **View.sort()** (radix subgraph) as builtins;
    only expand/range steps are library CustomNodes.
    """
    n = means2d.shape[0]
    ntx = (width + tile_size - 1) // tile_size
    nty = (height + tile_size - 1) // tile_size
    n_tiles = ntx * nty
    max_inst = n * max_tiles_per_gaussian

    counts = View.identity(
        TileCountNode(
            shape=(n_tiles,),
            etype=U4,
            args=(means2d, radii),
            width=width,
            height=height,
            tile_size=tile_size,
            n_gaussians=n,
        )
    )
    # Builtin exclusive prefix sum → per-tile write offsets.
    offsets = counts.prefix_sum(inclusive=False)

    fill = TileFillNode(
        shape=(max_inst,),
        etype=U4,
        args=(means2d, depths, radii, offsets),
        width=width,
        height=height,
        tile_size=tile_size,
        n_gaussians=n,
        max_instances=max_inst,
    )
    keys = View.port(fill, "keys")
    ids = View.port(fill, "ids")

    # Builtin radix-sort subgraph on keys; permute ids with the same perm via remap.
    _sorted_keys, perm = keys.sort()
    _ = _sorted_keys
    # sorted_ids[i] = ids[perm[i]]
    indices = perm.reshape((max_inst, 1))
    _gather_info = RemapGatherInfo(
        accessor=Accessor.dense((max_inst,)),
        source_shape=(max_inst,),
    )
    sorted_ids = View.remap(
        source=ids,
        info=_gather_info,
        indices=indices,
    )
    # Re-sort keys for range identification (same perm).
    sorted_keys = View.remap(
        source=keys,
        info=_gather_info,
        indices=indices,
    )

    ranges = View.identity(
        TileRangesNode(
            shape=(n_tiles * 2,),
            etype=U4,
            args=(sorted_keys,),
            n_tiles=n_tiles,
            n_instances=max_inst,
        )
    )

    return gaussian_blend_tiled(
        width=width,
        height=height,
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities,
        instances=sorted_ids,
        tile_ranges=ranges,
        tile_size=tile_size,
        n_tiles_x=ntx,
        n_tiles_y=nty,
    )
