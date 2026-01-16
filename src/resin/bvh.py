"""
Utilities for BVH construction.
"""

__all__ = [
    "Blas",
    "Bvh",
    "Tlas",
    "build_blas_bvh",
    "build_bvh",
    "build_tlas_bvh",
    "compute_points_aabb",
    "partition_instances",
    "partition_points",
    "partition_triangles",
    "transform_aabb",
]

from dataclasses import dataclass
import logging
import numba
import numpy.typing as npt
import numpy as np
import time

from .basic import NUMBA_CACHE_ENABLED
from . import trace


@dataclass
class Blas:
    """
    Bottom-Level Acceleration Structure (BLAS) for ray tracing over triangles.

    A BLAS is a BVH built over triangles of a single mesh. The triangles are reordered
    during construction for cache-friendly traversal.
    """

    t: npt.NDArray[np.uint32]  # (nt, 3)
    aabb: npt.NDArray[np.float32]  # (nb, 2, 3)
    children: npt.NDArray[np.uint32]  # (nb, 2)
    tri_span: npt.NDArray[np.uint32]  # (nb, 2)

    def __post_init__(self):
        assert self.aabb.ndim == 3 and self.aabb.shape == (self.node_count, 2, 3)
        assert self.children.ndim == 2 and self.children.shape == (self.node_count, 2)
        assert self.tri_span.ndim == 2 and self.tri_span.shape == (self.node_count, 2)

    @property
    def node_count(self) -> int:
        return self.aabb.shape[0]


@dataclass
class Tlas:
    """
    Top-Level Acceleration Structure (TLAS) for ray tracing over multiple object instances.

    Each instance references a BLAS (Bottom-Level Acceleration Structure) and has an associated
    world-space transform. The TLAS builds a BVH over the transformed instance AABBs.
    """

    instance_indices: npt.NDArray[np.uint32]  # (ni,) reordered instance indices
    aabb: npt.NDArray[np.float32]  # (nb, 2, 3) per-node AABBs
    children: npt.NDArray[np.uint32]  # (nb, 2) per-node child indices
    instance_span: npt.NDArray[np.uint32]  # (nb, 2) per-node instance ranges

    def __post_init__(self):
        assert self.aabb.ndim == 3 and self.aabb.shape == (self.node_count, 2, 3)
        assert self.children.ndim == 2 and self.children.shape == (self.node_count, 2)
        assert self.instance_span.ndim == 2 and self.instance_span.shape == (
            self.node_count,
            2,
        )

    @property
    def node_count(self) -> int:
        return self.aabb.shape[0]


def build_blas_bvh(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    copy_t: bool = True,
) -> Blas:
    """
    Constructs a BLAS (bottom-level acceleration structure) BVH for the given triangles and vertices.

    The triangles are reordered and returned in the BLAS structure.
    Use those triangles for BVH traversal instead of the input `t`.

    A BLAS is a BVH built over triangles of a single mesh. Each node contains an axis-aligned
    bounding box (AABB) that encloses a subset of the triangles. Leaf nodes contain the actual
    triangles, while internal nodes partition the triangles into two child nodes.

    :param t: Triangle index array of shape (nt, 3). Each element indexes into `v`.
    :param v: Vertex position array of shape (nv, 3).
    :param copy_t: If True, copies the triangle array before reordering.
    :return: The constructed BLAS, including re-ordered triangle indices.
    """

    t0 = time.perf_counter()

    # Copy `t`: if not specified, `t` will be modified in-place.
    if copy_t:
        t = t.copy()

    # Compute triangle centroids:
    nt = t.shape[0]
    c = v[t].mean(axis=-2)
    t1 = time.perf_counter()
    trace.add_time_span("bvh/blas/centroids", t0, t1)

    # Guess number of BVH nodes needed for initial capacity:
    # For a binary tree, worst case is 2L - 1 nodes (for L leaves).
    # Assume each leaf has at least one triangle.
    ESTIMATED_TRIANGLES_PER_LEAF = 1
    nl = (nt + ESTIMATED_TRIANGLES_PER_LEAF - 1) // ESTIMATED_TRIANGLES_PER_LEAF
    nb = max(128, 2 * nl - 1)

    # Allocate BVH arrays:
    bvh_count = np.zeros((1,), dtype=np.uint32)
    bvh_b = np.zeros((nb, 2, 3), dtype=np.float32)  # AABBs
    bvh_c = np.zeros((nb, 2), dtype=np.uint32)  # Child indices
    bvh_r = np.zeros((nb, 2), dtype=np.uint32)  # Triangle ranges
    bvh_s = np.zeros((nb,), dtype=np.float32)  # SAH costs

    # Initialize root node at index 0.
    # When BVH nodes have '0' as their child indices, they are leaf nodes since root nodes have no parents.
    t2 = time.perf_counter()
    bvh_count[0] += 1
    # Compute root AABB from all triangles
    bvh_b[0] = compute_points_aabb(v=v[t.ravel()])
    bvh_c[0, :] = (0, 0)
    bvh_r[0, :] = 0, nt
    bvh_s[0] = nt * single_aabb_surface_area(bvh_b[0])
    t3 = time.perf_counter()
    trace.add_time_span("bvh/blas/root_aabb", t2, t3)

    # Recursively build BVH subtree starting from root node:
    t4 = time.perf_counter()
    build_blas_bvh_subtree(
        t=t,
        c=c,
        v=v,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=0,
        _debug_depth=0,
    )
    t5 = time.perf_counter()
    trace.add_time_span("bvh/blas/recursive_build", t4, t5)

    # Ensure we did not exceed allocated BVH node buffer:
    assert bvh_count[0] <= nb, "BLAS node buffer overflow"

    # Done:
    return Blas(
        t=t,
        aabb=bvh_b[: bvh_count[0]],
        children=bvh_c[: bvh_count[0]],
        tri_span=bvh_r[: bvh_count[0]],
    )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def build_blas_bvh_subtree(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    bvh_count: npt.NDArray[np.uint32],  # (1,)
    bvh_b: npt.NDArray[np.float32],  # (nb, 2, 3)
    bvh_c: npt.NDArray[np.uint32],  # (nb, 2)
    bvh_r: npt.NDArray[np.uint32],  # (nb, 2)
    bvh_s: npt.NDArray[np.float32],  # (nb,)
    i_bvh_root: int,  # [0, bvh_n.item())
    _debug_depth: int = 0,
):
    """
    Given a BLAS BVH node, subdivides it into a binary subtree by partitioning its triangles.

    :param t: Triangle index array of shape (nt, 3). Each element indexes into `v`.
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param v: Vertex position array of shape (nv, 3).
    :param bvh_count: Number of nodes in the BVH (scalar).
    :param bvh_b: BVH per-node AABBs of shape (nb, 2, 3). Each node has min and max corners.
    :param bvh_c: BVH per-node child indices of shape (nb, 2). Each node has left and right child indices.
    :param bvh_r: BVH per-node triangle start and end of shape (nb, 2).
    :param bvh_s: BVH per-node surface area heuristic (SAH) cost of shape (nb,).
    :param i_bvh_root: Index of the BVH node to subdivide.
    """

    nt = t.shape[0]
    nb = bvh_b.shape[0]
    _ = _debug_depth

    assert t.ndim == 2 and t.shape[1] == 3
    assert c.ndim == 2 and c.shape[1] == 3
    assert v.ndim == 2 and v.shape[1] == 3
    assert t.shape[0] == c.shape[0] == nt
    assert bvh_b.ndim == 3 and bvh_b.shape[1:] == (2, 3)
    assert bvh_c.ndim == 2 and bvh_c.shape[1] == 2
    assert bvh_r.ndim == 2 and bvh_r.shape[1] == 2
    assert bvh_s.ndim == 1 and bvh_s.shape[0] == nb
    assert 0 <= i_bvh_root < bvh_count[0]

    # Gather triangle indices (and centroids) for the root node:
    t_root = t[bvh_r[i_bvh_root, 0] : bvh_r[i_bvh_root, 1]]
    c_root = c[bvh_r[i_bvh_root, 0] : bvh_r[i_bvh_root, 1]]
    bvh_r_root = bvh_r[i_bvh_root]

    # Find optimal partition for triangles in the root node:
    partition_parameters: tuple[int, float] = find_partition_parameters(
        t=t_root,
        c=c_root,
        v=v,
        aabb=bvh_b[i_bvh_root],
        bin_count=8,
    )
    partition_axis, partition_value = partition_parameters
    (
        i_lt_in_t_root,
        i_rt_in_t_root,
        aabb_lt,
        aabb_rt,
        sah_cost_lt,
        sah_cost_rt,
    ) = partition_triangles(
        t=t_root,
        c=c_root,
        v=v,
        z=partition_value,
        x=partition_axis,
    )

    # If the surface area heuristic (SAH) cost is not improved, do not subdivide.
    # This includes the case where no partitioning was possible (all triangles on one side).
    if sah_cost_lt + sah_cost_rt >= bvh_s[i_bvh_root]:
        assert np.all(bvh_c[i_bvh_root] == 0)
        return

    # Convert i_lt_root and i_rt_root to global triangle indices:
    i_lt = bvh_r_root[0] + i_lt_in_t_root
    i_rt = bvh_r_root[0] + i_rt_in_t_root

    # Gather triangles indices and centroids for each partition:
    t_lt, c_lt, n_lt = t[i_lt].copy(), c[i_lt].copy(), i_lt.shape[0]
    t_rt, c_rt, n_rt = t[i_rt].copy(), c[i_rt].copy(), i_rt.shape[0]
    assert n_lt + n_rt == bvh_r_root[1] - bvh_r_root[0]

    # Write triangles indices and centroids to t and c arrays, modifying in-place:
    lt_slice = slice(bvh_r_root[0], bvh_r_root[0] + n_lt)
    rt_slice = slice(bvh_r_root[0] + n_lt, bvh_r_root[1])
    t[lt_slice], c[lt_slice] = t_lt, c_lt
    t[rt_slice], c[rt_slice] = t_rt, c_rt

    # Emplace left and right child by bumping bvh_count:
    # (critical section)
    i_bvh_lt = bvh_count[0] + 0
    i_bvh_rt = bvh_count[0] + 1
    bvh_count[0] += 2
    assert bvh_count[0] <= nb, "BVH node buffer overflow"

    # Write bvh_b AABBs to child nodes:
    bvh_b[i_bvh_lt] = aabb_lt
    bvh_b[i_bvh_rt] = aabb_rt

    # Write bvh_c child indices to parent node:
    bvh_c[i_bvh_root, 0] = i_bvh_lt
    bvh_c[i_bvh_root, 1] = i_bvh_rt

    # Write bvh_r triangle ranges to child nodes:
    bvh_r[i_bvh_lt, 0] = bvh_r_root[0]
    bvh_r[i_bvh_lt, 1] = bvh_r_root[0] + n_lt
    bvh_r[i_bvh_rt, 0] = bvh_r_root[0] + n_lt
    bvh_r[i_bvh_rt, 1] = bvh_r_root[1]

    # Write bvh_s SAH costs to child nodes:
    bvh_s[i_bvh_lt] = sah_cost_lt
    bvh_s[i_bvh_rt] = sah_cost_rt

    # Recursively subdivide child nodes:
    build_blas_bvh_subtree(
        t=t,
        c=c,
        v=v,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=i_bvh_lt,
        _debug_depth=_debug_depth + 1,
    )
    build_blas_bvh_subtree(
        t=t,
        c=c,
        v=v,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=i_bvh_rt,
        _debug_depth=_debug_depth + 1,
    )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def find_partition_parameters(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    aabb: npt.NDArray[np.float32],  # (2, 3)
    bin_count: int = 256,
) -> tuple[int, float]:
    """
    Finds good partitioning parameters (pivot index and axis) for triangles based on their centroids.
    Uses either exhaustive search or approximate binning based on the `bin_count` parameter.

    :param t: Triangle index array of shape (nt, 3).
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param v: Vertex position array of shape (nv, 3). Each element in 't' indexes into this array.
    :param aabb: Axis-aligned bounding box of the triangles, shape (2, 3).
    :param bin_count: Number of bins to use for approximate binning. If <=0, uses exhaustive search.
    :return: A tuple containing:
        - Centroid axis index (0, 1, or 2) for partitioning.
        - Partition split value.
    """

    if bin_count <= 0:
        return find_partition_parameters_with_exhaustive_search(
            t=t, c=c, v=v, aabb=aabb
        )
    else:
        return find_partition_parameters_with_approximate_binning(
            t=t, c=c, v=v, aabb=aabb, bin_count=bin_count
        )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def find_partition_parameters_with_approximate_binning(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    aabb: npt.NDArray[np.float32],  # (2, 3)
    bin_count: int = 256,
) -> tuple[int, float]:
    """
    Finds good partitioning parameters (pivot index and axis) for triangles based on their centroids.

    Uses binning to find an approximate partition quickly. As bin_count -> ∞, results approach those of exhaustive
    search.

    :param t: Triangle index array of shape (nt, 3).
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param v: Vertex position array of shape (nv, 3). Each element in 't' indexes into this array.
    :param aabb: Axis-aligned bounding box of the triangles, shape (2, 3).
    :return: A tuple containing:
        - Centroid axis index (0, 1, or 2) for partitioning.
        - Partition split value
    """

    c_aabb = compute_points_aabb(c)

    best_sah_cost = np.inf
    best_sah_axis = 0
    best_sah_value = 0.0

    for x in range(3):
        # Compute bins:
        bin_freqs, bin_aabbs, bin_width = compute_node_bins(
            t=t,
            c=c,
            v=v,
            c_aabb=c_aabb,
            x=x,
            bin_count=bin_count,
        )

        # Sweep over bins to find cumulative AABBs and frequencies:
        bin_freqs_cum_asc = np.cumsum(bin_freqs)
        bin_freqs_cum_desc = np.cumsum(bin_freqs[::-1])[::-1]
        bin_aabbs_cum_asc = cum_union_aabbs(bin_aabbs)
        bin_aabbs_cum_desc = cum_union_aabbs(bin_aabbs[::-1])[::-1]
        bin_areas_cum_asc = aabbs_surface_areas(bin_aabbs_cum_asc)
        bin_areas_cum_desc = aabbs_surface_areas(bin_aabbs_cum_desc)

        # Evaluate SAH cost for each possible split in parallel:
        lt_sah_costs = bin_freqs_cum_asc * bin_areas_cum_asc
        rt_sah_costs = bin_freqs_cum_desc * bin_areas_cum_desc
        sah_costs = lt_sah_costs + rt_sah_costs

        # Find best split for this axis:
        axis_best_bin_idx = np.argmin(sah_costs)
        axis_best_sah_cost = sah_costs[axis_best_bin_idx]

        # If this axis is better than previous best, record it:
        if axis_best_sah_cost < best_sah_cost:
            best_sah_cost = axis_best_sah_cost
            best_sah_axis = x

            # Compute split value as the right of the bin
            best_sah_value = (1.0 + axis_best_bin_idx) * bin_width + c_aabb[0, x]

    return best_sah_axis, best_sah_value


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def compute_node_bins(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    c_aabb: npt.NDArray[np.float32],
    x: int,
    bin_count: int,
) -> tuple[
    npt.NDArray[np.uint32],  # bin_freqs (bin_count,)
    npt.NDArray[np.float32],  # bin_aabbs (bin_count, 2, 3)
    float,  # bin_width
]:
    """
    Bins triangle centroids along a given axis into the specified number of bins, returning the per-bin frequencies and
    per-bin _triangle_ AABBs (not centroid AABBs).

    :param t: Triangle index array of shape (nt, 3).
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param v: Vertex position array of shape (nv, 3). Each element in 't' indexes into this array.
    :param c_aabb: Axis-aligned bounding box of all the centroids, shape (2, 3).
    :param x: Centroid axis index (0, 1, or 2) along which to organize bins.
    :param bin_count: Number of bins to use.
    :return: A tuple containing:
        - Bin frequencies as array of shape (bin_count,).
        - Bin triangle AABBs as array of shape (bin_count, 2, 3).
        - Bin width as float.
    """

    bin_width = (c_aabb[1, x] - c_aabb[0, x]) / bin_count

    cx_normalized = (c[:, x] - c_aabb[0, x]) / (c_aabb[1, x] - c_aabb[0, x] + 1e-7)
    c_bin = (cx_normalized * bin_count).astype(np.int32)
    c_bin = np.clip(c_bin, 0, bin_count - 1)

    bin_freqs = np.bincount(c_bin, minlength=bin_count).astype(np.uint32)

    bin_aabbs = np.empty((bin_count, 2, 3), dtype=np.float32)
    bin_aabbs[:, 0, :] = +np.inf
    bin_aabbs[:, 1, :] = -np.inf
    for i_bin in range(bin_count):
        if bin_freqs[i_bin] > 0:
            bin_t_sel = c_bin == i_bin
            bin_aabbs[i_bin] = compute_points_aabb(v[t[bin_t_sel].ravel()])

    return bin_freqs, bin_aabbs, bin_width


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def find_partition_parameters_with_exhaustive_search(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    aabb: npt.NDArray[np.float32],  # (2, 3)
) -> tuple[int, float]:
    """
    Finds the optimal partitioning parameters (pivot index and axis) for triangles based on their centroids.

    Uses an exhaustive search over all possible splits to find the best partition. Slow, but gives high-quality results.

    :param t: Triangle index array of shape (nt, 3).
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param v: Vertex position array of shape (nv, 3). Each element in 't' indexes into this array.
    :param aabb: Axis-aligned bounding box of the triangles, shape (2, 3).
    :return: A tuple containing:
        - Centroid axis index (0, 1, or 2) for partitioning.
        - Partition split value
    """

    nt = t.shape[0]

    _ = aabb

    assert t.ndim == 2 and t.shape[1] == 3
    assert c.ndim == 2 and c.shape[1] == 3
    assert v.ndim == 2 and v.shape[1] == 3
    assert t.shape[0] == c.shape[0] == nt

    sah_cost_best = np.inf
    best_v = np.inf
    best_x = -1

    for i in range(nt):
        for x in range(3):
            (
                i_lt,
                i_rt,
                aabb_lt,
                aabb_rt,
                sah_cost_lt,
                sah_cost_rt,
            ) = partition_triangles(t=t, c=c, v=v, z=c[i, x], x=x)

            _ = i_lt, i_rt, aabb_lt, aabb_rt

            sah_cost = sah_cost_lt + sah_cost_rt

            if sah_cost < sah_cost_best:
                sah_cost_best = sah_cost
                best_v = c[i, x]
                best_x = x

    return best_x, best_v


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def partition_triangles(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    c: npt.NDArray[np.float32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    z: float,  # [0, nt)
    x: int,  # [0, 2)
) -> tuple[
    npt.NDArray[np.uint32],  # i_lt
    npt.NDArray[np.uint32],  # i_rt
    npt.NDArray[np.float32],  # aabb_lt (2, 3)
    npt.NDArray[np.float32],  # aabb_rt (2, 3)
    float,  # sah_cost_lt
    float,  # sah_cost_rt
]:
    """
    Partitions triangles into two sets based on their centroids.

    :param t: Triangle index array of shape (nt, 3).
    :param v: Vertex position array of shape (nv, 3). Each element in 't' indexes into this array.
    :param c: Triangle centroid array of shape (nt, 3). Equal to `v[t].mean(axis=-2)`.
    :param z: Value to partition triangles on.
    :param x: Triangle axis index (0, 1, or 2) for partitioning.
    :return: A tuple containing:
        - Triangle indices for left partition.
        - Triangle indices for right partition.
        - AABB for left partition as 2x3 array.
        - AABB for right partition as 2x3 array.
        - Surface Area Heuristic (SAH) cost for left partition.
        - Surface Area Heuristic (SAH) cost for right partition.
    """

    nt = t.shape[0]

    assert t.ndim == 2 and t.shape[1] == 3
    assert c.ndim == 2 and c.shape[1] == 3
    assert v.ndim == 2 and v.shape[1] == 3
    assert t.shape[0] == c.shape[0] == nt
    assert 0 <= x < 3

    i_lt, i_rt = partition_points(p=c, z=z, x=x)

    nt_lt = i_lt.shape[0]
    nt_rt = i_rt.shape[0]

    v_lt = v[t[i_lt].ravel()]
    aabb_lt = compute_points_aabb(v_lt)
    aabb_surface_area_lt = single_aabb_surface_area(aabb_lt)

    v_rt = v[t[i_rt].ravel()]
    aabb_rt = compute_points_aabb(v_rt)
    aabb_surface_area_rt = single_aabb_surface_area(aabb_rt)

    sah_cost_lt = float(nt_lt * aabb_surface_area_lt if nt_lt > 0 else float("inf"))
    sah_cost_rt = float(nt_rt * aabb_surface_area_rt if nt_rt > 0 else float("inf"))

    return i_lt, i_rt, aabb_lt, aabb_rt, sah_cost_lt, sah_cost_rt


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def partition_points(
    p: npt.NDArray[np.float32],
    z: float,
    x: int,
) -> tuple[
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]:
    """
    Splits a list of points into two partitions based on a pivot.

    :param p: Points array of shape (np, 3).
    :param i: Point pivot index for partitioning.
    :param x: Point axis index (0, 1, or 2) for partitioning.
    :return: A tuple containing the triangle indices for left and right partitions.
    """

    assert p.ndim == 2 and p.shape[1] == 3

    n = p.shape[0]

    lt_mask = p[:, x] < z
    rt_mask = ~lt_mask

    assert lt_mask.shape == (n,)
    assert rt_mask.shape == (n,)

    (lt,) = np.nonzero(lt_mask)
    (rt,) = np.nonzero(rt_mask)

    return lt.astype(np.uint32), rt.astype(np.uint32)


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def compute_points_aabb(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Compute axis-aligned bounding box (AABB) for given points. If those points are vertices of triangles, the AABB
    encloses the triangles.

    :param v: Vertex position array of shape (nv, 3).
    :return: AABB as 2x3 array where [0] is min corner and [1] is max corner.
    """

    assert v.ndim == 2 and v.shape[1] == 3

    nv = v.shape[0]

    if nv == 0:
        # Return infinite AABB for empty vertex set
        aabb = np.empty((2, 3), dtype=np.float32)
        aabb[0] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        aabb[1] = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        return aabb

    # Use vectorized min/max per column for performance
    aabb = np.empty((2, 3), dtype=np.float32)
    aabb[0, 0] = v[:, 0].min()
    aabb[0, 1] = v[:, 1].min()
    aabb[0, 2] = v[:, 2].min()
    aabb[1, 0] = v[:, 0].max()
    aabb[1, 1] = v[:, 1].max()
    aabb[1, 2] = v[:, 2].max()

    return aabb


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def aabbs_surface_areas(
    aabbs: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Compute surface areas of axis-aligned bounding boxes (AABBs).

    :param aabbs: Array of AABBs of shape (n, 2, 3) where [0] is min corner and [1] is max corner.
    :return: Surface areas of the AABBs as array of shape (n,).
    """

    n = aabbs.shape[0]
    surface_areas = np.empty((n,), dtype=np.float32)

    for i in range(n):
        surface_areas[i] = single_aabb_surface_area(aabbs[i])

    return surface_areas


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def single_aabb_surface_area(
    aabb: npt.NDArray[np.float32],
) -> np.float32:
    """
    Compute surface area of an axis-aligned bounding box (AABB).

    :param aabb: AABB as 2x3 array where [0] is min corner and [1] is max corner.
    :return: Surface area of the AABB.
    """

    extent = aabb[1] - aabb[0]
    surface_area = 2.0 * np.sum(extent)

    return surface_area


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def cum_union_aabbs(aabbs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Computes cumulative union of AABBs from a list of AABBs.

    :param aabbs: Array of AABBs of shape (n, 2, 3).
    :return: Cumulative AABBs of shape (n, 2, 3).
    """

    n = aabbs.shape[0]
    if n == 0:
        return aabbs

    cum_aabbs = np.empty((n, 2, 3), dtype=np.float32)
    cum_aabbs[:, 0] = +np.inf
    cum_aabbs[:, 1] = -np.inf

    cum_aabbs[0] = aabbs[0]
    for i in range(1, n):
        cum_aabbs[i] = aabb_union(cum_aabbs[i - 1], aabbs[i])

    return cum_aabbs


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def aabb_union(
    aabb1: npt.NDArray[np.float32],
    aabb2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Computes the union of two AABBs.

    :param aabb1: First AABB as 2x3 array.
    :param aabb2: Second AABB as 2x3 array.
    :return: Union AABB as 2x3 array.
    """

    aabb_union = np.empty((2, 3), dtype=np.float32)
    aabb_union[0] = np.minimum(aabb1[0], aabb2[0])
    aabb_union[1] = np.maximum(aabb1[1], aabb2[1])
    return aabb_union


def build_tlas_bvh(
    instance_aabbs: npt.NDArray[np.float32],  # (ni, 2, 3)
    copy_instances: bool = True,
) -> Tlas:
    """
    Constructs a TLAS (top-level acceleration structure) BVH over instance AABBs.

    The instances are reordered and returned in the TLAS structure.
    Use those instance indices for BVH traversal instead of the input order.

    A TLAS is a BVH built over object instances, where each instance has a world-space AABB.
    Typically, each instance references a BLAS and has an associated transform matrix.

    :param instance_aabbs: Per-instance world-space AABBs of shape (ni, 2, 3).
    :param copy_instances: If True, copies the instance indices before reordering.
    :return: The constructed TLAS, including re-ordered instance indices.
    """

    t0 = time.perf_counter()

    ni = instance_aabbs.shape[0]

    # Create instance indices array that will be reordered:
    instance_indices = np.arange(ni, dtype=np.uint32)
    if copy_instances:
        instance_indices = instance_indices.copy()

    # Compute instance centroids from AABBs:
    c = (instance_aabbs[:, 0, :] + instance_aabbs[:, 1, :]) * 0.5
    t1 = time.perf_counter()
    trace.add_time_span("bvh/tlas/centroids", t0, t1)

    # Guess number of BVH nodes needed for initial capacity:
    ESTIMATED_INSTANCES_PER_LEAF = 1
    nl = (ni + ESTIMATED_INSTANCES_PER_LEAF - 1) // ESTIMATED_INSTANCES_PER_LEAF
    nb = max(128, 2 * nl - 1)

    # Allocate BVH arrays:
    bvh_count = np.zeros((1,), dtype=np.uint32)
    bvh_b = np.zeros((nb, 2, 3), dtype=np.float32)  # AABBs
    bvh_c = np.zeros((nb, 2), dtype=np.uint32)  # Child indices
    bvh_r = np.zeros((nb, 2), dtype=np.uint32)  # Instance ranges
    bvh_s = np.zeros((nb,), dtype=np.float32)  # SAH costs

    # Initialize root node at index 0:
    t2 = time.perf_counter()
    bvh_count[0] += 1
    # Compute root AABB from all instances
    bvh_b[0] = compute_aabbs_union(instance_aabbs)
    bvh_c[0, :] = (0, 0)
    bvh_r[0, :] = 0, ni
    bvh_s[0] = ni * single_aabb_surface_area(bvh_b[0])
    t3 = time.perf_counter()
    trace.add_time_span("bvh/tlas/root_aabb", t2, t3)

    # Recursively build BVH subtree starting from root node:
    t4 = time.perf_counter()
    build_tlas_bvh_subtree(
        instance_indices=instance_indices,
        instance_aabbs=instance_aabbs,
        c=c,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=0,
        _debug_depth=0,
    )
    t5 = time.perf_counter()
    trace.add_time_span("bvh/tlas/recursive_build", t4, t5)

    # Ensure we did not exceed allocated BVH node buffer:
    assert bvh_count[0] <= nb, "TLAS node buffer overflow"

    # Done:
    return Tlas(
        instance_indices=instance_indices,
        aabb=bvh_b[: bvh_count[0]],
        children=bvh_c[: bvh_count[0]],
        instance_span=bvh_r[: bvh_count[0]],
    )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def build_tlas_bvh_subtree(
    instance_indices: npt.NDArray[np.uint32],  # (ni,)
    instance_aabbs: npt.NDArray[np.float32],  # (ni, 2, 3)
    c: npt.NDArray[np.float32],  # (ni, 3)
    bvh_count: npt.NDArray[np.uint32],  # (1,)
    bvh_b: npt.NDArray[np.float32],  # (nb, 2, 3)
    bvh_c: npt.NDArray[np.uint32],  # (nb, 2)
    bvh_r: npt.NDArray[np.uint32],  # (nb, 2)
    bvh_s: npt.NDArray[np.float32],  # (nb,)
    i_bvh_root: int,  # [0, bvh_count.item())
    _debug_depth: int = 0,
):
    """
    Given a TLAS BVH node, subdivides it into a binary subtree by partitioning its instances.

    :param instance_indices: Instance index array of shape (ni,).
    :param instance_aabbs: Per-instance world-space AABBs of shape (ni, 2, 3).
    :param c: Instance centroid array of shape (ni, 3).
    :param bvh_count: Number of nodes in the BVH (scalar).
    :param bvh_b: BVH per-node AABBs of shape (nb, 2, 3).
    :param bvh_c: BVH per-node child indices of shape (nb, 2).
    :param bvh_r: BVH per-node instance start and end of shape (nb, 2).
    :param bvh_s: BVH per-node SAH cost of shape (nb,).
    :param i_bvh_root: Index of the BVH node to subdivide.
    """

    ni = instance_indices.shape[0]
    nb = bvh_b.shape[0]
    _ = _debug_depth

    assert instance_indices.ndim == 1
    assert instance_aabbs.ndim == 3 and instance_aabbs.shape[1:] == (2, 3)
    assert c.ndim == 2 and c.shape[1] == 3
    assert instance_indices.shape[0] == c.shape[0] == ni
    assert bvh_b.ndim == 3 and bvh_b.shape[1:] == (2, 3)
    assert bvh_c.ndim == 2 and bvh_c.shape[1] == 2
    assert bvh_r.ndim == 2 and bvh_r.shape[1] == 2
    assert bvh_s.ndim == 1 and bvh_s.shape[0] == nb
    assert 0 <= i_bvh_root < bvh_count[0]

    # Gather instance indices (and centroids) for the root node:
    idx_root = instance_indices[bvh_r[i_bvh_root, 0] : bvh_r[i_bvh_root, 1]]
    c_root = c[bvh_r[i_bvh_root, 0] : bvh_r[i_bvh_root, 1]]
    bvh_r_root = bvh_r[i_bvh_root]

    # Find optimal partition for instances in the root node:
    partition_parameters: tuple[int, float] = find_tlas_partition_parameters(
        instance_aabbs=instance_aabbs,
        idx=idx_root,
        c=c_root,
        aabb=bvh_b[i_bvh_root],
        bin_count=8,
    )
    partition_axis, partition_value = partition_parameters
    (
        i_lt_in_root,
        i_rt_in_root,
        aabb_lt,
        aabb_rt,
        sah_cost_lt,
        sah_cost_rt,
    ) = partition_instances(
        instance_aabbs=instance_aabbs,
        idx=idx_root,
        c=c_root,
        z=partition_value,
        x=partition_axis,
    )

    # If the SAH cost is not improved, do not subdivide:
    if sah_cost_lt + sah_cost_rt >= bvh_s[i_bvh_root]:
        assert np.all(bvh_c[i_bvh_root] == 0)
        return

    # Convert local indices to global indices:
    i_lt = bvh_r_root[0] + i_lt_in_root
    i_rt = bvh_r_root[0] + i_rt_in_root

    # Gather instance indices and centroids for each partition:
    idx_lt, c_lt, n_lt = instance_indices[i_lt].copy(), c[i_lt].copy(), i_lt.shape[0]
    idx_rt, c_rt, n_rt = instance_indices[i_rt].copy(), c[i_rt].copy(), i_rt.shape[0]
    assert n_lt + n_rt == bvh_r_root[1] - bvh_r_root[0]

    # Write instance indices and centroids in-place:
    lt_slice = slice(bvh_r_root[0], bvh_r_root[0] + n_lt)
    rt_slice = slice(bvh_r_root[0] + n_lt, bvh_r_root[1])
    instance_indices[lt_slice], c[lt_slice] = idx_lt, c_lt
    instance_indices[rt_slice], c[rt_slice] = idx_rt, c_rt

    # Emplace left and right child by bumping bvh_count:
    i_bvh_lt = bvh_count[0] + 0
    i_bvh_rt = bvh_count[0] + 1
    bvh_count[0] += 2
    assert bvh_count[0] <= nb, "TLAS node buffer overflow"

    # Write bvh_b AABBs to child nodes:
    bvh_b[i_bvh_lt] = aabb_lt
    bvh_b[i_bvh_rt] = aabb_rt

    # Write bvh_c child indices to parent node:
    bvh_c[i_bvh_root, 0] = i_bvh_lt
    bvh_c[i_bvh_root, 1] = i_bvh_rt

    # Write bvh_r instance ranges to child nodes:
    bvh_r[i_bvh_lt, 0] = bvh_r_root[0]
    bvh_r[i_bvh_lt, 1] = bvh_r_root[0] + n_lt
    bvh_r[i_bvh_rt, 0] = bvh_r_root[0] + n_lt
    bvh_r[i_bvh_rt, 1] = bvh_r_root[1]

    # Write bvh_s SAH costs to child nodes:
    bvh_s[i_bvh_lt] = sah_cost_lt
    bvh_s[i_bvh_rt] = sah_cost_rt

    # Recursively subdivide child nodes:
    build_tlas_bvh_subtree(
        instance_indices=instance_indices,
        instance_aabbs=instance_aabbs,
        c=c,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=i_bvh_lt,
        _debug_depth=_debug_depth + 1,
    )
    build_tlas_bvh_subtree(
        instance_indices=instance_indices,
        instance_aabbs=instance_aabbs,
        c=c,
        bvh_count=bvh_count,
        bvh_b=bvh_b,
        bvh_c=bvh_c,
        bvh_r=bvh_r,
        bvh_s=bvh_s,
        i_bvh_root=i_bvh_rt,
        _debug_depth=_debug_depth + 1,
    )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def find_tlas_partition_parameters(
    instance_aabbs: npt.NDArray[np.float32],  # (ni, 2, 3)
    idx: npt.NDArray[np.uint32],  # (n,)
    c: npt.NDArray[np.float32],  # (n, 3)
    aabb: npt.NDArray[np.float32],  # (2, 3)
    bin_count: int = 8,
) -> tuple[int, float]:
    """
    Finds good partitioning parameters for instances based on their centroids using binning.

    :param instance_aabbs: Per-instance world-space AABBs of shape (ni, 2, 3).
    :param idx: Instance indices of shape (n,) into instance_aabbs.
    :param c: Instance centroid array of shape (n, 3).
    :param aabb: Parent node AABB of shape (2, 3).
    :param bin_count: Number of bins to use for approximate binning.
    :return: A tuple of (axis, split_value).
    """

    _ = aabb

    c_aabb = compute_points_aabb(c)

    best_sah_cost = np.inf
    best_sah_axis = 0
    best_sah_value = 0.0

    for x in range(3):
        bin_freqs, bin_aabbs, bin_width = compute_tlas_node_bins(
            instance_aabbs=instance_aabbs,
            idx=idx,
            c=c,
            c_aabb=c_aabb,
            x=x,
            bin_count=bin_count,
        )

        bin_freqs_cum_asc = np.cumsum(bin_freqs)
        bin_freqs_cum_desc = np.cumsum(bin_freqs[::-1])[::-1]
        bin_aabbs_cum_asc = cum_union_aabbs(bin_aabbs)
        bin_aabbs_cum_desc = cum_union_aabbs(bin_aabbs[::-1])[::-1]
        bin_areas_cum_asc = aabbs_surface_areas(bin_aabbs_cum_asc)
        bin_areas_cum_desc = aabbs_surface_areas(bin_aabbs_cum_desc)

        lt_sah_costs = bin_freqs_cum_asc * bin_areas_cum_asc
        rt_sah_costs = bin_freqs_cum_desc * bin_areas_cum_desc
        sah_costs = lt_sah_costs + rt_sah_costs

        axis_best_bin_idx = np.argmin(sah_costs)
        axis_best_sah_cost = sah_costs[axis_best_bin_idx]

        if axis_best_sah_cost < best_sah_cost:
            best_sah_cost = axis_best_sah_cost
            best_sah_axis = x
            best_sah_value = (1.0 + axis_best_bin_idx) * bin_width + c_aabb[0, x]

    return best_sah_axis, best_sah_value


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def compute_tlas_node_bins(
    instance_aabbs: npt.NDArray[np.float32],  # (ni, 2, 3)
    idx: npt.NDArray[np.uint32],  # (n,)
    c: npt.NDArray[np.float32],  # (n, 3)
    c_aabb: npt.NDArray[np.float32],  # (2, 3)
    x: int,
    bin_count: int,
) -> tuple[
    npt.NDArray[np.uint32],  # bin_freqs (bin_count,)
    npt.NDArray[np.float32],  # bin_aabbs (bin_count, 2, 3)
    float,  # bin_width
]:
    """
    Bins instance centroids along a given axis, returning per-bin frequencies and AABBs.

    :param instance_aabbs: Per-instance world-space AABBs of shape (ni, 2, 3).
    :param idx: Instance indices of shape (n,) into instance_aabbs.
    :param c: Instance centroid array of shape (n, 3).
    :param c_aabb: AABB of all centroids, shape (2, 3).
    :param x: Axis index (0, 1, or 2).
    :param bin_count: Number of bins.
    :return: (bin_freqs, bin_aabbs, bin_width)
    """

    bin_width = (c_aabb[1, x] - c_aabb[0, x]) / bin_count

    cx_normalized = (c[:, x] - c_aabb[0, x]) / (c_aabb[1, x] - c_aabb[0, x] + 1e-7)
    c_bin = (cx_normalized * bin_count).astype(np.int32)
    c_bin = np.clip(c_bin, 0, bin_count - 1)

    bin_freqs = np.bincount(c_bin, minlength=bin_count).astype(np.uint32)

    bin_aabbs = np.empty((bin_count, 2, 3), dtype=np.float32)
    bin_aabbs[:, 0, :] = +np.inf
    bin_aabbs[:, 1, :] = -np.inf
    for i_bin in range(bin_count):
        if bin_freqs[i_bin] > 0:
            bin_sel = c_bin == i_bin
            bin_idx = idx[bin_sel]
            bin_aabbs[i_bin] = compute_aabbs_union(instance_aabbs[bin_idx])

    return bin_freqs, bin_aabbs, bin_width


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def partition_instances(
    instance_aabbs: npt.NDArray[np.float32],  # (ni, 2, 3)
    idx: npt.NDArray[np.uint32],  # (n,)
    c: npt.NDArray[np.float32],  # (n, 3)
    z: float,
    x: int,
) -> tuple[
    npt.NDArray[np.uint32],  # i_lt
    npt.NDArray[np.uint32],  # i_rt
    npt.NDArray[np.float32],  # aabb_lt (2, 3)
    npt.NDArray[np.float32],  # aabb_rt (2, 3)
    float,  # sah_cost_lt
    float,  # sah_cost_rt
]:
    """
    Partitions instances into two sets based on their centroids.

    :param instance_aabbs: Per-instance world-space AABBs of shape (ni, 2, 3).
    :param idx: Instance indices of shape (n,) into instance_aabbs.
    :param c: Instance centroid array of shape (n, 3).
    :param z: Split value.
    :param x: Axis index (0, 1, or 2).
    :return: (i_lt, i_rt, aabb_lt, aabb_rt, sah_cost_lt, sah_cost_rt)
    """

    n = idx.shape[0]

    assert idx.ndim == 1
    assert c.ndim == 2 and c.shape == (n, 3)
    assert 0 <= x < 3

    i_lt, i_rt = partition_points(p=c, z=z, x=x)

    n_lt = i_lt.shape[0]
    n_rt = i_rt.shape[0]

    aabb_lt = compute_aabbs_union(instance_aabbs[idx[i_lt]])
    aabb_surface_area_lt = single_aabb_surface_area(aabb_lt)

    aabb_rt = compute_aabbs_union(instance_aabbs[idx[i_rt]])
    aabb_surface_area_rt = single_aabb_surface_area(aabb_rt)

    sah_cost_lt = float(n_lt * aabb_surface_area_lt if n_lt > 0 else float("inf"))
    sah_cost_rt = float(n_rt * aabb_surface_area_rt if n_rt > 0 else float("inf"))

    return i_lt, i_rt, aabb_lt, aabb_rt, sah_cost_lt, sah_cost_rt


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def compute_aabbs_union(aabbs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Computes the union of multiple AABBs.

    :param aabbs: Array of AABBs of shape (n, 2, 3).
    :return: Union AABB as 2x3 array.
    """

    n = aabbs.shape[0]

    if n == 0:
        result = np.empty((2, 3), dtype=np.float32)
        result[0] = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        result[1] = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        return result

    result = aabbs[0].copy()
    for i in range(1, n):
        result = aabb_union(result, aabbs[i])

    return result


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def transform_aabb(
    aabb: npt.NDArray[np.float32],  # (2, 3)
    transform: npt.NDArray[np.float32],  # (4, 4)
) -> npt.NDArray[np.float32]:
    """
    Transforms an AABB by a 4x4 transformation matrix, returning a new axis-aligned AABB
    that encloses the transformed box.

    :param aabb: Input AABB as 2x3 array where [0] is min corner and [1] is max corner.
    :param transform: 4x4 transformation matrix (affine).
    :return: Transformed AABB as 2x3 array.
    """

    assert aabb.shape == (2, 3)
    assert transform.shape == (4, 4)

    # Generate all 8 corners of the AABB:
    corners = np.empty((8, 3), dtype=np.float32)
    for i in range(8):
        corners[i, 0] = aabb[(i >> 0) & 1, 0]
        corners[i, 1] = aabb[(i >> 1) & 1, 1]
        corners[i, 2] = aabb[(i >> 2) & 1, 2]

    # Transform corners:
    transformed_corners = np.empty((8, 3), dtype=np.float32)
    for i in range(8):
        # Homogeneous coordinate transform
        px = corners[i, 0]
        py = corners[i, 1]
        pz = corners[i, 2]
        tx = (
            transform[0, 0] * px
            + transform[0, 1] * py
            + transform[0, 2] * pz
            + transform[0, 3]
        )
        ty = (
            transform[1, 0] * px
            + transform[1, 1] * py
            + transform[1, 2] * pz
            + transform[1, 3]
        )
        tz = (
            transform[2, 0] * px
            + transform[2, 1] * py
            + transform[2, 2] * pz
            + transform[2, 3]
        )
        transformed_corners[i, 0] = tx
        transformed_corners[i, 1] = ty
        transformed_corners[i, 2] = tz

    # Compute new AABB from transformed corners:
    return compute_points_aabb(transformed_corners)


# Backward compatibility aliases for old API
# TODO: Clean this up
Bvh = Blas
build_bvh = build_blas_bvh

LOG = logging.getLogger(__name__)
