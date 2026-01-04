"""
Utilities for BVH construction.
"""

__all__ = [
    "Bvh",
    "build_bvh",
    "compute_points_aabb",
    "partition_points",
    "partition_triangles",
]

from dataclasses import dataclass
import logging
import numba
import numpy.typing as npt
import numpy as np
import time

from .basic import logger

NUMBA_CACHE_ENABLED = True


@dataclass
class Bvh:
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


def build_bvh(
    t: npt.NDArray[np.uint32],  # (nt, 3)
    v: npt.NDArray[np.float32],  # (nv, 3)
    copy_t: bool = True,
    metrics_log_level: int = logging.DEBUG,
) -> Bvh:
    """
    Constructs a BVH (bounding volume hierarchy) for the given triangles and vertices.

    The triangles are reordered and returned in the BVH structure.
    Use those triangles for BVH traversal instead of the input `t`.

    A BVH is a binary tree where each node contains an axis-aligned bounding box (AABB)
    that encloses a subset of the triangles. Leaf nodes contain the actual triangles,
    while internal nodes partition the triangles into two child nodes.

    :param t: Triangle index array of shape (nt, 3). Each element indexes into `v`.
    :param v: Vertex position array of shape (nv, 3).
    :return: The constructed BVH, including re-ordered triangle indices.
    """

    t0 = time.perf_counter()

    # Copy `t`: if not specified, `t` will be modified in-place.
    if copy_t:
        t = t.copy()

    nt = t.shape[0]
    nv = v.shape[0]
    LOG.log(
        msg=f"Starting BVH construction: {nv=}, {nt=}",
        level=metrics_log_level,
    )

    # Compute triangle centroids:
    c = v[t].mean(axis=-2)
    assert c.shape == (nt, 3)
    t1 = time.perf_counter()
    LOG.log(
        msg=f"BVH construction: centroids: {(t1 - t0) * 1000:.2f}ms",
        level=metrics_log_level,
    )

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
    LOG.log(
        msg=f"BVH construction: root AABB: {(t3 - t2) * 1000:.2f}ms",
        level=metrics_log_level,
    )

    # Recursively build BVH subtree starting from root node:
    t4 = time.perf_counter()
    build_bvh_subtree(
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
    LOG.log(
        msg=f"BVH construction: recursive build: {(t5 - t4) * 1000:.2f}ms",
        level=metrics_log_level,
    )

    # Ensure we did not exceed allocated BVH node buffer:
    assert bvh_count[0] <= nb, "BVH node buffer overflow"

    # Done:
    return Bvh(
        t=t,
        aabb=bvh_b[: bvh_count[0]],
        children=bvh_c[: bvh_count[0]],
        tri_span=bvh_r[: bvh_count[0]],
    )


@numba.njit(cache=NUMBA_CACHE_ENABLED)
def build_bvh_subtree(
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
    Given a BVH node, subdivides it into a binary subtree by partitioning its triangles.

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
    build_bvh_subtree(
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
    build_bvh_subtree(
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

    sah_cost_lt = nt_lt * aabb_surface_area_lt if nt_lt > 0 else np.inf
    sah_cost_rt = nt_rt * aabb_surface_area_rt if nt_rt > 0 else np.inf

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


LOG = logger(__name__)
