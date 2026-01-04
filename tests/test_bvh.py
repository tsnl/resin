import logging
import zfw

import time
import numpy as np
import rich
from pathlib import Path


def test_compute_triangles_aabb_performance():
    """
    Test AABB computation performance in isolation.
    This should take microseconds, not milliseconds.
    """
    # Load Suzanne mesh
    base_path = Path(__file__).parent / "data" / "glTF-Sample-Assets" / "Models"
    mesh_path = base_path / "Suzanne" / "glTF" / "Suzanne.gltf"
    meshes_dict = zfw.load_gltf(mesh_path)

    for (geometry, _), _ in meshes_dict.items():
        v = geometry.v_p_array
        t = geometry.t_indices

        # Get triangle vertices (what root AABB computation needs)
        v_triangles = v[t.flatten()]

        # Time the AABB computation (cold - first run with JIT)
        aabb1 = zfw.compute_triangles_aabb(v_triangles)

        # Time the AABB computation (warm - second run without JIT)
        start_time = time.monotonic_ns()
        aabb2 = zfw.compute_triangles_aabb(v_triangles)
        end_time = time.monotonic_ns()
        elapsed_ms_warm = (end_time - start_time) * 1e-6

        # Run it multiple times to get average
        num_iterations = 100
        for _ in range(num_iterations):
            _ = zfw.compute_triangles_aabb(v_triangles)

        # Verify AABBs are the same
        assert np.allclose(aabb1, aabb2), "AABBs should be identical"

        # This should take at most a few milliseconds even cold, and microseconds warm
        assert elapsed_ms_warm < 10.0, (
            f"AABB computation too slow: {elapsed_ms_warm:.2f} ms"
        )


def test_build_bvh(mesh_name: str = "Suzanne.gltf"):
    """
    Test BVH construction on real meshes from glTF sample assets.
    Verifies the generated BVH structure satisfies key invariants.
    """
    # Test meshes from glTF sample assets
    test_meshes = [
        mesh_name,
    ]

    base_path = Path(__file__).parent / "data" / "glTF-Sample-Assets" / "Models"

    for mesh_name in test_meshes:
        if mesh_name == "Box.gltf":
            mesh_path = base_path / "Box" / "glTF" / "Box.gltf"
        elif mesh_name == "Suzanne.gltf":
            mesh_path = base_path / "Suzanne" / "glTF" / "Suzanne.gltf"
        else:
            continue

        # Load mesh using resources
        meshes_dict = zfw.load_gltf(mesh_path)
        assert len(meshes_dict) > 0, f"Failed to load mesh {mesh_name}"

        # Extract geometry from the first mesh (we only care about geometry, not materials)
        for (geometry, _), _ in meshes_dict.items():
            v = geometry.v_p_array
            t = geometry.t_indices

            # Build BVH with timing
            start_time = time.monotonic_ns()
            bvh = zfw.build_bvh(t=t, v=v, metrics_log_level=logging.INFO)
            end_time = time.monotonic_ns()
            elapsed_ms = (end_time - start_time) * 1e-6

            rich.print(
                f"[dark_blue]BVH build for {mesh_name}: {elapsed_ms:.2f} ms "
                f"({t.shape[0]} triangles, {bvh.node_count} nodes)[/dark_blue]"
            )

            # Run all verification checks
            _verify_leaf_nodes_contain_triangles(bvh, v, t, mesh_name)
            _verify_all_triangles_represented(bvh, t, mesh_name)
            _verify_intermediate_node_aabbs(bvh, mesh_name)
            _verify_children_have_fewer_triangles(bvh, mesh_name)


def _verify_leaf_nodes_contain_triangles(
    bvh: zfw.Bvh,
    v: np.ndarray,
    t_original: np.ndarray,
    mesh_name: str,
) -> None:
    """
    Verify that for each leaf BVH node:
    - All triangles in the BVH's triangle list lie within the AABB
    - The AABB is the tightest possible (exact bounding box of the triangles)
    """
    for i_node in range(bvh.node_count):
        # Check if this is a leaf node (no children)
        if np.all(bvh.children[i_node] == 0):
            # This is a leaf node
            tri_start, tri_end = bvh.tri_span[i_node]
            leaf_triangles = bvh.t[tri_start:tri_end]
            aabb_min, aabb_max = bvh.aabb[i_node]

            # Get all vertices in this leaf
            v_leaf = v[leaf_triangles.flatten()]

            # Verify all vertices are within the AABB
            assert np.all(v_leaf >= aabb_min - 1e-6), (
                f"{mesh_name}: Leaf node {i_node} contains vertices "
                "outside AABB min bound"
            )
            assert np.all(v_leaf <= aabb_max + 1e-6), (
                f"{mesh_name}: Leaf node {i_node} contains vertices "
                "outside AABB max bound"
            )

            # Verify AABB is the tightest (exact bounding box)
            aabb_exact = zfw.bvh.compute_triangles_aabb(v_leaf)
            assert np.allclose(aabb_min, aabb_exact[0], atol=1e-6), (
                f"{mesh_name}: Leaf node {i_node} AABB min is not tight"
            )
            assert np.allclose(aabb_max, aabb_exact[1], atol=1e-6), (
                f"{mesh_name}: Leaf node {i_node} AABB max is not tight"
            )


def _verify_all_triangles_represented(
    bvh: zfw.Bvh,
    t_original: np.ndarray,
    mesh_name: str,
) -> None:
    """
    Verify that all triangles from the original mesh are represented in the BVH.
    """
    # Collect all triangle indices from all leaf nodes
    represented_tri_indices = set()

    for i_node in range(bvh.node_count):
        # Check if this is a leaf node
        if np.all(bvh.children[i_node] == 0):
            tri_start, tri_end = bvh.tri_span[i_node]
            leaf_triangles = bvh.t[tri_start:tri_end]

            # Add the global indices (0 to nt-1)
            for global_idx in range(tri_start, tri_end):
                represented_tri_indices.add(global_idx)

    # Verify we have exactly the right number of triangles
    assert len(represented_tri_indices) == t_original.shape[0], (
        f"{mesh_name}: Not all triangles are represented in BVH. "
        f"Expected {t_original.shape[0]}, got {len(represented_tri_indices)}"
    )

    # Verify all indices are in the valid range
    assert (
        max(represented_tri_indices) < t_original.shape[0]
        and min(represented_tri_indices) >= 0
    ), f"{mesh_name}: Invalid triangle indices in BVH"


def _verify_intermediate_node_aabbs(
    bvh: zfw.Bvh,
    mesh_name: str,
) -> None:
    """
    Verify that for each intermediate BVH node:
    - Its AABB is the exact union of its children AABBs (within tight tolerance)
    """
    for i_node in range(bvh.node_count):
        # Check if this is an intermediate node (has children)
        if not np.all(bvh.children[i_node] == 0):
            i_child_left, i_child_right = bvh.children[i_node]
            aabb_parent_min, aabb_parent_max = bvh.aabb[i_node]

            aabb_left_min, aabb_left_max = bvh.aabb[i_child_left]
            aabb_right_min, aabb_right_max = bvh.aabb[i_child_right]

            # Compute the union of children AABBs
            union_min = np.minimum(aabb_left_min, aabb_right_min)
            union_max = np.maximum(aabb_left_max, aabb_right_max)

            # Verify parent AABB matches the union
            assert np.allclose(aabb_parent_min, union_min, atol=1e-6), (
                f"{mesh_name}: Intermediate node {i_node} AABB min is not "
                "the exact union of children"
            )
            assert np.allclose(aabb_parent_max, union_max, atol=1e-6), (
                f"{mesh_name}: Intermediate node {i_node} AABB max is not "
                "the exact union of children"
            )


def _verify_children_have_fewer_triangles(
    bvh: zfw.Bvh,
    mesh_name: str,
) -> None:
    """
    Verify that for each intermediate BVH node:
    - Each child has fewer triangles than its parent
    """
    for i_node in range(bvh.node_count):
        # Check if this is an intermediate node (has children)
        if not np.all(bvh.children[i_node] == 0):
            i_child_left, i_child_right = bvh.children[i_node]

            # Get triangle counts
            parent_tri_start, parent_tri_end = bvh.tri_span[i_node]
            parent_tri_count = parent_tri_end - parent_tri_start

            left_tri_start, left_tri_end = bvh.tri_span[i_child_left]
            left_tri_count = left_tri_end - left_tri_start

            right_tri_start, right_tri_end = bvh.tri_span[i_child_right]
            right_tri_count = right_tri_end - right_tri_start

            # Verify each child has fewer triangles than parent
            assert left_tri_count < parent_tri_count, (
                f"{mesh_name}: Left child node {i_child_left} has {left_tri_count} "
                f"triangles, not less than parent node {i_node} with {parent_tri_count}"
            )
            assert right_tri_count < parent_tri_count, (
                f"{mesh_name}: Right child node {i_child_right} has {right_tri_count} "
                f"triangles, not less than parent node {i_node} with {parent_tri_count}"
            )

            # Verify children sum to parent count
            assert left_tri_count + right_tri_count == parent_tri_count, (
                f"{mesh_name}: Children triangle counts ({left_tri_count} + {right_tri_count}) "
                f"don't sum to parent count {parent_tri_count}"
            )


def test_partition_triangles():
    # Triangle vertices:
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Triangles:
    t = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [4, 5, 6],
            [5, 7, 6],
        ],
        dtype=np.uint32,
    )

    # Triangle centroids:
    c = v[t].mean(axis=-2)

    # Partition triangles at pivot index 2 along x-axis:
    i_lt, i_rt, aabb_lt, aabb_rt, sah_cost_lt, sah_cost_rt = zfw.partition_triangles(
        t=t, v=v, c=c, i=2, x=0
    )

    # Check left partition:
    assert i_lt.shape[0] == 2
    assert set(i_lt.tolist()) == {0, 1}

    # Check right partition:
    assert i_rt.shape[0] == 2
    assert set(i_rt.tolist()) == {2, 3}


def test_partition_points():
    # Number of points:
    n = 10

    # Generate points:
    v_min = -5.0
    v_max = +5.0
    v_ext = v_max - v_min
    v = v_min + (v_ext * np.linspace(0, 1, num=n, endpoint=True, dtype=np.float32))
    p = np.array([v, v, v], dtype=np.float32).T

    # Test all pivots and axes:
    for i in range(n):
        for x in range(3):
            i_lt, i_rt = zfw.partition_points(p=p, i=i, x=x)
            assert i_lt.shape[0] + i_rt.shape[0] == n
            assert np.all(i_lt < i)
            assert np.all(i_rt >= i)


def test_compute_triangles_abbb():
    v = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 4.0],
            [0.5, 1.5, -2.0],
        ],
        dtype=np.float32,
    )

    v_min, v_max = zfw.bvh.compute_triangles_aabb(v)

    assert np.allclose(np.asarray(v_min), [-1.0, 0.0, -2.0])
    assert np.allclose(np.asarray(v_max), [1.0, 2.0, 4.0])


if __name__ == "__main__":
    test_build_bvh("Suzanne.gltf")
