import zfw

import numpy as np


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
