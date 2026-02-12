"""Expect tests for graph.py debug_print output."""

import textwrap
from io import StringIO

from resin.graph import Node


def debug_str(node: Node) -> str:
    out = StringIO()
    node.debug_print(out=out)
    return out.getvalue().strip()


class TestConstants:
    def test_scalar(self) -> None:
        t = Node.const(42, dtype="fp32")
        expected = "const(value=42) :: (fp32 () ())"
        assert debug_str(t) == expected.strip()

    def test_1d(self) -> None:
        t = Node.const([1, 2, 3], dtype="fp32")
        expected = "const(value=[1, 2, 3]) :: (fp32 (3,) (1,))"
        assert debug_str(t) == expected.strip()

    def test_2d(self) -> None:
        t = Node.const([[1, 2], [3, 4]], dtype="fp32")
        expected = "const(value=[[1, 2], [3, 4]]) :: (fp32 (2, 2) (2, 1))"
        assert debug_str(t) == expected.strip()

    def test_fp16(self) -> None:
        t = Node.const([1, 2], dtype="fp16")
        expected = "const(value=[1, 2]) :: (fp16 (2,) (1,))"
        assert debug_str(t) == expected.strip()


class TestElementwiseOps:
    def test_add(self) -> None:
        t1 = Node.const([1, 2], dtype="fp32")
        t2 = Node.const([3, 4], dtype="fp32")
        t = t1 + t2
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: (fp32 (2,) (1,))
            ├ view() :: (fp32 (2,) (1,))
            │ └ const(value=[1, 2]) :: (fp32 (2,) (1,))
            └ view() :: (fp32 (2,) (1,))
              └ const(value=[3, 4]) :: (fp32 (2,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_chained(self) -> None:
        t1 = Node.const([1, 2], dtype="fp32")
        t2 = Node.const([3, 4], dtype="fp32")
        t3 = Node.const([5, 6], dtype="fp32")
        t = (t1 + t2) * t3
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: (fp32 (2,) (1,))
            ├ view() :: (fp32 (2,) (1,))
            │ └ elementwise(operator='add') :: (fp32 (2,) (1,))
            │   ├ view() :: (fp32 (2,) (1,))
            │   │ └ const(value=[1, 2]) :: (fp32 (2,) (1,))
            │   └ view() :: (fp32 (2,) (1,))
            │     └ const(value=[3, 4]) :: (fp32 (2,) (1,))
            └ view() :: (fp32 (2,) (1,))
              └ const(value=[5, 6]) :: (fp32 (2,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_unary_chain(self) -> None:
        t = Node.const([1, 2, 3, 4], dtype="fp32")
        t = t.exp().log().exp().log()
        expected = textwrap.dedent(
            """
            elementwise(operator='log') :: (fp32 (4,) (1,))
            └ elementwise(operator='exp') :: (fp32 (4,) (1,))
              └ elementwise(operator='log') :: (fp32 (4,) (1,))
                └ elementwise(operator='exp') :: (fp32 (4,) (1,))
                  └ const(value=[1, 2, 3, 4]) :: (fp32 (4,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_scalar_broadcast(self) -> None:
        t = Node.const([[1, 2], [3, 4]], dtype="fp32")
        t = t + 10
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: (fp32 (2, 2) (2, 1))
            ├ view() :: (fp32 (2, 2) (2, 1))
            │ └ const(value=[[1, 2], [3, 4]]) :: (fp32 (2, 2) (2, 1))
            └ view() :: (fp32 (2, 2) (0, 0))
              └ const(value=10) :: (fp32 () ())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_max_with_scalar(self) -> None:
        t = Node.const([1, 2, 3], dtype="fp32")
        t = t.max(0)
        expected = textwrap.dedent(
            """
            elementwise(operator='max') :: (fp32 (3,) (1,))
            ├ view() :: (fp32 (3,) (1,))
            │ └ const(value=[1, 2, 3]) :: (fp32 (3,) (1,))
            └ view() :: (fp32 (3,) (0,))
              └ const(value=0) :: (fp32 () ())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_comparison(self) -> None:
        t1 = Node.const([1, 2], dtype="fp32")
        t2 = Node.const([2, 1], dtype="fp32")
        t = t1.gt(t2)
        expected = textwrap.dedent(
            """
            elementwise(operator='gt') :: (fp32 (2,) (1,))
            ├ view() :: (fp32 (2,) (1,))
            │ └ const(value=[1, 2]) :: (fp32 (2,) (1,))
            └ view() :: (fp32 (2,) (1,))
              └ const(value=[2, 1]) :: (fp32 (2,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()


class TestReduction:
    def test_sum(self) -> None:
        t = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t.reduce(axes=(1,), operator="add")
        expected = textwrap.dedent(
            """
            reduction(operator='add', axes=(1,)) :: (fp32 (2, 1) (3, 1))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestIndexing:
    def test_integer_index(self) -> None:
        t = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t[0]
        expected = textwrap.dedent(
            """
            index(key=(0,)) :: (fp32 (3,) (1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = Node.const([1, 2, 3, 4, 5, 6], dtype="fp32")
        t = t[::2]
        expected = textwrap.dedent(
            """
            index(key=(slice(None, None, 2),)) :: (fp32 (3,) (2,))
            └ const(value=[1, 2, 3, 4, 5, 6]) :: (fp32 (6,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_multi_dim(self) -> None:
        t = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t[1, 1:3]
        expected = textwrap.dedent(
            """
            index(key=(1, slice(1, 3, None))) :: (fp32 (2,) (1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestPermute:
    def test_transpose(self) -> None:
        t = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t.permute((1, 0))
        expected = textwrap.dedent(
            """
            view() :: (fp32 (3, 2) (1, 3))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestCompact:
    def test_compact_after_slice(self) -> None:
        t = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t[(slice(None, None, 2), slice(None))]
        t = t.copy()
        expected = textwrap.dedent(
            """
            copy() :: (fp32 (1, 3) (3, 1))
            └ index(key=(slice(None, None, 2), slice(None, None, None))) :: (fp32 (1, 3) (6, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_compact_noop_on_contiguous(self) -> None:
        t = Node.const([1, 2, 3], dtype="fp32")
        assert t.copy() is t


class TestSharedSubexpressions:
    def test_shared_subgraph(self) -> None:
        t1 = Node.const([1, 2], dtype="fp32")
        t2 = Node.const([3, 4], dtype="fp32")
        t = t1 + t2
        t = t * t
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: (fp32 (2,) (1,))
            ├ view() :: (fp32 (2,) (1,))
            │ └ %0
            └ view() :: (fp32 (2,) (1,))
              └ %0
            %0 := elementwise(operator='add') :: (fp32 (2,) (1,))
            ├ view() :: (fp32 (2,) (1,))
            │ └ const(value=[1, 2]) :: (fp32 (2,) (1,))
            └ view() :: (fp32 (2,) (1,))
              └ const(value=[3, 4]) :: (fp32 (2,) (1,))
            """
        )
        assert debug_str(t) == expected.strip()


class TestMatmul:
    def test_matmul_2d(self) -> None:
        t1 = Node.const([[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: (fp32 (3, 3) (3, 1))
            ├ view() :: (fp32 (3, 2) (2, 1))
            │ └ const(value=[[1, 2], [3, 4], [5, 6]]) :: (fp32 (3, 2) (2, 1))
            └ view() :: (fp32 (2, 3) (3, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_batched(self) -> None:
        t1 = Node.const(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype="fp32"
        )
        t2 = Node.const(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype="fp32"
        )
        r = t1 @ t2
        assert r.shape == (2, 3, 3)

    def test_matmul_2d_broadcast_left(self) -> None:
        """2D @ 3D: left operand is broadcast along the batch dim."""
        t1 = Node.const([[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = Node.const(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype="fp32"
        )
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: (fp32 (2, 3, 3) (9, 3, 1))
            ├ view() :: (fp32 (2, 3, 2) (0, 2, 1))
            │ └ const(value=[[1, 2], [3, 4], [5, 6]]) :: (fp32 (3, 2) (2, 1))
            └ view() :: (fp32 (2, 2, 3) (6, 3, 1))
              └ const(value=[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) :: (fp32 (2, 2, 3) (6, 3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_2d_broadcast_right(self) -> None:
        """3D @ 2D: right operand is broadcast along the batch dim."""
        t1 = Node.const(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype="fp32"
        )
        t2 = Node.const([[1, 2, 3], [4, 5, 6]], dtype="fp32")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: (fp32 (2, 3, 3) (9, 3, 1))
            ├ view() :: (fp32 (2, 3, 2) (6, 2, 1))
            │ └ const(value=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]) :: (fp32 (2, 3, 2) (6, 2, 1))
            └ view() :: (fp32 (2, 2, 3) (0, 3, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: (fp32 (2, 3) (3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestBroadcast:
    def test_full(self) -> None:
        t = Node.full((2, 3), v=1, dtype="fp32")
        expected = textwrap.dedent(
            """
            view() :: (fp32 (2, 3) (0, 0))
            └ const(value=1) :: (fp32 () ())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_zeros(self) -> None:
        t = Node.zeros((4,), dtype="fp32")
        expected = textwrap.dedent(
            """
            view() :: (fp32 (4,) (0,))
            └ const(value=0) :: (fp32 () ())
            """
        )
        assert debug_str(t) == expected.strip()


class TestParam:
    def test_param_node(self) -> None:
        t = Node.param((4,), "fp32", label="weights")
        expected = "param(label='weights') :: (fp32 (4,) (1,))"
        assert debug_str(t) == expected.strip()
