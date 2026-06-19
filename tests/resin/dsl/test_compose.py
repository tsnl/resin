"""Expect tests for dsl debug_print output."""

import textwrap
from io import StringIO

from resin.dsl.dsl import View, const, full, param, zeros


def debug_str(view: View) -> str:
    out = StringIO()
    view.debug_print(out=out)
    return out.getvalue().strip()


class TestConstants:
    def test_scalar(self) -> None:
        t = const(42, dtype="f4")
        expected = "const(value=42) :: f4()"
        assert debug_str(t) == expected.strip()

    def test_1d(self) -> None:
        t = const([1, 2, 3], dtype="f4")
        expected = "const(value=[1, 2, 3]) :: f4(3,)"
        assert debug_str(t) == expected.strip()

    def test_2d(self) -> None:
        t = const([[1, 2], [3, 4]], dtype="f4")
        expected = "const(value=[[1, 2], [3, 4]]) :: f4(2, 2)"
        assert debug_str(t) == expected.strip()

    def test_fp16(self) -> None:
        t = const([1, 2], dtype="f2")
        expected = "const(value=[1, 2]) :: f2(2,)"
        assert debug_str(t) == expected.strip()


class TestElementwiseOps:
    def test_add(self) -> None:
        t1 = const([1, 2], dtype="f4")
        t2 = const([3, 4], dtype="f4")
        t = t1 + t2
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: f4(2,)
            ├ const(value=[1, 2]) :: f4(2,)
            └ const(value=[3, 4]) :: f4(2,)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_chained(self) -> None:
        t1 = const([1, 2], dtype="f4")
        t2 = const([3, 4], dtype="f4")
        t3 = const([5, 6], dtype="f4")
        t = (t1 + t2) * t3
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: f4(2,)
            ├ elementwise(operator='add') :: f4(2,)
            │ ├ const(value=[1, 2]) :: f4(2,)
            │ └ const(value=[3, 4]) :: f4(2,)
            └ const(value=[5, 6]) :: f4(2,)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_unary_chain(self) -> None:
        t = const([1, 2, 3, 4], dtype="f4")
        t = t.exp().log().exp().log()
        expected = textwrap.dedent(
            """
            elementwise(operator='log') :: f4(4,)
            └ elementwise(operator='exp') :: f4(4,)
              └ elementwise(operator='log') :: f4(4,)
                └ elementwise(operator='exp') :: f4(4,)
                  └ const(value=[1, 2, 3, 4]) :: f4(4,)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_scalar_broadcast(self) -> None:
        t = const([[1, 2], [3, 4]], dtype="f4")
        t = t + 10
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: f4(2, 2)
            ├ const(value=[[1, 2], [3, 4]]) :: f4(2, 2)
            └ view(offset=0, shape=(2, 2), pitch=(0, 0))
              └ const(value=10) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()

    def test_max_with_scalar(self) -> None:
        t = const([1, 2, 3], dtype="f4")
        t = t.max(0)
        expected = textwrap.dedent(
            """
            elementwise(operator='max') :: f4(3,)
            ├ const(value=[1, 2, 3]) :: f4(3,)
            └ view(offset=0, shape=(3,), pitch=(0,))
              └ const(value=0) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()

    def test_comparison(self) -> None:
        t1 = const([1, 2], dtype="f4")
        t2 = const([2, 1], dtype="f4")
        t = t1.gt(t2)
        expected = textwrap.dedent(
            """
            elementwise(operator='gt') :: f4(2,)
            ├ const(value=[1, 2]) :: f4(2,)
            └ const(value=[2, 1]) :: f4(2,)
            """
        )
        assert debug_str(t) == expected.strip()


class TestReduction:
    def test_sum(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t.reduce(axes=(1,), operator="add")
        expected = textwrap.dedent(
            """
            reduction(operator='add', axes=(1,)) :: f4(2, 1)
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()


class TestIndexing:
    def test_integer_index(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t[0]
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(3,), pitch=(1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = const([1, 2, 3, 4, 5, 6], dtype="f4")
        t = t[::2]
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(3,), pitch=(2,))
            └ const(value=[1, 2, 3, 4, 5, 6]) :: f4(6,)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_multi_dim(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t[1, 1:3]
        expected = textwrap.dedent(
            """
            view(offset=4, shape=(2,), pitch=(1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()


class TestPermute:
    def test_transpose(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t.permute((1, 0))
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(3, 2), pitch=(1, 3))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()


class TestCompact:
    def test_compact_after_slice(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t[(slice(None, None, 2), slice(None))]
        t = t.copy()
        expected = textwrap.dedent(
            """
            scatter(operator=None, woffset=0, wpitch=(3, 1)) :: f4(1, 3)
            └ view(offset=0, shape=(1, 3), pitch=(6, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_compact_noop_on_contiguous(self) -> None:
        t = const([1, 2, 3], dtype="f4")
        t2 = t.copy()
        expected = textwrap.dedent(
            """
            scatter(operator=None, woffset=0, wpitch=(1,)) :: f4(3,)
            └ const(value=[1, 2, 3]) :: f4(3,)
            """
        )
        assert debug_str(t2) == expected.strip()


class TestSharedSubexpressions:
    def test_shared_subgraph(self) -> None:
        t1 = const([1, 2], dtype="f4")
        t2 = const([3, 4], dtype="f4")
        t = t1 + t2
        t = t * t
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: f4(2,)
            ├ %0
            └ %0
            %0 := elementwise(operator='add') :: f4(2,)
            ├ const(value=[1, 2]) :: f4(2,)
            └ const(value=[3, 4]) :: f4(2,)
            """
        )
        assert debug_str(t) == expected.strip()


class TestMatmul:
    def test_matmul_2d(self) -> None:
        t1 = const([[1, 2], [3, 4], [5, 6]], dtype="f4")
        t2 = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: f4(3, 3)
            ├ const(value=[[1, 2], [3, 4], [5, 6]]) :: f4(3, 2)
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_batched(self) -> None:
        t1 = const([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype="f4")
        t2 = const([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype="f4")
        r = t1 @ t2
        assert r.shape == (2, 3, 3)

    def test_matmul_2d_broadcast_left(self) -> None:
        t1 = const([[1, 2], [3, 4], [5, 6]], dtype="f4")
        t2 = const([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype="f4")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: f4(2, 3, 3)
            ├ view(offset=0, shape=(2, 3, 2), pitch=(0, 2, 1))
            │ └ const(value=[[1, 2], [3, 4], [5, 6]]) :: f4(3, 2)
            └ const(value=[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) :: f4(2, 2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_2d_broadcast_right(self) -> None:
        t1 = const([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype="f4")
        t2 = const([[1, 2, 3], [4, 5, 6]], dtype="f4")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: f4(2, 3, 3)
            ├ const(value=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]) :: f4(2, 3, 2)
            └ view(offset=0, shape=(2, 2, 3), pitch=(0, 3, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()


class TestBroadcast:
    def test_full(self) -> None:
        t = full((2, 3), v=1, dtype="f4")
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(2, 3), pitch=(0, 0))
            └ const(value=1) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()

    def test_zeros(self) -> None:
        t = zeros((4,), dtype="f4")
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(4,), pitch=(0,))
            └ const(value=0) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()


class TestParam:
    def test_param_node(self) -> None:
        t = param(shape=(4,), dtype="f4", label="weights")
        expected = "param(label='weights') :: f4(4,)"
        assert debug_str(t) == expected.strip()
