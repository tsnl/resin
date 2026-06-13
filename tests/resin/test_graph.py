"""Expect tests for graph.py debug_print output."""

import textwrap
from io import StringIO

from resin.graph import (
    Accessor,
    ConstNode,
    ElementwiseNode,
    Node,
    ParamNode,
    ViewNode,
    c_contiguous_pitch_for_shape,
    invert_permutation,
    permute,
)


def debug_str(node: Node) -> str:
    out = StringIO()
    node.debug_print(out=out)
    return out.getvalue().strip()


class TestConstants:
    def test_scalar(self) -> None:
        t = ConstNode.new(42, stype="fp32")
        expected = "const(value=42) :: fp32(0,(),())"
        assert debug_str(t) == expected.strip()

    def test_1d(self) -> None:
        t = ConstNode.new([1, 2, 3], stype="fp32")
        expected = "const(value=[1, 2, 3]) :: fp32(0,(3,),(1,))"
        assert debug_str(t) == expected.strip()

    def test_2d(self) -> None:
        t = ConstNode.new([[1, 2], [3, 4]], stype="fp32")
        expected = "const(value=[[1, 2], [3, 4]]) :: fp32(0,(2, 2),(2, 1))"
        assert debug_str(t) == expected.strip()

    def test_fp16(self) -> None:
        t = ConstNode.new([1, 2], stype="fp16")
        expected = "const(value=[1, 2]) :: fp16(0,(2,),(1,))"
        assert debug_str(t) == expected.strip()


class TestElementwiseOps:
    def test_add(self) -> None:
        t1 = ConstNode.new([1, 2], stype="fp32")
        t2 = ConstNode.new([3, 4], stype="fp32")
        t = t1 + t2
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: fp32(0,(2,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │ └ const(value=[1, 2]) :: fp32(0,(2,),(1,))
            └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
              └ const(value=[3, 4]) :: fp32(0,(2,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_chained(self) -> None:
        t1 = ConstNode.new([1, 2], stype="fp32")
        t2 = ConstNode.new([3, 4], stype="fp32")
        t3 = ConstNode.new([5, 6], stype="fp32")
        t = (t1 + t2) * t3
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: fp32(0,(2,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │ └ elementwise(operator='add') :: fp32(0,(2,),(1,))
            │   ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │   │ └ const(value=[1, 2]) :: fp32(0,(2,),(1,))
            │   └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │     └ const(value=[3, 4]) :: fp32(0,(2,),(1,))
            └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
              └ const(value=[5, 6]) :: fp32(0,(2,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_unary_chain(self) -> None:
        t = ConstNode.new([1, 2, 3, 4], stype="fp32")
        t = t.exp().log().exp().log()
        expected = textwrap.dedent(
            """
            elementwise(operator='log') :: fp32(0,(4,),(1,))
            └ elementwise(operator='exp') :: fp32(0,(4,),(1,))
              └ elementwise(operator='log') :: fp32(0,(4,),(1,))
                └ elementwise(operator='exp') :: fp32(0,(4,),(1,))
                  └ const(value=[1, 2, 3, 4]) :: fp32(0,(4,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_scalar_broadcast(self) -> None:
        t = ConstNode.new([[1, 2], [3, 4]], stype="fp32")
        t = t + 10
        expected = textwrap.dedent(
            """
            elementwise(operator='add') :: fp32(0,(2, 2),(2, 1))
            ├ view(accessor=Accessor(offset=0, pitch=(2, 1), shape=(2, 2))) :: fp32(0,(2, 2),(2, 1))
            │ └ const(value=[[1, 2], [3, 4]]) :: fp32(0,(2, 2),(2, 1))
            └ view(accessor=Accessor(offset=0, pitch=(0, 0), shape=(2, 2))) :: fp32(0,(2, 2),(0, 0))
              └ const(value=10) :: fp32(0,(),())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_max_with_scalar(self) -> None:
        t = ConstNode.new([1, 2, 3], stype="fp32")
        t = t.max(0)
        expected = textwrap.dedent(
            """
            elementwise(operator='max') :: fp32(0,(3,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(3,))) :: fp32(0,(3,),(1,))
            │ └ const(value=[1, 2, 3]) :: fp32(0,(3,),(1,))
            └ view(accessor=Accessor(offset=0, pitch=(0,), shape=(3,))) :: fp32(0,(3,),(0,))
              └ const(value=0) :: fp32(0,(),())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_comparison(self) -> None:
        t1 = ConstNode.new([1, 2], stype="fp32")
        t2 = ConstNode.new([2, 1], stype="fp32")
        t = t1.gt(t2)
        expected = textwrap.dedent(
            """
            elementwise(operator='gt') :: fp32(0,(2,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │ └ const(value=[1, 2]) :: fp32(0,(2,),(1,))
            └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
              └ const(value=[2, 1]) :: fp32(0,(2,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()


class TestReduction:
    def test_sum(self) -> None:
        t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t.reduce(axes=(1,), operator="add")
        expected = textwrap.dedent(
            """
            reduction(operator='add', axes=(1,)) :: fp32(0,(2, 1),(1, 1))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestIndexing:
    def test_integer_index(self) -> None:
        t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t[0]
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=0, pitch=(1,), shape=(3,))) :: fp32(0,(3,),(1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = ConstNode.new([1, 2, 3, 4, 5, 6], stype="fp32")
        t = t[::2]
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=0, pitch=(2,), shape=(3,))) :: fp32(0,(3,),(2,))
            └ const(value=[1, 2, 3, 4, 5, 6]) :: fp32(0,(6,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_multi_dim(self) -> None:
        t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t[1, 1:3]
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=4, pitch=(1,), shape=(2,))) :: fp32(4,(2,),(1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestPermute:
    def test_transpose(self) -> None:
        t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t.permute((1, 0))
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=0, pitch=(1, 3), shape=(3, 2))) :: fp32(0,(3, 2),(1, 3))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestCompact:
    def test_compact_after_slice(self) -> None:
        t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t[(slice(None, None, 2), slice(None))]
        t = t.copy()
        expected = textwrap.dedent(
            """
            scatter(accessor=Accessor(offset=0, pitch=(3, 1), shape=(1, 3))) :: fp32(0,(1, 3),(3, 1))
            └ view(accessor=Accessor(offset=0, pitch=(6, 1), shape=(1, 3))) :: fp32(0,(1, 3),(6, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_compact_noop_on_contiguous(self) -> None:
        t = ConstNode.new([1, 2, 3], stype="fp32")
        t2 = t.copy()
        expected = textwrap.dedent(
            """
            scatter(accessor=Accessor(offset=0, pitch=(1,), shape=(3,))) :: fp32(0,(3,),(1,))
            └ const(value=[1, 2, 3]) :: fp32(0,(3,),(1,))
            """
        )
        assert debug_str(t2) == expected.strip()


class TestSharedSubexpressions:
    def test_shared_subgraph(self) -> None:
        t1 = ConstNode.new([1, 2], stype="fp32")
        t2 = ConstNode.new([3, 4], stype="fp32")
        t = t1 + t2
        t = t * t
        expected = textwrap.dedent(
            """
            elementwise(operator='mul') :: fp32(0,(2,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │ └ %0
            └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
              └ %0
            %0 := elementwise(operator='add') :: fp32(0,(2,),(1,))
            ├ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
            │ └ const(value=[1, 2]) :: fp32(0,(2,),(1,))
            └ view(accessor=Accessor(offset=0, pitch=(1,), shape=(2,))) :: fp32(0,(2,),(1,))
              └ const(value=[3, 4]) :: fp32(0,(2,),(1,))
            """
        )
        assert debug_str(t) == expected.strip()


class TestMatmul:
    def test_matmul_2d(self) -> None:
        t1 = ConstNode.new([[1, 2], [3, 4], [5, 6]], stype="fp32")
        t2 = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: fp32(0,(3, 3),(3, 1))
            ├ view(accessor=Accessor(offset=0, pitch=(2, 1), shape=(3, 2))) :: fp32(0,(3, 2),(2, 1))
            │ └ const(value=[[1, 2], [3, 4], [5, 6]]) :: fp32(0,(3, 2),(2, 1))
            └ view(accessor=Accessor(offset=0, pitch=(3, 1), shape=(2, 3))) :: fp32(0,(2, 3),(3, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_batched(self) -> None:
        t1 = ConstNode.new(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], stype="fp32"
        )
        t2 = ConstNode.new(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], stype="fp32"
        )
        r = t1 @ t2
        assert r.shape == (2, 3, 3)

    def test_matmul_2d_broadcast_left(self) -> None:
        """2D @ 3D: left operand is broadcast along the batch dim."""
        t1 = ConstNode.new([[1, 2], [3, 4], [5, 6]], stype="fp32")
        t2 = ConstNode.new(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], stype="fp32"
        )
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: fp32(0,(2, 3, 3),(9, 3, 1))
            ├ view(accessor=Accessor(offset=0, pitch=(0, 2, 1), shape=(2, 3, 2))) :: fp32(0,(2, 3, 2),(0, 2, 1))
            │ └ const(value=[[1, 2], [3, 4], [5, 6]]) :: fp32(0,(3, 2),(2, 1))
            └ view(accessor=Accessor(offset=0, pitch=(6, 3, 1), shape=(2, 2, 3))) :: fp32(0,(2, 2, 3),(6, 3, 1))
              └ const(value=[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) :: fp32(0,(2, 2, 3),(6, 3, 1))
            """
        )
        assert debug_str(t) == expected.strip()

    def test_matmul_2d_broadcast_right(self) -> None:
        """3D @ 2D: right operand is broadcast along the batch dim."""
        t1 = ConstNode.new(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], stype="fp32"
        )
        t2 = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
        t = t1 @ t2
        expected = textwrap.dedent(
            """
            matmul() :: fp32(0,(2, 3, 3),(9, 3, 1))
            ├ view(accessor=Accessor(offset=0, pitch=(6, 2, 1), shape=(2, 3, 2))) :: fp32(0,(2, 3, 2),(6, 2, 1))
            │ └ const(value=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]) :: fp32(0,(2, 3, 2),(6, 2, 1))
            └ view(accessor=Accessor(offset=0, pitch=(0, 3, 1), shape=(2, 2, 3))) :: fp32(0,(2, 2, 3),(0, 3, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: fp32(0,(2, 3),(3, 1))
            """
        )
        assert debug_str(t) == expected.strip()


class TestBroadcast:
    def test_full(self) -> None:
        t = ConstNode.full((2, 3), v=1, stype="fp32")
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=0, pitch=(0, 0), shape=(2, 3))) :: fp32(0,(2, 3),(0, 0))
            └ const(value=1) :: fp32(0,(),())
            """
        )
        assert debug_str(t) == expected.strip()

    def test_zeros(self) -> None:
        t = ConstNode.zeros((4,), stype="fp32")
        expected = textwrap.dedent(
            """
            view(accessor=Accessor(offset=0, pitch=(0,), shape=(4,))) :: fp32(0,(4,),(0,))
            └ const(value=0) :: fp32(0,(),())
            """
        )
        assert debug_str(t) == expected.strip()


class TestParam:
    def test_param_node(self) -> None:
        t = ParamNode.new(shape=(4,), stype="fp32", label="weights")
        expected = "param(label='weights') :: fp32(0,(4,),(1,))"
        assert debug_str(t) == expected.strip()


class TestViewDfDo:
    """Verify ViewNode.df_do() produces correct gradient graphs."""

    def test_broadcast_reduces(self) -> None:
        """Broadcast dims (pitch=0) should be sum-reduced in the backward pass."""
        x = ParamNode.new(shape=(3,), stype="fp32", label="x")
        y = x.broadcast((4,))  # shape (4, 3), pitch (0, 1)
        df_dn = ParamNode.new(shape=y.shape, stype="fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_scalar_broadcast(self) -> None:
        """Broadcasting a scalar to a matrix should reduce all dims."""
        x = ParamNode.new(shape=(), stype="fp32", label="x")
        y = x.broadcast((2, 3))  # shape (2, 3), pitch (0, 0)
        df_dn = ParamNode.new(shape=y.shape, stype="fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert grad_x.shape == x.shape


class TestReductionDfDo:
    """Verify ReductionNode.df_do() produces correct gradient graphs.

    Each test builds a small forward graph (param -> reduce), then calls
    df_do() with a mock upstream gradient and checks the resulting graph
    structure: node type, operator, and shape.
    """

    def _make(self, axes, operator):
        """Helper: build input, reduction node, and upstream gradient."""
        x = ParamNode.new(shape=(2, 3), stype="fp32", label="x")
        n = x.reduce(axes=axes, operator=operator)
        df_dn = ParamNode.new(shape=n.shape, stype="fp32", label="df_dn")
        return x, n, df_dn

    # -- add ------------------------------------------------------------------

    def test_add_returns_view_broadcast(self) -> None:
        """∂(Σxᵢ)/∂xⱼ = 1 — gradient is just df_dn broadcast to input shape."""
        x, n, df_dn = self._make(axes=(1,), operator="add")
        (grad_x,) = n.df_do(df_dn)

        # Broadcasting df_dn from (2,1) back to (2,3) is a pure view.
        assert isinstance(grad_x, ViewNode)
        assert grad_x.shape == x.shape

    # -- mul ------------------------------------------------------------------

    def test_mul_uses_quotient(self) -> None:
        """∂(∏xᵢ)/∂xⱼ = (∏xᵢ)/xⱼ — top-level node is df_dn * (n / x)."""
        x, n, df_dn = self._make(axes=(1,), operator="mul")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x, ElementwiseNode)
        assert grad_x.operator == "mul"
        assert grad_x.shape == x.shape

    # -- max / min ------------------------------------------------------------

    def test_max_masks_by_equality(self) -> None:
        """∂(max xᵢ)/∂xⱼ = 1{xⱼ == max} — top node is mul (broadcast * mask)."""
        x, n, df_dn = self._make(axes=(1,), operator="max")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x, ElementwiseNode)
        assert grad_x.operator == "mul"
        assert grad_x.shape == x.shape

    def test_min_masks_by_equality(self) -> None:
        """∂(min xᵢ)/∂xⱼ = 1{xⱼ == min} — same structure as max."""
        x, n, df_dn = self._make(axes=(1,), operator="min")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x, ElementwiseNode)
        assert grad_x.operator == "mul"
        assert grad_x.shape == x.shape

    # -- multi-axis -----------------------------------------------------------

    def test_add_multi_axis(self) -> None:
        """Reducing over all axes: gradient broadcasts scalar back to full shape."""
        x = ParamNode.new(shape=(2, 3), stype="fp32", label="x")
        n = x.reduce(axes=(0, 1), operator="add")  # shape (1, 1)
        df_dn = ParamNode.new(shape=n.shape, stype="fp32", label="df_dn")
        (grad_x,) = n.df_do(df_dn)

        assert grad_x.shape == x.shape

    # -- single operand -------------------------------------------------------

    def test_always_returns_single_element_tuple(self) -> None:
        """ReductionNode has one input, so df_do always returns a 1-tuple."""
        for op in ("add", "mul", "max", "min"):
            _, n, df_dn = self._make(axes=(0,), operator=op)
            result = n.df_do(df_dn)
            assert result is not None
            assert len(result) == 1


class TestAccessorMath:
    def test_address_index_inverse(self) -> None:
        """
        Applying an accessor and then its inverse should yield the original index for a C-contiguous
        accessor (i.e. 1:1 relationship between index and address).
        """

        offset = 5
        shape = (4, 5)

        accessor = Accessor(
            offset=offset,
            pitch=permute(c_contiguous_pitch_for_shape(shape), (1, 0)),
            shape=shape,
        )

        for i in range(accessor.shape[0]):
            for j in range(accessor.shape[1]):
                index = (i, j)
                address = accessor.address(index)
                index_ref = accessor.index(address)
                assert index == index_ref


class TestPermutationMath:
    def test_permute_inverse(self) -> None:
        """Permuting by a given order and then by the inverse should yield the original shape."""
        permutation = (3, 2, 0, 1)
        inverse_permutation = invert_permutation(permutation)

        xs = (100, 101, 102, 103)
        ys = permute(xs, permutation)
        xs_ref = permute(ys, inverse_permutation)

        assert xs == xs_ref
