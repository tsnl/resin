"""Expect tests for graph.py debug_print output."""

import textwrap
from io import StringIO

from resin.graph import ElementwiseNode, Node, ViewNode


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


class TestViewDfDo:
    """Verify ViewNode.df_do() produces correct gradient graphs."""

    def test_broadcast_reduces(self) -> None:
        """Broadcast dims (pitch=0) should be sum-reduced in the backward pass."""
        x = Node.param((3,), "fp32", label="x")
        y = x.broadcast((4,))  # shape (4, 3), pitch (0, 1)
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_permutation_views_back(self) -> None:
        """Permutation has no broadcast dims — backward is just a view."""
        x = Node.param((2, 3), "fp32", label="x")
        y = x.permute((1, 0))  # shape (3, 2), pitch (1, 2)
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert isinstance(grad_x, ViewNode)
        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_identity_view(self) -> None:
        """Viewing with same shape/pitch is a no-op on the gradient."""
        x = Node.param((2, 3), "fp32", label="x")
        y = x.view(shape=x.shape, pitch=x.pitch)
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert grad_x.shape == x.shape

    def test_reshape_1d_to_2d(self) -> None:
        """Reshaping 1D to 2D is one-to-one — backward is just a view back."""
        x = Node.param((6,), "fp32", label="x")
        y = x.view(shape=(2, 3), pitch=(3, 1))
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert isinstance(grad_x, ViewNode)
        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_reshape_1d_to_3d(self) -> None:
        """Reshaping 1D to 3D is one-to-one — backward is just a view back."""
        x = Node.param((24,), "fp32", label="x")
        y = x.view(shape=(2, 3, 4), pitch=(12, 4, 1))
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
        (grad_x,) = y.df_do(df_dn)

        assert isinstance(grad_x, ViewNode)
        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_scalar_broadcast(self) -> None:
        """Broadcasting a scalar to a matrix should reduce all dims."""
        x = Node.param((), "fp32", label="x")
        y = x.broadcast((2, 3))  # shape (2, 3), pitch (0, 0)
        df_dn = Node.param(y.shape, "fp32", label="df_dn")
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
        x = Node.param((2, 3), "fp32", label="x")
        n = x.reduce(axes=axes, operator=operator)
        df_dn = Node.param(n.shape, "fp32", label="df_dn")
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
        x = Node.param((2, 3), "fp32", label="x")
        n = x.reduce(axes=(0, 1), operator="add")  # shape (1, 1)
        df_dn = Node.param(n.shape, "fp32", label="df_dn")
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
