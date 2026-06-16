"""Expect tests for front.py debug_print output."""

import textwrap
from io import StringIO

from resin import gpu
from resin.gpu.front import (
    ElementwiseNode,
    View,
    const,
    full,
    invert_permutation,
    param,
    permute,
    zeros,
)


def debug_str(view: View) -> str:
    out = StringIO()
    view.debug_print(out=out)
    return out.getvalue().strip()


class TestConstants:
    def test_scalar(self) -> None:
        t = const(42, stype="f4")
        expected = "const(value=42) :: f4()"
        assert debug_str(t) == expected.strip()

    def test_1d(self) -> None:
        t = const([1, 2, 3], stype="f4")
        expected = "const(value=[1, 2, 3]) :: f4(3,)"
        assert debug_str(t) == expected.strip()

    def test_2d(self) -> None:
        t = const([[1, 2], [3, 4]], stype="f4")
        expected = "const(value=[[1, 2], [3, 4]]) :: f4(2, 2)"
        assert debug_str(t) == expected.strip()

    def test_fp16(self) -> None:
        t = const([1, 2], stype="f2")
        expected = "const(value=[1, 2]) :: f2(2,)"
        assert debug_str(t) == expected.strip()


class TestElementwiseOps:
    def test_add(self) -> None:
        t1 = const([1, 2], stype="f4")
        t2 = const([3, 4], stype="f4")
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
        t1 = const([1, 2], stype="f4")
        t2 = const([3, 4], stype="f4")
        t3 = const([5, 6], stype="f4")
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
        t = const([1, 2, 3, 4], stype="f4")
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
        t = const([[1, 2], [3, 4]], stype="f4")
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
        t = const([1, 2, 3], stype="f4")
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
        t1 = const([1, 2], stype="f4")
        t2 = const([2, 1], stype="f4")
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
        t = const([[1, 2, 3], [4, 5, 6]], stype="f4")
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
        t = const([[1, 2, 3], [4, 5, 6]], stype="f4")
        t = t[0]
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(3,), pitch=(1,))
            └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = const([1, 2, 3, 4, 5, 6], stype="f4")
        t = t[::2]
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(3,), pitch=(2,))
            └ const(value=[1, 2, 3, 4, 5, 6]) :: f4(6,)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_multi_dim(self) -> None:
        t = const([[1, 2, 3], [4, 5, 6]], stype="f4")
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
        t = const([[1, 2, 3], [4, 5, 6]], stype="f4")
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
        t = const([[1, 2, 3], [4, 5, 6]], stype="f4")
        t = t[(slice(None, None, 2), slice(None))]
        t = t.copy()
        expected = textwrap.dedent(
            """
            scatter(woffset=0, wpitch=(3, 1)) :: f4(1, 3)
            └ view(offset=0, shape=(1, 3), pitch=(6, 1))
              └ const(value=[[1, 2, 3], [4, 5, 6]]) :: f4(2, 3)
            """
        )
        assert debug_str(t) == expected.strip()

    def test_compact_noop_on_contiguous(self) -> None:
        t = const([1, 2, 3], stype="f4")
        t2 = t.copy()
        expected = textwrap.dedent(
            """
            scatter(woffset=0, wpitch=(1,)) :: f4(3,)
            └ const(value=[1, 2, 3]) :: f4(3,)
            """
        )
        assert debug_str(t2) == expected.strip()


class TestSharedSubexpressions:
    def test_shared_subgraph(self) -> None:
        t1 = const([1, 2], stype="f4")
        t2 = const([3, 4], stype="f4")
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
        t1 = const([[1, 2], [3, 4], [5, 6]], stype="f4")
        t2 = const([[1, 2, 3], [4, 5, 6]], stype="f4")
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
        t1 = const([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], stype="f4")
        t2 = const([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], stype="f4")
        r = t1 @ t2
        assert r.shape == (2, 3, 3)

    def test_matmul_2d_broadcast_left(self) -> None:
        """2D @ 3D: left operand is broadcast along the batch dim."""
        t1 = const([[1, 2], [3, 4], [5, 6]], stype="f4")
        t2 = const([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], stype="f4")
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
        """3D @ 2D: right operand is broadcast along the batch dim."""
        t1 = const([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], stype="f4")
        t2 = const([[1, 2, 3], [4, 5, 6]], stype="f4")
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
        t = full((2, 3), v=1, stype="f4")
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(2, 3), pitch=(0, 0))
            └ const(value=1) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()

    def test_zeros(self) -> None:
        t = zeros((4,), stype="f4")
        expected = textwrap.dedent(
            """
            view(offset=0, shape=(4,), pitch=(0,))
            └ const(value=0) :: f4()
            """
        )
        assert debug_str(t) == expected.strip()


class TestParam:
    def test_param_node(self) -> None:
        t = param(shape=(4,), stype="f4", label="weights")
        expected = "param(label='weights') :: f4(4,)"
        assert debug_str(t) == expected.strip()


class TestViewAdjoint:
    """Verify View._accessor_adjoint() pushes gradients back into the backing node."""

    def test_broadcast_reduces(self) -> None:
        """Broadcast dims (pitch=0) should be sum-reduced in the backward pass."""
        x = param(shape=(3,), stype="f4", label="x")
        y = x.broadcast((4,))  # shape (4, 3), pitch (0, 1)
        g = param(shape=y.shape, stype="f4", label="g")
        grad_x = y._accessor_adjoint(g)

        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_scalar_broadcast(self) -> None:
        """Broadcasting a scalar to a matrix should reduce all dims."""
        x = param(shape=(), stype="f4", label="x")
        y = x.broadcast((2, 3))  # shape (2, 3), pitch (0, 0)
        g = param(shape=y.shape, stype="f4", label="g")
        grad_x = y._accessor_adjoint(g)

        assert grad_x.shape == x.shape


class TestReductionDfDo:
    """Verify ReductionNode.df_do() produces correct gradient graphs.

    Each test builds a small forward graph (param -> reduce), then calls df_do() on the
    resulting view with a mock upstream gradient and checks the resulting graph structure:
    backing node type, operator, and shape.
    """

    def _make(self, axes, operator):
        """Helper: build input, reduction view, and upstream gradient."""
        x = param(shape=(2, 3), stype="f4", label="x")
        n = x.reduce(axes=axes, operator=operator)
        df_dn = param(shape=n.shape, stype="f4", label="df_dn")
        return x, n, df_dn

    # -- add ------------------------------------------------------------------

    def test_add_returns_broadcast_view(self) -> None:
        """∂(Σxᵢ)/∂xⱼ = 1 — gradient is just df_dn broadcast to input shape."""
        x, n, df_dn = self._make(axes=(1,), operator="add")
        (grad_x,) = n.df_do(df_dn)

        # Broadcasting df_dn from (2,1) back to (2,3) is a pure view (broadcast pitch 0).
        assert isinstance(grad_x, View)
        assert grad_x.shape == x.shape
        assert 0 in grad_x.pitch

    # -- mul ------------------------------------------------------------------

    def test_mul_uses_quotient(self) -> None:
        """∂(∏xᵢ)/∂xⱼ = (∏xᵢ)/xⱼ — top-level node is df_dn * (n / x)."""
        x, n, df_dn = self._make(axes=(1,), operator="mul")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    # -- max / min ------------------------------------------------------------

    def test_max_masks_by_equality(self) -> None:
        """∂(max xᵢ)/∂xⱼ = 1{xⱼ == max} — top node is mul (broadcast * mask)."""
        x, n, df_dn = self._make(axes=(1,), operator="max")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    def test_min_masks_by_equality(self) -> None:
        """∂(min xᵢ)/∂xⱼ = 1{xⱼ == min} — same structure as max."""
        x, n, df_dn = self._make(axes=(1,), operator="min")
        (grad_x,) = n.df_do(df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    # -- multi-axis -----------------------------------------------------------

    def test_add_multi_axis(self) -> None:
        """Reducing over all axes: gradient broadcasts scalar back to full shape."""
        x = param(shape=(2, 3), stype="f4", label="x")
        n = x.reduce(axes=(0, 1), operator="add")  # shape (1, 1)
        df_dn = param(shape=n.shape, stype="f4", label="df_dn")
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


class TestGrad:
    """End-to-end gradient checks: grad is keyed by backing Node."""

    def _scalar(self, view: View) -> View:
        s = view.sum()
        return s.squeeze(axes=tuple(range(s.rank)))

    def test_matmul_grad_shapes(self) -> None:
        x = param(shape=(2, 3), stype="f4", label="x")
        w = param(shape=(3, 4), stype="f4", label="w")
        g = gpu.front.grad(self._scalar(x @ w))
        assert g[x.node].shape == x.shape
        assert g[w.node].shape == w.shape

    def test_broadcast_and_reduce_grad(self) -> None:
        p = param(shape=(3,), stype="f4", label="p")
        b = param(shape=(2, 3), stype="f4", label="b")
        g = gpu.front.grad(self._scalar(b + p))
        assert g[p.node].shape == p.shape
        assert g[b.node].shape == b.shape

    def test_copy_then_sum_grad(self) -> None:
        x = param(shape=(4,), stype="f4", label="x")
        g = gpu.front.grad(self._scalar(x.copy()))
        assert g[x.node].shape == x.shape


class TestViewMath:
    def test_chained_slice_offsets(self) -> None:
        """View composition stays on the same backing node with correct offset/pitch."""
        t = const(list(range(10)), stype="f4")
        v = t[2:8][::2]  # base indices 2, 4, 6

        assert v.node is t.node
        assert v.offset == 2
        assert v.shape == (3,)
        assert v.pitch == (2,)

    def test_permute_after_slice(self) -> None:
        """Permuting a sliced view composes correctly onto the backing node."""
        t = const([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], stype="f4")  # (3, 4)
        v = t[1:].permute((1, 0))

        assert v.node is t.node
        assert v.offset == 4
        assert v.shape == (4, 2)
        assert v.pitch == (1, 4)


class TestPermutationMath:
    def test_permute_inverse(self) -> None:
        """Permuting by a given order and then by the inverse should yield the original."""
        permutation = (3, 2, 0, 1)
        inverse_permutation = invert_permutation(permutation)

        xs = (100, 101, 102, 103)
        ys = permute(xs, permutation)
        xs_ref = permute(ys, inverse_permutation)

        assert xs == xs_ref
