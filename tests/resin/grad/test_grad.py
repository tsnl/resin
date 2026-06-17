from resin import grad
from resin.dsl import ElementwiseNode, View, param


class TestAccessorAdjoint:
    def test_broadcast_reduces(self) -> None:
        x = param(shape=(3,), stype="f4", label="x")
        y = x.broadcast((4,))
        g = param(shape=y.shape, stype="f4", label="g")
        grad_x = grad.accessor_adjoint(y, g)

        assert grad_x.shape == x.shape
        assert grad_x.pitch == x.pitch

    def test_scalar_broadcast(self) -> None:
        x = param(shape=(), stype="f4", label="x")
        y = x.broadcast((2, 3))
        g = param(shape=y.shape, stype="f4", label="g")
        grad_x = grad.accessor_adjoint(y, g)

        assert grad_x.shape == x.shape


class TestReductionDfDo:
    def _make(self, axes, operator):
        x = param(shape=(2, 3), stype="f4", label="x")
        n = x.reduce(axes=axes, operator=operator)
        df_dn = param(shape=n.shape, stype="f4", label="df_dn")
        return x, n, df_dn

    def test_add_returns_broadcast_view(self) -> None:
        x, n, df_dn = self._make(axes=(1,), operator="add")
        (grad_x,) = grad.df_do(n.node, df_dn)

        assert isinstance(grad_x, View)
        assert grad_x.shape == x.shape
        assert 0 in grad_x.pitch

    def test_mul_uses_quotient(self) -> None:
        x, n, df_dn = self._make(axes=(1,), operator="mul")
        (grad_x,) = grad.df_do(n.node, df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    def test_max_masks_by_equality(self) -> None:
        x, n, df_dn = self._make(axes=(1,), operator="max")
        (grad_x,) = grad.df_do(n.node, df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    def test_min_masks_by_equality(self) -> None:
        x, n, df_dn = self._make(axes=(1,), operator="min")
        (grad_x,) = grad.df_do(n.node, df_dn)

        assert isinstance(grad_x.node, ElementwiseNode)
        assert grad_x.node.operator == "mul"
        assert grad_x.shape == x.shape

    def test_add_multi_axis(self) -> None:
        x = param(shape=(2, 3), stype="f4", label="x")
        n = x.reduce(axes=(0, 1), operator="add")
        df_dn = param(shape=n.shape, stype="f4", label="df_dn")
        (grad_x,) = grad.df_do(n.node, df_dn)

        assert grad_x.shape == x.shape

    def test_always_returns_single_element_tuple(self) -> None:
        for op in ("add", "mul", "max", "min"):
            _, n, df_dn = self._make(axes=(0,), operator=op)
            result = grad.df_do(n.node, df_dn)
            assert result is not None
            assert len(result) == 1


class TestGrad:
    def _scalar(self, view: View) -> View:
        s = view.sum()
        return s.squeeze(axes=tuple(range(s.rank)))

    def test_matmul_grad_shapes(self) -> None:
        x = param(shape=(2, 3), stype="f4", label="x")
        w = param(shape=(3, 4), stype="f4", label="w")
        g = grad.grad(self._scalar(x @ w))
        assert g[x.node].shape == x.shape
        assert g[w.node].shape == w.shape

    def test_broadcast_and_reduce_grad(self) -> None:
        p = param(shape=(3,), stype="f4", label="p")
        b = param(shape=(2, 3), stype="f4", label="b")
        g = grad.grad(self._scalar(b + p))
        assert g[p.node].shape == p.shape
        assert g[b.node].shape == b.shape

    def test_copy_then_sum_grad(self) -> None:
        x = param(shape=(4,), stype="f4", label="x")
        g = grad.grad(self._scalar(x.copy()))
        assert g[x.node].shape == x.shape
