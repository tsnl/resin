import pytest

from resin.core.pytree import flatten_pytree
from resin.dsl.node import ParamNode
from resin.core.etype import F4
from resin.dsl import View, param
from resin.dsl.functional import trace, typecheck
from .fixtures import mlp_step


class TestTrace:
    def test_mlp_step_bindings_are_params(self) -> None:
        bindings, output = trace(mlp_step)

        assert set(bindings.keys()) == {"x", "y", "params"}
        assert output.shape == ()

        for view in flatten_pytree(bindings):
            assert isinstance(view.node, ParamNode)


class TestTypecheck:
    def test_rejects_invalid_shape(self) -> None:
        @typecheck
        def add(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, (2,)]:
            return x + y

        x = param(shape=(2,), etype=F4, label="x")
        y = param(shape=(3,), etype=F4, label="y")
        with pytest.raises(ValueError, match="expected shape"):
            add(x=x, y=y)

    def test_accepts_valid_views(self) -> None:
        @typecheck
        def add(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, (2,)]:
            return x + y

        x = param(shape=(2,), etype=F4, label="x")
        y = param(shape=(2,), etype=F4, label="y")
        assert (add(x=x, y=y) - x - y).shape == (2,)

    def test_accepts_positional_args(self) -> None:
        @typecheck
        def add(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, (2,)]:
            return x + y

        x = param(shape=(2,), etype=F4, label="x")
        y = param(shape=(2,), etype=F4, label="y")
        assert (add(x, y) - x - y).shape == (2,)