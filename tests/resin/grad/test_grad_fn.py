import pytest

from resin.core.etype import F4
from resin.dsl import View, param
from resin.grad import grad, grad_fn
from ..dsl.fixtures import LinearParams, mlp_step


def _param_tree(spec_name: str, shape: tuple[int, ...]) -> View:
    return param(shape=shape, etype=F4, label=spec_name)


def _mlp_x() -> View:
    return _param_tree("x", (64, 784))


def _mlp_y() -> View:
    return _param_tree("y", (64, 10))


def _mlp_params() -> LinearParams:
    return {
        "weight": _param_tree("weight", (10, 784)),
        "bias": _param_tree("bias", (10,)),
    }


class TestGradFn:
    def test_forward_and_grad_args(self) -> None:
        x, y, params = _mlp_x(), _mlp_y(), _mlp_params()
        forward, grad_args = grad_fn(mlp_step, grads_for=["params"])(
            x=x,
            y=y,
            params=params,
        )

        assert forward.shape == ()
        assert set(grad_args.keys()) == {"x", "y", "params"}
        assert grad_args["x"] is x
        assert grad_args["y"] is y
        assert set(params.keys()) == {"weight", "bias"}

    def test_build_grad_args_matches_grad_map(self) -> None:
        x, y, params = _mlp_x(), _mlp_y(), _mlp_params()
        forward = mlp_step(x=x, y=y, params=params)
        grad_map = grad(forward)
        _, grad_args = grad_fn(mlp_step, grads_for=["params"])(x=x, y=y, params=params)

        assert grad_args["params"] is not params
        for name in ("weight", "bias"):
            assert grad_map[params[name].node] is not None

    def test_zero_grad_for_unreachable_input(self) -> None:
        def sum_x(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, ()]:
            return x.sum().squeeze(axes=(0,))

        x = param(shape=(2,), etype=F4, label="x")
        y = param(shape=(2,), etype=F4, label="y")
        _, grad_args = grad_fn(sum_x, grads_for=["x", "y"])(x=x, y=y)

        grad_map = grad(sum_x(x=x, y=y))
        assert grad_map.get(y.node) is None
        assert grad_args["y"] is not y

    def test_empty_grads_for_computes_no_gradients(self) -> None:
        def sum_both(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, ()]:
            return (x + y).sum().squeeze(axes=(0,))

        x = param(shape=(2,), etype=F4, label="x")
        y = param(shape=(2,), etype=F4, label="y")
        _, grad_args = grad_fn(sum_both, grads_for=[])(x=x, y=y)

        assert grad_args["x"] is x
        assert grad_args["y"] is y

    def test_rejects_unknown_grads_for(self) -> None:
        with pytest.raises(ValueError, match="unknown grads_for"):
            grad_fn(mlp_step, grads_for=["param"])

    def test_accepts_positional_args(self) -> None:
        x, y, params = _mlp_x(), _mlp_y(), _mlp_params()
        forward, grad_args = grad_fn(mlp_step, grads_for=["params"])(
            x,
            y,
            params,
        )
        assert forward.shape == ()
        assert set(grad_args.keys()) == {"x", "y", "params"}