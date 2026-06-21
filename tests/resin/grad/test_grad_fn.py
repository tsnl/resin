import pytest

from resin.dsl.spec import parse_signature
from resin.core.etype import F4
from resin.dsl import View, param
from resin.grad import _build_grad_args, grad, grad_fn
from ..dsl.fixtures import mlp_step


def _param_tree(spec_name: str, shape: tuple[int, ...]) -> View:
    return param(shape=shape, etype="f4", label=spec_name)


def _mlp_views() -> dict[str, object]:
    return {
        "x": _param_tree("x", (64, 784)),
        "y": _param_tree("y", (64, 10)),
        "params": {
            "weight": _param_tree("weight", (10, 784)),
            "bias": _param_tree("bias", (10,)),
        },
    }


class TestGradFn:
    def test_forward_and_grad_args(self) -> None:
        views = _mlp_views()
        forward, grad_args = grad_fn(mlp_step, grads_for=["params"])(**views)

        assert forward.shape == ()
        assert set(grad_args.keys()) == {"x", "y", "params"}
        assert grad_args["x"] is views["x"]
        assert grad_args["y"] is views["y"]
        assert set(grad_args["params"].keys()) == {"weight", "bias"}

    def test_build_grad_args_matches_grad_map(self) -> None:
        views = _mlp_views()
        spec = parse_signature(mlp_step)
        forward = mlp_step(**views)
        grad_map = grad(forward)
        grad_args = _build_grad_args(views, grad_map, ["params"], spec)

        for name, param_view in views["params"].items():
            grad_view = grad_args["params"][name]
            assert grad_map[param_view.node] is grad_view
            assert grad_map[param_view.node].node is grad_view.node

    def test_zero_grad_for_unreachable_input(self) -> None:
        def sum_x(x: View[F4, (2,)], y: View[F4, (2,)]) -> View[F4, ()]:
            return x.sum().squeeze(axes=(0,))

        x = param(shape=(2,), etype="f4", label="x")
        y = param(shape=(2,), etype="f4", label="y")
        _, grad_args = grad_fn(sum_x, grads_for=["x", "y"])(x=x, y=y)

        grad_map = grad(sum_x(x=x, y=y))
        assert grad_map.get(y.node) is None
        assert grad_args["y"].node is not y.node

    def test_rejects_unknown_grads_for(self) -> None:
        with pytest.raises(ValueError, match="unknown grads_for"):
            grad_fn(mlp_step, grads_for=["param"])

    def test_accepts_positional_args(self) -> None:
        views = _mlp_views()
        forward, grad_args = grad_fn(mlp_step, grads_for=["params"])(
            views["x"],
            views["y"],
            views["params"],
        )
        assert forward.shape == ()
        assert set(grad_args.keys()) == {"x", "y", "params"}