from resin.core.pytree import flatten_pytree
from resin.dsl.dsl import ParamNode
from resin.dsl.functional import trace
from .fixtures import mlp_step


class TestTrace:
    def test_mlp_step_bindings_are_params(self) -> None:
        bindings, output = trace(mlp_step)

        assert set(bindings.keys()) == {"x", "y", "params"}
        assert output.shape == ()

        for view in flatten_pytree(bindings):
            assert isinstance(view.node, ParamNode)