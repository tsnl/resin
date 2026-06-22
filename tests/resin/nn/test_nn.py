from resin.core.etype import F4
from resin.dsl import param
from resin.nn import linear, linear_new


class TestLinear:
    def test_new_with_bias(self) -> None:
        layer = linear_new(3, 2)
        assert layer["weight"].shape == (2, 3)
        assert "bias" in layer
        assert layer["bias"].shape == (2,)

    def test_new_without_bias(self) -> None:
        layer = linear_new(3, 2, bias=False)
        assert layer["weight"].shape == (2, 3)
        assert "bias" not in layer

    def test_forward_traces_graph(self) -> None:
        layer = linear_new(3, 2)
        x = param(shape=(4, 3), etype=F4)
        output = linear(layer, x)
        assert output.shape == (4, 2)