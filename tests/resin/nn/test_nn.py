from resin.core.etype import F4
from resin.dsl import param
from resin.nn import Linear


class TestLinear:
    def test_new_with_bias(self) -> None:
        layer = Linear.new(3, 2)
        assert layer.weight.shape == (2, 3)
        assert layer.bias is not None
        assert layer.bias.shape == (2,)

    def test_new_without_bias(self) -> None:
        layer = Linear.new(3, 2, bias=False)
        assert layer.weight.shape == (2, 3)
        assert layer.bias is None

    def test_forward_traces_graph(self) -> None:
        layer = Linear.new(3, 2)
        x = param(shape=(4, 3), etype=F4)
        output = layer(x)
        assert output.shape == (4, 2)
