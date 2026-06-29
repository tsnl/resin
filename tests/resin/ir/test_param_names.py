import pytest

from resin.core.etype import F4
from resin.dsl import param
from resin.ir.ir import IrProgramBuilder


def test_finish_raises_on_duplicate_param_names() -> None:
    builder = IrProgramBuilder()
    a = param(shape=(2,), etype=F4, name="x")
    b = param(shape=(3,), etype=F4, name="x")
    builder.build_sink("out", a.sum() + b.sum())
    with pytest.raises(ValueError, match="duplicate param name: 'x'"):
        _ = builder.finish()


def test_register_param_raises_on_duplicate_registered_name() -> None:
    builder = IrProgramBuilder()
    a = param(shape=(2,), etype=F4)
    b = param(shape=(3,), etype=F4)
    builder.register_param("weights", a)
    with pytest.raises(ValueError, match="duplicate param name: 'weights'"):
        builder.register_param("weights", b)
