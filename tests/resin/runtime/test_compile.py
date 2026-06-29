import struct

import resin_rt_pybind

from resin.core.etype import F4
from resin.dsl import param
from resin.grad import grad
from resin.nn import linear, linear_new, mean
from resin.opt import sgd
from resin.runtime import compile_program


def test_param_buffers_use_stable_names() -> None:
    xs = param(shape=(2,), etype=F4, name="xs")
    builder_program = compile_program(
        params={"xs": xs},
        sinks={"out": xs},
    )
    assert builder_program.manifest.names() == ("xs",)
    assert builder_program.artifact.param_buffers["xs"] == 0


def test_tree_params_and_commit() -> None:
    layer = linear_new(2, 3, bias=True)
    model = [layer]
    xs = param(shape=(1, 2), etype=F4, name="xs")
    out = linear(layer, xs)
    loss = mean(out)

    grads = grad(loss, wrt=model)
    new_model = sgd(model, grads, lr=0.1)

    compiled = compile_program(
        params={"model": model, "xs": xs},
        sinks={"new_model": new_model, "loss": loss},
    )

    assert "model.0.weight" in compiled.artifact.param_buffers
    assert "new_model.0.weight" in compiled.artifact.sinks

    interp = resin_rt_pybind.Interp("wgpu")
    program_id = compiled.admit(interp)
    binding = compiled.binding(interp, program_id)
    binding.write(
        {
            "xs": struct.pack("<2f", 1.0, 2.0),
            "model.0.weight": struct.pack("<6f", *([0.1] * 6)),
            "model.0.bias": struct.pack("<3f", 0.0, 0.0, 0.0),
        }
    )
    interp.run(program_id)
    binding.commit(from_prefix="new_model", to_prefix="model", tree=model)
    _ = binding.read_sink("loss")
