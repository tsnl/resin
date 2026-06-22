"""SGD training helpers for the Resin MNIST demo."""

import math
import random
import struct
from collections.abc import Callable
from typing import cast

import resin.dsl as dsl
import resin.grad as grad_mod

from resin.core.etype import ElementType, Scalar, spell_etype_in_pystruct
from resin.core.pytree import (
    PyTensor,
    PyTree,
    flatten_pytree_paths,
    infer_pytensor_shape,
    map_pytree,
    map_pytree_paths,
    marshall_pytensor,
)
from resin.dsl import ParamNode, View
from resin.interp import Interp, ProgramId
from resin.ir import IrProgramBuilder
from resin.wgpu import (
    WgpuProgram,
    WgslKernelConfig,
    build_wgpu_program,
    param_buffer_index,
)


def sgd_step(loss: View, params: object) -> WgpuProgram:
    if loss.shape != ():
        raise ValueError(f"loss must be scalar, got shape {loss.shape!r}")

    param_tree = cast(PyTree[View], params)
    return build_loss_grad_program(loss, params=param_tree)


def run_sgd_step(
    interp: Interp,
    program: WgpuProgram,
    param_views: object,
    *,
    program_id: ProgramId,
) -> tuple[float, PyTree[PyTensor]]:
    interp.run(program_id)
    loss_bytes = interp.read_buffer(
        program_id,
        sink_buffer_index(program, "loss"),
    )
    loss = cast(float, struct.unpack("<f", loss_bytes)[0])
    param_tree = cast(PyTree[View], param_views)

    def read_grad(path: str, view: View) -> PyTensor:
        return read_sink_tensor(
            interp,
            program,
            view,
            grad_sink_name(path),
            program_id=program_id,
        )

    grads = map_pytree_paths(param_tree, read_grad)
    return loss, grads


def sgd_update(
    params: PyTree[PyTensor],
    grads: PyTree[PyTensor],
    learning_rate: float,
) -> PyTree[PyTensor]:
    def update_leaf(param: PyTensor, grad: PyTensor) -> PyTensor:
        return _subtract_scaled(param, grad, learning_rate)

    return map_pytree_pair(params, grads, update_leaf)


def sink_buffer_index(program: WgpuProgram, sink_name: str) -> int:
    view_index = program.sinks[sink_name]
    return program.buffer_views[view_index]["buffer_index"]


def param_view_buffer_index(program: WgpuProgram, view: View) -> int:
    node = view.node
    assert isinstance(node, ParamNode)
    return param_buffer_index(program, node)


def random_params(
    param_views: object,
    *,
    seed: int | None = None,
) -> PyTree[PyTensor]:
    rng = random.Random(seed)
    views = cast(PyTree[View], param_views)

    def random_leaf(view: View) -> PyTensor:
        raw = _random_param_bytes(view.shape, view.etype, rng)
        return _unpack_view_tensor(raw, view)

    return map_pytree(views, random_leaf)


def write_params(
    interp: Interp,
    program: WgpuProgram,
    param_views: object,
    values: PyTree[PyTensor],
    *,
    program_id: ProgramId,
) -> None:
    views = cast(PyTree[View], param_views)
    validate_values_against_views(views, values)

    def write_leaf(view: View, value: PyTensor) -> None:
        node = view.node
        assert isinstance(node, ParamNode)
        data = marshall_pytensor(value, etype=view.etype)
        interp.write_buffer(
            program_id,
            param_buffer_index(program, node),
            data,
        )

    map_pytree_pair_views(views, values, write_leaf)


def grad_sink_name(path: str) -> str:
    return f"grad.{path}"


def build_loss_grad_program(
    loss: dsl.View,
    *,
    params: PyTree[View],
    wgsl_kernel_config: WgslKernelConfig | None = None,
) -> WgpuProgram:
    grads = grad_mod.grad(loss)

    builder = IrProgramBuilder()
    builder.build_sink("loss", loss)

    for path, param_view in flatten_pytree_paths(params):
        node = param_view.node
        if not isinstance(node, dsl.ParamNode):
            raise TypeError(f"trainable param must be a ParamNode, got {type(node)}")
        grad_view = grads.get(node)
        if grad_view is None:
            raise ValueError(f"no gradient for param {path!r}")
        builder.build_sink(grad_sink_name(path), grad_view)

    ir_program = builder.finish()
    return build_wgpu_program(
        ir_program,
        wgsl_kernel_config=wgsl_kernel_config,
    )


def read_sink_tensor(
    interp: Interp,
    program: WgpuProgram,
    view: View,
    sink_name: str,
    *,
    program_id: ProgramId,
) -> PyTensor:
    raw = interp.read_buffer(
        program_id,
        sink_buffer_index(program, sink_name),
    )
    return _unpack_view_tensor(raw, view)


def validate_values_against_views(
    weights: PyTree[View],
    values: PyTree[PyTensor],
) -> None:
    if isinstance(weights, View):
        if not isinstance(values, (float, int, list, tuple)):
            raise TypeError(f"expected PyTensor leaf, got {type(values).__name__}")
        actual_shape = infer_pytensor_shape(cast(PyTensor, values))
        if actual_shape != weights.shape:
            raise ValueError(
                f"expected weight shape {weights.shape!r}, got {actual_shape!r}"
            )
    elif isinstance(weights, dict):
        if not isinstance(values, dict):
            raise TypeError("weight values tree shape mismatch")
        weights_dict: dict[str, PyTree[View]] = weights
        for key, child_view in weights_dict.items():
            if key not in values:
                raise ValueError(f"missing weight value for key {key!r}")
            validate_values_against_views(child_view, values[key])
        extra = set(values) - set(weights_dict)
        if extra:
            raise ValueError(f"unexpected weight value keys {sorted(extra)!r}")
    elif isinstance(weights, tuple):
        if not isinstance(values, tuple):
            raise TypeError("weight values tree shape mismatch")
        weights_tuple: tuple[PyTree[View], ...] = weights
        if len(weights_tuple) != len(values):
            raise TypeError("weight values tree shape mismatch")
        for child_view, child_value in zip(weights_tuple, values, strict=True):
            validate_values_against_views(child_view, child_value)
    else:
        if not isinstance(values, list):
            raise TypeError("weight values tree shape mismatch")
        if len(weights) != len(values):
            raise TypeError("weight values tree shape mismatch")
        for child_view, child_value in zip(weights, values, strict=True):
            validate_values_against_views(child_view, child_value)


def map_pytree_pair[T, U, V](
    left: PyTree[T],
    right: PyTree[U],
    fn: Callable[[T, U], V],
) -> PyTree[V]:
    if isinstance(left, (int, float)):
        return fn(left, cast(U, right))
    if isinstance(left, dict):
        left_dict = cast(dict[str, PyTree[T]], left)
        right_dict = cast(dict[str, PyTree[U]], right)
        return {
            key: map_pytree_pair(left_dict[key], right_dict[key], fn)
            for key in left_dict
        }
    if isinstance(left, tuple):
        left_tuple = cast(tuple[PyTree[T], ...], left)
        right_tuple = cast(tuple[PyTree[U], ...], right)
        return tuple(
            map_pytree_pair(left_item, right_item, fn)
            for left_item, right_item in zip(left_tuple, right_tuple, strict=True)
        )
    if isinstance(left, list):
        left_list = cast(list[PyTree[T]], left)
        right_list = cast(list[PyTree[U]], right)
        return [
            map_pytree_pair(left_item, right_item, fn)
            for left_item, right_item in zip(left_list, right_list, strict=True)
        ]
    raise TypeError(f"unsupported pytree node: {type(left).__name__}")


def map_pytree_pair_views[T](
    views: PyTree[View],
    values: PyTree[T],
    fn: Callable[[View, T], None],
) -> None:
    if isinstance(views, View):
        fn(views, cast(T, values))
    elif isinstance(views, dict):
        if not isinstance(values, dict):
            raise TypeError("params tree shape mismatch")
        views_dict: dict[str, PyTree[View]] = views
        values_dict = cast(dict[str, PyTree[T]], values)
        for key in views_dict:
            if key not in values_dict:
                raise ValueError(f"missing params key {key!r}")
            map_pytree_pair_views(views_dict[key], values_dict[key], fn)
    elif isinstance(views, tuple):
        if not isinstance(values, tuple):
            raise TypeError("params tree shape mismatch")
        views_tuple: tuple[PyTree[View], ...] = views
        values_tuple = cast(tuple[PyTree[T], ...], values)
        if len(views_tuple) != len(values_tuple):
            raise TypeError("params tree shape mismatch")
        for view_child, value_child in zip(views_tuple, values_tuple, strict=True):
            map_pytree_pair_views(view_child, value_child, fn)
    else:
        if not isinstance(values, list):
            raise TypeError("params tree shape mismatch")
        values_list = cast(list[PyTree[T]], values)
        if len(views) != len(values_list):
            raise TypeError("params tree shape mismatch")
        for view_child, value_child in zip(views, values_list, strict=True):
            map_pytree_pair_views(view_child, value_child, fn)


def _subtract_scaled(param: PyTensor, grad: PyTensor, learning_rate: float) -> PyTensor:
    if isinstance(param, (int, float)):
        if not isinstance(grad, (int, float)):
            raise TypeError("grad leaf must be scalar when param leaf is scalar")
        return param - learning_rate * grad
    if not isinstance(param, list):
        raise TypeError(f"unsupported param leaf: {type(param).__name__}")
    if not isinstance(grad, list):
        raise TypeError("grad tree shape mismatch")
    return [
        _subtract_scaled(p, g, learning_rate) for p, g in zip(param, grad, strict=True)
    ]


def _unpack_view_tensor(raw: bytes, view: View) -> PyTensor:
    shape = view.shape
    count = math.prod(shape)
    fmt = spell_etype_in_pystruct(view.etype)
    values = cast(tuple[Scalar, ...], struct.unpack(f"<{count}{fmt}", raw))
    if len(shape) == 0:
        return values[0]
    if len(shape) == 1:
        return cast(PyTensor, list(values))
    if len(shape) == 2:
        row_count = shape[0]
        row_size = count // row_count
        return [
            list(values[row_index * row_size : (row_index + 1) * row_size])
            for row_index in range(row_count)
        ]
    raise ValueError(f"unsupported tensor rank: {len(shape)}")


def _random_param_bytes(
    shape: tuple[int, ...],
    etype: ElementType | str,
    rng: random.Random,
) -> bytes:
    count = math.prod(shape)
    values = [rng.uniform(-0.1, 0.1) for _ in range(count)]
    fmt = spell_etype_in_pystruct(etype)
    return struct.pack(f"<{count}{fmt}", *values)
