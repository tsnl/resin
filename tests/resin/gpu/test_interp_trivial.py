"""End-to-end GPU tests for individual graph operations."""

import math

import pytest

import resin.grad as grad
import resin.nn as nn
from resin import dsl
from resin.core.accessor import Accessor
from resin.core.etype import F4, U4

from resin.core.pytree import PyTensor

from tests.resin.gpu.interp_helpers import run_graph, run_scalar


class TestConst:
    def test_1d(self) -> None:
        out = dsl.const([1.0, 2.0, 3.0], etype=F4)
        assert run_graph(out) == pytest.approx([1.0, 2.0, 3.0])

    def test_2d(self) -> None:
        out = dsl.const([[1.0, 2.0], [3.0, 4.0]], etype=F4)
        assert run_graph(out) == pytest.approx([1.0, 2.0, 3.0, 4.0])


class TestElementwiseBinary:
    def test_add(self) -> None:
        a = dsl.param(shape=(3,), etype=F4)
        b = dsl.param(shape=(3,), etype=F4)
        assert run_graph(
            a + b, params={a: [1.0, 2.0, 3.0], b: [4.0, 5.0, 6.0]}
        ) == pytest.approx([5.0, 7.0, 9.0])

    def test_mul(self) -> None:
        a = dsl.param(shape=(2,), etype=F4)
        b = dsl.param(shape=(2,), etype=F4)
        assert run_graph(a * b, params={a: [2.0, 3.0], b: [4.0, 5.0]}) == pytest.approx(
            [8.0, 15.0]
        )

    def test_sub(self) -> None:
        a = dsl.param(shape=(2,), etype=F4)
        b = dsl.param(shape=(2,), etype=F4)
        assert run_graph(a - b, params={a: [5.0, 3.0], b: [1.0, 4.0]}) == pytest.approx(
            [4.0, -1.0]
        )

    def test_div(self) -> None:
        a = dsl.param(shape=(2,), etype=F4)
        b = dsl.param(shape=(2,), etype=F4)
        assert run_graph(a / b, params={a: [8.0, 9.0], b: [2.0, 3.0]}) == pytest.approx(
            [4.0, 3.0]
        )


class TestFloorCeilBitcast:
    def test_floor(self) -> None:
        x = dsl.param(shape=(4,), etype=F4)
        assert run_graph(x.floor(), params={x: [1.2, 2.8, -1.5, 0.0]}) == pytest.approx(
            [1.0, 2.0, -2.0, 0.0]
        )

    def test_ceil(self) -> None:
        x = dsl.param(shape=(4,), etype=F4)
        assert run_graph(x.ceil(), params={x: [1.2, 2.8, -1.5, 0.0]}) == pytest.approx(
            [2.0, 3.0, -1.0, 0.0]
        )

    def test_bitcast_round_trip(self) -> None:
        x = dsl.param(shape=(2,), etype=F4)
        out = x.bitcast(U4).bitcast(F4)
        assert run_graph(out, params={x: [1.0, -2.5]}) == pytest.approx([1.0, -2.5])

    def test_bitcast_preserves_bits(self) -> None:
        import struct

        x = dsl.param(shape=(1,), etype=F4)
        bits = run_graph(x.bitcast(U4), params={x: [3.14]})
        assert len(bits) == 1
        assert bits[0] == struct.unpack("<I", struct.pack("<f", 3.14))[0]


class TestBitwiseOps:
    def test_and(self) -> None:
        a = dsl.param(shape=(3,), etype=U4)
        b = dsl.param(shape=(3,), etype=U4)
        out = a & b
        raw = run_graph(out, params={a: [0b1100, 0b1010, 0b1111], b: [0b1010, 0b0110, 0b0101]})
        assert raw == [0b1000, 0b0010, 0b0101]

    def test_or(self) -> None:
        a = dsl.param(shape=(2,), etype=U4)
        b = dsl.param(shape=(2,), etype=U4)
        out = a | b
        assert run_graph(out, params={a: [0b1100, 0b1010], b: [0b1010, 0b0100]}) == [
            0b1110,
            0b1110,
        ]

    def test_xor(self) -> None:
        a = dsl.param(shape=(2,), etype=U4)
        b = dsl.param(shape=(2,), etype=U4)
        out = a ^ b
        assert run_graph(out, params={a: [0b1100, 0b1111], b: [0b1010, 0b0101]}) == [
            0b0110,
            0b1010,
        ]

    def test_lshift(self) -> None:
        a = dsl.param(shape=(2,), etype=U4)
        b = dsl.param(shape=(2,), etype=U4)
        out = a << b
        assert run_graph(out, params={a: [1, 3], b: [2, 1]}) == [4, 6]

    def test_rshift(self) -> None:
        a = dsl.param(shape=(2,), etype=U4)
        b = dsl.param(shape=(2,), etype=U4)
        out = a >> b
        assert run_graph(out, params={a: [8, 7], b: [1, 2]}) == [4, 1]


class TestElementwiseUnary:
    def test_neg(self) -> None:
        x = dsl.param(shape=(3,), etype=F4)
        assert run_graph(-x, params={x: [1.0, -2.0, 3.0]}) == pytest.approx(
            [-1.0, 2.0, -3.0]
        )

    def test_exp(self) -> None:
        x = dsl.param(shape=(2,), etype=F4)
        assert run_graph(x.exp(), params={x: [0.0, 1.0]}) == pytest.approx(
            [1.0, math.e]
        )

    def test_log(self) -> None:
        x = dsl.param(shape=(2,), etype=F4)
        assert run_graph(x.log(), params={x: [1.0, math.e]}) == pytest.approx(
            [0.0, 1.0]
        )

    def test_sqrt(self) -> None:
        x = dsl.param(shape=(2,), etype=F4)
        assert run_graph(x.sqrt(), params={x: [4.0, 9.0]}) == pytest.approx([2.0, 3.0])

    def test_relu(self) -> None:
        x = dsl.param(shape=(4,), etype=F4)
        out = nn.relu(x)
        assert run_graph(out, params={x: [-1.0, 0.0, 2.0, -3.0]}) == pytest.approx(
            [0.0, 0.0, 2.0, 0.0]
        )


class TestMatmul:
    def test_2d(self) -> None:
        a = dsl.param(shape=(2, 3), etype=F4)
        b = dsl.param(shape=(3, 2), etype=F4)
        out = a @ b
        assert run_graph(
            out,
            params={
                a: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                b: [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            },
        ) == pytest.approx([58.0, 64.0, 139.0, 154.0])


class TestReduction:
    def test_sum_all(self) -> None:
        x = dsl.param(shape=(2, 2), etype=F4)
        out = x.sum()
        while out.rank > 0:
            out = out.squeeze(axes=(0,))
        assert run_scalar(out, params={x: [[1.0, 2.0], [3.0, 4.0]]}) == pytest.approx(
            10.0
        )

    def test_sum_axis_1(self) -> None:
        x = dsl.param(shape=(2, 3), etype=F4)
        out = x.sum(axes=(1,))
        assert run_graph(
            out, params={x: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
        ) == pytest.approx([6.0, 15.0])

    def test_max_axis_0(self) -> None:
        x = dsl.param(shape=(2, 2), etype=F4)
        out = x.reduce(axes=(0,), operator="max")
        assert run_graph(out, params={x: [[1.0, 4.0], [3.0, 2.0]]}) == pytest.approx(
            [3.0, 4.0]
        )


class TestViewOps:
    def test_broadcast_add(self) -> None:
        a = dsl.param(shape=(2, 3), etype=F4)
        b = dsl.param(shape=(3,), etype=F4)
        out = a + b
        assert run_graph(
            out,
            params={
                a: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                b: [10.0, 20.0, 30.0],
            },
        ) == pytest.approx([11.0, 22.0, 33.0, 14.0, 25.0, 36.0])

    def test_transpose_matmul(self) -> None:
        x = dsl.param(shape=(2, 3), etype=F4)
        w = dsl.param(shape=(4, 3), etype=F4)
        out = x @ w.transpose()
        assert run_graph(
            out,
            params={
                x: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                w: [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
            },
        ) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_copy(self) -> None:
        x = dsl.param(shape=(3,), etype=F4)
        assert run_graph(x.copy(), params={x: [1.0, 2.0, 3.0]}) == pytest.approx(
            [1.0, 2.0, 3.0]
        )


class TestRemapLayout:
    def test_scatter_with_accessor(self) -> None:
        src = dsl.param(shape=(2,), etype=F4)
        out = dsl.View.remap(
            source=src,
            info=dsl.RemapScatterInfo(
                accessor=Accessor(offset=1, shape=(2,), pitch=(1,)),
            ),
            out_shape=(4,),
        )
        assert run_graph(out, params={src: [10.0, 20.0]}) == pytest.approx(
            [0.0, 10.0, 20.0, 0.0]
        )

    def test_scatter_accumulate_with_accessor(self) -> None:
        src = dsl.param(shape=(2,), etype=F4)
        out = dsl.View.remap(
            source=src,
            info=dsl.RemapScatterInfo(
                accessor=Accessor(offset=0, shape=(2,), pitch=(1,)),
                operator="add",
            ),
            out_shape=(3,),
        )
        assert run_graph(
            out,
            params={src: [1.0, 2.0]},
        ) == pytest.approx([1.0, 2.0, 0.0])


class TestRemapIndices:
    def test_scatter_with_indices(self) -> None:
        src = dsl.param(shape=(2,), etype=F4)
        idx = dsl.const([[2], [0]], etype=U4)
        out = dsl.View.remap(
            source=src,
            info=dsl.RemapScatterInfo(),
            indices=idx,
            out_shape=(3,),
        )
        assert run_graph(out, params={src: [5.0, 9.0]}) == pytest.approx(
            [9.0, 0.0, 5.0]
        )

    def test_gather_with_indices(self) -> None:
        src = dsl.param(shape=(4,), etype=F4)
        idx = dsl.const([[2], [0], [3]], etype=U4)
        out = dsl.View.remap(
            source=src,
            info=dsl.RemapGatherInfo(
                accessor=Accessor.dense((4,)),
                source_shape=(4,),
            ),
            indices=idx,
        )
        assert run_graph(
            out, params={src: [1.0, 2.0, 3.0, 4.0]}
        ) == pytest.approx([3.0, 1.0, 4.0])

    def test_scatter_accumulate_with_indices(self) -> None:
        src = dsl.param(shape=(2,), etype=F4)
        idx = dsl.const([[0], [0]], etype=U4)
        out = dsl.View.remap(
            source=src,
            info=dsl.RemapScatterInfo(operator="add"),
            indices=idx,
            out_shape=(2,),
        )
        assert run_graph(out, params={src: [1.0, 2.0]}) == pytest.approx([3.0, 0.0])


class TestScatterAccumulate:
    def test_broadcast_adjoint_accumulates_to_scalar(self) -> None:
        x = dsl.param(shape=(), etype=F4)
        y = x.broadcast((5,))
        g = dsl.param(shape=y.shape, etype=F4)
        out = grad.accessor_adjoint(y, g)
        upstream: PyTensor = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert run_scalar(out, params={g: upstream}) == pytest.approx(15.0)


class TestSoftmax:
    def test_rows_sum_to_one(self) -> None:
        x = dsl.param(shape=(2, 3), etype=F4)
        probs = nn.softmax(x, axes=(1,))
        row_sums = probs.sum(axes=(1,))
        values = run_graph(row_sums, params={x: [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]})
        assert values == pytest.approx([1.0, 1.0])
