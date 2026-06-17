"""End-to-end GPU tests for composed graphs."""

import math

import pytest

import resin.nn as nn
from resin import dsl, grad
from resin.nn import Linear

from tests.resin.gpu.interp_helpers import run_graph, run_scalar


def _scalar(view: dsl.View) -> dsl.View:
    out = view.sum()
    while out.rank > 0:
        out = out.squeeze(axes=(0,))
    return out


class TestLinear:
    def test_forward(self) -> None:
        x = dsl.param(shape=(2, 3), stype="f4")
        layer = Linear.new(3, 2, bias=True)
        out = layer(x)
        values = run_graph(
            out,
            params={
                x: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                layer.weight: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                layer.bias: [10.0, 20.0],
            },
        )
        assert values == pytest.approx([11.0, 20.0, 10.0, 21.0])


class TestSmallMlp:
    def test_relu_linear_stack(self) -> None:
        x = dsl.param(shape=(2, 2), stype="f4")
        l1 = Linear.new(2, 2, bias=False)
        l2 = Linear.new(2, 1, bias=True)
        out = l2(nn.relu(l1(x)))
        values = run_graph(
            out,
            params={
                x: [[1.0, -1.0], [2.0, 3.0]],
                l1.weight: [[1.0, 0.0], [0.0, 1.0]],
                l2.weight: [[1.0, -1.0]],
                l2.bias: [0.5],
            },
        )
        assert values == pytest.approx([1.5, -0.5])


class TestLossOps:
    def test_cross_entropy_single_sample(self) -> None:
        probs = dsl.param(shape=(1, 3), stype="f4")
        label = dsl.param(shape=(1, 3), stype="f4")
        loss = nn.cross_entropy(probs, label)
        value = run_scalar(
            loss,
            params={
                probs: [[0.7, 0.2, 0.1]],
                label: [[1.0, 0.0, 0.0]],
            },
        )
        expected = -math.log(0.7)
        assert value == pytest.approx(expected)

    def test_mean_cross_entropy_batch(self) -> None:
        probs = dsl.param(shape=(2, 2), stype="f4")
        label = dsl.param(shape=(2, 2), stype="f4")
        loss = nn.mean(nn.cross_entropy(probs, label))
        value = run_scalar(
            loss,
            params={
                probs: [[0.6, 0.4], [0.2, 0.8]],
                label: [[1.0, 0.0], [0.0, 1.0]],
            },
        )
        per_sample = [-math.log(0.6), -math.log(0.8)]
        expected = sum(per_sample) / 2.0
        assert value == pytest.approx(expected)


class TestGradExecution:
    def test_sum_param_grad_via_update(self) -> None:
        p = dsl.param(shape=(4,), stype="f4")
        g = grad.grad(_scalar(p))[p.node]
        updated = p + dsl.const(1.0, stype="f4") * g
        values = run_graph(updated, params={p: [1.0, 2.0, 3.0, 4.0]})
        assert values == pytest.approx([2.0, 3.0, 4.0, 5.0])

    def test_matmul_grad_via_update(self) -> None:
        x = dsl.param(shape=(2, 2), stype="f4")
        w = dsl.param(shape=(2, 2), stype="f4")
        g = grad.grad(_scalar(x @ w))[w.node]
        updated = w + dsl.const(1.0, stype="f4") * g
        values = run_graph(
            updated,
            params={
                x: [[1.0, 2.0], [3.0, 4.0]],
                w: [[0.0, 0.0], [0.0, 0.0]],
            },
        )
        assert values == pytest.approx([4.0, 4.0, 6.0, 6.0])

    def test_param_update_adds_scaled_grad(self) -> None:
        p = dsl.param(shape=(3,), stype="f4")
        loss = _scalar(p * dsl.const([1.0, 2.0, 3.0], stype="f4"))
        g = grad.grad(loss)[p.node]
        lr = dsl.const(0.1, stype="f4")
        updated = p + lr * g
        values = run_graph(updated, params={p: [1.0, 1.0, 1.0]})
        assert values == pytest.approx([1.1, 1.2, 1.3])

    def test_param_update_subtracts_scaled_grad(self) -> None:
        p = dsl.param(shape=(3,), stype="f4")
        loss = _scalar(p * dsl.const([1.0, 2.0, 3.0], stype="f4"))
        g = grad.grad(loss)[p.node]
        lr = dsl.const(0.1, stype="f4")
        updated = p - lr * g
        values = run_graph(updated, params={p: [1.0, 1.0, 1.0]})
        assert values == pytest.approx([0.9, 0.8, 0.7])