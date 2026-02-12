"""Tests for the resin interpreter."""

import numpy as np
import pytest

import resin.comp as rc


def evaluate(tensor: rc.Tensor) -> np.ndarray:
    """Helper: evaluate a tensor with a fresh interpreter."""
    return rc.Interpreter().evaluate(tensor)


class TestConstant:
    def test_scalar(self) -> None:
        t = rc.Tensor.const(value=3.0, dtype="fp32")
        result = evaluate(t)
        assert result.shape == ()
        assert result == pytest.approx(3.0)

    def test_1d(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = evaluate(t)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert result.dtype == np.float32

    def test_2d(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = evaluate(t)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_fp16(self) -> None:
        t = rc.Tensor.const(value=[1, 2], dtype="fp16")
        result = evaluate(t)
        assert result.dtype == np.float16
        np.testing.assert_array_equal(result, [1, 2])


class TestElementwiseBinaryOps:
    def test_add(self) -> None:
        a = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        b = rc.Tensor.const(value=[4, 5, 6], dtype="fp32")
        result = evaluate(a + b)
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_sub(self) -> None:
        a = rc.Tensor.const(value=[10, 20, 30], dtype="fp32")
        b = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = evaluate(a - b)
        np.testing.assert_array_equal(result, [9, 18, 27])

    def test_mul(self) -> None:
        a = rc.Tensor.const(value=[2, 3, 4], dtype="fp32")
        b = rc.Tensor.const(value=[5, 6, 7], dtype="fp32")
        result = evaluate(a * b)
        np.testing.assert_array_equal(result, [10, 18, 28])

    def test_div(self) -> None:
        a = rc.Tensor.const(value=[10, 20, 30], dtype="fp32")
        b = rc.Tensor.const(value=[2, 4, 5], dtype="fp32")
        result = evaluate(a / b)
        np.testing.assert_array_almost_equal(result, [5, 5, 6])

    def test_pow(self) -> None:
        a = rc.Tensor.const(value=[2, 3, 4], dtype="fp32")
        b = rc.Tensor.const(value=[3, 2, 1], dtype="fp32")
        result = evaluate(a**b)
        np.testing.assert_array_almost_equal(result, [8, 9, 4])

    def test_scalar_add(self) -> None:
        a = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = evaluate(a + 10)
        np.testing.assert_array_equal(result, [[11, 12], [13, 14]])

    def test_2d_ops(self) -> None:
        a = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        b = rc.Tensor.const(value=[[10, 20], [30, 40]], dtype="fp32")
        result = evaluate(a * b)
        np.testing.assert_array_equal(result, [[10, 40], [90, 160]])


class TestElementwiseUnaryOps:
    def test_neg(self) -> None:
        a = rc.Tensor.const(value=[1, -2, 3], dtype="fp32")
        result = evaluate(-a)
        np.testing.assert_array_equal(result, [-1, 2, -3])

    def test_exp(self) -> None:
        a = rc.Tensor.const(value=[0, 1, 2], dtype="fp32")
        result = evaluate(a.exp())
        np.testing.assert_array_almost_equal(result, np.exp([0, 1, 2]))

    def test_log(self) -> None:
        a = rc.Tensor.const(value=[1, 2, 4], dtype="fp32")
        result = evaluate(a.log())
        np.testing.assert_array_almost_equal(result, np.log([1, 2, 4]))

    def test_exp_log_roundtrip(self) -> None:
        a = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = evaluate(a.exp().log())
        np.testing.assert_array_almost_equal(result, [1, 2, 3])


class TestChainedOps:
    def test_add_then_mul(self) -> None:
        a = rc.Tensor.const(value=[1, 2], dtype="fp32")
        b = rc.Tensor.const(value=[3, 4], dtype="fp32")
        c = rc.Tensor.const(value=[5, 6], dtype="fp32")
        result = evaluate((a + b) * c)
        np.testing.assert_array_equal(result, [20, 36])

    def test_deep_chain(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
        result = evaluate(t.exp().log().exp().log())
        np.testing.assert_array_almost_equal(result, [1, 2, 3, 4])


class TestMatrixMultiplication:
    def test_2d_matmul(self) -> None:
        a = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        b = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(a @ b)
        expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32) @ np.array(
            [[1, 2, 3], [4, 5, 6]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_3d_batched_matmul(self) -> None:
        a = rc.Tensor.const(value=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="fp32")
        b = rc.Tensor.const(value=[[[1, 0], [0, 1]], [[2, 0], [0, 2]]], dtype="fp32")
        result = evaluate(a @ b)
        expected = np.array(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32
        ) @ np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_identity_matmul(self) -> None:
        a = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        identity = rc.Tensor.const(value=[[1, 0], [0, 1]], dtype="fp32")
        result = evaluate(a @ identity)
        np.testing.assert_array_almost_equal(result, [[1, 2], [3, 4]])


class TestBroadcast:
    def test_broadcast_1d(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = evaluate(t.broadcast(n=4))
        expected = np.broadcast_to([1, 2, 3], (4, 3))
        np.testing.assert_array_equal(result, expected)

    def test_broadcast_scalar_to_matrix(self) -> None:
        t = rc.Tensor.const(value=42, dtype="fp32")
        b = t.broadcast(n=3).broadcast(n=2)
        result = evaluate(b)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, np.full((2, 3), 42))

    def test_ones(self) -> None:
        t = rc.Tensor.ones(shape=(3, 4), dtype="fp32")
        result = evaluate(t)
        np.testing.assert_array_equal(result, np.ones((3, 4)))

    def test_zeros(self) -> None:
        t = rc.Tensor.zeros(shape=(2, 3), dtype="fp32")
        result = evaluate(t)
        np.testing.assert_array_equal(result, np.zeros((2, 3)))


class TestExpand:
    def test_expand_1d_to_2d(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        t = t.broadcast(n=1).expand(n=2)
        result = evaluate(t)
        expected = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestSqueeze:
    def test_squeeze_first_dim(self) -> None:
        t = rc.Tensor.const(value=[[[1, 2, 3]]], dtype="fp32")
        result = evaluate(t.squeeze(dim=0))
        np.testing.assert_array_equal(result, [[1, 2, 3]])
        assert result.shape == (1, 3)

    def test_squeeze_middle_dim(self) -> None:
        t = rc.Tensor.const(value=[[[1, 2, 3]]], dtype="fp32")
        result = evaluate(t.squeeze(dim=1))
        np.testing.assert_array_equal(result, [[1, 2, 3]])
        assert result.shape == (1, 3)


class TestReduction:
    def test_sum_axis0(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t.reduce(axis=0, op="add"))
        np.testing.assert_array_equal(result, [5, 7, 9])

    def test_sum_axis1(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t.reduce(axis=1, op="add"))
        np.testing.assert_array_equal(result, [6, 15])

    def test_prod_axis0(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = evaluate(t.reduce(axis=0, op="mul"))
        np.testing.assert_array_equal(result, [3, 8])

    def test_prod_axis1(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = evaluate(t.reduce(axis=1, op="mul"))
        np.testing.assert_array_equal(result, [2, 12])


class TestIndex:
    def test_integer_index(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t[0])
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_negative_index(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t[-1])
        np.testing.assert_array_equal(result, [4, 5, 6])

    def test_slice(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        result = evaluate(t[1:4])
        np.testing.assert_array_equal(result, [2, 3, 4])

    def test_slice_with_step(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        result = evaluate(t[::2])
        np.testing.assert_array_equal(result, [1, 3, 5])

    def test_multi_dim_index(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t[1, 1:3])
        np.testing.assert_array_equal(result, [5, 6])


class TestCompact:
    def test_compact_contiguous(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        # Already contiguous, compact returns self
        result = evaluate(t.compact())
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])

    def test_compact_after_slice(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        sliced = t[::2, :]
        result = evaluate(sliced.compact())
        np.testing.assert_array_equal(result, [[1, 2, 3]])


class TestReshape:
    def test_reshape_2d_to_1d(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = evaluate(t.reshape((4,)))
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_reshape_2d_to_2d(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t.reshape((3, 2)))
        np.testing.assert_array_equal(result, [[1, 2], [3, 4], [5, 6]])

    def test_reshape_1d_to_2d(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        result = evaluate(t.reshape((2, 3)))
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


class TestPermute:
    def test_transpose_2d(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t.transpose())
        np.testing.assert_array_equal(result, [[1, 4], [2, 5], [3, 6]])

    def test_permute_3d(self) -> None:
        t = rc.Tensor.const(value=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="fp32")
        result = evaluate(t.permute((2, 0, 1)))
        expected = np.transpose(
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
            (2, 0, 1),
        )
        np.testing.assert_array_equal(result, expected)

    def test_transpose_matmul_identity(self) -> None:
        """(A^T)^T == A."""
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t.transpose().transpose())
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


class TestMemoization:
    def test_shared_subexpression(self) -> None:
        a = rc.Tensor.const(value=[1, 2], dtype="fp32")
        b = rc.Tensor.const(value=[3, 4], dtype="fp32")
        shared = a + b
        result = evaluate(shared * shared)
        np.testing.assert_array_equal(result, [16, 36])

    def test_memo_returns_same_array(self) -> None:
        interp = rc.Interpreter()
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        r1 = interp.evaluate(t)
        r2 = interp.evaluate(t)
        assert r1 is r2


class TestParameterTensor:
    def test_raises_on_unsubstituted(self) -> None:
        p = rc.VarTensor.new(name="x", dtype="fp32", shape=(3,))
        with pytest.raises(ValueError, match="Cannot evaluate unsubstituted parameter"):
            evaluate(p)


class TestComplexGraphs:
    def test_matmul_plus_scalar(self) -> None:
        """(t1 @ t2 + 42) -- the basic example graph."""
        t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate(t1 @ t2 + 42)
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, a @ b + 42)

    def test_matmul_plus_scalar_indexed(self) -> None:
        """(t1 @ t2 + 42)[::2, ::-1]"""
        t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = evaluate((t1 @ t2 + 42)[::2, ::-1])
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        expected = (a @ b + 42)[::2, ::-1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_softmax_like(self) -> None:
        """exp(x) / sum(exp(x)) pattern."""
        x = rc.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
        exp_x = x.exp()
        sum_exp = exp_x.reduce(axis=0, op="add")
        # sum_exp is scalar-shaped (,), broadcast to match exp_x (4,)
        sum_bcast = sum_exp.broadcast(n=4)
        softmax = exp_x / sum_bcast
        result = evaluate(softmax)

        x_np = np.array([1, 2, 3, 4], dtype=np.float32)
        expected = np.exp(x_np) / np.sum(np.exp(x_np))
        np.testing.assert_array_almost_equal(result, expected)

    def test_reshape_then_matmul(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        a = t.reshape((2, 3))
        b = t.reshape((3, 2))
        result = evaluate(a @ b)
        a_np = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(2, 3)
        b_np = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(3, 2)
        np.testing.assert_array_almost_equal(result, a_np @ b_np)
