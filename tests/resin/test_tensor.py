"""Expect tests for tensor debug_print output."""

import textwrap
from io import StringIO

import resin.tensor as rt


def debug_str(tensor: rt.Tensor) -> str:
    """Capture debug_print output as a string."""
    out = StringIO()
    tensor.debug_print(out=out)
    return out.getvalue().strip()


class TestConstantTensor:
    def test_scalar_constant(self) -> None:
        t = rt.Tensor.const(value=[42], dtype="fp32")
        assert debug_str(t) == "(constant [42]) :: (fp32 (1))"

    def test_1d_constant(self) -> None:
        t = rt.Tensor.const(value=[1, 2, 3], dtype="fp32")
        assert debug_str(t) == "(constant [1 2 3]) :: (fp32 (3))"

    def test_2d_constant(self) -> None:
        t = rt.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        assert debug_str(t) == "(constant [[1 2] [3 4]]) :: (fp32 (2 2))"

    def test_fp16_constant(self) -> None:
        t = rt.Tensor.const(value=[1, 2], dtype="fp16")
        assert debug_str(t) == "(constant [1 2]) :: (fp16 (2))"


class TestElementwiseOperations:
    def test_add_tensors(self) -> None:
        t1 = rt.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rt.Tensor.const(value=[3, 4], dtype="fp32")
        result = t1 + t2
        expected = textwrap.dedent(
            """
            (elementwise "add") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [3 4]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_mul_tensors(self) -> None:
        t1 = rt.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rt.Tensor.const(value=[3, 4], dtype="fp32")
        result = t1 * t2
        expected = textwrap.dedent(
            """
            (elementwise "mul") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [3 4]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_chained_operations(self) -> None:
        t1 = rt.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rt.Tensor.const(value=[3, 4], dtype="fp32")
        t3 = rt.Tensor.const(value=[5, 6], dtype="fp32")
        result = (t1 + t2) * t3
        expected = textwrap.dedent(
            """
            (elementwise "mul") :: (fp32 (2))
            ├ (elementwise "add") :: (fp32 (2))
            │ ├ (constant [1 2]) :: (fp32 (2))
            │ └ (constant [3 4]) :: (fp32 (2))
            └ (constant [5 6]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_exp_log(self) -> None:
        t = rt.Tensor.const(value=[1, 2], dtype="fp32")
        result = t.exp().log()
        expected = textwrap.dedent(
            """
            (elementwise "log") :: (fp32 (2))
            └ (elementwise "exp") :: (fp32 (2))
              └ (constant [1 2]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_scalar_broadcast(self) -> None:
        t = rt.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = t + 10
        expected = textwrap.dedent(
            """
            (elementwise "add") :: (fp32 (2 2))
            ├ (constant [[1 2] [3 4]]) :: (fp32 (2 2))
            └ (broadcast 2) :: (fp32 (2 2))
              └ (broadcast 2) :: (fp32 (2))
                └ (constant [10]) :: (fp32 (1))
            """
        )
        assert debug_str(result) == expected.strip()


class TestMatrixMultiplication:
    def test_matmul_2d(self) -> None:
        t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t1 @ t2
        expected = textwrap.dedent(
            """
            (squeeze 0) :: (fp32 (3 3))
            └ (batched-matmul) :: (fp32 (1 3 3))
              ├ (broadcast 1) :: (fp32 (1 3 2))
              │ └ (constant [[1 2] [3 4] [5 6]]) :: (fp32 (3 2))
              └ (broadcast 1) :: (fp32 (1 2 3))
                └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()


class TestBroadcastSqueeze:
    def test_broadcast(self) -> None:
        t = rt.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t.broadcast(n=4)
        expected = textwrap.dedent(
            """
            (broadcast 4) :: (fp32 (4 3))
            └ (constant [1 2 3]) :: (fp32 (3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_squeeze(self) -> None:
        t = rt.Tensor.const(value=[[[1, 2, 3]]], dtype="fp32")
        result = t.squeeze(dim=0)
        expected = textwrap.dedent(
            """
            (squeeze 0) :: (fp32 (1 3))
            └ (constant [[[1 2 3]]]) :: (fp32 (1 1 3))
            """
        )
        assert debug_str(result) == expected.strip()


class TestReduction:
    def test_sum_reduction(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t.reduce(axis=1, op="add")
        expected = textwrap.dedent(
            """
            (reduction add 1) :: (fp32 (2))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_product_reduction(self) -> None:
        t = rt.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = t.reduce(axis=0, op="mul")
        expected = textwrap.dedent(
            """
            (reduction mul 0) :: (fp32 (2))
            └ (constant [[1 2] [3 4]]) :: (fp32 (2 2))
            """
        )
        assert debug_str(result) == expected.strip()


class TestIndexing:
    def test_integer_index(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t[0]
        expected = textwrap.dedent(
            """
            (index [0]) :: (fp32 (3))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_negative_index(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t[-1]
        expected = textwrap.dedent(
            """
            (index [-1]) :: (fp32 (3))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = rt.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        result = t[::2]
        expected = textwrap.dedent(
            """
            (index [(slice () () 2)]) :: (fp32 (3))
            └ (constant [1 2 3 4 5 6]) :: (fp32 (6))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_reverse_slice(self) -> None:
        t = rt.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t[::-1]
        expected = textwrap.dedent(
            """
            (index [(slice () () -1)]) :: (fp32 (3))
            └ (constant [1 2 3]) :: (fp32 (3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_multi_dim_index(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t[1, 1:3]
        expected = textwrap.dedent(
            """
            (index [1 (slice 1 3 ())]) :: (fp32 (2))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()


class TestCompactReshape:
    def test_compact(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        sliced = t[::2, :]
        result = sliced.compact()
        expected = textwrap.dedent(
            """
            (compact) :: (fp32 (1 3))
            └ (index [(slice () () 2) (slice () () ())]) :: (fp32 (1 3))
              └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_reshape(self) -> None:
        t = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t.reshape((3, 2))
        expected = textwrap.dedent(
            """
            (reshape (3 2)) :: (fp32 (3 2))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_reshape_to_1d(self) -> None:
        t = rt.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = t.reshape((4,))
        expected = textwrap.dedent(
            """
            (reshape (4)) :: (fp32 (4))
            └ (constant [[1 2] [3 4]]) :: (fp32 (2 2))
            """
        )
        assert debug_str(result) == expected.strip()


class TestComplexGraphs:
    def test_basic_example(self) -> None:
        """Test the example from basic.py"""
        t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = (t1 @ t2 + 42)[::2, ::-1]
        expected = textwrap.dedent(
            """
            (index [(slice () () 2) (slice () () -1)]) :: (fp32 (2 3))
            └ (elementwise "add") :: (fp32 (3 3))
              ├ (squeeze 0) :: (fp32 (3 3))
              │ └ (batched-matmul) :: (fp32 (1 3 3))
              │   ├ (broadcast 1) :: (fp32 (1 3 2))
              │   │ └ (constant [[1 2] [3 4] [5 6]]) :: (fp32 (3 2))
              │   └ (broadcast 1) :: (fp32 (1 2 3))
              │     └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
              └ (broadcast 3) :: (fp32 (3 3))
                └ (broadcast 3) :: (fp32 (3))
                  └ (constant [42]) :: (fp32 (1))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_deep_chain(self) -> None:
        """Test a deeply nested computation chain."""
        t = rt.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
        result = t.exp().log().exp().log()
        expected = textwrap.dedent(
            """
            (elementwise "log") :: (fp32 (4))
            └ (elementwise "exp") :: (fp32 (4))
              └ (elementwise "log") :: (fp32 (4))
                └ (elementwise "exp") :: (fp32 (4))
                  └ (constant [1 2 3 4]) :: (fp32 (4))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_shared_subexpressions(self) -> None:
        """Test a computation graph that contains shared sub-expressions."""
        t1 = rt.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rt.Tensor.const(value=[3, 4], dtype="fp32")
        sum_t = t1 + t2
        result = sum_t * sum_t
        expected = textwrap.dedent(
            """
            (elementwise "mul") :: (fp32 (2))
            ├ %0
            └ %0
            %0 := (elementwise "add") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [3 4]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()
