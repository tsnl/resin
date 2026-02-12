"""Expect tests for tensor debug_print output."""

import textwrap
from io import StringIO

import resin.comp as rc


def debug_str(tensor: rc.Tensor) -> str:
    """Capture debug_print output as a string."""
    out = StringIO()
    tensor.debug_print(out=out)
    return out.getvalue().strip()


class TestConstantTensor:
    def test_scalar_constant(self) -> None:
        t = rc.Tensor.const(value=[42], dtype="fp32")
        assert debug_str(t) == "(constant [42]) :: (fp32 (1))"

    def test_1d_constant(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        assert debug_str(t) == "(constant [1 2 3]) :: (fp32 (3))"

    def test_2d_constant(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        assert debug_str(t) == "(constant [[1 2] [3 4]]) :: (fp32 (2 2))"

    def test_fp16_constant(self) -> None:
        t = rc.Tensor.const(value=[1, 2], dtype="fp16")
        assert debug_str(t) == "(constant [1 2]) :: (fp16 (2))"


class TestElementwiseOperations:
    def test_add_tensors(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
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
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
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
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
        t3 = rc.Tensor.const(value=[5, 6], dtype="fp32")
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
        t = rc.Tensor.const(value=[1, 2], dtype="fp32")
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
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
        result = t + 10
        expected = textwrap.dedent(
            """
            (elementwise "add") :: (fp32 (2 2))
            ├ (constant [[1 2] [3 4]]) :: (fp32 (2 2))
            └ (broadcast 2) :: (fp32 (2 2))
              └ (broadcast 2) :: (fp32 (2))
                └ (constant 10) :: (fp32 ())
            """
        )
        assert debug_str(result) == expected.strip()


class TestMatrixMultiplication:
    def test_matmul_2d(self) -> None:
        t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
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
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t.broadcast(n=4)
        expected = textwrap.dedent(
            """
            (broadcast 4) :: (fp32 (4 3))
            └ (constant [1 2 3]) :: (fp32 (3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_squeeze(self) -> None:
        t = rc.Tensor.const(value=[[[1, 2, 3]]], dtype="fp32")
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
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t.reduce(axis=1, op="add")
        expected = textwrap.dedent(
            """
            (reduction add 1) :: (fp32 (2))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_product_reduction(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
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
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t[0]
        expected = textwrap.dedent(
            """
            (index [0]) :: (fp32 (3))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_negative_index(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t[-1]
        expected = textwrap.dedent(
            """
            (index [-1]) :: (fp32 (3))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_slice_with_step(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3, 4, 5, 6], dtype="fp32")
        result = t[::2]
        expected = textwrap.dedent(
            """
            (index [(slice () () 2)]) :: (fp32 (3))
            └ (constant [1 2 3 4 5 6]) :: (fp32 (6))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_reverse_slice(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t[::-1]
        expected = textwrap.dedent(
            """
            (index [(slice () () -1)]) :: (fp32 (3))
            └ (constant [1 2 3]) :: (fp32 (3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_multi_dim_index(self) -> None:
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
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
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
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
        t = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
        result = t.reshape((3, 2))
        expected = textwrap.dedent(
            """
            (reshape (3 2)) :: (fp32 (3 2))
            └ (constant [[1 2 3] [4 5 6]]) :: (fp32 (2 3))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_reshape_to_1d(self) -> None:
        t = rc.Tensor.const(value=[[1, 2], [3, 4]], dtype="fp32")
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
        t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
        t2 = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
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
                  └ (constant 42) :: (fp32 ())
            """
        )
        assert debug_str(result) == expected.strip()

    def test_deep_chain(self) -> None:
        """Test a deeply nested computation chain."""
        t = rc.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
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
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
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


class TestRewriteSimple:
    def test_substitute_single_parameter(self) -> None:
        """A parameter is replaced by a constant."""
        x = rc.VarTensor.new(name="x", dtype="fp32", shape=(4,))
        y = rc.VarTensor.new(name="y", dtype="fp32", shape=(4,))
        graph = x + y

        replacement = rc.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
        rewritten = rc.Substitution(subs={"x": replacement}).rewrite(graph)

        assert isinstance(rewritten, rc.ElementwiseOperationTensor)
        assert rewritten is not graph
        assert rewritten.operands[0] is replacement
        assert rewritten.operands[1] is y
        assert rewritten.shape == (4,)

    def test_leaf_constant_unchanged(self) -> None:
        """Constants pass through rewrite untouched."""
        c = rc.Tensor.const(value=[1, 2], dtype="fp32")
        assert c is rc.Substitution(subs={"x": c}).rewrite(c)


class TestRewriteComplexGraph:
    def test_softmax_like_pattern(self) -> None:
        """Rewrite through a multi-layer softmax-like graph."""
        x = rc.VarTensor.new(name="x", dtype="fp32", shape=(8,))
        w = rc.VarTensor.new(name="w", dtype="fp32", shape=(8,))

        z = x * w
        exp_z = z.exp()
        sum_exp = exp_z.reduce(axis=0, op="add")
        sum_bcast = sum_exp.broadcast(n=8)
        softmax = exp_z / sum_bcast

        x_val = rc.Tensor.const(value=[1] * 8, dtype="fp32")
        w_val = rc.Tensor.const(value=[0] * 8, dtype="fp32")
        rewritten = rc.Substitution(subs={"x": x_val, "w": w_val}).rewrite(softmax)

        # No ParameterTensor should remain in the rewritten graph
        for tensor in rewritten.topological_sort():
            assert not isinstance(tensor, rc.VarTensor)

        assert rewritten.shape == (8,)

        # Debug-print should show the substituted constants
        output = debug_str(rewritten)
        assert "(variable x)" not in output
        assert "(variable w)" not in output


class TestRewriteSharedSubexpressions:
    def test_shared_subexpr_stays_shared(self) -> None:
        """Shared sub-expressions remain shared (same identity) after rewrite."""
        x = rc.VarTensor.new(name="x", dtype="fp32", shape=(4,))
        shared = x.exp()
        graph = shared + shared

        replacement = rc.Tensor.const(value=[1, 2, 3, 4], dtype="fp32")
        rewritten = rc.Substitution(subs={"x": replacement}).rewrite(graph)

        assert isinstance(rewritten, rc.ElementwiseOperationTensor)
        # Both operands must be the *same* object (identity, not equality)
        assert rewritten.operands[0] is rewritten.operands[1]

        rewritten_exp = rewritten.operands[0]
        assert isinstance(rewritten_exp, rc.ElementwiseOperationTensor)
        assert rewritten_exp.operator == "exp"
        assert rewritten_exp.operands[0] is replacement

    def test_shared_parameter_substituted_once(self) -> None:
        """A parameter used in multiple places maps to the same replacement."""
        x = rc.VarTensor.new(name="x", dtype="fp32", shape=(2,))
        graph = (x + x) * x

        replacement = rc.Tensor.const(value=[5, 6], dtype="fp32")
        rewritten = rc.Substitution(subs={"x": replacement}).rewrite(graph)

        # All three original uses of x should resolve to the same replacement
        mul = rewritten
        assert isinstance(mul, rc.ElementwiseOperationTensor)
        add = mul.operands[0]
        assert isinstance(add, rc.ElementwiseOperationTensor)
        assert add.operands[0] is replacement
        assert add.operands[1] is replacement
        assert mul.operands[1] is replacement


class TestMaxMinOperations:
    def test_max_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
        result = t1.max(t2)
        expected = textwrap.dedent(
            """
            (elementwise "max") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [3 4]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_min_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[3, 4], dtype="fp32")
        result = t1.min(t2)
        expected = textwrap.dedent(
            """
            (elementwise "min") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [3 4]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_max_with_scalar(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t.max(0)
        expected = textwrap.dedent(
            """
            (elementwise "max") :: (fp32 (3))
            ├ (constant [1 2 3]) :: (fp32 (3))
            └ (broadcast 3) :: (fp32 (3))
              └ (constant 0) :: (fp32 ())
            """
        )
        assert debug_str(result) == expected.strip()

    def test_min_with_scalar(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t.min(10)
        expected = textwrap.dedent(
            """
            (elementwise "min") :: (fp32 (3))
            ├ (constant [1 2 3]) :: (fp32 (3))
            └ (broadcast 3) :: (fp32 (3))
              └ (constant 10) :: (fp32 ())
            """
        )
        assert debug_str(result) == expected.strip()


class TestComparisonOperations:
    def test_gt_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[2, 1], dtype="fp32")
        result = t1 > t2
        expected = textwrap.dedent(
            """
            (elementwise "gt") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [2 1]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_lt_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[2, 1], dtype="fp32")
        result = t1 < t2
        expected = textwrap.dedent(
            """
            (elementwise "lt") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [2 1]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_eq_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[1, 3], dtype="fp32")
        result = t1.eq(t2)
        expected = textwrap.dedent(
            """
            (elementwise "eq") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [1 3]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_ge_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[2, 1], dtype="fp32")
        result = t1 >= t2
        expected = textwrap.dedent(
            """
            (elementwise "ge") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [2 1]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_le_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[2, 1], dtype="fp32")
        result = t1 <= t2
        expected = textwrap.dedent(
            """
            (elementwise "le") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [2 1]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_ne_debug_print(self) -> None:
        t1 = rc.Tensor.const(value=[1, 2], dtype="fp32")
        t2 = rc.Tensor.const(value=[1, 3], dtype="fp32")
        result = t1.ne(t2)
        expected = textwrap.dedent(
            """
            (elementwise "ne") :: (fp32 (2))
            ├ (constant [1 2]) :: (fp32 (2))
            └ (constant [1 3]) :: (fp32 (2))
            """
        )
        assert debug_str(result) == expected.strip()

    def test_comparison_with_scalar(self) -> None:
        t = rc.Tensor.const(value=[1, 2, 3], dtype="fp32")
        result = t > 2
        expected = textwrap.dedent(
            """
            (elementwise "gt") :: (fp32 (3))
            ├ (constant [1 2 3]) :: (fp32 (3))
            └ (broadcast 3) :: (fp32 (3))
              └ (constant 2) :: (fp32 ())
            """
        )
        assert debug_str(result) == expected.strip()
