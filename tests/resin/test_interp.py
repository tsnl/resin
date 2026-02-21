import numpy as np
import numpy.testing as npt

import resin.graph as rg
from resin.interp import NumpyInterp


def eval_node(node: rg.Node) -> np.ndarray:
    interp = NumpyInterp({"out": node})
    result = interp.run()
    return np.array(result["out"].view(node))


class TestEvalSingleNode:
    # ConstNode:

    def test_const_scalar(self) -> None:
        result = eval_node(rg.ConstNode.new(42.0))
        npt.assert_array_equal(result, 42.0)

    def test_const_2d(self) -> None:
        result = eval_node(rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    # ParamNode:

    def test_param(self) -> None:
        p = rg.ParamNode.new(shape=(2, 3), dtype="fp32")
        interp = NumpyInterp({"out": p})
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        interp.buffer(p).view(p)[:] = data
        result = interp.run()
        npt.assert_array_equal(result["out"].view(p), data)

    # ElementwiseNode

    def test_elementwise_neg(self) -> None:
        result = eval_node(-rg.ConstNode.new([1.0, -2.0, 3.0]))
        npt.assert_array_equal(result, [-1.0, 2.0, -3.0])

    def test_elementwise_exp(self) -> None:
        result = eval_node(rg.ConstNode.new([0.0, 1.0]).exp())
        npt.assert_allclose(result, [1.0, np.e])

    def test_log(self) -> None:
        result = eval_node(rg.ConstNode.new([1.0, np.e, np.e**2]).log())
        npt.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_not(self) -> None:
        result = eval_node(~rg.ConstNode.new([0.0, 1.0, -1.0]))
        npt.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_sub(self) -> None:
        a = rg.ConstNode.new([5.0, 3.0, 1.0])
        b = rg.ConstNode.new([1.0, 2.0, 3.0])
        result = eval_node(a - b)
        npt.assert_array_equal(result, [4.0, 1.0, -2.0])

    def test_mul(self) -> None:
        a = rg.ConstNode.new([2.0, 3.0, 4.0])
        b = rg.ConstNode.new([5.0, 6.0, 7.0])
        result = eval_node(a * b)
        npt.assert_array_equal(result, [10.0, 18.0, 28.0])

    def test_div(self) -> None:
        a = rg.ConstNode.new([6.0, 9.0, 12.0])
        b = rg.ConstNode.new([2.0, 3.0, 4.0])
        result = eval_node(a / b)
        npt.assert_array_equal(result, [3.0, 3.0, 3.0])

    def test_pow(self) -> None:
        a = rg.ConstNode.new([2.0, 3.0, 4.0])
        b = rg.ConstNode.new([3.0, 2.0, 0.5])
        result = eval_node(a**b)
        npt.assert_allclose(result, [8.0, 9.0, 2.0])

    def test_max(self) -> None:
        a = rg.ConstNode.new([1.0, 5.0, 3.0])
        b = rg.ConstNode.new([4.0, 2.0, 3.0])
        result = eval_node(a.max(b))
        npt.assert_array_equal(result, [4.0, 5.0, 3.0])

    def test_min(self) -> None:
        a = rg.ConstNode.new([1.0, 5.0, 3.0])
        b = rg.ConstNode.new([4.0, 2.0, 3.0])
        result = eval_node(a.min(b))
        npt.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_eq(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0])
        b = rg.ConstNode.new([1.0, 0.0, 3.0])
        result = eval_node(a.eq(b))
        npt.assert_array_equal(result, [1.0, 0.0, 1.0])

    def test_gt(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0])
        b = rg.ConstNode.new([2.0, 2.0, 1.0])
        result = eval_node(a.gt(b))
        npt.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_scalar_operand(self) -> None:
        result = eval_node(rg.ConstNode.new([1.0, 2.0, 3.0]) * 2)
        npt.assert_array_equal(result, [2.0, 4.0, 6.0])

    # ViewNode:

    def test_view_index(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = eval_node(a[1])
        npt.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_view_transpose(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.transpose())
        npt.assert_array_equal(result, [[1.0, 3.0], [2.0, 4.0]])

    # ReductionNode:

    def test_reduction_sum(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.reduce(axes=(1,), operator="add"))
        npt.assert_array_equal(result, [[3.0], [7.0]])

    def test_reduction_max(self) -> None:
        a = rg.ConstNode.new([[3.0, 1.0], [2.0, 4.0]])
        result = eval_node(a.reduce(axes=(1,), operator="max"))
        npt.assert_array_equal(result, [[3.0], [4.0]])

    # ScatterNode:

    def test_scatter_copy(self) -> None:
        result = eval_node(rg.ConstNode.new([1.0, 2.0, 3.0]).copy())
        npt.assert_array_equal(result, [1.0, 2.0, 3.0])

    # MatmulNode (requires ViewNode for shape joining):

    def test_matmul(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        b = rg.ConstNode.new([[5.0, 6.0], [7.0, 8.0]])
        result = eval_node(a @ b)
        npt.assert_array_equal(result, [[19.0, 22.0], [43.0, 50.0]])

    # ElementwiseNode binary (requires ViewNode for shape joining):

    def test_elementwise_add(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0])
        b = rg.ConstNode.new([4.0, 5.0, 6.0])
        result = eval_node(a + b)
        npt.assert_array_equal(result, [5.0, 7.0, 9.0])


class TestBroadcasting:
    def test_elementwise_broadcast(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = rg.ConstNode.new([10.0, 20.0])  # (2,)
        result = eval_node(a + b)
        npt.assert_array_equal(result, [[11.0, 22.0], [13.0, 24.0]])

    def test_ones(self) -> None:
        result = eval_node(rg.ConstNode.ones((2, 3)))
        npt.assert_array_equal(result, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_full(self) -> None:
        result = eval_node(rg.ConstNode.full((3,), 7.0))
        npt.assert_array_equal(result, [7.0, 7.0, 7.0])


class TestReductionOps:
    def test_min(self) -> None:
        a = rg.ConstNode.new([[3.0, 1.0], [2.0, 4.0]])
        result = eval_node(a.reduce(axes=(1,), operator="min"))
        npt.assert_array_equal(result, [[1.0], [2.0]])

    def test_mul(self) -> None:
        a = rg.ConstNode.new([[2.0, 3.0], [4.0, 5.0]])
        result = eval_node(a.reduce(axes=(1,), operator="mul"))
        npt.assert_array_equal(result, [[6.0], [20.0]])

    def test_sum_axis0(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.reduce(axes=(0,), operator="add"))
        npt.assert_array_equal(result, [[4.0, 6.0]])

    def test_sum_multi_axis(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.reduce(axes=(0, 1), operator="add"))
        npt.assert_array_equal(result, [[10.0]])


class TestViewOps:
    def test_slice(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0, 4.0, 5.0])
        result = eval_node(a[1:4])
        npt.assert_array_equal(result, [2.0, 3.0, 4.0])

    def test_slice_with_step(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = eval_node(a[::2])
        npt.assert_array_equal(result, [1.0, 3.0, 5.0])

    def test_broadcast(self) -> None:
        a = rg.ConstNode.new(5.0)
        result = eval_node(a.broadcast((3,)))
        npt.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_multi_dim_index(self) -> None:
        a = rg.ConstNode.new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = eval_node(a[0, 1:3])
        npt.assert_array_equal(result, [2.0, 3.0])


class TestComposed:
    """Test graphs that chain multiple node types."""

    def test_chained_arithmetic(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0])
        b = rg.ConstNode.new([4.0, 5.0, 6.0])
        c = rg.ConstNode.new([2.0, 2.0, 2.0])
        result = eval_node((a + b) * c)
        npt.assert_array_equal(result, [10.0, 14.0, 18.0])

    def test_matmul_then_reduce(self) -> None:
        a = rg.ConstNode.new([[1.0, 0.0], [0.0, 1.0]])
        b = rg.ConstNode.new([[3.0, 4.0], [5.0, 6.0]])
        result = eval_node((a @ b).reduce(axes=(1,), operator="add"))
        npt.assert_array_equal(result, [[7.0], [11.0]])

    def test_param_in_expression(self) -> None:
        p = rg.ParamNode.new(shape=(3,), dtype="fp32")
        t = p * 2 + rg.ConstNode.new([10.0, 20.0, 30.0])
        interp = NumpyInterp({"out": t})
        interp.buffer(p).view(p)[:] = [1.0, 2.0, 3.0]
        result = interp.run()
        npt.assert_array_equal(np.array(result["out"].view(t)), [12.0, 24.0, 36.0])

    def test_multiple_outputs(self) -> None:
        a = rg.ConstNode.new([1.0, 2.0, 3.0])
        s = a.reduce(axes=(0,), operator="add")
        m = a.reduce(axes=(0,), operator="max")
        interp = NumpyInterp({"sum": s, "max": m})
        result = interp.run()
        npt.assert_array_equal(np.array(result["sum"].view(s)), [6.0])
        npt.assert_array_equal(np.array(result["max"].view(m)), [3.0])

    def test_shared_operand_through_views(self) -> None:
        """ViewNode must reuse its operand's buffer even when the operand has
        refcount > 1 (e.g. two views of the same tensor)."""
        a = rg.ConstNode.new([[1.0, 2.0], [3.0, 4.0]])
        v0 = a[0]  # ViewNode, shares a's buffer
        v1 = a[1]  # ViewNode, shares a's buffer (a has refcount > 1)
        interp = NumpyInterp({"row0": v0, "row1": v1})
        result = interp.run()
        npt.assert_array_equal(np.array(result["row0"].view(v0)), [1.0, 2.0])
        npt.assert_array_equal(np.array(result["row1"].view(v1)), [3.0, 4.0])

    def test_shared_const_not_corrupted(self) -> None:
        """Elementwise ops sharing const operands must not corrupt each other's
        data through buffer reuse (ConstNode buffers are not reusable)."""
        a = rg.ConstNode.new([1.0, 2.0])
        b = rg.ConstNode.new([3.0, 4.0])
        s = a + b
        p = a * b
        interp = NumpyInterp({"sum": s, "prod": p})
        result = interp.run()
        npt.assert_array_equal(np.array(result["sum"].view(s)), [4.0, 6.0])
        npt.assert_array_equal(np.array(result["prod"].view(p)), [3.0, 8.0])
