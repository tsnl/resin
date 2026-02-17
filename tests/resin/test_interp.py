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
        result = eval_node(rg.Node.const(42.0))
        npt.assert_array_equal(result, 42.0)

    def test_const_2d(self) -> None:
        result = eval_node(rg.Node.const([[1.0, 2.0], [3.0, 4.0]]))
        npt.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    # ParamNode:

    def test_param(self) -> None:
        p = rg.Node.param((2, 3), dtype="fp32")
        interp = NumpyInterp({"out": p})
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        interp.buffer(p).view(p)[:] = data
        result = interp.run()
        npt.assert_array_equal(result["out"].view(p), data)

    # ElementwiseNode (unary does not introduce ViewNodes):

    def test_elementwise_neg(self) -> None:
        result = eval_node(-rg.Node.const([1.0, -2.0, 3.0]))
        npt.assert_array_equal(result, [-1.0, 2.0, -3.0])

    def test_elementwise_exp(self) -> None:
        result = eval_node(rg.Node.const([0.0, 1.0]).exp())
        npt.assert_allclose(result, [1.0, np.e])

    # ViewNode:

    def test_view_index(self) -> None:
        a = rg.Node.const([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = eval_node(a[1])
        npt.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_view_transpose(self) -> None:
        a = rg.Node.const([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.transpose())
        npt.assert_array_equal(result, [[1.0, 3.0], [2.0, 4.0]])

    # ReductionNode:

    def test_reduction_sum(self) -> None:
        a = rg.Node.const([[1.0, 2.0], [3.0, 4.0]])
        result = eval_node(a.reduce(axes=(1,), operator="add"))
        npt.assert_array_equal(result, [[3.0], [7.0]])

    def test_reduction_max(self) -> None:
        a = rg.Node.const([[3.0, 1.0], [2.0, 4.0]])
        result = eval_node(a.reduce(axes=(1,), operator="max"))
        npt.assert_array_equal(result, [[3.0], [4.0]])

    # ScatterNode:

    def test_scatter_copy(self) -> None:
        result = eval_node(rg.Node.const([1.0, 2.0, 3.0]).copy())
        npt.assert_array_equal(result, [1.0, 2.0, 3.0])

    # MatmulNode (requires ViewNode for shape joining):

    def test_matmul(self) -> None:
        a = rg.Node.const([[1.0, 2.0], [3.0, 4.0]])
        b = rg.Node.const([[5.0, 6.0], [7.0, 8.0]])
        result = eval_node(a @ b)
        npt.assert_array_equal(result, [[19.0, 22.0], [43.0, 50.0]])

    # ElementwiseNode binary (requires ViewNode for shape joining):

    def test_elementwise_add(self) -> None:
        a = rg.Node.const([1.0, 2.0, 3.0])
        b = rg.Node.const([4.0, 5.0, 6.0])
        result = eval_node(a + b)
        npt.assert_array_equal(result, [5.0, 7.0, 9.0])
