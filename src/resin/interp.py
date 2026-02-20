from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import numpy.lib.stride_tricks as nps

from . import graph as rg

#
# API:
#


class Interp[B: Buffer](ABC):
    output_dict: Mapping[str, rg.Node]
    buffer_dict: Mapping[rg.Node, B]

    def __init__(self, output_dict: Mapping[str, rg.Node]):
        super().__init__()
        self.output_dict = output_dict
        self.buffer_dict = self._build_buffer_dict(list(output_dict.values()))

    def buffer(self, param: rg.Node) -> B:
        return self.buffer_dict[param]

    def run(self) -> Mapping[str, B]:
        return self._run()

    #
    # Customization points:
    #

    @abstractmethod
    def _create_buffer(self, node: rg.Node, can_be_reused: bool) -> B: ...

    @abstractmethod
    def _run(self) -> Mapping[str, B]: ...

    #
    # Implementation:
    #

    def _build_buffer_dict(self, nodes: list[rg.Node]) -> Mapping[rg.Node, B]:
        """
        Creates buffers for all nodes in the graph.
        """

        buffer_dict: dict[rg.Node, B] = {}
        ref_count_map = rg.refcount(nodes)

        for node in rg.toposort(nodes):
            # A node can reuse one of its operands' storage if:
            #   0.  The operator allows operand memory reuse
            #   1.  The operand is used only once in the entire graph (ref count == 1)
            if isinstance(node, (rg.ElementwiseNode, rg.ViewNode)):
                reused_buffer = None

                for operand in node.input:
                    operand_buffer = buffer_dict[operand]

                    if isinstance(node, rg.ViewNode):
                        reused_buffer = operand_buffer
                        break

                    if ref_count_map[operand] == 1 and operand_buffer.can_be_reused:
                        reused_buffer = operand_buffer
                        break

                if reused_buffer:
                    buffer_dict[node] = reused_buffer
                    continue

            # If reuse analysis fails, create a new buffer for the node.
            buffer_dict[node] = self._create_buffer(
                node,
                can_be_reused=not isinstance(node, (rg.ConstNode, rg.ParamNode)),
            )

        return buffer_dict


@dataclass
class Buffer(ABC):
    interp: Interp
    can_be_reused: bool

    def write(self, data: bytes) -> None:
        raise NotImplementedError()

    def read(self) -> bytes:
        raise NotImplementedError()


#
# NumPy Interpreter:
#


class NumpyInterp(Interp["NumpyBuffer"]):
    def _create_buffer(self, node: rg.Node, can_be_reused: bool) -> NumpyBuffer:
        assert rg.is_contiguous(node.shape, node.pitch)
        numel = math.prod(node.shape)
        storage = np.full((numel,), float("NaN"), dtype=to_numpy_dtype(node.dtype))
        return NumpyBuffer(interp=self, can_be_reused=can_be_reused, storage=storage)

    def _run(self) -> Mapping[str, "NumpyBuffer"]:
        for node in rg.toposort(list(self.output_dict.values())):
            self._eval_node(node)

        return {
            name: self.buffer_dict[node]  #
            for name, node in self.output_dict.items()
        }

    def _eval_node(self, node: rg.Node) -> None:
        match node:
            case rg.ConstNode():
                self._eval_const_node(node)
            case rg.ParamNode():
                self._eval_param_node(node)
            case rg.MatmulNode():
                self._eval_matmul_node(node)
            case rg.ElementwiseNode():
                self._eval_elementwise_node(node)
            case rg.ReductionNode():
                self._eval_reduction_node(node)
            case rg.ViewNode():
                self._eval_view_node(node)
            case rg.ScatterNode():
                self._eval_scatter_node(node)
            case _:
                raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def _eval_const_node(self, node: rg.ConstNode) -> None:
        n = self.buffer(node).view(node)
        np.copyto(n, node.value)

    def _eval_param_node(self, node: rg.ParamNode) -> None:
        # ParamNode is a placeholder for user-provided data, so we don't write anything
        # to it during evaluation.
        pass

    def _eval_elementwise_node(self, node: rg.ElementwiseNode) -> None:
        # FIXME: we explicitly copy input operands' data because NumPy may not handle
        # in-place operations correctly. Need to verify this.

        o = [self.buffer(operand).view(operand).copy() for operand in node.input]
        r = self.buffer(node).view(node)

        match node.operator:
            case "neg":
                np.negative(o[0], out=r)
            case "exp":
                np.exp(o[0], out=r)
            case "log":
                np.log(o[0], out=r)
            case "not":
                np.logical_not(o[0], out=r)

            case "pow":
                np.power(o[0], o[1], out=r)
            case "mul":
                np.multiply(o[0], o[1], out=r)
            case "div":
                np.divide(o[0], o[1], out=r)
            case "add":
                np.add(o[0], o[1], out=r)
            case "sub":
                np.subtract(o[0], o[1], out=r)
            case "max":
                np.maximum(o[0], o[1], out=r)
            case "min":
                np.minimum(o[0], o[1], out=r)
            case "eq":
                np.equal(o[0], o[1], out=r)
            case "ne":
                np.not_equal(o[0], o[1], out=r)
            case "gt":
                np.greater(o[0], o[1], out=r)
            case "lt":
                np.less(o[0], o[1], out=r)
            case "ge":
                np.greater_equal(o[0], o[1], out=r)
            case "le":
                np.less_equal(o[0], o[1], out=r)
            case _:
                raise NotImplementedError(f"Unsupported op: {node.operator}")

    def _eval_reduction_node(self, node: rg.ReductionNode) -> None:
        o = self.buffer(node.input[0]).view(node.input[0])
        r = self.buffer(node).view(node)

        match node.operator:
            case "mul":
                np.copyto(r, np.prod(o, axis=tuple(node.axes), keepdims=True))
            case "add":
                np.copyto(r, np.sum(o, axis=tuple(node.axes), keepdims=True))
            case "max":
                np.copyto(r, np.amax(o, axis=tuple(node.axes), keepdims=True))
            case "min":
                np.copyto(r, np.amin(o, axis=tuple(node.axes), keepdims=True))
            case _:
                raise NotImplementedError(f"Unsupported reduction op: {node.operator}")

    def _eval_matmul_node(self, node: rg.MatmulNode) -> None:
        a = self.buffer(node.input[0]).view(node.input[0])
        b = self.buffer(node.input[1]).view(node.input[1])
        r = self.buffer(node).view(node)
        np.matmul(a, b, out=r)

    def _eval_view_node(self, node: rg.ViewNode) -> None:
        # ViewNode doesn't perform any computation, so we don't need to do anything during
        # evaluation. The buffer's `view()` method will take care of returning the correct
        # view of the data.
        pass

    def _eval_scatter_node(self, node: rg.ScatterNode) -> None:
        o = self.buffer(node.input[0]).view(node.input[0])
        r = self.buffer(node).view(node)
        np.copyto(
            nps.as_strided(
                r,
                shape=node.shape,
                strides=tuple(
                    x * to_numpy_dtype(node.dtype).itemsize for x in node.pitch
                ),
            ),
            o,
        )


@dataclass
class NumpyBuffer(Buffer):
    storage: np.ndarray

    def view(self, node: rg.Node) -> np.ndarray:
        return nps.as_strided(
            self.storage[node.offset :],
            shape=node.shape,
            strides=tuple(x * to_numpy_dtype(node.dtype).itemsize for x in node.pitch),
        )


def to_numpy_dtype(dtype: rg.DType) -> np.dtype:
    if dtype == "fp32":
        return np.dtype(np.float32)
    elif dtype == "fp16":
        return np.dtype(np.float16)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")

