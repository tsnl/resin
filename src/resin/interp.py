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
    def _create_buffer(self, node: rg.Node) -> B: ...

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
            if node.can_reuse_operand_memory() and node.input:
                reused_buffer = None

                for operand in node.input:
                    if ref_count_map[operand] == 1:
                        reused_buffer = buffer_dict[operand]
                        break

                if reused_buffer:
                    buffer_dict[node] = reused_buffer
                    continue

            # Other types of tensors always get their own storage.
            buffer_dict[node] = self._create_buffer(node)

        return buffer_dict


@dataclass
class Buffer(ABC):
    interp: Interp

    def write(self, data: bytes) -> None:
        raise NotImplementedError()

    def read(self) -> bytes:
        raise NotImplementedError()


#
# NumPy Interpreter:
#


class NumpyInterp(Interp["NumpyBuffer"]):
    def _create_buffer(self, node: rg.Node) -> NumpyBuffer:
        assert rg.is_contiguous(node.shape, node.pitch)
        numel = math.prod(node.shape)
        storage = np.full((numel,), float("NaN"), dtype=to_numpy_dtype(node.dtype))
        return NumpyBuffer(interp=self, storage=storage)

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
            case _:
                raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def _eval_const_node(self, node: rg.ConstNode) -> None:
        n = self.buffer(node).view(node)
        np.copyto(n, node.value)


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
