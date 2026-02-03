"""
resin.interp implements an optimizing compiler and WebGPU execution engine for
resin.tensor computational graphs.
"""

from dataclasses import dataclass
from typing import Literal
from .tensor import (
    BroadcastTensor,
    ConstantTensor,
    ElementwiseOperationTensor,
    ReshapeTensor,
    SqueezeTensor,
    Tensor,
    DType,
)


class Interp:
    def __init__(self):
        super().__init__()


@dataclass
class Storage:
    type Mode = Literal["ro", "wo", "rw"]

    shape: tuple[int, ...]
    dtype: DType
    mode: Mode

    @property
    def is_readable(self) -> bool:
        return self.mode in ("ro", "rw")

    @property
    def is_writable(self) -> bool:
        return self.mode in ("wo", "rw")


def compute_storage(tensor: Tensor) -> dict[Tensor, Storage]:
    """
    Computes the storage requirements for each tensor in the graph.
    """

    storage_dict: dict[Tensor, Storage] = {}

    ref_count_map = tensor.compute_subgraph_reference_counts()

    for node in reversed(tensor.topological_sort()):
        # A node can reuse one of its operands' storage if:
        #   0.  The operator allows operand memory reuse
        #   1.  The operand is used only once in the entire graph (ref count == 1)
        #   2.  Shapes and dtypes match
        if node_type_supports_operand_memory_reuse(node):
            operand_storage_reused = False
            for operand in node.operands:
                operand_storage = storage_dict[operand]
                if (
                    ref_count_map[operand] == 1
                    and operand.shape == node.shape
                    and operand.pitch == node.pitch
                    and operand.dtype == node.dtype
                    and operand_storage.mode == node_storage_mode(node)
                ):
                    storage_dict[node] = operand_storage
                    operand_storage_reused = True
                    break
            if operand_storage_reused:
                continue

        # Other types of tensors always get their own storage.
        storage_dict[node] = Storage(
            shape=node.shape,
            dtype=node.dtype,
            mode=node_storage_mode(node),
        )

    return storage_dict


def node_type_supports_operand_memory_reuse(node: Tensor) -> bool:
    """
    Returns whether the given node type supports reusing one of its operands' memory.
    This means there are no read-after-write hazards when reusing operand memory.
    """

    match node:
        # Elementwise operations can always reuse one of their operands' memory because
        # each thread reads from and writes to a single element.
        case ElementwiseOperationTensor():
            return True
        # Other types of tensors that only change shape can also reuse operand memory
        # trivially.
        case BroadcastTensor() | SqueezeTensor() | ReshapeTensor():
            return True
        # By default, other node types do not support operand memory reuse.
        case _:
            return False


def node_storage_mode(node: Tensor) -> Storage.Mode:
    """
    Returns the storage mode for the given node.
    """

    match node:
        case ConstantTensor():
            return "ro"
        case _:
            return "rw"
