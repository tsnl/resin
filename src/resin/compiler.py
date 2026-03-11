from dataclasses import dataclass, field
from typing import Literal

from . import graph as rg
from .scalar import ScalarOperator, ScalarType


def build_trivial_execution_plan(outs: rg.PyTree[rg.Node]) -> list["Kernel"]:
    node_to_kernel_map = {}

    def map_node(node: rg.Node) -> Kernel:
        nonlocal node_to_kernel_map

        if k := node_to_kernel_map.get(node):
            return k

        k = build_kernel_for_node(node)
        node_to_kernel_map[node] = k
        return k

    def build_kernel_for_node(node: rg.Node) -> Kernel:
        match node:
            case rg.ConstNode():
                return ConstKernel(
                    shape=node.shape,
                    pitch=node.shape,
                    stype=node.stype,
                    input=(),
                    init=node,
                )
            case rg.ElementwiseNode():
                input = tuple(map_node(input_node) for input_node in node.input)
                return FusedElementwiseKernel(
                    shape=node.shape,
                    pitch=node.shape,
                    stype=node.stype,
                    input=input,
                    ops=((node.operator, tuple(range(len(input)))),),
                )
            case _:
                raise NotImplementedError()

    nodes = rg.toposort(rg.flatten_pytree(outs))
    kernels = [map_node(node) for node in nodes]

    return kernels


def fuse_elementwise_kernels(kernels: list["Kernel"]) -> list["Kernel"]:
    raise NotImplementedError()


type Mode = Literal["r", "rw"]


@dataclass
class Kernel:
    shape: tuple[int, ...]
    pitch: tuple[int, ...]
    stype: ScalarType
    input: tuple["Kernel", ...]


@dataclass
class ConstKernel(Kernel):
    init: rg.ConstNode


@dataclass
class FusedElementwiseKernel(Kernel):
    type Op = tuple[ScalarOperator, tuple[int, ...]]
    # ^- int >= 0: load input param
    # ^- int <  0: load intermediate result **relative to current slot**

    ops: tuple[Op, ...]
