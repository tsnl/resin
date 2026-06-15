from dataclasses import dataclass
from typing import Literal

from . import graph as rg
from .scalar import ScalarOperator, ScalarType


def build_trivial_execution_plan(outs: rg.PyTree[rg.View]) -> list["Kernel"]:
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
                    pitch=rg.c_contiguous_pitch_for_shape(node.shape),
                    stype=node.stype,
                    input=(),
                    init=node,
                )
            case rg.ElementwiseNode():
                # TODO: plumb each operand View's read accessor (offset/shape/pitch) into
                # the kernel so it can read sparsely; for now operands are mapped by their
                # backing node only.
                input = tuple(map_node(iv.node) for iv in node.input)
                return FusedElementwiseKernel(
                    shape=node.shape,
                    pitch=rg.c_contiguous_pitch_for_shape(node.shape),
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
    fuse_min_index = None
    fuse_max_index = None

    output_kernels = []

    def finalize_fusion() -> Kernel | None:
        nonlocal fuse_min_index, fuse_max_index

        # If no fuse range found, early out.
        if fuse_min_index is None:
            return None
        assert fuse_min_index is not None
        assert fuse_max_index is not None

        # Fuse kernels in the min, max range
        fused_kernel = fuse_active_kernel_range()

        # Reset the accumulator
        fuse_min_index = fuse_max_index = None

        # Return the fused kernel
        return fused_kernel

    def fuse_active_kernel_range() -> Kernel:
        """
        Fuses the kernels in `kernels[fuse_min_index:fuse_max_index]` into a single kernel.
        Returns the single fused kernel.
        """

        assert fuse_min_index is not None and fuse_max_index is not None

        fused_ins_list = []
        fused_ops_list = []

        fusible_kernels_list = kernels[fuse_min_index:fuse_max_index]
        kernel_to_fused_op_index = {}

        for kernel in fusible_kernels_list:
            assert isinstance(kernel, FusedElementwiseKernel)

            # Rewrite each op in the smaller kernel and append to the fused ops list.
            kernel_ops_offset_in_fused_ops_list = len(fused_ops_list)
            for op in kernel.ops:
                new_operands = []
                for ref in op.operands:
                    match ref:
                        case FusedElementwiseKernel.OpRef():
                            new_index = kernel_ops_offset_in_fused_ops_list + ref.index
                            new_ref = FusedElementwiseKernel.OpRef(index=new_index)
                        case FusedElementwiseKernel.InputRef():
                            input_kernel = kernel.input[ref.index]
                            if op_index := kernel_to_fused_op_index.get(input_kernel):
                                new_ref = FusedElementwiseKernel.OpRef(index=op_index)
                            else:
                                index = len(fused_ins_list)
                                fused_ins_list.append(input_kernel)
                                new_ref = FusedElementwiseKernel.InputRef(index=index)
                        case _:
                            raise NotImplementedError()
                    new_operands.append(new_ref)

                new_op = FusedElementwiseKernel.Op(
                    operator=op.operator,
                    operands=tuple(new_operands),
                )
                fused_ops_list.append(new_op)

            # The small kernel's return value op is the last op in the op list.
            # Future ops that depend on this small kernel as an input should instead
            # use this small kernel's last op index in the fused op list instead.
            kernel_to_fused_op_index[kernel] = len(fused_ops_list) - 1

        return FusedElementwiseKernel(
            stype=kernels[fuse_min_index].stype,
            shape=kernels[fuse_min_index].shape,
            pitch=kernels[fuse_min_index].pitch,
            input=tuple(fused_ins_list),
            ops=tuple(fused_ops_list),
        )

    for i, k in enumerate(kernels):
        if isinstance(k, FusedElementwiseKernel):
            fuse_min_index = fuse_min_index or i
            fuse_max_index = i
        else:
            # TODO: what about view nodes?

            # If we had a fuse range built up, terminate it and append to output kernels.
            if fk := finalize_fusion():
                output_kernels.append(fk)

            # Append the current kernel.
            output_kernels.append(k)

    # Done:
    return output_kernels


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
    @dataclass
    class Op:
        operator: ScalarOperator
        operands: tuple["FusedElementwiseKernel.Ref", ...]

    @dataclass
    class Ref: ...

    @dataclass
    class InputRef(Ref):
        index: int

    @dataclass
    class OpRef(Ref):
        index: int

    ops: tuple[Op, ...]
