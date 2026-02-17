from dataclasses import dataclass

from . import graph as rg


class Interp:
    output_dict: dict[str, rg.Node]
    buffer_dict: dict[rg.Node, "Buffer"]

    def __init__(self, output_dict: dict[str, rg.Node]):
        super().__init__()

        self.output_dict = output_dict
        self.buffer_dict = self._build_buffer_dict(list(output_dict.values()))

    def _build_buffer_dict(self, nodes: list[rg.Node]) -> dict[rg.Node, Buffer]:
        """
        Creates buffers for all nodes in the graph.
        """

        buffer_dict: dict[rg.Node, Buffer] = {}
        ref_count_map = rg.refcount(nodes)

        for node in reversed(rg.toposort(nodes)):
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
            buffer_dict[node] = Buffer.new(interp=self, node=node)

        return buffer_dict

    def buffer(self, param: rg.Node) -> "Buffer":
        return self.buffer_dict[param]

    def run(self) -> "Buffer":
        raise NotImplementedError()


@dataclass
class Buffer:
    interp: Interp
    node: rg.Node

    @staticmethod
    def new(*, interp: Interp, node: rg.Node) -> "Buffer":
        return Buffer(interp=interp, node=node)

    def write(self, data: bytes) -> None:
        raise NotImplementedError()

    def read(self) -> bytes:
        raise NotImplementedError()
