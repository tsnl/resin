import wgpu

from . import graph as rg


class Processor:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        output_tree: rg.PyTree[rg.Node],
    ) -> None:
        super().__init__()

        self._device = device
        self._output_tree = Processor._optimize_graph(output_tree)
        self._param_nodes = Processor._compute_param_nodes(self._output_tree)

    @staticmethod
    def _compute_param_nodes(pytree: rg.PyTree[rg.Node]) -> list[rg.ParamNode]:
        return [
            node  #
            for node in rg.pytree_leaves(pytree)
            if isinstance(node, rg.ParamNode)
        ]

    def run(
        self,
        params: dict[rg.ParamNode, wgpu.GPUBuffer],
    ) -> rg.PyTree[wgpu.GPUBuffer]:
        self._bind_params_buffers(params)
        self._flood_buffers()
        return self._gather_output()

    def _bind_params_buffers(self, params: dict[rg.ParamNode, wgpu.GPUBuffer]) -> None:
        for param_node in self._param_nodes:
            if param_node not in params:
                raise ValueError(f"Missing buffer for parameter node {param_node}")

        raise NotImplementedError("Binding parameter buffers is not implemented yet")

    def _flood_buffers(self) -> None:
        raise NotImplementedError(
            "Flooding buffers through the graph is not implemented yet"
        )

    def _gather_output(self) -> rg.PyTree[wgpu.GPUBuffer]:
        raise NotImplementedError("Gathering output buffers is not implemented yet")

    #
    # Compilation pipeline:
    #

    @staticmethod
    def _optimize_graph(graph: rg.PyTree[rg.Node]) -> rg.PyTree[rg.Node]:
        # TODO: implement graph optimizations
        return graph


class ExecutionPlan:
    def __init__(self, device: wgpu.GPUDevice, output_tree: rg.PyTree[rg.Node]) -> None:
        super().__init__()


class ExecutionNode:
    pass


class FusedElementwiseOperation:
    def __init__(self, device: wgpu.GPUDevice, output_tree: rg.PyTree[rg.Node]) -> None:
        super().__init__()
