import wgpu

from . import front as rf


class Processor:
    def __init__(
        self,
        device: wgpu.GPUDevice,
        output_tree: rf.PyTree[rf.View],
    ) -> None:
        super().__init__()

        self._device = device
        self._output_tree = Processor._optimize_graph(output_tree)
        self._param_nodes = Processor._compute_param_nodes(self._output_tree)

    @staticmethod
    def _compute_param_nodes(pytree: rf.PyTree[rf.View]) -> list[rf.ParamNode]:
        return [
            node  #
            for node in rf.toposort(rf.flatten_pytree(pytree))
            if isinstance(node, rf.ParamNode)
        ]

    def run(
        self,
        params: dict[rf.ParamNode, wgpu.GPUBuffer],
    ) -> rf.PyTree[wgpu.GPUBuffer]:
        self._bind_params_buffers(params)
        self._flood_buffers()
        return self._gather_output()

    def _bind_params_buffers(self, params: dict[rf.ParamNode, wgpu.GPUBuffer]) -> None:
        for param_node in self._param_nodes:
            if param_node not in params:
                raise ValueError(f"Missing buffer for parameter node {param_node}")

        raise NotImplementedError("Binding parameter buffers is not implemented yet")

    def _flood_buffers(self) -> None:
        raise NotImplementedError(
            "Flooding buffers through the graph is not implemented yet"
        )

    def _gather_output(self) -> rf.PyTree[wgpu.GPUBuffer]:
        raise NotImplementedError("Gathering output buffers is not implemented yet")

    #
    # Compilation pipeline:
    #

    @staticmethod
    def _optimize_graph(graph: rf.PyTree[rf.View]) -> rf.PyTree[rf.View]:
        # TODO: implement graph optimizations
        return graph
