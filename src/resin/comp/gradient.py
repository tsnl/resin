"""
resin.gradient: computes the gradient of a tensor function with respect to its inputs.
"""

from dataclasses import dataclass

from .tensor import (
    BroadcastTensor,
    ConstantTensor,
    ElementwiseOperationTensor,
    ExpandTensor,
    TensorDict,
    TensorDictType,
    VarTensor,
    Tensor,
    TensorFunction,
    Substitution,
    BatchedMatrixMultiplicationTensor,
    tensordict_flatten,
    tensordict_type,
    tensordict_type_instantiate,
)


def grad(
    f: TensorFunction[TensorFunction[Tensor]],
) -> TensorFunction[TensorFunction[tuple[TensorDict, Tensor]]]:
    """
    Transforms a scalar valued function 'f: TensorDict -> TensorDict -> Tensor' into a
    function 'g: TensorDict -> TensorDict -> (TensorDict, Tensor)' that computes both
    the value of 'f' and the gradients of 'f' with respect to its inputs.

    Useful for backpropagation in neural networks, where we need to compute the
    gradients of a loss function with respect to the model parameters.

    The gradient graph holds references to tensor objects in the forward graph for
    efficiency: this means that the forward pass' expressions are reused in the backward
    pass.
    """

    grad_graph_cache: dict[tuple[TensorDictType, TensorDictType], GradCacheValue] = {}

    def new_grad_graph(
        params_types: TensorDictType,
        inputs_types: TensorDictType,
    ) -> GradCacheValue:
        params_vars_td = tensordict_type_instantiate(
            params_types,
            name=f.__qualname__,
        )
        f_prime = f(params_vars_td)

        inputs_vars_td = tensordict_type_instantiate(
            inputs_types,
            name=f_prime.__qualname__,
        )
        f_graph = f_prime(inputs_vars_td)

        grad_dict = differentiate_graph(f_graph)

        sub = Substitution(
            subs={
                param_var.name: grad_dict[param_var]
                for param_var in tensordict_flatten(params_vars_td)
                if isinstance(param_var, VarTensor) and param_var in grad_dict
            }
        )
        backward_graph_dict = sub.rewrite_tensordict(params_vars_td)

        return GradCacheValue(
            forward_graph=f_graph,
            backward_graph_dict=backward_graph_dict,
            params_vars_td=params_vars_td,
            inputs_vars_td=inputs_vars_td,
        )

    def get_grad_graph(
        params_types: TensorDictType,
        inputs_types: TensorDictType,
    ) -> GradCacheValue:
        if grad_cache_value := grad_graph_cache.get((params_types, inputs_types)):
            return grad_cache_value
        else:
            grad_cache_value = new_grad_graph(params_types, inputs_types)
            grad_graph_cache[params_types, inputs_types] = grad_cache_value
            return grad_cache_value

    def gradient(params: TensorDict) -> TensorFunction[tuple[TensorDict, Tensor]]:
        params_types = tensordict_type(params)

        def inner(inputs: TensorDict) -> tuple[TensorDict, Tensor]:
            inputs_types = tensordict_type(inputs)

            grad_cache_value = get_grad_graph(
                params_types=params_types,
                inputs_types=inputs_types,
            )

            rewriter = Substitution.zip_tensordict(
                {
                    "params": grad_cache_value.params_vars_td,
                    "inputs": grad_cache_value.inputs_vars_td,
                },
                {
                    "params": params,
                    "inputs": inputs,
                },
            )
            concrete_forward_graph = rewriter.rewrite(grad_cache_value.forward_graph)
            concrete_backward_graph_dict = rewriter.rewrite_tensordict(
                grad_cache_value.backward_graph_dict
            )
            return concrete_backward_graph_dict, concrete_forward_graph

        return inner

    return gradient


@dataclass
class GradCacheValue:
    forward_graph: Tensor
    backward_graph_dict: TensorDict
    params_vars_td: TensorDict
    inputs_vars_td: TensorDict


def differentiate_graph(f_graph: Tensor) -> dict[Tensor, Tensor]:
    """
    Given a forward graph 'f_graph' that computes a scalar output, returns a mapping
    from each input tensor in 'f_graph' to its gradient with respect to the output.

    The returned gradient graph holds references to tensor objects in the forward graph
    for efficient reuse of forward pass expressions in the backward pass.
    """

    # grad[t] = ∂f / ∂t
    grad: dict[Tensor, Tensor] = {}

    # Initialize: ∂f / ∂f = 1
    grad[f_graph] = Tensor.ones_like(f_graph)

    def visit_node(node: Tensor) -> None:
        """
        Visits a node and writes to `grad`.

        For each operand 'o' and for node 'n'...
            ∂f/∂o = ∂f/∂n * ∂n/∂o   (chain rule)
        We have:
            ∂f/∂n = grad[node]
            ∂n/∂o can be computed depending on node 'n'.

        Note that when an operand is shared between multiple nodes, we need to
        accumulate the gradients from each node that depends on it:
            ∂f/∂o = (∂f/∂n₁ * ∂n₁/∂o) + (∂f/∂n₂ * ∂n₂/∂o) + ...
        This is handled by the `accumulate_gradient` helper, which adds the new gradient
        contribution to the existing gradient for that operand in `grad`.
        """

        match node:
            case ConstantTensor():
                visit_constant(node)

            case VarTensor():
                visit_parameter(node)

            case ElementwiseOperationTensor():
                visit_elementwise_operation(node)

            case BatchedMatrixMultiplicationTensor():
                visit_batched_matmul_operation(node)

            case BroadcastTensor():
                visit_broadcast_operation(node)

            case ExpandTensor():
                visit_expand_operation(node)

            case _:
                raise NotImplementedError()

    def accumulate_gradient(t: Tensor, increment: Tensor) -> None:
        if base := grad.get(t):
            grad[t] = base + increment
        else:
            grad[t] = increment

    def visit_constant(node: ConstantTensor) -> None:
        # Constants do not contribute to gradients.
        _ = node

    def visit_parameter(node: VarTensor) -> None:
        # Parameters' gradients are computed by backpropagation and are written
        # by nodes that depend on them.
        assert node in grad

    def visit_elementwise_operation(node: ElementwiseOperationTensor) -> None:
        # From chain rule, we have
        #   ∂f/∂o₁ += ∂f/∂n * ∂n/∂o₁
        #   ∂f/∂o₂ += ∂f/∂n * ∂n/∂o₂
        #
        # Thus, we just evaluate ∂n/∂o₁ and ∂n/∂o₂ for the given node, and then
        # accumulate the contributions to the operands' gradients.

        df_dn = grad[node]

        match node.operator:
            case "neg":
                # ∂n/∂o₁ = -1
                accumulate_gradient(node.operands[0], -df_dn)
            case "exp":
                # ∂n/∂o₁ = exp(o₁) = n
                accumulate_gradient(node.operands[0], df_dn * node)
            case "log":
                # ∂n/∂o₁ = 1 / o₁
                accumulate_gradient(node.operands[0], df_dn / node.operands[0])
            case "pow":
                # ∂n/∂o₁ = o₂ * o₁^(o₂ - 1) = o₂ * n / o₁
                # ∂n/∂o₂ = log(o₁) * o₁^o₂ = log(o₁) * n
                accumulate_gradient(
                    node.operands[0],
                    df_dn * node.operands[1] * node / node.operands[0],
                )
                accumulate_gradient(
                    node.operands[1],
                    df_dn * node.operands[0].log() * node,
                )
            case "mul":
                # ∂n/∂o₁ = o₂, ∂n/∂o₂ = o₁
                accumulate_gradient(node.operands[0], df_dn * node.operands[1])
                accumulate_gradient(node.operands[1], df_dn * node.operands[0])
            case "div":
                # ∂n/∂o₁ = 1 / o₂, ∂n/∂o₂ = -o₁ / o₂² = -n / o₂
                accumulate_gradient(node.operands[0], df_dn / node.operands[1])
                accumulate_gradient(node.operands[1], df_dn * -node / node.operands[1])
            case "add":
                # ∂n/∂o₁ = ∂n/∂o₂ = 1
                accumulate_gradient(node.operands[0], df_dn)
                accumulate_gradient(node.operands[1], df_dn)
            case "sub":
                # ∂n/∂o₁ = 1, ∂n/∂o₂ = -1
                accumulate_gradient(node.operands[0], df_dn)
                accumulate_gradient(node.operands[1], -df_dn)
            case "max":
                # ∂n/∂o₁ = 1 if o₁ > o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ > o₁ else 0
                accumulate_gradient(
                    node.operands[0],
                    df_dn * (node.operands[0] > node.operands[1]),
                )
                accumulate_gradient(
                    node.operands[1],
                    df_dn * (node.operands[1] > node.operands[0]),
                )
            case "min":
                # ∂n/∂o₁ = 1 if o₁ < o₂ else 0
                # ∂n/∂o₂ = 1 if o₂ < o₁ else 0
                accumulate_gradient(
                    node.operands[0],
                    df_dn * (node.operands[0] < node.operands[1]),
                )
                accumulate_gradient(
                    node.operands[1],
                    df_dn * (node.operands[1] < node.operands[0]),
                )
            case "eq" | "ne" | "lt" | "gt" | "le" | "ge":
                raise ValueError(
                    f"Cannot compute gradients for comparison operator {node.operator}"
                )
            case _:
                raise NotImplementedError(f"{node.operator=}")

    def visit_batched_matmul_operation(node: BatchedMatrixMultiplicationTensor) -> None:
        df_dn = grad[node]

        # ∂n/∂o₁ = df/dn @ o₂.T
        # ∂n/∂o₂ = o₁.T @ df/dn

        accumulate_gradient(node.operands[0], df_dn @ node.operands[1].transpose())
        accumulate_gradient(node.operands[1], node.operands[0].transpose() @ df_dn)

    def visit_broadcast_operation(node: BroadcastTensor) -> None:
        df_dn = grad[node]

        # Broadcasting replicates the input N times along axis 0.
        # The gradient sums the contribution from each replication:
        # ∂n/∂o = sum(df/dn, axis=0)
        accumulate_gradient(
            node.operands[0],
            df_dn.reduce(axis=0, op="add"),
        )

    def visit_expand_operation(node: ExpandTensor) -> None:
        df_dn = grad[node]

        # Expanding replicates the input N times along axis 0.
        # The gradient sums the contribution from each replication:
        # ∂n/∂o = sum(df/dn, axis=0)
        accumulate_gradient(
            node.operands[0],
            df_dn.reduce(axis=0, op="add").broadcast(n=1),
        )

    for node in f_graph.topological_sort():
        visit_node(node)

    return grad
