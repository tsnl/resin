"""
resin.gradient: computes the gradient of a tensor function with respect to its inputs.
"""

__all__ = [
    "create_parameter_tensor_list",
    "differentiate_graph",
    "grad",
]

from collections import deque
import inspect

from .tensor import (
    BroadcastTensor,
    ConstantTensor,
    DType,
    ElementwiseOperationTensor,
    ExpandTensor,
    ParameterTensor,
    Shape,
    Tensor,
    TensorFunction,
    Substitution,
    BatchedMatrixMultiplicationTensor,
)


def grad(
    f: TensorFunction[Tensor],
) -> TensorFunction[tuple[Tensor, tuple[Tensor, ...]]]:
    """
    Transforms a scalar valued function 'f: (*Tensor) -> Tensor' into a function
    'g: (*Tensor) -> (Tensor, (Tensor, ...))' that computes both the value of 'f' and
    the gradients of 'f' with respect to its inputs.

    Useful for backpropagation in neural networks, where we need to compute the
    gradients of a loss function with respect to the model parameters.

    The gradient graph holds references to tensor objects in the forward graph for
    efficiency: this means that the forward pass' expressions are reused in the backward
    pass.
    """

    grad_graph_cache = {}

    def new_grad_graph(
        a: tuple[tuple[DType, Shape], ...],
    ) -> tuple[
        tuple[Tensor, tuple[Tensor, ...]],
        list[ParameterTensor],
    ]:
        params = create_parameter_tensor_list(f, a)
        f_graph = f(*params)
        g_graph_dict = differentiate_graph(f_graph)
        return (
            (f_graph, tuple(g_graph_dict[param] for param in params)),
            params,
        )

    def get_grad_graph(
        a: tuple[tuple[DType, Shape], ...],
    ) -> tuple[
        tuple[Tensor, tuple[Tensor, ...]],
        list[ParameterTensor],
    ]:
        if grad_graph := grad_graph_cache.get(a):
            return grad_graph
        else:
            grad_graph = new_grad_graph(a)
            grad_graph_cache[a] = grad_graph
            return grad_graph

    def gradient(*inputs: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        a: tuple[tuple[DType, Shape], ...] = tuple((x.dtype, x.shape) for x in inputs)

        (value_graph, gradient_graph_tuple), grad_params = get_grad_graph(a=a)

        rewriter = Substitution.zip(grad_params, inputs)
        concrete_value_graph = rewriter.rewrite(value_graph)
        concrete_gradient_graph_tuple = rewriter.rewrite_tuple(gradient_graph_tuple)
        return concrete_value_graph, concrete_gradient_graph_tuple

    return gradient


def create_parameter_tensor_list[T](
    f: TensorFunction[T],
    a: tuple[tuple[DType, Shape], ...],
) -> list[ParameterTensor]:
    signature = inspect.signature(f)
    f_name = f.__qualname__

    queue = deque(a)
    out = []

    for param_name, param_spec in signature.parameters.items():
        name = f"{f_name}.{param_name}"

        if param_spec.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            # Pop one arg type and shape for this parameter
            dtype, shape = queue.popleft()
            param = ParameterTensor.new(name=name, dtype=dtype, shape=shape)
            out.append(param)

        elif param_spec.kind == inspect.Parameter.VAR_POSITIONAL:
            # Pop all remaining arg types and shapes for *args
            i = 0
            while queue:
                (dtype, shape) = queue.popleft()
                elt_name = f"{name}[{i}]"
                param = ParameterTensor.new(name=elt_name, dtype=dtype, shape=shape)
                out.append(param)
                i += 1
            break

        else:
            # ignore: this argument type is not traced
            pass

    return out


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
                pass

            case ParameterTensor():
                # Parameters' gradients are computed by backpropagation and are written
                # by nodes that depend on them.
                assert node in grad

            case ElementwiseOperationTensor():
                visit_elementwise_operation(node)

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

    def visit_parameter(node: ParameterTensor) -> None:
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
        match node:
            case ConstantTensor():
                visit_constant(node)

            case ParameterTensor():
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

    return grad
