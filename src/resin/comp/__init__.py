"""
resin.comp models tensor computational graphs and provides tools for building,
transforming, interpreting, and compiling them.
"""

__all__ = [
    "BatchedMatrixMultiplicationKernel",
    "BatchedMatrixMultiplicationTensor",
    "BinaryOperator",
    "BroadcastTensor",
    "CompactTensor",
    "ConstantTensor",
    "DType",
    "ElementwiseOperationKernel",
    "ElementwiseOperationTensor",
    "ExecutionPlan",
    "IndexTensor",
    "Interpreter",
    "Kernel",
    "Operator",
    "ParameterTensor",
    "Pitch",
    "ReductionTensor",
    "ReshapeTensor",
    "Shape",
    "SqueezeTensor",
    "Storage",
    "Substitution",
    "Tensor",
    "TensorFunction",
    "UnaryOperator",
    "compute_storage",
    "create_parameter_tensor_list",
    "differentiate_graph",
    "dtype_bytes",
    "grad",
    "plan_execution",
]

from .gradient import (
    create_parameter_tensor_list,
    differentiate_graph,
    grad,
)
from .interp import (
    Interpreter,
)
from .planner import (
    BatchedMatrixMultiplicationKernel,
    ElementwiseOperationKernel,
    ExecutionPlan,
    Kernel,
    Storage,
    compute_storage,
    plan_execution,
)
from .tensor import (
    BatchedMatrixMultiplicationTensor,
    BinaryOperator,
    BroadcastTensor,
    CompactTensor,
    ConstantTensor,
    DType,
    ElementwiseOperationTensor,
    IndexTensor,
    Operator,
    ParameterTensor,
    Pitch,
    ReductionTensor,
    ReshapeTensor,
    Shape,
    SqueezeTensor,
    Substitution,
    Tensor,
    TensorFunction,
    UnaryOperator,
    dtype_bytes,
)
