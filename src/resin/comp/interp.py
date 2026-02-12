"""
resin.interp evaluates a tensor computational graph by recursively computing the values
of tensors based on their definitions and the values of their dependencies, using
memoization to avoid redundant computations.
"""

__all__ = [
    "Interpreter",
]

import numpy as np

from .tensor import (
    BatchedMatrixMultiplicationTensor,
    BroadcastTensor,
    CompactTensor,
    ConstantTensor,
    DType,
    ElementwiseOperationTensor,
    ExpandTensor,
    IndexTensor,
    VarTensor,
    PermuteTensor,
    ReductionTensor,
    ReshapeTensor,
    SqueezeTensor,
    Tensor,
)


class Interpreter:
    memo: dict[Tensor, np.ndarray]

    def __init__(self) -> None:
        self.memo = {}

    def evaluate_dict(
        self,
        tensor_dict: dict[str, Tensor],
    ) -> dict[str, np.ndarray]:
        return {name: self.evaluate(tensor) for name, tensor in tensor_dict.items()}

    def evaluate_list(
        self,
        tensor_list: list[Tensor],
    ) -> list[np.ndarray]:
        return [self.evaluate(tensor) for tensor in tensor_list]

    def evaluate_tuple(
        self,
        tensor_tuple: tuple[Tensor, ...],
    ) -> tuple[np.ndarray, ...]:
        return tuple(self.evaluate(tensor) for tensor in tensor_tuple)

    def evaluate(self, tensor: Tensor) -> np.ndarray:
        if tensor in self.memo:
            return self.memo[tensor]

        value = self._evaluate(tensor)
        assert value.shape == tensor.shape
        self.memo[tensor] = value
        return value

    def _evaluate(self, tensor: Tensor) -> np.ndarray:
        match tensor:
            case ConstantTensor():
                return self._evaluate_constant(tensor)
            case VarTensor():
                raise ValueError(
                    f"Cannot evaluate unsubstituted parameter: {tensor.name}"
                )
            case ElementwiseOperationTensor():
                return self._evaluate_elementwise_operation(tensor)
            case BatchedMatrixMultiplicationTensor():
                return self._evaluate_batched_matrix_multiplication(tensor)
            case BroadcastTensor():
                return self._evaluate_broadcast(tensor)
            case ExpandTensor():
                return self._evaluate_expand(tensor)
            case SqueezeTensor():
                return self._evaluate_squeeze(tensor)
            case ReductionTensor():
                return self._evaluate_reduction(tensor)
            case IndexTensor():
                return self._evaluate_index(tensor)
            case CompactTensor():
                return self._evaluate_compact(tensor)
            case ReshapeTensor():
                return self._evaluate_reshape(tensor)
            case PermuteTensor():
                return self._evaluate_permute(tensor)
            case _:
                raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")

    def _evaluate_constant(self, tensor: ConstantTensor) -> np.ndarray:
        return np.array(tensor.value, dtype=Interpreter.np_dtype(tensor.dtype))

    def _evaluate_elementwise_operation(
        self, tensor: ElementwiseOperationTensor
    ) -> np.ndarray:
        operands = [self.evaluate(operand) for operand in tensor.operands]
        dtype = Interpreter.np_dtype(tensor.dtype)

        match tensor.operator:
            case "pow":
                return np.power(*operands, dtype=dtype)
            case "mul":
                return np.multiply(*operands, dtype=dtype)
            case "div":
                return np.divide(*operands, dtype=dtype)
            case "add":
                return np.add(*operands, dtype=dtype)
            case "sub":
                return np.subtract(*operands, dtype=dtype)
            case "neg":
                return np.negative(operands[0], dtype=dtype)
            case "exp":
                return np.exp(operands[0], dtype=dtype)
            case "log":
                return np.log(operands[0], dtype=dtype)
            case "not":
                return (~operands[0].astype(bool)).astype(dtype)
            case "max":
                return np.maximum(*operands).astype(dtype)
            case "min":
                return np.minimum(*operands).astype(dtype)
            case "eq":
                return np.equal(*operands).astype(dtype)
            case "ne":
                return np.not_equal(*operands).astype(dtype)
            case "lt":
                return np.less(*operands).astype(dtype)
            case "gt":
                return np.greater(*operands).astype(dtype)
            case "le":
                return np.less_equal(*operands).astype(dtype)
            case "ge":
                return np.greater_equal(*operands).astype(dtype)
            case _:
                raise NotImplementedError(f"Unsupported operator: {tensor.operator}")

    def _evaluate_batched_matrix_multiplication(
        self, tensor: BatchedMatrixMultiplicationTensor
    ) -> np.ndarray:
        operand_a = self.evaluate(tensor.operands[0])
        operand_b = self.evaluate(tensor.operands[1])
        dtype = Interpreter.np_dtype(tensor.dtype)
        return np.matmul(operand_a, operand_b, dtype=dtype)

    def _evaluate_broadcast(self, tensor: BroadcastTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.broadcast_to(operand, tensor.shape).copy()

    def _evaluate_expand(self, tensor: ExpandTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.broadcast_to(operand, tensor.shape).copy()

    def _evaluate_squeeze(self, tensor: SqueezeTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.squeeze(operand, axis=tensor.dim)

    def _evaluate_reduction(self, tensor: ReductionTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        dtype = Interpreter.np_dtype(tensor.dtype)
        match tensor.operator:
            case "add":
                return np.sum(operand, axis=tensor.axis, dtype=dtype)
            case "mul":
                result = np.prod(operand, axis=tensor.axis)
                return result.astype(dtype)
            case _:
                raise NotImplementedError(
                    f"Unsupported reduction operator: {tensor.operator}"
                )

    def _evaluate_index(self, tensor: IndexTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return operand[tensor.key].copy()

    def _evaluate_compact(self, tensor: CompactTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.ascontiguousarray(operand)

    def _evaluate_reshape(self, tensor: ReshapeTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.reshape(operand, tensor.new_shape)

    def _evaluate_permute(self, tensor: PermuteTensor) -> np.ndarray:
        operand = self.evaluate(tensor.operands[0])
        return np.transpose(operand, tensor.new_order)

    @staticmethod
    def np_dtype(dtype: DType) -> type:
        match dtype:
            case "fp16":
                return np.float16
            case "fp32":
                return np.float32
            case _:
                raise NotImplementedError(f"Unsupported dtype: {dtype}")
