from typing import Callable
import inspect

from .type import Type, FnType, VOID_TYPE


class Function:
    """A C/GLSL function."""

    raw: Callable

    def __init__(self, raw: Callable):
        super().__init__()
        self.raw = raw
        self.signature = Function._parse_signature(raw)

    @staticmethod
    def _parse_signature(func: Callable) -> FnType:
        """Compute a string signature for the given function."""

        python_signature = inspect.signature(func)

        param_types: list[Type] = []
        for param in python_signature.parameters.values():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Parameter '{param.name}' of function '{func.__name__}' is missing a type annotation."
                )
            param_types.append(param.annotation)

        if python_signature.return_annotation is inspect.Signature.empty:
            return_type: Type = VOID_TYPE
        else:
            return_type = python_signature.return_annotation

        return FnType(param_types=param_types, return_type=return_type)

    def __call__(self, *args, **kwargs):
        _ = args
        _ = kwargs
        raise RuntimeError(
            "CFunction call expressions cannot be called outside other CFunction definitions."
        )
