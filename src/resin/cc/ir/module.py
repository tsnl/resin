__all__ = ["Module"]

from typing import Callable

from .function import Function
from .type import StructType


class Module:
    """A C/GLSL module containing functions and structs."""

    name: str
    functions: dict[str, Function]
    structs: dict[str, StructType]

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.functions = {}
        self.structs = {}

    @staticmethod
    def get(name: str) -> "Module":
        """Get or create the Module instance for the given Python module name."""
        global _all_modules
        if m := _all_modules.get(name):
            return m
        m = Module(name)
        _all_modules[name] = m
        return m

    def define_function(self, func: Callable) -> Function:
        """Define a Function for the given Python function."""
        f = Function(func)
        self.functions[func.__name__] = f
        return f

    def define_struct(self, cls: type) -> StructType:
        """Define a StructType for the given Python class."""
        s = StructType(cls)
        self.structs[cls.__name__] = s
        return s

    @staticmethod
    def all() -> dict[str, "Module"]:
        """Get all defined modules."""
        global _all_modules
        return _all_modules


_all_modules: dict[str, Module] = {}
"""Mapping from Python module names (i.e. __name__) to CModule instances."""
