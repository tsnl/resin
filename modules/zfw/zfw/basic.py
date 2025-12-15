__all__ = [
    "BaseResource",
    "ColorSpace",
    "SupportsWrite",
    "expect",
    "round_up_to_po2",
]

from abc import ABC
from typing import Protocol, TypeVar, Self, TypeAlias, Literal
from weakref import ref as WeakRef
import warnings

import numpy as np
import numpy.typing as npt

from .excepts import LogicError

#
# Typing (from _typeshed)
#

# Why define these _typeshed types?
# It's not possible to install _typeshed at runtime.
# We don't want users to have `if TYPE_CHECKING` blocks everywhere.
# So we just copy the relevant definitions here.

_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> object: ...


#
# BaseResource
#


class BaseResource(ABC):
    """
    All resources in Zfw inherit from this base class to create a clear tree hierarchy.

    Each resource must be disposed of before its parent resource is disposed.
    Resource disposal is triggered either automatically by `__del__` or manually by
    calling `dispose()`. Disposal of a resource will also dispose all its child
    resources in reverse order of their creation.

    Explicit disposal is recommended to ensure timely release of resources. This must be
    done with care to avoid disposing a parent before its children.
    """

    def __init__(self, *, parent: "BaseResource | None"):
        super().__init__()

        self._parent: "BaseResource | None" = parent
        self._children: list[WeakRef[BaseResource]] = []
        self._children_cleanup_threshold: int = 64

        self._is_disposed: bool = False

        self._post_init()

    def _post_init(self) -> None:
        if self._parent:
            self._parent._notify_child_added(self)

    def _notify_child_added(self, child: "BaseResource") -> None:
        if len(self._children) >= self._children_cleanup_threshold:
            # Clean up dead weak references.
            # Do not modify the order of existing children: critical for disposal.
            self._children = [
                child_ref for child_ref in self._children if child_ref() is not None
            ]

            # Adjust threshold if needed.
            if len(self._children) >= self._children_cleanup_threshold:
                # Increase threshold to avoid frequent cleanups.
                self._children_cleanup_threshold *= 2
            elif len(self._children) < self._children_cleanup_threshold // 4:
                # Decrease threshold to avoid excessive memory usage.
                self._children_cleanup_threshold //= 2

        self._children.append(WeakRef(child))

    def __del__(self) -> None:
        if self._parent is not None:
            parent = self._parent
            del parent
        self.dispose()

    def dispose(self) -> None:
        # If already disposed, no-op.
        if self._is_disposed:
            return

        # If 'self' is not yet disposed, ensure self._parent has not yet been disposed
        # either.
        if self._parent is not None and self._parent._is_disposed:
            warnings.warn(
                f"Cannot dispose resource {self} after its parent {self._parent}",
            )

        # Dispose children in reverse order of creation.
        for child_ref in reversed(self._children):
            child = child_ref()
            if child is not None:
                child.dispose()
        self._children.clear()

        # Dispose self.
        self._on_dispose()
        self._parent = None
        self._is_disposed = True

    def _on_dispose(self) -> None:
        pass


#
# expect: check for None values
#


def expect[T](opt_value: T | None, message: str = "Expected value to not be None") -> T:
    if opt_value is None:
        raise LogicError(message)
    return opt_value


#
# Simple math utilities
#


def round_up_to_po2(x: int) -> int:
    """Return the next power of two greater than or equal to x."""
    if x <= 0:
        return 1
    v = 1
    while v < x:
        v *= 2
    return v


#
# StructuredNDArray
#


class StructuredNDArray(np.ndarray, ABC):
    DTYPE: np.dtype

    def __new__(cls, shape: tuple[int, ...] | int) -> Self:
        return np.zeros(shape, dtype=cls.DTYPE).view(cls)

    @classmethod
    def of(cls, arr: npt.NDArrayLike) -> Self:
        return np.asarray(arr, dtype=cls.DTYPE).view(cls)


#
# Constants
#

ColorSpace: TypeAlias = Literal["srgb", "linear"]
Font: TypeAlias = Literal["sans-serif", "serif"]
