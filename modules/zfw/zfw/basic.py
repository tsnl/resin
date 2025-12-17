__all__ = [
    "BaseResource",
    "ButtonAction",
    "ColorSpace",
    "Key",
    "KeyModifier",
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

    def __init__(self, *, parent_resource: "BaseResource | None"):
        super().__init__()

        self._parent_resource: "BaseResource | None" = parent_resource
        self._child_resources: list[WeakRef[BaseResource]] = []
        self._child_resources_cleanup_threshold: int = 64

        self._resource_is_disposed: bool = False

        self._resource_post_init()

    def _resource_post_init(self) -> None:
        if self._parent_resource:
            self._parent_resource._notify_child_resource_added(self)

    def _notify_child_resource_added(self, child: "BaseResource") -> None:
        if len(self._child_resources) >= self._child_resources_cleanup_threshold:
            # Clean up dead weak references.
            # Do not modify the order of existing children: critical for disposal.
            self._child_resources = [
                child_ref
                for child_ref in self._child_resources
                if child_ref() is not None
            ]

            # Adjust threshold if needed.
            if len(self._child_resources) >= self._child_resources_cleanup_threshold:
                # Increase threshold to avoid frequent cleanups.
                self._child_resources_cleanup_threshold *= 2
            elif (
                len(self._child_resources)
                < self._child_resources_cleanup_threshold // 4
            ):
                # Decrease threshold to avoid excessive memory usage.
                self._child_resources_cleanup_threshold //= 2

        self._child_resources.append(WeakRef(child))

    def __del__(self) -> None:
        if self._parent_resource is not None:
            parent = self._parent_resource
            del parent
        self.dispose_resource()

    def dispose_resource(self) -> None:
        # If already disposed, no-op.
        if self._resource_is_disposed:
            return

        # If 'self' is not yet disposed, ensure self._parent has not yet been disposed
        # either.
        if (
            self._parent_resource is not None
            and self._parent_resource._resource_is_disposed
        ):
            warnings.warn(
                f"Cannot dispose resource {self} after its parent {self._parent_resource}",
            )

        # Dispose children in reverse order of creation.
        for child_ref in reversed(self._child_resources):
            child = child_ref()
            if child is not None:
                child.dispose_resource()
        self._child_resources.clear()

        # Dispose self.
        self._on_dispose_resource()
        self._parent_resource = None
        self._resource_is_disposed = True

    def _on_dispose_resource(self) -> None:
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
# Camel-case to snake-case conversion
#


def camel_to_snake(name: str) -> str:
    small_chunks = []
    for large_chunk in name.split("_"):
        last_small_chunk_start_index = 0
        for i in range(1, len(large_chunk)):
            if large_chunk[i].isupper():
                small_chunks.append(large_chunk[last_small_chunk_start_index:i].lower())
                last_small_chunk_start_index = i
        small_chunks.append(large_chunk[last_small_chunk_start_index:].lower())
    return "_".join(small_chunks)


#
# Constants
#

ColorSpace: TypeAlias = Literal["srgb", "linear"]

Font: TypeAlias = Literal["sans-serif", "serif"]

Key: TypeAlias = Literal[
    # Printable keys (US layout)
    "space",
    "apostrophe",
    "comma",
    "minus",
    "period",
    "slash",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "semicolon",
    "equal",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "left-bracket",
    "backslash",
    "right-bracket",
    "grave-accent",
    "world-1",
    "world-2",
    # Function keys and special keys
    "escape",
    "enter",
    "tab",
    "backspace",
    "insert",
    "delete",
    "right",
    "left",
    "down",
    "up",
    "page-up",
    "page-down",
    "home",
    "end",
    "caps-lock",
    "scroll-lock",
    "num-lock",
    "print-screen",
    "pause",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    "f25",
    # Keypad keys
    "kp-0",
    "kp-1",
    "kp-2",
    "kp-3",
    "kp-4",
    "kp-5",
    "kp-6",
    "kp-7",
    "kp-8",
    "kp-9",
    "kp-decimal",
    "kp-divide",
    "kp-multiply",
    "kp-subtract",
    "kp-add",
    "kp-enter",
    "kp-equal",
    # Modifier keys
    "left-shift",
    "left-control",
    "left-alt",
    "left-super",
    "right-shift",
    "right-control",
    "right-alt",
    "right-super",
    "menu",
]
ButtonAction: TypeAlias = Literal["press", "release", "repeat"]
KeyModifier: TypeAlias = Literal["shift", "control", "alt", "super"]

MouseButton: TypeAlias = Literal[
    "left",  # left mouse button, aka button-1
    "right",  # right mouse button, aka button-2
    "middle",  # middle mouse button, aka button-3
    "button-4",
    "button-5",
    "button-6",
    "button-7",
    "button-8",
]
