__all__ = [
    "SupportsWrite",
    "BaseResource",
    "expect",
]

from abc import ABC
from weakref import ref as WeakRef
from typing import TypeVar, Protocol

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
# Context, ContextResource
#


class BaseResource(ABC):
    def __init__(self, *, parent: "BaseResource | None" = None):
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
        self.dispose()

    def dispose(self) -> None:
        # If already disposed, no-op.
        if self._is_disposed:
            return

        # Dispose children in reverse order of creation.
        for child_ref in reversed(self._children):
            child = child_ref()
            if child is not None:
                child.dispose()
        self._children.clear()

        # Dispose self.
        self._on_dispose()
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
