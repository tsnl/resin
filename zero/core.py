from abc import ABC, abstractmethod
from typing import cast
from weakref import ref as WeakRef

from .excepts import LogicError

#
# Context, ContextResource
#


class BaseContextResource[TContext: "BaseContext"](ABC):
    def __init__(self, *, parent: "BaseContextResource[TContext] | None" = None):
        super().__init__()

        if parent is None:
            # Only `TContext` instances are allowed to have no parent.
            self._parent = None
            self._context: TContext = cast(TContext, self)
        else:
            self._parent: "BaseContextResource[TContext] | None" = parent
            self._context: TContext = parent._context

        self._children: list[WeakRef[BaseContextResource[TContext]]] = []
        self._children_cleanup_threshold: int = 64

        self._is_disposed: bool = False

        self._post_init()

    def _post_init(self) -> None:
        if self._parent:
            self._parent._notify_child_added(self)

    def _notify_child_added(self, child: "BaseContextResource[TContext]") -> None:
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

    @property
    def context(self) -> TContext:
        return self._context

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

    @abstractmethod
    def _on_dispose(self) -> None:
        pass


class BaseContext[TContext](BaseContextResource[TContext]):
    def __init__(self):
        super().__init__(parent=None)


#
# expect: check for None values
#


def expect[T](opt_value: T | None, message: str = "Expected value to not be None") -> T:
    if opt_value is None:
        raise LogicError(message)
    return opt_value
