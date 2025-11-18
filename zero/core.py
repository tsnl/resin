from abc import ABC, abstractmethod
import atexit
import functools
from typing import cast, Type

import glfw

from .excepts import LogicError, GlfwError


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

        self._children: set[BaseContextResource[TContext]] = set()
        self._is_disposed: bool = False

        self._post_init()

    def _post_init(self) -> None:
        if self._parent:
            self._parent._notify_child_created(self)

    def _notify_child_created(self, child: "BaseContextResource[TContext]") -> None:
        self._children.add(child)

    def _notify_child_disposed(self, child: "BaseContextResource[TContext]") -> None:
        assert child._is_disposed
        self._children.remove(child)

    @property
    def context(self) -> TContext:
        return self._context

    def dispose(self) -> None:
        # If already disposed, no-op.
        if self._is_disposed:
            return

        # Dispose children.
        # Each child's `dispose()` method will mutate 'self._children`, so we need to
        # iterate over a copy of the set. We also expect all children to be disposed by
        # the end.
        for child in set(self._children):
            child.dispose()
        assert not self._children, "Expected all children to be disposed."

        # Dispose self.
        self._on_dispose()
        self._is_disposed = True

        # Notify parent AFTER disposing self.
        if self._parent:
            self._parent._notify_child_disposed(self)

    @abstractmethod
    def _on_dispose(self) -> None:
        pass


class BaseContext[TContext](BaseContextResource[TContext]):
    def __init__(self):
        super().__init__(parent=None)


@functools.cache
def ensure_glfw_init():
    ok = bool(glfw.init())
    if not ok:
        raise GlfwError("Failed to initialize GLFW")

    atexit.register(glfw.terminate)
