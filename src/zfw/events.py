__all__ = [
    "EventHandler",
    "EventHub",
]

import logging
from typing import Callable

LOG = logging.getLogger(__name__)


class EventHub[TEvent]:
    _callbacks: list["EventHandler[TEvent]"]

    def __init__(self) -> None:
        super().__init__()
        self._callbacks = []

    def subscribe(self) -> Callable[["EventHandler[TEvent]"], "EventHandler[TEvent]"]:
        """Register a listener function for events."""

        def decorator(fn: "EventHandler[TEvent]") -> "EventHandler[TEvent]":
            self._callbacks.append(fn)
            return fn

        return decorator

    def unsubscribe(self, fn: "EventHandler[TEvent]") -> None:
        """Unregister a listener function from events."""
        self._callbacks.remove(fn)

    def publish(self, event: TEvent) -> None:
        """Publish an event to all registered listeners."""
        for fn in self._callbacks:
            fn(event)


type EventHandler[TEvent] = Callable[[TEvent], None]
