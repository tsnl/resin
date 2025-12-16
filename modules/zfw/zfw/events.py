from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, is_dataclass
from typing import Callable

from .basic import camel_to_snake


_event_name_to_cls_map: dict[str, type["Event"]] = {}


@dataclass
class Event(ABC):
    @classmethod
    def __init_subclass__(cls: type["Event"]) -> None:
        if not is_dataclass(cls):
            raise TypeError("Event subclasses must be dataclasses")
        if (c := _event_name_to_cls_map.setdefault(cls.hook_name(), cls)) is not cls:
            raise ValueError(
                f"Event unique name '{cls.hook_name()}' is already registered "
                f"by {c} in {c.__module__!r}"
            )

    @classmethod
    def hook_name(cls) -> str:
        """
        Return a unique name for the event, used for dispatch.
        Must be all lowercase with optional underscores and ending with '_event'.
        E.g. 'key_event', 'mouse_move_event'.
        """
        return camel_to_snake(cls.__name__)


class EventRouter[TEvent: Event]:
    _callback_registry: defaultdict[str, list["EventHandler"]]

    def __init__(self) -> None:
        super().__init__()
        self._callback_registry = defaultdict(list)

    def subscribe[T: Event](self) -> Callable[["EventHandler[T]"], "EventHandler[T]"]:
        """Register a listener function for events."""

        def decorator(fn: "EventHandler[T]") -> "EventHandler[T]":
            self._callback_registry[fn.__name__].append(fn)
            return fn

        return decorator

    def unsubscribe(self, fn: "EventHandler[TEvent]") -> None:
        """Unregister a listener function from events."""
        self._callback_registry[fn.__name__].remove(fn)

    def publish(self, event: TEvent) -> None:
        """Publish an event to all registered listeners."""
        for fn in self._callback_registry[event.hook_name()]:
            fn(event)


type EventHandler[TEvent: Event] = Callable[[TEvent], None]
