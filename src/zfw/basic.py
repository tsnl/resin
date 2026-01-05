__all__ = [
    "NUMBA_CACHE_ENABLED",
    "BaseDisposable",
    "ButtonAction",
    "ColorSpace",
    "Font",
    "FontSize",
    "FontWeight",
    "Key",
    "KeyModifier",
    "MouseButton",
    "SupportsWrite",
    "expect",
    "logger",
    "round_up_to_po2",
    "setup_logging",
]

from abc import ABC
from typing import Protocol, TypeVar, Self, Literal, Sequence
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rich.logging

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
# Disposable
#


class BaseDisposable:
    _is_disposed: bool

    def __init__(self) -> None:
        super().__init__()
        self._is_disposed = False

    def dispose(self) -> None:
        """Dispose of the resource, freeing any associated GPU memory."""
        if self._is_disposed:
            return
        self._is_disposed = True
        self._on_dispose()

    def _on_dispose(self) -> None:
        """Hook called when dispose() is called."""
        pass


#
# unwrap: check for None values
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

    def __new__(
        cls,
        data: npt.ArrayLike,
        *,
        copy: bool | np._CopyMode | None = True,
    ) -> Self:
        """
        Replacement for np.array() that returns an instance of this subclass.
        """

        return np.array(data, dtype=cls.DTYPE, copy=copy).view(cls)

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> Self:
        return np.empty(shape, dtype=cls.DTYPE).view(cls)

    @classmethod
    def array_size(cls, *, shape: tuple[int, ...]) -> int:
        n = 1
        for dim in shape:
            n *= dim
        return n * cls.DTYPE.itemsize


#
# Numba config
#

NUMBA_CACHE_ENABLED = True


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

type ColorSpace = Literal["srgb", "linear"]
type Font = Literal["sans-serif", "serif", "monospaced"]
type FontWeight = Literal["light", "regular", "bold"]
type FontSize = Literal["regular", "large", "extra-large"]
type Key = Literal[
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
type ButtonAction = Literal["press", "release", "repeat"]
type KeyModifier = Literal["shift", "control", "alt", "super"]

type MouseButton = Literal[
    "left",  # left mouse button, aka button-1
    "right",  # right mouse button, aka button-2
    "middle",  # middle mouse button, aka button-3
    "button-4",
    "button-5",
    "button-6",
    "button-7",
    "button-8",
]


type HorizontalAlignment = Literal["left", "center", "right"]
type VerticalAlignment = Literal["top", "middle", "bottom"]


#
# Json types
#

type JsonObject = dict[str, Json]
type JsonArray = Sequence[Json]
type Json = JsonObject | JsonArray | str | int | float | bool | None


#
# Logging utilities (and own logger)
#


def logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Args:
        name: The logger name, typically __name__.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    file: Path | None = None,
    console: bool = True,
    suppress_noisy_third_party_loggers: bool = True,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Path to log file. If None and file_enabled=True, creates a default.
        console_enabled: Whether to log to console.
        file_enabled: Whether to log to file.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stderr)
    if console:
        console_handler = rich.logging.RichHandler()
        console_handler.setLevel(level)
        # console_formatter = logging.Formatter(
        #     fmt="%(asctime)s|%(levelname)s|%(name)s| %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
        # console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if file:
        file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    if suppress_noisy_third_party_loggers:
        noisy_loggers = [
            "numba",
            "wgpu",
            "PIL",
            "Pillow",
        ]
        for logger_name in noisy_loggers:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.ERROR)


#
# Log for this module:
#

LOG = logger(__name__)
