__all__ = [
    "TraceHub",
    "TraceStats",
    "add_time_span",
    "clear",
    "compute_execution_time",
    "enable_chromium_trace",
    "log",
    "decorator",
    "save_chromium_trace",
    "span",
]

import atexit
import functools
import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, ParamSpec, TypeVar

import numpy as np


LOG = logging.getLogger(__name__)


@dataclass
class TraceStats:
    """Statistics computed from trace samples."""

    mean_sec: float
    p5_sec: float
    p50_sec: float
    p95_sec: float
    sample_count: int


@dataclass
class _TimeSample:
    """A single time span sample with wall-clock timestamp."""

    wall_time_sec: float
    duration_sec: float


class TraceHub:
    """
    Collects time span samples for multiple streams and computes statistics.

    Streams are named hierarchically, e.g. "Draw3dRenderer/record/tlas_build".

    Usage:
        hub = TraceHub()
        hub.add_time_span("MyClass/method/operation", t0, t1)
        stats = hub.compute_execution_time("MyClass/method/operation", truncate_window_sec=5.0)
        print(f"Mean: {stats.mean_sec * 1000:.2f}ms")
    """

    _streams: dict[str, list[_TimeSample]]

    def __init__(self) -> None:
        self._streams = {}

    def add_time_span(self, stream: str, beg_sec: float, end_sec: float) -> None:
        """
        Add a time span sample to a named stream.

        :param stream: Name of the stream (e.g., "Draw3dRenderer/record/tlas_build").
        :param beg_sec: Start time in seconds.
        :param end_sec: End time in seconds.
        """
        if stream not in self._streams:
            self._streams[stream] = []

        duration = end_sec - beg_sec
        wall_time = time.monotonic()
        self._streams[stream].append(
            _TimeSample(wall_time_sec=wall_time, duration_sec=duration)
        )

    def compute_execution_time(
        self,
        stream: str,
        *,
        truncate_window_sec: float | None = None,
    ) -> TraceStats | None:
        """
        Compute execution time statistics for a stream.

        :param stream: Name of the stream.
        :param truncate_window_sec: If specified, only consider samples from the last
            N seconds of wall-clock time. If None, use all samples.
        :return: TraceStats with mean, p5, p50, p95 durations, or None if no samples.
        """
        if stream not in self._streams:
            return None

        samples = self._streams[stream]
        if len(samples) == 0:
            return None

        # Filter by wall-clock time window if specified
        if truncate_window_sec is not None:
            current_time = time.monotonic()
            cutoff_time = current_time - truncate_window_sec
            samples = [s for s in samples if s.wall_time_sec >= cutoff_time]

            # Prune old samples from storage
            self._streams[stream] = samples

        if len(samples) == 0:
            return None

        durations = np.array([s.duration_sec for s in samples], dtype=np.float64)

        return TraceStats(
            mean_sec=float(durations.mean()),
            p5_sec=float(np.percentile(durations, 5)),
            p50_sec=float(np.percentile(durations, 50)),
            p95_sec=float(np.percentile(durations, 95)),
            sample_count=len(samples),
        )

    def clear(self, stream: str | None = None) -> None:
        """
        Clear samples from a stream or all streams.

        :param stream: If specified, clear only this stream. Otherwise clear all.
        """
        if stream is not None:
            if stream in self._streams:
                self._streams[stream] = []
        else:
            self._streams.clear()

    def log(
        self,
        *,
        filter_prefixes: list[str] | None = None,
        truncate_window_sec: float | None = None,
    ) -> None:
        """
        Log all stream statistics to DEBUG level.

        :param filter_prefixes: If specified, only log streams whose names start with
            one of these prefixes. If None, log all streams.
        :param truncate_window_sec: Time window for computing statistics.
        """
        streams_to_log = sorted(self._streams.keys())

        if filter_prefixes is not None:
            streams_to_log = [
                s
                for s in streams_to_log
                if any(s.startswith(prefix) for prefix in filter_prefixes)
            ]

        for stream in streams_to_log:
            stats = self.compute_execution_time(
                stream, truncate_window_sec=truncate_window_sec
            )
            if stats is not None:
                LOG.debug(
                    "%s: p5/p50/p95/mean=%.2f/%.2f/%.2f/%.2fms (n=%d)",
                    stream,
                    stats.p5_sec * 1000,
                    stats.p50_sec * 1000,
                    stats.p95_sec * 1000,
                    stats.mean_sec * 1000,
                    stats.sample_count,
                )


# Global singleton instance
_global_hub = TraceHub()


def add_time_span(stream: str, beg_sec: float, end_sec: float) -> None:
    """Add a time span sample to the global trace hub and Chromium tracer."""
    _global_hub.add_time_span(stream, beg_sec, end_sec)
    # Also add to Chromium tracer if enabled (uses forward reference)
    _add_to_chromium_tracer(stream, beg_sec, end_sec)


def compute_execution_time(
    stream: str,
    *,
    truncate_window_sec: float | None = None,
) -> TraceStats | None:
    """Compute execution time statistics from the global trace hub."""
    return _global_hub.compute_execution_time(
        stream, truncate_window_sec=truncate_window_sec
    )


def clear(stream: str | None = None) -> None:
    """Clear samples from the global trace hub."""
    _global_hub.clear(stream)


def log(
    *,
    filter_prefixes: list[str] | None = None,
    truncate_window_sec: float | None = None,
) -> None:
    """Log all stream statistics from the global trace hub to DEBUG level."""
    _global_hub.log(
        filter_prefixes=filter_prefixes, truncate_window_sec=truncate_window_sec
    )


# =============================================================================
# Chromium Trace Format Support
# =============================================================================


@dataclass
class _ChromiumTraceEvent:
    """
    A single event in Chromium trace format.

    See: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """

    name: str
    cat: str  # Category
    ph: str  # Phase: 'B' = begin, 'E' = end, 'X' = complete, 'i' = instant
    ts: float  # Timestamp in microseconds
    dur: float | None = None  # Duration in microseconds (for 'X' phase)
    pid: int = 0  # Process ID
    tid: int = 0  # Thread ID
    args: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str | int | float | dict[str, str | int | float]]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, str | int | float | dict[str, str | int | float]] = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
        }
        if self.dur is not None:
            result["dur"] = self.dur
        if self.args:
            result["args"] = self.args
        return result


class ChromiumTracer:
    """
    Collects trace events and exports them in Chromium trace format.

    The Chromium trace format is a JSON array of trace events that can be
    loaded in Chrome's chrome://tracing or Perfetto UI (https://ui.perfetto.dev).

    Usage:
        tracer = ChromiumTracer()
        tracer.add_complete_event("my_function", "category", start_us, duration_us)
        tracer.save("trace.json")

    Or use the context manager:
        with tracer.span("my_operation", "category"):
            do_work()
    """

    _events: list[_ChromiumTraceEvent]
    _enabled: bool
    _output_path: Path | None
    _start_time_sec: float
    _atexit_registered: bool

    def __init__(self) -> None:
        self._events = []
        self._enabled = False
        self._output_path = None
        self._start_time_sec = time.perf_counter()
        self._atexit_registered = False

    def enable(
        self,
        output_path: Path | str | None = None,
        register_atexit: bool = True,
    ) -> None:
        """
        Enable trace collection.

        :param output_path: Path to save trace file. If None, uses 'trace.json'
            in the current directory.
        :param register_atexit: If True, automatically save trace on program exit.
        """
        self._enabled = True
        self._start_time_sec = time.perf_counter()
        self._events.clear()

        if output_path is None:
            output_path = Path("trace.json")
        self._output_path = Path(output_path)

        if register_atexit and not self._atexit_registered:
            atexit.register(self._atexit_save)
            self._atexit_registered = True

        LOG.info("Chromium tracing enabled, output: %s", self._output_path)

    def disable(self) -> None:
        """Disable trace collection."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled

    def add_complete_event(
        self,
        name: str,
        category: str,
        start_sec: float,
        duration_sec: float,
        *,
        tid: int = 0,
        args: dict[str, str | int | float] | None = None,
    ) -> None:
        """
        Add a complete duration event (phase 'X').

        :param name: Event name (displayed in trace viewer).
        :param category: Category for filtering.
        :param start_sec: Start time in seconds (relative to tracer start).
        :param duration_sec: Duration in seconds.
        :param tid: Thread ID for grouping events.
        :param args: Optional dictionary of additional event data.
        """
        if not self._enabled:
            return

        # Convert seconds to microseconds
        ts_us = (start_sec - self._start_time_sec) * 1_000_000
        dur_us = duration_sec * 1_000_000

        self._events.append(
            _ChromiumTraceEvent(
                name=name,
                cat=category,
                ph="X",  # Complete event
                ts=ts_us,
                dur=dur_us,
                pid=os.getpid(),
                tid=tid,
                args=args or {},
            )
        )

    def add_instant_event(
        self,
        name: str,
        category: str,
        timestamp_sec: float,
        *,
        scope: str = "t",  # 't' = thread, 'p' = process, 'g' = global
        tid: int = 0,
        args: dict[str, str | int | float] | None = None,
    ) -> None:
        """
        Add an instant event (phase 'i').

        :param name: Event name.
        :param category: Category for filtering.
        :param timestamp_sec: Timestamp in seconds.
        :param scope: Event scope: 't' (thread), 'p' (process), 'g' (global).
        :param tid: Thread ID.
        :param args: Optional additional data.
        """
        if not self._enabled:
            return

        ts_us = (timestamp_sec - self._start_time_sec) * 1_000_000

        event_args = dict(args) if args else {}
        event_args["s"] = scope

        self._events.append(
            _ChromiumTraceEvent(
                name=name,
                cat=category,
                ph="i",
                ts=ts_us,
                pid=os.getpid(),
                tid=tid,
                args=event_args,
            )
        )

    @contextmanager
    def span(
        self,
        name: str,
        category: str = "default",
        *,
        tid: int = 0,
        args: dict[str, str | int | float] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for timing a code block.

        Usage:
            with tracer.span("my_operation", "render"):
                do_work()

        :param name: Event name.
        :param category: Category for filtering.
        :param tid: Thread ID.
        :param args: Optional additional data.
        """
        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.add_complete_event(
                name=name,
                category=category,
                start_sec=start,
                duration_sec=end - start,
                tid=tid,
                args=args,
            )

    def save(self, output_path: Path | str | None = None) -> None:
        """
        Save collected trace events to a JSON file.

        :param output_path: Path to save. If None, uses the path from enable().
        """
        path = Path(output_path) if output_path else self._output_path
        if path is None:
            path = Path("trace.json")

        if not self._events:
            LOG.warning("No trace events to save")
            return

        # Build Chromium trace format
        trace_data = {
            "traceEvents": [e.to_dict() for e in self._events],
            "displayTimeUnit": "ms",
            "metadata": {
                "process_name": "resin",
            },
        }

        with open(path, "w") as f:
            json.dump(trace_data, f, indent=None)

        LOG.info("Saved %d trace events to %s", len(self._events), path)

    def _atexit_save(self) -> None:
        """Called at program exit to save trace."""
        if self._enabled and self._events:
            self.save()

    def clear(self) -> None:
        """Clear all collected events."""
        self._events.clear()


# Global singleton instance
_global_tracer = ChromiumTracer()


def _add_to_chromium_tracer(stream: str, beg_sec: float, end_sec: float) -> None:
    """Helper to add time span to Chromium tracer (called from add_time_span)."""
    # Parse category from stream name (use first component before '/')
    parts = stream.split("/", 1)
    category = parts[0] if parts else "default"
    _global_tracer.add_complete_event(
        name=stream,
        category=category,
        start_sec=beg_sec,
        duration_sec=end_sec - beg_sec,
    )


def enable_chromium_trace(
    output_path: Path | str | None = None,
    register_atexit: bool = True,
) -> None:
    """
    Enable global Chromium trace collection.

    :param output_path: Path to save trace file on exit.
    :param register_atexit: If True, automatically save on program exit.
    """
    _global_tracer.enable(output_path=output_path, register_atexit=register_atexit)


def save_chromium_trace(output_path: Path | str | None = None) -> None:
    """Save the global Chromium trace to a file."""
    _global_tracer.save(output_path)


@contextmanager
def span(
    name: str,
    category: str = "default",
    *,
    tid: int = 0,
    args: dict[str, str | int | float] | None = None,
) -> Generator[None, None, None]:
    """
    Context manager for timing a code block in the global tracer.

    Also adds the time span to the global TraceHub for statistics.

    Usage:
        with trace.span("my_operation", "render"):
            do_work()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        # Add to Chromium tracer
        _global_tracer.add_complete_event(
            name=name,
            category=category,
            start_sec=start,
            duration_sec=end - start,
            tid=tid,
            args=args,
        )
        # Also add to TraceHub for statistics
        _global_hub.add_time_span(name, start, end)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def decorator(
    name: str,
    category: str = "default",
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Decorator to automatically trace function execution.

    Usage:
        @trace.decorator("MyClass/my_method")
        def my_method(self, x: int) -> int:
            return x * 2

    :param name: Event name (displayed in trace viewer).
    :param category: Category for filtering (default: first component of name).
    """

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                # Add to Chromium tracer
                _global_tracer.add_complete_event(
                    name=name,
                    category=category,
                    start_sec=start,
                    duration_sec=end - start,
                )
                # Also add to TraceHub for statistics
                _global_hub.add_time_span(name, start, end)

        return wrapper

    return decorator
