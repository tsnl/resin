__all__ = [
    "TraceHub",
    "TraceStats",
    "add_time_span",
    "clear",
    "compute_execution_time",
    "log",
]

import logging
import time
from dataclasses import dataclass

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
    """Add a time span sample to the global trace hub."""
    _global_hub.add_time_span(stream, beg_sec, end_sec)


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
