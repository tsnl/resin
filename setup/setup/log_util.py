"""Shared logger configuration for setup scripts.

Provides get_logger(name) which configures a RichHandler (no timestamps)
and falls back to a StreamHandler if `rich` is not installed.
"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    - Uses rich.logging.RichHandler(show_time=False) if available.
    - Otherwise falls back to a basic StreamHandler.
    - Ensures handlers are only added once per logger.
    """
    logger = logging.getLogger(name or __name__)
    if logger.handlers:
        return logger

    try:
        from rich.logging import RichHandler

        handler = RichHandler(show_time=False)
    except Exception:
        handler = logging.StreamHandler()

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
