"""Unified logging configuration for langchain-chat.

Provides a single ``setup_logging()`` entry point that configures the
Python ``logging`` module with both console and file handlers.  All
project modules obtain loggers via ``logging.getLogger(__name__)`` —
there is no custom logger abstraction.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "logs/app.log",
    *,
    fmt: str | None = None,
) -> None:
    """Configure root and project loggers with console + file output.

    Should be called once at application startup (e.g. from ``main.py``).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Path for the file handler (``None`` disables file output).
        fmt: Optional custom format string.
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"

    formatter = logging.Formatter(fmt)

    # Console handler.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    # Configure the project logger (all langchain_chat.* loggers).
    project_logger = logging.getLogger("langchain_chat")
    project_logger.setLevel(_level_value(level))
    project_logger.addHandler(console)
    project_logger.propagate = False  # don't bubble to root

    # File handler (optional).
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        project_logger.addHandler(fh)

    # Keep noisy third-party libraries quiet.
    _silence_third_party()


def _level_value(name: str) -> int:
    """Convert a level name to its numeric value, defaulting to INFO."""
    return getattr(logging, name.upper(), logging.INFO)


def _silence_third_party() -> None:
    """Set higher log thresholds for chatty dependencies."""
    for name in ("aiosqlite", "aiomysql", "httpx", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
