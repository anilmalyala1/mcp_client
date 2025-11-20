"""
Centralized logging configuration for the MCP client.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from constants import DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        # Format the message
        formatted = super().format(record)

        return formatted


def setup_logging(
    level: str | int | None = None,
    log_file: str | None = None,
    colored: bool = True,
    format_string: str | None = None,
) -> None:
    """
    Configure logging for the MCP client.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
        log_file: Optional file path to write logs to
        colored: Whether to use colored output for console
        format_string: Custom log format string

    Example:
        >>> setup_logging(level="DEBUG", log_file="mcp.log")
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Ready to go!")
    """
    # Determine log level
    if level is None:
        level = os.getenv("MCP_LOG_LEVEL", DEFAULT_LOG_LEVEL)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create format
    if format_string is None:
        format_string = os.getenv("MCP_LOG_FORMAT", DEFAULT_LOG_FORMAT)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if colored and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set level for specific loggers to reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info("Logging configured: level=%s, colored=%s, file=%s", level, colored, log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello!")
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: logging.Logger, level: int | str):
        """
        Initialize log context.

        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.original_level = logger.level

    def __enter__(self) -> logging.Logger:
        """Enter context and set new level."""
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore original level."""
        self.logger.setLevel(self.original_level)
