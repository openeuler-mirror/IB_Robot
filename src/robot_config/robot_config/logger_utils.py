"""Logging utilities for robot_config.

Provides a ColoredLogger wrapper around launch.logging to ensure ANSI colors
are preserved for warnings and errors, using a cursor-return hack to overwrite
default ROS 2 prefixes. INFO logs are kept in their original white format.
"""

import os
import sys
import launch.logging


class ColoredLogger:
    """Wrapper for launch.logging that colors warning and error lines including prefix."""

    # ANSI color codes (High Intensity)
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def __init__(self, name: str):
        self._logger = launch.logging.get_logger(name)
        self._name = name

    def should_colorize(self) -> bool:
        """Check if output should be colorized."""
        return sys.stdout.isatty() or os.environ.get("RCUTILS_COLORIZED_OUTPUT") == "1"

    def _format_line(self, level: str, color: str, msg: str) -> str:
        """Construct a full log line and use \r to overwrite the default prefix."""
        # Standard ROS 2 launch prefix format is "[LEVEL] [name]: "
        # We use \r to go to start of line and overwrite the white prefix with our colored one.
        return f"\r{color}[{level}] [{self._name}]: {msg}{self.RESET}"

    def info(self, msg: str):
        """Log info message (original white format)."""
        self._logger.info(msg)

    def warning(self, msg: str):
        """Log warning message (full line yellow)."""
        if self.should_colorize():
            self._logger.warning(self._format_line("WARNING", self.YELLOW, msg))
        else:
            self._logger.warning(msg)

    def warn(self, msg: str):
        """Alias for warning()."""
        self.warning(msg)

    def error(self, msg: str):
        """Log error message (full line red)."""
        if self.should_colorize():
            self._logger.error(self._format_line("ERROR", self.RED, msg))
        else:
            self._logger.error(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self._logger.debug(msg)


def get_colored_logger(name: str):
    """Factory function to get a ColoredLogger instance."""
    return ColoredLogger(name)
