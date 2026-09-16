"""SO-101 Robot Hardware Package

This package provides a unified hardware interface for SO-101 robot arms,
including both C++ ros2_control hardware interface and Python utilities.
"""

from .calibration import constants

__all__ = ["constants"]
