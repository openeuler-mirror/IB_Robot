"""Compatibility alias for the shared NVIDIA video codec backend."""

import sys

from observation_transport import nvidia_video_codec as _implementation

sys.modules[__name__] = _implementation
