"""Compatibility alias for the shared software video codec backend."""

import sys

from observation_transport import software_video_codec as _implementation

sys.modules[__name__] = _implementation
