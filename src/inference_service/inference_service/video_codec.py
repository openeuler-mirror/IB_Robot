"""Compatibility alias for the IB-Robot observation transport codec SSOT."""

import sys

from observation_transport import video_codec as _implementation

sys.modules[__name__] = _implementation
