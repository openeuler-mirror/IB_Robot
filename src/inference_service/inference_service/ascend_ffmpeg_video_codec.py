"""Compatibility alias for the shared Ascend FFmpeg video codec backend."""

import sys

from observation_transport import ascend_ffmpeg_video_codec as _implementation

sys.modules[__name__] = _implementation
