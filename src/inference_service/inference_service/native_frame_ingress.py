"""Compatibility alias for the production IB-Robot FrameIngress."""

import sys

from observation_transport import native_frame_ingress as _implementation

sys.modules[__name__] = _implementation
