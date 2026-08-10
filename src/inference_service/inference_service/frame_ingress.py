"""Compatibility alias for the public IB-Robot FrameIngress contract."""

import sys

from observation_transport import frame_ingress as _implementation

sys.modules[__name__] = _implementation
