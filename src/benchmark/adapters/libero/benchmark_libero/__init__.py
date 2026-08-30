"""LIBERO child adapter package for IB-Robot benchmark evaluation.

LIBERO runtime scope: this package implements the production LIBERO plugin/adapter,
observation/action codecs, version probe and a no-op native reporter. It
is discovered through the ``ibrobot.benchmark_adapters`` entry-point group
with name ``libero``.

Heavy LIBERO/robosuite/MuJoCo imports are deferred to
``LiberoAdapter.configure()`` so plugin discovery never creates a MuJoCo
context. The package must NOT be imported by ``benchmark_runtime`` or
``robot_config``; discovery is exclusively through the entry-point group.
"""

from benchmark_libero.adapter import LiberoAdapter
from benchmark_libero.observation_codec import (
    POLICY_IMAGE2_KEY,
    POLICY_IMAGE_KEY,
    POLICY_STATE_KEY,
    ObservationCodecError,
    convert_observation,
    quat_to_axis_angle_xyzw,
)
from benchmark_libero.plugin import (
    create_adapter,
    create_plugin,
    validate_environment_config,
)
from benchmark_libero.version_probe import (
    ProviderIdentity,
    ProviderProbeError,
    probe_libero_provider,
)

__all__ = [
    "LiberoAdapter",
    "ObservationCodecError",
    "POLICY_IMAGE2_KEY",
    "POLICY_IMAGE_KEY",
    "POLICY_STATE_KEY",
    "ProviderIdentity",
    "ProviderProbeError",
    "convert_observation",
    "create_adapter",
    "create_plugin",
    "probe_libero_provider",
    "quat_to_axis_angle_xyzw",
    "validate_environment_config",
]
