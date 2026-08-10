"""Generic ROS runtime for IB-Robot benchmark evaluation.

benchmark plugin contract adds the frozen wire contract, pure-Python adapter/native-report ABCs,
data models, plugin descriptor and production registry. All four protocol
modules (``models``, ``adapter``, ``native_report``, ``registry``) are
importable without initializing ``rclpy``.

The environment/evaluator nodes (``environment_node``, ``evaluator_node``)
import ``rclpy`` and are only loaded via their console-script entry points;
they are deliberately NOT re-exported here so that ``import benchmark_runtime``
works in a plain Python process.
"""

from benchmark_runtime.adapter import BenchmarkAdapter
from benchmark_runtime.capability_negotiation import (
    BenchmarkCapabilityAgreement,
    CapabilityNegotiationError,
    negotiate_provider_capabilities,
)
from benchmark_runtime.io_descriptor import (
    BenchmarkIODescriptor,
    FeatureDescriptor,
    IOCompatibilityError,
    IODescriptorError,
    ObservationBatch,
    ObservationSample,
)
from benchmark_runtime.models import (
    BenchmarkCapabilities,
    BenchmarkEnvironmentConfig,
    BenchmarkTask,
    EpisodeResult,
    MetricRecord,
    NativeArtifact,
    ResetRequest,
    ResetResult,
    RunManifest,
    RunSummary,
    StepEvent,
    StepResult,
)
from benchmark_runtime.native_report import NativeReportExporter
from benchmark_runtime.observation_router import (
    DeliveryContext,
    DeliveryReceipt,
    ObservationCommitError,
    ObservationPrepareError,
    ObservationRoute,
    ObservationRouter,
    ObservationRouterError,
    ObservationSink,
    PreparedBatch,
    PreparedRoute,
    RouteDeliveryReceipt,
)
from benchmark_runtime.registry import (
    BENCHMARK_ADAPTER_ENTRY_POINT_GROUP,
    BENCHMARK_PLUGIN_API_VERSION,
    BenchmarkPlugin,
    BenchmarkPluginDescriptorError,
    BenchmarkPluginDuplicateError,
    BenchmarkPluginLoadError,
    BenchmarkPluginNotFoundError,
    BenchmarkRegistryError,
    discover_plugins,
    load_plugin,
)

__all__ = [
    "BENCHMARK_ADAPTER_ENTRY_POINT_GROUP",
    "BENCHMARK_PLUGIN_API_VERSION",
    "BenchmarkAdapter",
    "BenchmarkCapabilityAgreement",
    "CapabilityNegotiationError",
    "BenchmarkIODescriptor",
    "FeatureDescriptor",
    "DeliveryContext",
    "DeliveryReceipt",
    "ObservationCommitError",
    "ObservationPrepareError",
    "ObservationRoute",
    "ObservationSink",
    "ObservationRouter",
    "ObservationRouterError",
    "PreparedBatch",
    "PreparedRoute",
    "RouteDeliveryReceipt",
    "IOCompatibilityError",
    "IODescriptorError",
    "ObservationBatch",
    "ObservationSample",
    "BenchmarkCapabilities",
    "BenchmarkEnvironmentConfig",
    "BenchmarkPlugin",
    "BenchmarkPluginDescriptorError",
    "BenchmarkPluginDuplicateError",
    "BenchmarkPluginLoadError",
    "BenchmarkPluginNotFoundError",
    "BenchmarkRegistryError",
    "BenchmarkTask",
    "EpisodeResult",
    "MetricRecord",
    "NativeArtifact",
    "NativeReportExporter",
    "ResetRequest",
    "ResetResult",
    "RunManifest",
    "RunSummary",
    "StepEvent",
    "StepResult",
    "discover_plugins",
    "load_plugin",
    "negotiate_provider_capabilities",
]
