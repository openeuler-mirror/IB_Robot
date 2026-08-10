#!/usr/bin/env python3
"""Cloud endpoint for one validated distributed inference pipeline."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    DistributedInferenceResult,
    InferencePipelineStatus,
    VideoStreamDescriptor,
    VideoStreamStatus,
)
from inference_manifest import load_inference_manifest, load_inference_manifest_metadata
from inference_service.compute_video_streams import ComputeVideoStreamManager
from inference_service.distributed import (
    CloudBackendRuntime,
    DistributedCloudService,
    build_pipeline_identity,
    structured_error_from_exception,
)
from inference_service.distributed.ros_protocol import (
    decode_failure_result,
    request_from_message,
    result_to_message,
    status_from_message,
    status_to_message,
    transport_failure_result,
    video_descriptor_from_message,
    video_status_from_message,
    video_status_to_message,
)
from inference_service.runtime_composition import (
    build_policy_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders
from robot_config.contract_utils import contract_fingerprint, iter_specs


@dataclass(frozen=True)
class CloudNodeConfig:
    pipeline_id: str
    model_path: str
    deployment: str
    request_timeout: float
    runtime_options_json: str
    request_topic: str
    result_topic: str
    heartbeat_topic: str
    robot_config_path: str
    video_descriptor_topic: str
    video_status_topic: str


class PureInferenceNode(Node):
    """Own one cloud backend and its session-scoped request transport."""

    def __init__(
        self,
        config: CloudNodeConfig,
        *,
        node_name: str,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ) -> None:
        super().__init__(node_name)
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers
        try:
            runtime_options = json.loads(config.runtime_options_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"runtime_options_json is invalid: {exc}") from exc
        if not isinstance(runtime_options, dict):
            raise RuntimeError("runtime_options_json must decode to an object")

        metadata = load_inference_manifest_metadata(config.model_path, config.deployment)
        identity = build_pipeline_identity(config.pipeline_id, metadata)
        from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict

        startup_error = None
        stream_manager = None
        try:
            if config.robot_config_path:
                contract = build_contract_from_robot_config_dict(load_robot_config_dict(config.robot_config_path))
                required_inputs = set(metadata.policy.input_features)
                observation_specs = tuple(
                    spec for spec in iter_specs(contract) if not spec.is_action and spec.key in required_inputs
                )
                stream_manager = ComputeVideoStreamManager(
                    pipeline_id=config.pipeline_id,
                    session_id="pending",
                    session_generation=1,
                    contract_fingerprint=contract_fingerprint(contract),
                    deployment_fingerprint=metadata.fingerprint,
                    observation_specs=observation_specs,
                    rate_hz=float(contract.rate_hz),
                    n_obs_steps=metadata.policy.n_obs_steps,
                )
            validated = load_inference_manifest(config.model_path, config.deployment)
            runtime = CloudBackendRuntime(
                config.pipeline_id,
                validated,
                request_timeout=config.request_timeout,
                runtime_options=runtime_options,
                registry_set=self._registry_set,
                providers=self._providers,
            )
        except Exception as exc:
            runtime = None
            startup_error = structured_error_from_exception(exc, "startup")
            self.get_logger().error(f"distributed cloud backend failed to start: {exc}")
        self._config = config
        self._stream_manager = stream_manager
        self._service = DistributedCloudService(
            identity,
            runtime,
            startup_error=startup_error,
            stream_manager=stream_manager,
        )
        self._fingerprint = metadata.fingerprint
        self._result_pub = self.create_publisher(DistributedInferenceResult, config.result_topic, 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        descriptor_qos = QoSProfile(
            depth=max(1, len(stream_manager.diagnostic_snapshots()) if stream_manager is not None else 1),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            DistributedInferenceRequest,
            config.request_topic,
            self._request_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_subscription(
            VideoStreamDescriptor,
            config.video_descriptor_topic,
            self._video_descriptor_callback,
            descriptor_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_subscription(
            VideoStreamStatus,
            config.video_status_topic,
            self._video_status_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )
        self._video_status_pub = self.create_publisher(VideoStreamStatus, config.video_status_topic, 10)
        self._status_pub = self.create_publisher(InferencePipelineStatus, config.heartbeat_topic, status_qos)
        self.create_subscription(
            InferencePipelineStatus,
            config.heartbeat_topic,
            self._edge_status_callback,
            status_qos,
            callback_group=ReentrantCallbackGroup(),
        )
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._video_status_timer = self.create_timer(0.02, self._publish_video_status)
        self._log_video_stream_diagnostics()
        self.get_logger().info(
            f"Distributed cloud pipeline loaded: id={config.pipeline_id}, deployment={config.deployment}, "
            f"backend={metadata.deployment.backend}, request={config.request_topic}, result={config.result_topic}"
        )

    def _log_video_stream_diagnostics(self) -> None:
        if self._stream_manager is None:
            return
        for snapshot in self._stream_manager.diagnostic_snapshots():
            self.get_logger().info(
                f"Video stream startup: observation={snapshot.observation_key}, "
                f"stream_id={snapshot.stream_id}, mode={snapshot.mode}, "
                f"encoder={snapshot.configured_encoder_backend}/{snapshot.selected_encoder_backend}, "
                f"decoder={snapshot.configured_decoder_backend}/{snapshot.selected_decoder_backend}, "
                f"endpoint={snapshot.endpoint[0]}:{snapshot.endpoint[1]}, "
                f"contract_fingerprint={snapshot.contract_fingerprint}, "
                f"deployment_fingerprint={snapshot.deployment_fingerprint}, security={snapshot.security}, "
                f"lifecycle={snapshot.lifecycle_state}, ready={snapshot.ready}"
            )

    def _edge_status_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_EDGE:
            return
        try:
            status = status_from_message(message)
            response = self._service.observe_edge(status)
            self._status_pub.publish(status_to_message(response, stamp=self.get_clock().now().to_msg()))
        except Exception as exc:
            self.get_logger().error(f"invalid edge handshake status: {exc}\n{traceback.format_exc()}")

    def _publish_status(self) -> None:
        try:
            status = self._service.status()
            self._status_pub.publish(status_to_message(status, stamp=self.get_clock().now().to_msg()))
        except Exception as exc:
            self.get_logger().error(f"failed to publish cloud status: {exc}")

    def _video_descriptor_callback(self, message: VideoStreamDescriptor) -> None:
        if self._stream_manager is None:
            return
        try:
            self._stream_manager.observe_descriptor(video_descriptor_from_message(message))
        except Exception as exc:
            self.get_logger().error(f"invalid video stream descriptor: {exc}")

    def _video_status_callback(self, message: VideoStreamStatus) -> None:
        if self._stream_manager is None:
            return
        try:
            self._stream_manager.observe_status(
                video_status_from_message(message),
                receive_time_ns=self.get_clock().now().nanoseconds,
            )
        except Exception as exc:
            self.get_logger().error(f"invalid video stream status: {exc}")

    def _publish_video_status(self) -> None:
        if self._stream_manager is None:
            return
        stamp = self.get_clock().now().to_msg()
        for status in self._stream_manager.statuses():
            self._video_status_pub.publish(video_status_to_message(status, stamp=stamp))
        for observation_key, metrics in self._stream_manager.decoder_diagnostics():
            self.get_logger().info(
                f"Decoder runtime: observation={observation_key}, "
                f"input_frames={metrics.input_frames}, output_frames={metrics.output_frames}, "
                f"input_fps={metrics.input_frame_rate_hz:.2f}, output_fps={metrics.output_frame_rate_hz:.2f}, "
                f"backlog={metrics.decoder_backlog_depth}, output_age_ms={metrics.decoder_output_age_ns / 1e6:.1f}, "
                f"dropped_stale={metrics.dropped_stale_decoder_frames}, metadata_depth={metrics.metadata_fifo_depth}",
                throttle_duration_sec=2.0,
            )

    def _request_callback(self, message: DistributedInferenceRequest) -> None:
        request = None
        try:
            request = request_from_message(message)
            self.get_logger().info(
                "[IBROBOT_INFERENCE][DISTRIBUTED_REQUEST] "
                f"request={request.request_id} tensor_keys={sorted(request.inputs)} "
                f"stream_keys={[item.observation_key for item in request.stream_references]}"
            )
            result = self._service.handle(request)
            if request.operation.name == "INFER":
                performance = dict(result.performance)
                self.get_logger().info(
                    "[IBROBOT_INFERENCE][INFERENCE_TIMING] "
                    f"request={request.request_id} "
                    f"request_start_mono_ns={performance.get('request_start_monotonic_ns', 0)} "
                    f"assembly_start_mono_ns={performance.get('stream_assembly_start_monotonic_ns', 0)} "
                    f"assembly_end_mono_ns={performance.get('stream_assembly_end_monotonic_ns', 0)} "
                    f"inference_start_mono_ns={performance.get('inference_start_monotonic_ns', 0)} "
                    f"inference_end_mono_ns={performance.get('inference_end_monotonic_ns', 0)} "
                    f"result_mono_ns={performance.get('result_monotonic_ns', 0)} "
                    f"backend_latency_ms={result.backend_latency_ms:.6f}"
                )
            if request.operation.name == "INFER" and self._stream_manager is not None:
                for status in self._stream_manager.statuses():
                    self.get_logger().info(
                        "[IBROBOT_INFERENCE][VIDEO_RECEIVER] "
                        f"request={request.request_id} observation={status.observation_key} "
                        f"stream={status.stream_id} ready={status.ready} "
                        f"received_packets={status.received_packets} decoded_frames={status.decoded_frames} "
                        f"lost_packets={status.lost_packets} dropped_packets={status.dropped_packets} "
                        f"queue_overflow={status.receiver_queue_overflow_drops} "
                        f"sequence_gaps={status.sequence_gap_events} reordered={status.reordered_packets} "
                        f"recovery_keyframes={status.recovery_keyframes} jitter_ns={status.jitter_ns} "
                        f"receive_mono_ns={status.receive_monotonic_ns} "
                        f"decode_start_mono_ns={status.decode_start_monotonic_ns} "
                        f"decode_end_mono_ns={status.decode_end_monotonic_ns} "
                        f"decoded_capture_ns={status.last_decoded_capture_timestamp_ns}"
                    )
        except Exception as exc:
            result = decode_failure_result(message, exc, self._fingerprint, self._config.pipeline_id)
            self.get_logger().error(f"distributed request decode failed: {exc}")
        try:
            result_message = result_to_message(result)
        except Exception as exc:
            result = transport_failure_result(
                message,
                exc,
                self._fingerprint,
                self._config.pipeline_id,
                code="encode_failed",
                stage="encode",
                operation=request.operation if request is not None else None,
            )
            result_message = result_to_message(result)
            self.get_logger().error(f"distributed result encode failed: {exc}")
        self._result_pub.publish(result_message)

    def destroy_node(self) -> None:
        try:
            self._service.close()
        except Exception as exc:
            self.get_logger().error(f"cloud pipeline shutdown failed: {exc}")
        super().destroy_node()


def _read_config() -> tuple[CloudNodeConfig, str]:
    reader = Node("_pure_inference_param_reader")
    defaults: dict[str, object] = {
        "pipeline_id": "policy",
        "model_path": "",
        "deployment": "cpu",
        "request_timeout": 5.0,
        "runtime_options_json": "{}",
        "node_name": "inference_policy_cloud",
        "request_topic": "/inference/policy/request",
        "result_topic": "/inference/policy/result",
        "heartbeat_topic": "/inference/policy/heartbeat",
        "robot_config_path": "",
        "video_descriptor_topic": "/inference/policy/video/descriptors",
        "video_status_topic": "/inference/policy/video/status",
    }
    for name, default in defaults.items():
        reader.declare_parameter(name, default)
    values = {name: reader.get_parameter(name).value for name in defaults}
    reader.destroy_node()
    node_name = str(values.pop("node_name"))
    return CloudNodeConfig(**values), node_name


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: PureInferenceNode | None = None
    dependencies = None
    try:
        config, node_name = _read_config()
        dependencies = build_policy_runtime_dependencies()
        node = PureInferenceNode(
            config,
            node_name=node_name,
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if dependencies is not None:
            dependencies.providers.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
