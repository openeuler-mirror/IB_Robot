#!/usr/bin/env python3
"""Standalone RTP video + DDS action/state recording node.

Composes ComputeVideoStreamManager (decode=False, with recorders) and
EpisodeRecorderServer in a single process. Used for cross-device data
collection where video arrives via RTP unicast from edge, while action/state
arrive via DDS.

Architecture:
  - H264RtpReceiver (decode=False) × N → H264StreamRecorder × N
  - VideoRecordingCoordinator (manages all recorders)
  - EpisodeRecorderServer (rosbag writer + RecordEpisode action server)
  - Episode dir layout: episode_XXXXXX/{*.mcap, observation.*.h264, observation.*.h264.json}

NOT for same-device DDS recording (use episode_recorder directly) or
pure inference (use pure_inference_node).
"""

from __future__ import annotations

import sys
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from dataset_tools.episode_recorder import EpisodeRecorderServer
from ibrobot_msgs.msg import InferencePipelineStatus, VideoStreamDescriptor, VideoStreamStatus
from inference_service.compute_video_streams import ComputeVideoStreamManager
from inference_service.distributed import CloudSession, PipelineStatus
from inference_service.distributed.ros_protocol import (
    status_from_message,
    status_to_message,
    video_descriptor_from_message,
    video_status_from_message,
    video_status_to_message,
)
from inference_service.video_recording_coordinator import VideoRecordingCoordinator
from robot_config.contract_utils import contract_fingerprint, iter_specs
from robot_config.loader import load_robot_config, resolve_robot_config_path


class RecordingCloudHandshake:
    """Provide the cloud heartbeat required to activate edge RTP streams.

    The edge only binds a video session after it observes a ROLE_CLOUD status
    (see pipeline_policy_node._cloud_status_callback). Until then
    EdgeVideoStreamManager.descriptors() returns an empty tuple, so the edge
    publishes no descriptors, the receiver never learns the stream parameters,
    and the recorded .h264 stays zero bytes with nothing logged on either side.

    Recording runs no inference backend, so this stands in for the cloud peer
    and reports ready unconditionally.
    """

    def __init__(self, pipeline_id: str) -> None:
        self._pipeline_id = pipeline_id
        self._session: CloudSession | None = None
        self._lock = threading.Lock()

    def observe_edge(self, status: PipelineStatus) -> PipelineStatus:
        if status.identity.pipeline_id != self._pipeline_id:
            raise ValueError(
                f"recording pipeline mismatch: expected {self._pipeline_id!r}, got {status.identity.pipeline_id!r}"
            )
        with self._lock:
            if self._session is None:
                self._session = CloudSession(status.identity)
            self._session.observe_edge(status, backend_ready=True)
            return self._status_locked()

    def status(self) -> PipelineStatus | None:
        with self._lock:
            if self._session is None:
                return None
            return self._status_locked()

    def _status_locked(self) -> PipelineStatus:
        assert self._session is not None
        return self._session.status(
            backend_ready=True,
            backend_state="recording",
            reset_supported=False,
            cancellation_supported=False,
        )


class RecordingNodeOrchestrator(Node):
    """Orchestrate RTP video stream manager and episode recorder lifecycle."""

    def __init__(self) -> None:
        super().__init__("recording_node_orchestrator")

        # Load robot config and contract
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("pipeline_id", "policy")
        self.declare_parameter("heartbeat_topic", "/inference/policy/heartbeat")
        self.declare_parameter("video_descriptor_topic", "/inference/policy/video/descriptors")
        self.declare_parameter("video_status_topic", "/inference/policy/video/status")
        self.declare_parameter("runtime_options_json", "{}")
        config_path_param = self.get_parameter("robot_config_path").value
        if not config_path_param:
            raise RuntimeError("recording_node requires robot_config_path parameter")

        resolved_path = resolve_robot_config_path(None, config_path_param)
        pipeline_id = str(self.get_parameter("pipeline_id").value)
        heartbeat_topic = str(self.get_parameter("heartbeat_topic").value)
        descriptor_topic = str(self.get_parameter("video_descriptor_topic").value)
        status_topic = str(self.get_parameter("video_status_topic").value)
        self.get_logger().info(f"Loading robot config: {resolved_path}")
        robot_config = load_robot_config(resolved_path)
        contract = robot_config.to_contract()

        # Validate: require at least one RTP observation
        observation_specs = [spec for spec in iter_specs(contract) if not spec.is_action]
        rtp_observations = [spec for spec in observation_specs if spec.transport and spec.transport.mode == "rtp"]
        if not rtp_observations:
            raise RuntimeError(
                "recording_node requires at least one observation with transport.mode=rtp; "
                "use episode_recorder directly for DDS-only recording"
            )

        # Validate: action/state observations must use header timestamp
        for obs in observation_specs:
            if obs.transport and obs.transport.mode == "rtp":
                continue  # RTP observations are fine (sidecar has RTP-mapped timestamps)
            # DDS observations: check stamp_src
            if obs.stamp_src == "receive":
                self.get_logger().warning(
                    f"Observation '{obs.key}' uses stamp_src=receive; this may cause video/state "
                    "misalignment in cross-device deployments; use stamp_src=header for RTP recording"
                )

        # Create coordinator (manages recorders for all RTP streams)
        self._coordinator = VideoRecordingCoordinator()
        self._handshake = RecordingCloudHandshake(pipeline_id)

        # Create stream manager (RTP receivers + recorders, decode=False)
        self._stream_manager = ComputeVideoStreamManager(
            pipeline_id=pipeline_id,
            session_id="recording_session",
            session_generation=1,
            contract_fingerprint=contract_fingerprint(contract),
            deployment_fingerprint="recording",
            observation_specs=rtp_observations,
            rate_hz=contract.rate_hz,
            n_obs_steps=1,
            recording_coordinator=self._coordinator,
            decode=False,  # Recording-only: skip decode, save CPU
            validate_deployment_fingerprint=False,
        )
        self._session_identity = ("recording_session", 1)
        descriptor_qos = QoSProfile(
            depth=max(1, len(rtp_observations)),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            InferencePipelineStatus,
            heartbeat_topic,
            self._edge_status_callback,
            descriptor_qos,
            callback_group=callback_group,
        )
        self._heartbeat_pub = self.create_publisher(InferencePipelineStatus, heartbeat_topic, descriptor_qos)
        self._heartbeat_timer = self.create_timer(0.5, self._publish_cloud_status)
        self.create_subscription(
            VideoStreamDescriptor,
            descriptor_topic,
            self._video_descriptor_callback,
            descriptor_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            VideoStreamStatus,
            status_topic,
            self._video_status_callback,
            10,
            callback_group=callback_group,
        )
        self._video_status_pub = self.create_publisher(VideoStreamStatus, status_topic, 10)
        self._video_status_timer = self.create_timer(0.25, self._publish_video_status)

        self.get_logger().info(
            f"RTP video stream manager initialized with {len(rtp_observations)} streams (decode=False)"
        )

        # Note: EpisodeRecorderServer will be created and wired externally in main()
        # (it's a separate Node, not a sub-component)

    def get_coordinator(self) -> VideoRecordingCoordinator:
        """Return the coordinator for wiring into EpisodeRecorderServer."""
        return self._coordinator

    def _edge_status_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_EDGE:
            return
        try:
            status = self._handshake.observe_edge(status_from_message(message))
            self._heartbeat_pub.publish(status_to_message(status, stamp=self.get_clock().now().to_msg()))
        except Exception as exc:
            self.get_logger().error(f"invalid edge handshake status: {exc}")

    def _publish_cloud_status(self) -> None:
        status = self._handshake.status()
        if status is None:
            return
        self._heartbeat_pub.publish(status_to_message(status, stamp=self.get_clock().now().to_msg()))

    def _video_descriptor_callback(self, message: VideoStreamDescriptor) -> None:
        try:
            descriptor = video_descriptor_from_message(message)
            identity = (descriptor.session_id, descriptor.session_generation)
            if identity != self._session_identity:
                self._stream_manager.reset_session(*identity)
                self._session_identity = identity
            self._stream_manager.observe_descriptor(descriptor)
        except Exception as exc:
            self.get_logger().error(f"invalid video stream descriptor: {exc}")

    def _video_status_callback(self, message: VideoStreamStatus) -> None:
        try:
            self._stream_manager.observe_status(
                video_status_from_message(message),
                receive_time_ns=self.get_clock().now().nanoseconds,
            )
        except Exception as exc:
            self.get_logger().error(f"invalid video stream status: {exc}")

    def _publish_video_status(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for status in self._stream_manager.statuses():
            self._video_status_pub.publish(video_status_to_message(status, stamp=stamp))

    def destroy_node(self) -> None:
        self._stream_manager.close()
        super().destroy_node()


def main(args=None):
    """Entry point for recording_node."""
    rclpy.init(args=args)

    try:
        # Create orchestrator node (loads config, creates stream manager + coordinator)
        orchestrator = RecordingNodeOrchestrator()

        # Create episode recorder node (rosbag writer + action server)
        episode_recorder = EpisodeRecorderServer()

        # Wire coordinator into episode recorder (in-process direct call)
        episode_recorder.set_video_recording_coordinator(orchestrator.get_coordinator())

        # Spin both nodes in a multi-threaded executor
        executor = MultiThreadedExecutor()
        executor.add_node(orchestrator)
        executor.add_node(episode_recorder)

        orchestrator.get_logger().info("Recording node ready (RTP video + DDS action/state)")

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            orchestrator.destroy_node()
            episode_recorder.destroy_node()

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
