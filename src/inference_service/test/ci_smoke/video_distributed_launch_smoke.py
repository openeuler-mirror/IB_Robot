from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).parent))

from smoke_support import (  # noqa: E402
    assert_successful_action,
    prepare_smoke_runtime,
    publish_video_observations,
    send_inference,
    send_inference_when_observations_ready,
)

_VIDEO_ENCODER_BACKEND = os.environ.get("IBROBOT_SMOKE_VIDEO_ENCODER_BACKEND", "software")

from ibrobot_msgs.msg import (  # noqa: E402
    DistributedInferenceRequest,
    InferencePipelineStatus,
    VideoStreamDescriptor,
    VideoStreamStatus,
)
from tensormsg.converter import TensorMsgConverter  # noqa: E402


def generate_test_description():
    runtime = prepare_smoke_runtime(video_transport=True, video_encoder_backend=_VIDEO_ENCODER_BACKEND)
    pipeline_id = "video_distributed"
    topics = {
        "request_topic": f"/inference/{pipeline_id}/request",
        "result_topic": f"/inference/{pipeline_id}/result",
        "heartbeat_topic": f"/inference/{pipeline_id}/heartbeat",
        "video_descriptor_topic": f"/inference/{pipeline_id}/video/descriptors",
        "video_status_topic": f"/inference/{pipeline_id}/video/status",
    }
    shared = {
        "pipeline_id": pipeline_id,
        "model_path": str(runtime.bundle_path),
        "deployment": "cpu",
        "request_timeout": 5.0,
        "robot_config_path": str(runtime.robot_config_path),
        **topics,
    }
    edge = Node(
        package="inference_service",
        executable="pipeline_policy_node",
        name=f"inference_{pipeline_id}",
        output="screen",
        env=runtime.process_environment,
        parameters=[
            {
                **shared,
                "execution_mode": "distributed",
                "action_server": f"/inference/{pipeline_id}/dispatch",
                "reset_service": f"/inference/{pipeline_id}/reset",
                "health_topic": f"/inference/{pipeline_id}/health",
                "action_topic": f"/actions/{pipeline_id}",
            }
        ],
    )
    cloud = Node(
        package="inference_service",
        executable="pure_inference_node",
        name=f"inference_{pipeline_id}_cloud",
        output="screen",
        env=runtime.process_environment,
        parameters=[shared],
    )
    return (
        launch.LaunchDescription([edge, cloud, launch_testing.actions.ReadyToTest()]),
        {"edge": edge, "cloud": cloud},
    )


class TestVideoDistributedLaunch(unittest.TestCase):
    def test_dds_control_and_rtp_data_plane(self) -> None:
        rclpy.init()
        node = rclpy.create_node("inference_video_distributed_smoke_client")
        pipeline_id = "video_distributed"
        state = {
            "edge_ready": False,
            "cloud_ready": False,
            "descriptor": None,
            "video_status": None,
            "request": None,
            "health": None,
        }

        def heartbeat_callback(message: InferencePipelineStatus) -> None:
            if message.role == InferencePipelineStatus.ROLE_EDGE:
                state["edge_ready"] = message.ready
            elif message.role == InferencePipelineStatus.ROLE_CLOUD:
                state["cloud_ready"] = message.ready

        def descriptor_callback(message: VideoStreamDescriptor) -> None:
            state["descriptor"] = message

        def video_status_callback(message: VideoStreamStatus) -> None:
            state["video_status"] = message

        def request_callback(message: DistributedInferenceRequest) -> None:
            if message.operation == DistributedInferenceRequest.OP_INFER:
                state["request"] = message

        def health_callback(message: DiagnosticStatus) -> None:
            state["health"] = message

        subscriptions = [
            node.create_subscription(
                InferencePipelineStatus,
                f"/inference/{pipeline_id}/heartbeat",
                heartbeat_callback,
                10,
            ),
            node.create_subscription(
                VideoStreamDescriptor,
                f"/inference/{pipeline_id}/video/descriptors",
                descriptor_callback,
                10,
            ),
            node.create_subscription(
                VideoStreamStatus,
                f"/inference/{pipeline_id}/video/status",
                video_status_callback,
                10,
            ),
            node.create_subscription(
                DistributedInferenceRequest,
                f"/inference/{pipeline_id}/request",
                request_callback,
                10,
            ),
            node.create_subscription(
                DiagnosticStatus,
                f"/inference/{pipeline_id}/health",
                health_callback,
                10,
            ),
        ]
        stop_observations = None
        try:
            stop_images, stop_observations = publish_video_observations(node, expected_subscriptions=1)
            self._spin_until(
                node,
                lambda: (
                    state["edge_ready"]
                    and state["cloud_ready"]
                    and state["descriptor"] is not None
                    and state["video_status"] is not None
                    and state["video_status"].timestamp_mapping_valid
                    and state["video_status"].sent_packets > 0
                ),
                "DDS session and RTP sender readiness",
            )

            result, request_id = send_inference_when_observations_ready(
                node,
                f"/inference/{pipeline_id}/dispatch",
                "video-distributed-request",
            )
            assert_successful_action(self, result, pipeline_id, request_id)
            self._spin_until(node, lambda: state["request"] is not None, "distributed request observation")

            descriptor = state["descriptor"]
            request = state["request"]
            video_status = state["video_status"]
            self.assertEqual(request.session_id, descriptor.session_id)
            self.assertEqual(request.session_generation, descriptor.session_generation)
            self.assertEqual(["observation.images.top"], list(request.stream_observation_keys))
            self.assertEqual(["top"], list(request.stream_ids))
            self.assertGreater(request.observation_timestamp.sec, 0)
            self.assertEqual(
                {"observation.state"},
                set(TensorMsgConverter.from_variant(request.tensors)),
                "RTP image payload must not be duplicated in the DDS request",
            )
            self.assertEqual(_VIDEO_ENCODER_BACKEND, descriptor.encoder_backend)
            self.assertEqual(_VIDEO_ENCODER_BACKEND, video_status.selected_backend)
            self.assertTrue(video_status.keyframe_ready)
            self.assertGreater(video_status.encoded_frames, 0)
            self.assertGreater(video_status.sent_packets, 0)

            health_values = {item.key: item.value for item in state["health"].values}
            video_health = json.loads(health_values["video_stream.observation.images.top"])
            self.assertEqual("rtp", video_health["mode"])
            self.assertTrue(video_health["ready"])

            stop_images()
            time.sleep(0.7)
            interrupted = send_inference(
                node,
                f"/inference/{pipeline_id}/dispatch",
                "video-distributed-interrupted",
            )
            self.assertFalse(interrupted.success)
            self.assertEqual("observation_not_ready", interrupted.error.code)
        finally:
            if stop_observations is not None:
                stop_observations()
            for subscription in subscriptions:
                node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()

    @staticmethod
    def _spin_until(node, predicate, description: str, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not predicate():
            raise AssertionError(f"timed out waiting for {description}")


@launch_testing.post_shutdown_test()
class TestVideoDistributedShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info, edge, cloud) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, process=edge)
        launch_testing.asserts.assertExitCodes(proc_info, process=cloud)
