from __future__ import annotations

import sys
import unittest
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import rclpy
from launch_ros.actions import Node

sys.path.insert(0, str(Path(__file__).parent))

from smoke_support import (
    assert_successful_action,
    prepare_smoke_runtime,
    publish_required_observations,
    send_inference_when_observations_ready,
    wait_for_distributed_ready,
)


def generate_test_description():
    runtime = prepare_smoke_runtime()
    pipeline_id = "distributed"
    request_topic = f"/inference/{pipeline_id}/request"
    result_topic = f"/inference/{pipeline_id}/result"
    heartbeat_topic = f"/inference/{pipeline_id}/heartbeat"
    edge = Node(
        package="inference_service",
        executable="pipeline_policy_node",
        name=f"inference_{pipeline_id}",
        output="screen",
        env=runtime.process_environment,
        parameters=[
            {
                "pipeline_id": pipeline_id,
                "model_path": str(runtime.bundle_path),
                "deployment": "cpu",
                "execution_mode": "distributed",
                "request_timeout": 5.0,
                "robot_config_path": str(runtime.robot_config_path),
                "action_server": f"/inference/{pipeline_id}/dispatch",
                "reset_service": f"/inference/{pipeline_id}/reset",
                "health_topic": f"/inference/{pipeline_id}/health",
                "action_topic": f"/actions/{pipeline_id}",
                "request_topic": request_topic,
                "result_topic": result_topic,
                "heartbeat_topic": heartbeat_topic,
            }
        ],
    )
    cloud = Node(
        package="inference_service",
        executable="pure_inference_node",
        name=f"inference_{pipeline_id}_cloud",
        output="screen",
        env=runtime.process_environment,
        parameters=[
            {
                "pipeline_id": pipeline_id,
                "model_path": str(runtime.bundle_path),
                "deployment": "cpu",
                "request_timeout": 5.0,
                "request_topic": request_topic,
                "result_topic": result_topic,
                "heartbeat_topic": heartbeat_topic,
            }
        ],
    )
    return (
        launch.LaunchDescription(
            [
                edge,
                cloud,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"edge": edge, "cloud": cloud},
    )


class TestDistributedLaunch(unittest.TestCase):
    def test_handshake_and_request(self) -> None:
        rclpy.init()
        node = rclpy.create_node("inference_distributed_smoke_client")
        stop_observations = None
        try:
            health = wait_for_distributed_ready(node, "/inference/distributed/health")
            self.assertEqual("distributed", health["pipeline_id"])
            stop_observations = publish_required_observations(node, expected_subscriptions=1)
            result, request_id = send_inference_when_observations_ready(
                node, "/inference/distributed/dispatch", "distributed-request"
            )
            assert_successful_action(self, result, "distributed", request_id)
        finally:
            if stop_observations is not None:
                stop_observations()
            node.destroy_node()
            rclpy.shutdown()


@launch_testing.post_shutdown_test()
class TestDistributedShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info, edge, cloud) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, process=edge)
        launch_testing.asserts.assertExitCodes(proc_info, process=cloud)
