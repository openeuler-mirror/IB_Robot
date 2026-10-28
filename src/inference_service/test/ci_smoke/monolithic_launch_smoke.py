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
)


def _pipeline_node(runtime, pipeline_id: str) -> Node:
    return Node(
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
                "execution_mode": "monolithic",
                "request_timeout": 5.0,
                "robot_config_path": str(runtime.robot_config_path),
                "action_server": f"/inference/{pipeline_id}/dispatch",
                "reset_service": f"/inference/{pipeline_id}/reset",
                "health_topic": f"/inference/{pipeline_id}/health",
                "action_topic": f"/actions/{pipeline_id}",
            }
        ],
    )


def generate_test_description():
    runtime = prepare_smoke_runtime()
    primary = _pipeline_node(runtime, "primary")
    secondary = _pipeline_node(runtime, "secondary")
    return (
        launch.LaunchDescription(
            [
                primary,
                secondary,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"primary": primary, "secondary": secondary},
    )


class TestMonolithicLaunch(unittest.TestCase):
    def test_two_cpu_pipelines_accept_requests(self) -> None:
        rclpy.init()
        node = rclpy.create_node("inference_monolithic_smoke_client")
        stop_observations = None
        try:
            stop_observations = publish_required_observations(node, expected_subscriptions=2)
            primary, primary_request_id = send_inference_when_observations_ready(
                node, "/inference/primary/dispatch", "primary-request"
            )
            secondary, secondary_request_id = send_inference_when_observations_ready(
                node, "/inference/secondary/dispatch", "secondary-request"
            )
            assert_successful_action(self, primary, "primary", primary_request_id)
            assert_successful_action(self, secondary, "secondary", secondary_request_id)
        finally:
            if stop_observations is not None:
                stop_observations()
            node.destroy_node()
            rclpy.shutdown()


@launch_testing.post_shutdown_test()
class TestMonolithicShutdown(unittest.TestCase):
    def test_exit_codes(self, proc_info, primary, secondary) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, process=primary)
        launch_testing.asserts.assertExitCodes(proc_info, process=secondary)
