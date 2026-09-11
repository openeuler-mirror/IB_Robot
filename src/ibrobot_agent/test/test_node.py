from __future__ import annotations

import json
import time

import rclpy
from rclpy.context import Context
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_agent.contracts import RequestKey
from ibrobot_agent.node import IncubationAgentNode


class FakeCatalog:
    def get_status(self):
        return {
            "robot_name": "so101_single_arm",
            "active_control_mode": "moveit_planning",
            "control_plane_ready": True,
            "motion_authorized": False,
            "registry_epoch": "epoch",
            "registry_generation": 1,
            "registry_digest": "digest",
        }

    def get_catalog(self, status):
        return {"robot_name": "so101_single_arm", "skills": [], "pose_names": ["home", "zero"]}


def test_node_binds_runtime_identity_and_answers_read_only(tmp_path):
    context = Context()
    rclpy.init(context=context)
    node = IncubationAgentNode(
        catalog_port=FakeCatalog(),
        context=context,
        parameter_overrides=[
            Parameter("ledger_path", value=str(tmp_path / "requests.sqlite3")),
            Parameter("conversation_path", value=str(tmp_path / "conversation.sqlite3")),
            Parameter("deployment_lock_path", value=str(tmp_path / "agent.lock")),
            Parameter("robot_scope", value="so101_single_arm"),
            Parameter("channel_id", value="bound-channel"),
            Parameter("principal_id", value="bound-principal"),
        ],
    )
    try:
        ready = node._ready_callback(Trigger.Request(), Trigger.Response())
        assert ready.success

        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": "node-request-1",
                "session_id": "node-session-1",
                "channel_id": "spoofed-channel",
                "principal_id": "spoofed-principal",
                "robot_scope": "spoofed-robot",
                "text": "当前状态",
            }
        )
        node._request_callback(message)
        key = RequestKey("so101_single_arm", "bound-channel", "bound-principal", "node-request-1")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and node._service.get_request(key).state != "ANSWERED":
            time.sleep(0.01)
        record = node._service.get_request(key)
        assert record.state == "ANSWERED"
        assert "运动未授权" in record.terminal.message
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown(context=context)
