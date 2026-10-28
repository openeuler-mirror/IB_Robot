from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState

from ibrobot_msgs.action import DispatchInfer
from ibrobot_msgs.msg import (
    DistributedInferenceRequest,
    InferencePipelineStatus,
    VideoStreamDescriptor,
    VideoStreamStatus,
)
from tensormsg.converter import TensorMsgConverter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-id", default="video_distributed")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--interrupt-images", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("video_e2e_client")
    state: dict[str, object] = {
        "edge_ready": False,
        "cloud_ready": False,
        "descriptor": None,
        "sender_status": None,
        "receiver_status": None,
        "request": None,
        "health": None,
    }
    base = f"/inference/{args.pipeline_id}"

    def heartbeat_callback(message: InferencePipelineStatus) -> None:
        if message.role == InferencePipelineStatus.ROLE_EDGE:
            state["edge_ready"] = message.ready
        elif message.role == InferencePipelineStatus.ROLE_CLOUD:
            state["cloud_ready"] = message.ready

    def request_callback(message: DistributedInferenceRequest) -> None:
        if message.operation == DistributedInferenceRequest.OP_INFER:
            state["request"] = message

    def video_status_callback(message: VideoStreamStatus) -> None:
        if message.sent_packets > 0:
            state["sender_status"] = message
        if message.received_packets > 0:
            state["receiver_status"] = message

    subscriptions = [
        node.create_subscription(InferencePipelineStatus, f"{base}/heartbeat", heartbeat_callback, 10),
        node.create_subscription(
            VideoStreamDescriptor, f"{base}/video/descriptors", lambda message: state.update(descriptor=message), 10
        ),
        node.create_subscription(VideoStreamStatus, f"{base}/video/status", video_status_callback, 10),
        node.create_subscription(DistributedInferenceRequest, f"{base}/request", request_callback, 10),
        node.create_subscription(DiagnosticStatus, f"{base}/health", lambda message: state.update(health=message), 10),
    ]
    joint_publisher = node.create_publisher(JointState, "/ci_smoke/joint_states", 10)
    image_publisher = node.create_publisher(Image, "/ci_smoke/camera/top", 10)
    image_enabled = True

    def publish() -> None:
        stamp = node.get_clock().now().to_msg()
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = [f"joint_{index}" for index in range(6)]
        joints.position = [0.0] * 6
        joint_publisher.publish(joints)
        if not image_enabled:
            return
        image = Image()
        image.header.stamp = stamp
        image.height = 48
        image.width = 64
        image.encoding = "rgb8"
        image.step = 64 * 3
        y, x = np.indices((48, 64))
        image.data = np.stack((x * 4, y * 5, (x + y) * 2), axis=-1).astype(np.uint8).tobytes()
        image_publisher.publish(image)

    timer = node.create_timer(0.05, publish)
    action_client = ActionClient(node, DispatchInfer, f"{base}/dispatch")
    try:
        _wait_until(
            node,
            lambda: joint_publisher.get_subscription_count() >= 1 and image_publisher.get_subscription_count() >= 1,
            args.timeout,
            "edge observation subscriptions",
        )
        _wait_until(
            node,
            lambda: (
                bool(state["edge_ready"])
                and bool(state["cloud_ready"])
                and state["descriptor"] is not None
                and state["sender_status"] is not None
                and state["sender_status"].timestamp_mapping_valid
                and state["receiver_status"] is not None
                and state["receiver_status"].ready
                and state["receiver_status"].decoded_frames > 0
            ),
            args.timeout,
            "DDS session and RTP sender readiness",
        )
        success = _send_when_streams_ready(node, action_client, args.timeout)
        _wait_until(node, lambda: state["request"] is not None, args.timeout, "distributed request")

        request = state["request"]
        descriptor = state["descriptor"]
        sender_status = state["sender_status"]
        receiver_status = state["receiver_status"]
        health = state["health"]
        tensor_keys = sorted(TensorMsgConverter.from_variant(request.tensors))
        if tensor_keys != ["observation.state"]:
            raise AssertionError(f"DDS request unexpectedly contains {tensor_keys}")
        evidence = {
            "success": True,
            "pipeline_id": success.pipeline_id,
            "request_id": success.request_id,
            "chunk_size": success.chunk_size,
            "session_id": request.session_id,
            "session_generation": request.session_generation,
            "stream_observation_keys": list(request.stream_observation_keys),
            "stream_ids": list(request.stream_ids),
            "dds_tensor_keys": tensor_keys,
            "encoder_backend": descriptor.encoder_backend,
            "decoder_status_backend": receiver_status.selected_backend,
            "encoded_frames": sender_status.encoded_frames,
            "decoded_frames": receiver_status.decoded_frames,
            "sent_packets": sender_status.sent_packets,
            "received_packets": receiver_status.received_packets,
            "lost_packets": receiver_status.lost_packets,
            "dropped_packets": receiver_status.dropped_packets,
            "health": {item.key: item.value for item in health.values} if health is not None else {},
        }

        if args.interrupt_images:
            image_enabled = False
            time.sleep(0.7)
            interrupted = _send_goal(node, action_client, "cross-host-interrupted", args.timeout)
            evidence["interrupted_success"] = interrupted.success
            evidence["interrupted_error_code"] = interrupted.error.code
            if interrupted.success or interrupted.error.code != "observation_not_ready":
                raise AssertionError(
                    f"interrupted inference did not fail closed: {interrupted.success}, {interrupted.error.code}"
                )
        print(json.dumps(evidence, sort_keys=True))
    finally:
        action_client.destroy()
        node.destroy_timer(timer)
        node.destroy_publisher(image_publisher)
        node.destroy_publisher(joint_publisher)
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def _send_goal(node, client: ActionClient, request_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    if not client.wait_for_server(timeout_sec=timeout):
        raise AssertionError("inference action server did not become available")
    goal = DispatchInfer.Goal()
    goal.inference_id = request_id
    goal_future = client.send_goal_async(goal)
    _spin_future(node, goal_future, deadline, request_id)
    goal_handle = goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise AssertionError(f"inference goal {request_id!r} was rejected")
    result_future = goal_handle.get_result_async()
    _spin_future(node, result_future, deadline, request_id)
    return result_future.result().result


def _send_when_streams_ready(node, client: ActionClient, timeout: float):
    deadline = time.monotonic() + timeout
    attempt = 0
    while True:
        result = _send_goal(node, client, f"cross-host-success-{attempt}", max(0.1, deadline - time.monotonic()))
        if result.success:
            return result
        if result.error.code != "observation_not_ready" or time.monotonic() >= deadline:
            raise AssertionError(f"successful inference failed: {result.error.code}: {result.message}")
        attempt += 1
        rclpy.spin_once(node, timeout_sec=0.1)


def _spin_future(node, future, deadline: float, description: str) -> None:
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        raise AssertionError(f"timed out waiting for {description}")


def _wait_until(node, predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not predicate():
        raise AssertionError(f"timed out waiting for {description}")


if __name__ == "__main__":
    main()
