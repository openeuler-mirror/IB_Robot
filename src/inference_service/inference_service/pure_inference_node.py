#!/usr/bin/env python3
"""
Pure inference node for distributed/composed mode.

This node:
- Subscribes to preprocessed VariantsList
- Runs pure inference using PureInferenceEngine
- Publishes raw action as VariantsList

Designed to work with LeRobotPolicyNode in distributed mode.

Request-Response Matching:
- If input batch contains "_request_id", it will be passed through to output
- This enables the edge node to match responses to pending requests
"""

from __future__ import annotations

import time
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.msg import VariantsList
from inference_service.core._policy_config import read_local_policy_config_device
from inference_service.core.pure_inference_engine import (
    PureInferenceEngine,
)
from robot_config.tracing_utils import create_trace_logger
from tensormsg.converter import TensorMsgConverter

_trace = create_trace_logger("ib_trace.inference")


class PureInferenceNode(Node):
    """
    Pure inference node without preprocessing/postprocessing.

    Subscribes: /preprocessed/batch (VariantsList)
    Publishes: /inference/action (VariantsList)

    Passes through "_request_id" from input to output for request matching.
    """

    def __init__(
        self,
        node_name: str = "pure_inference",
        policy_path: str | None = None,
        input_topic: str = "/preprocessed/batch",
        output_topic: str = "/inference/action",
        device: str = "auto",
    ):
        super().__init__(node_name)

        if not policy_path:
            raise ValueError("policy_path is required for PureInferenceNode")

        self._input_topic = input_topic
        self._output_topic = output_topic
        self._inference_backend = str(device)
        model_config_device = read_local_policy_config_device(policy_path) or "<unset>"

        self.get_logger().info(f"Loading policy from {policy_path} with inference_backend={self._inference_backend}...")
        self._engine = PureInferenceEngine(policy_path=policy_path, device=device)
        self.get_logger().info(
            f"Engine loaded: policy_type={self._engine.policy_type}, "
            f"backend_type={self._engine.backend_type or self._inference_backend}, "
            f"chunk_size={self._engine.chunk_size}"
        )
        self.get_logger().info(
            f"Runtime device contract: model_config_device={model_config_device}, "
            f"runtime_backend={self._engine.backend_type or self._inference_backend}, "
            f"runtime_tensor_device={self._engine.device}; "
            f"model_config_device is training metadata"
        )

        self._sub = self.create_subscription(
            VariantsList,
            input_topic,
            self._inference_cb,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self._pub = self.create_publisher(VariantsList, output_topic, 10)

        self._inference_count = 0
        self._total_latency_ms = 0.0
        self._last_log_time = 0.0
        self._log_interval_s = 5.0

        self.get_logger().info(
            f"PureInferenceNode ready: input={input_topic}, output={output_topic}, "
            f"inference_backend={self._inference_backend}, tensor_device={self._engine.device}"
        )
        self.get_logger().info("Waiting for preprocessed batches from edge node...")

    def _inference_cb(self, msg: VariantsList):
        """Run inference on preprocessed input."""
        try:
            start_time = time.perf_counter()

            batch = TensorMsgConverter.from_variant(msg, self._engine._device)

            req_list = batch.pop("task.request_id", None)
            request_id = req_list[0] if req_list and isinstance(req_list, list) else None

            _trace.info(
                "[inference_begin] request_id=%s model=%s inference_backend=%s tensor_device=%s",
                request_id or "",
                self._engine.policy_type,
                self._inference_backend,
                self._engine.device,
            )

            result = self._engine(batch)

            inference_latency_ms = (time.perf_counter() - start_time) * 1000.0

            _trace.info(
                "[inference_end] request_id=%s latency_ms=%.2f shape=%s",
                request_id or "",
                inference_latency_ms,
                list(result.action.shape),
            )

            out_batch: dict[str, Any] = {"action": result.action}

            if request_id is not None:
                out_batch["action.request_id"] = [request_id]

            out_batch["_latency_ms"] = inference_latency_ms

            out_msg = TensorMsgConverter.to_variant(out_batch)
            self._pub.publish(out_msg)

            self._inference_count += 1
            self._total_latency_ms += inference_latency_ms

            if self._inference_count == 1:
                self.get_logger().info(
                    f"✓ First inference completed: "
                    f"latency={inference_latency_ms:.1f}ms, "
                    f"action_shape={list(result.action.shape)}"
                )

            now = time.monotonic()
            if now - self._last_log_time >= self._log_interval_s:
                avg_latency = self._total_latency_ms / self._inference_count
                self.get_logger().info(
                    f"[stats] count={self._inference_count}, avg={avg_latency:.1f}ms, last={inference_latency_ms:.1f}ms"
                )
                self._last_log_time = now

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()

    from rclpy.node import Node

    temp = Node("_pure_inference_param_reader")
    temp.declare_parameter("policy_path", "")
    temp.declare_parameter("input_topic", "/preprocessed/batch")
    temp.declare_parameter("output_topic", "/inference/action")
    temp.declare_parameter("device", "auto")

    if not temp.has_parameter("use_sim_time"):
        temp.declare_parameter("use_sim_time", False)

    params = {
        "policy_path": temp.get_parameter("policy_path").value or None,
        "input_topic": temp.get_parameter("input_topic").value,
        "output_topic": temp.get_parameter("output_topic").value,
        "device": temp.get_parameter("device").value,
    }
    temp.destroy_node()

    node = PureInferenceNode(
        node_name="pure_inference",
        policy_path=params["policy_path"],
        input_topic=params["input_topic"],
        output_topic=params["output_topic"],
        device=params["device"],
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
