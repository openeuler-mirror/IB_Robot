#!/usr/bin/env python3
"""
LeRobot policy inference node - Supports both Monolithic and Distributed modes.

This node provides ROS 2 integration for LeRobot policies with two execution modes:

    Mode A: Monolithic (default)
    ─────────────────────────────
    DispatchInfer Action Server → InferenceCoordinator (Pre → Infer → Post)

    All processing in one process for zero-copy tensor passing.

    Mode B: Distributed (cloud-edge)
    ────────────────────────────────
    Edge Node (this node):
        - Preprocessing (local CPU)
        - Publish to cloud via /preprocessed/batch
        - Await cloud result (async)
        - Postprocessing (local CPU)
        - Return to action_dispatch

    Cloud Node (pure_inference_node):
        - Subscribe /preprocessed/batch
        - GPU inference
        - Publish to /inference/action

ROS Interface Compatibility (MUST NOT CHANGE):
- Action: ibrobot_msgs/action/DispatchInfer
- Parameters: name, node_name, model_type, repo_id, checkpoint,
              contract_path, device, frequency, use_header_time,
              execution_mode, request_timeout, cloud_inference_topic,
              cloud_result_topic
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import rclpy.action
import torch
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

from ibrobot_msgs.action import DispatchInfer
from ibrobot_msgs.msg import VariantsList
from inference_service.core import (
    CoordinatorResult,
    InferenceCoordinator,
    resolve_device,
)
from inference_service.core.postprocessor import TensorPostprocessor
from inference_service.core.preprocessor import TensorPreprocessor
from robot_config.contract_utils import (
    SpecView,
    StreamBuffer,
    decode_value,
    feature_from_spec,
    iter_specs,
    qos_profile_from_dict,
    stamp_from_header_ns,
    zero_pad,
)
from robot_config.tracing_utils import create_trace_logger
from robot_config.utils import (
    build_joint_conversion_table,
    resolve_calibration_path_from_config,
    resolve_gripper_joints_from_config,
    resolve_joint_names_from_config,
)
from tensormsg.converter import TensorMsgConverter

_trace = create_trace_logger("ib_trace.policy")


def _trace_shape(value: Any) -> str:
    """Return a compact shape string for tracing fields."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 0:
            return "scalar"
        return "x".join(str(int(dim)) for dim in shape)
    if isinstance(value, list | tuple):
        return str(len(value))
    return "scalar"


@dataclass
class _SubState:
    """Subscription state for a single observation stream."""

    spec: SpecView
    buf: StreamBuffer


@dataclass
class _NodeConfig:
    """Configuration parsed from ROS parameters."""

    name: str = "lerobot_policy"
    node_name: str = "lerobot_policy_node"
    model_type: str = "lerobot_policy"
    repo_id: str | None = None
    checkpoint: str | None = None
    robot_config_path: str | None = None
    lerobot_norm_mode: str = "range_m100_100"
    device: str = "auto"
    frequency: float = 10.0
    use_header_time: bool = True
    execution_mode: str = "monolithic"
    request_timeout: float = 5.0
    cloud_inference_topic: str = "/preprocessed/batch"
    cloud_result_topic: str = "/inference/action"


class LeRobotPolicyNode(Node):
    """
    LeRobot policy inference node with DispatchInfer Action Server.

    Supports two execution modes:

    1. Monolithic (default): All inference in one process (zero-copy)
    2. Distributed: Edge preprocessing → Cloud inference → Edge postprocessing

    The distributed mode is transparent to action_dispatch - it always sees
    the same Action Server interface.

    Observation Filtering:
    ----------------------
    The node automatically filters observations based on the model's config.json.
    - robot_config.yaml defines ALL available observations (hardware config)
    - Model's config.json defines REQUIRED input_features
    - Node only subscribes to observations that match input_features

    This allows a single robot_config.yaml to support multiple models with
    different observation requirements.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config.get("node_name", "lerobot_policy_node"))

        self._config = _NodeConfig(**{k: v for k, v in model_config.items() if k in _NodeConfig.__dataclass_fields__})

        self.get_logger().info(f"Initializing {self._config.name} node")
        self.get_logger().info(f"Execution mode: {self._config.execution_mode}")

        self._device = resolve_device(self._config.device)
        self.get_logger().info(f"Using device: {self._device}")

        self._last_inference_time: float | None = None
        self._inference_count = 0
        self._consecutive_timeouts = 0
        self._cloud_connected = False
        self._health_status = DiagnosticStatus.OK
        self._error_message = ""

        self._contract = None
        self._obs_specs: list[SpecView] = []
        self._obs_zero: dict[str, np.ndarray] = {}
        self._subs: dict[str, _SubState] = {}
        self._state_specs: list[SpecView] = []
        self._policy_config: dict | None = None
        self._required_inputs: set = set()  # Input features required by model
        self._joint_rad_limits: list = []  # populated in _load_contract()

        # Load model config first to get required inputs
        self._load_policy_config()

        if self._config.robot_config_path:
            self._load_contract(self._config.robot_config_path)
            self._setup_observation_subscriptions()

        if self._config.execution_mode == "distributed":
            self._setup_distributed_mode()
        else:
            self._setup_monolithic_mode()

        self._setup_publishers()

        self._setup_action_server()

        self._health_timer = self.create_timer(1.0, self._health_callback)

        mode_str = "distributed (edge proxy)" if self._config.execution_mode == "distributed" else "monolithic"
        self.get_logger().info(
            f"{self._config.name} node ready ({mode_str}): "
            f"policy_type={self._policy_type}, "
            f"chunk_size={self._chunk_size}"
        )

    def _load_policy_config(self):
        """Load model config.json to get required input_features."""
        import json

        policy_path = self._config.repo_id or self._config.checkpoint
        if not policy_path:
            self.get_logger().warn("No policy path provided, cannot load model config")
            return

        config_path = Path(policy_path) / "config.json"
        if not config_path.exists():
            self.get_logger().warn(f"Model config not found: {config_path}")
            return

        with open(config_path) as f:
            self._policy_config = json.load(f)

        # Extract required input features
        input_features = self._policy_config.get("input_features", {})
        self._required_inputs = set(input_features.keys())

        self.get_logger().info(f"Model requires {len(self._required_inputs)} input features:")
        for key in self._required_inputs:
            self.get_logger().info(f"  - {key}")

    def _load_contract(self, robot_config_path: str):
        """Load contract from robot_config YAML file and filter by model's input_features."""
        p = Path(robot_config_path)
        if not p.exists():
            raise RuntimeError(f"Robot config file not found: {robot_config_path}")

        from robot_config.loader import (
            build_contract_from_robot_config_dict,
            load_robot_config_dict,
        )

        robot_cfg = load_robot_config_dict(robot_config_path)
        self._contract = build_contract_from_robot_config_dict(robot_cfg)

        # Build joint conversion table from calibration file
        calib_file = resolve_calibration_path_from_config(robot_cfg)
        joint_names = resolve_joint_names_from_config(robot_cfg)
        gripper_joints = resolve_gripper_joints_from_config(robot_cfg) or ["6"]
        norm_mode = self._config.lerobot_norm_mode
        if calib_file and joint_names:
            self._joint_rad_limits = build_joint_conversion_table(
                calib_file,
                joint_names,
                gripper_joints,
                norm_mode=norm_mode,
            )
            self.get_logger().info(
                f"Loaded joint conversion table (mode={norm_mode}): {len(self._joint_rad_limits)} joints"
            )

            # Append base velocity normalization entries using physical units (rad/s).
            # The raw steps ↔ physical unit conversion is handled by lekiwi_hardware.
            # Here we only need the physical range for LeRobot [-100, +100] mapping.
            velocity_joints = robot_cfg.get("ros2_control", {}).get("velocity_joints", [])
            base_vel_max_rad = robot_cfg.get("ros2_control", {}).get("base_vel_max_rad", 0)
            if velocity_joints and base_vel_max_rad > 0:
                # (rad_min, rad_max, span, offset) with span=200, offset=-100
                # maps [-max_rad/s, +max_rad/s] → [-100, +100]
                for _vj in velocity_joints:
                    self._joint_rad_limits.append((-float(base_vel_max_rad), float(base_vel_max_rad), 200.0, -100.0))
                self.get_logger().info(
                    f"Appended {len(velocity_joints)} velocity joints "
                    f"(max_rad/s={base_vel_max_rad}) → total {len(self._joint_rad_limits)} entries"
                )
        else:
            self._joint_rad_limits = []
            self.get_logger().warn("Missing calib_file or joint_names; rad↔pct conversion disabled")

        # Get all observation specs from contract
        all_obs_specs = [s for s in iter_specs(self._contract) if not s.is_action]

        # Filter by model's required inputs
        if self._required_inputs:
            self._obs_specs = [s for s in all_obs_specs if s.key in self._required_inputs]
            skipped = len(all_obs_specs) - len(self._obs_specs)
            if skipped > 0:
                self.get_logger().info(f"Filtered observations: {len(self._obs_specs)} required, {skipped} skipped")
        else:
            # No model config, use all observations
            self._obs_specs = all_obs_specs

        self._state_specs = [s for s in self._obs_specs if s.key == "observation.state"]

        self._topic_to_qos = {}
        for obs in self._contract.observations or []:
            self._topic_to_qos[obs.topic] = obs.qos

        self.get_logger().info(
            f"Loaded contract with {len(self._obs_specs)} observation specs (from {len(all_obs_specs)} total)"
        )

    def _setup_observation_subscriptions(self):
        """Setup observation subscriptions from loaded contract."""
        if not self._contract:
            self.get_logger().warn("No contract loaded, skipping observation subscriptions")
            return

        from rosidl_runtime_py.utilities import get_message

        for s in self._obs_specs:
            k, meta, _ = feature_from_spec(s, use_videos=False)

            if s.key == "observation.state" and len(self._state_specs) > 1:
                dict_key = f"{s.key}_{s.topic.replace('/', '_')}"
            else:
                dict_key = s.key

            self._obs_zero[dict_key] = zero_pad(meta)

            msg_cls = get_message(s.ros_type)
            qos_dict = self._topic_to_qos.get(s.topic, {})
            qos = qos_profile_from_dict(qos_dict) or QoSProfile(depth=10)

            self.create_subscription(
                msg_cls,
                s.topic,
                lambda m, sv=s: self._obs_cb(m, sv),
                qos,
                callback_group=ReentrantCallbackGroup(),
            )

            tol_ns = int(max(0, getattr(s, "asof_tol_ms", 0)) * 1_000_000)

            self._subs[dict_key] = _SubState(
                spec=s,
                buf=StreamBuffer(
                    policy=getattr(s, "resample_policy", "hold"),
                    step_ns=int(1e9 / self._config.frequency),
                    tol_ns=tol_ns,
                ),
            )

        self.get_logger().info(f"Subscribed to {len(self._subs)} observation streams")

    def _setup_monolithic_mode(self):
        """Setup for monolithic (single-process) inference."""
        policy_path = self._config.repo_id or self._config.checkpoint
        if not policy_path:
            raise RuntimeError("LeRobotPolicyNode: 'repo_id' or 'checkpoint' is required")

        self._coordinator = InferenceCoordinator(
            policy_path=policy_path,
            device=str(self._device),
        )

        self._policy_type = self._coordinator.policy_type
        self._chunk_size = self._coordinator.chunk_size
        self._use_action_chunking = self._coordinator.use_action_chunking

        self._preprocessor = None
        self._postprocessor = None
        self._pending_requests: dict[str, Any] = {}
        self._pub_batch = None
        self._sub_result = None

    def _setup_distributed_mode(self):
        """Setup for distributed (cloud-edge) inference."""
        policy_path = self._config.repo_id or self._config.checkpoint
        if not policy_path:
            raise RuntimeError("LeRobotPolicyNode: 'repo_id' or 'checkpoint' is required")

        self._preprocessor = TensorPreprocessor(
            policy_path=policy_path,
            device=self._device,
        )
        self._postprocessor = TensorPostprocessor(
            policy_path=policy_path,
            device=self._device,
        )

        self._coordinator = None

        temp_engine = InferenceCoordinator(
            policy_path=policy_path,
            device="cpu",
        )
        self._policy_type = temp_engine.policy_type
        self._chunk_size = temp_engine.chunk_size
        self._use_action_chunking = temp_engine.use_action_chunking

        self._pending_requests: dict[str, Any] = {}

        self._pub_batch = self.create_publisher(
            VariantsList,
            self._config.cloud_inference_topic,
            10,
        )

        self._sub_result = self.create_subscription(
            VariantsList,
            self._config.cloud_result_topic,
            self._cloud_result_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            f"Distributed mode: "
            f"publishing to {self._config.cloud_inference_topic}, "
            f"subscribed to {self._config.cloud_result_topic}"
        )

    def _setup_publishers(self):
        """Setup ROS publishers."""
        self._action_pub = self.create_publisher(
            VariantsList,
            f"/actions/{self._config.name}",
            10,
        )

        self._health_pub = self.create_publisher(
            DiagnosticStatus,
            f"/{self._config.node_name}/health",
            10,
        )

    def _setup_action_server(self):
        """Setup DispatchInfer Action Server."""
        self._action_server = rclpy.action.ActionServer(
            self,
            DispatchInfer,
            "~/DispatchInfer",
            execute_callback=self._dispatch_infer_callback,
            goal_callback=lambda req: rclpy.action.GoalResponse.ACCEPT,
            cancel_callback=lambda handle: rclpy.action.CancelResponse.ACCEPT,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info("DispatchInfer Action Server ready")

    def _obs_cb(self, msg, spec: SpecView):
        """Observation callback - push to StreamBuffer."""
        recv_ts_ns = self.get_clock().now().nanoseconds
        use_header = (spec.stamp_src == "header") or self._config.use_header_time

        if use_header:
            ts = stamp_from_header_ns(msg)
            ts_ns = int(ts) if ts is not None else recv_ts_ns
        else:
            ts_ns = recv_ts_ns

        val = decode_value(spec.ros_type, msg, spec)
        if val is not None:
            if spec.key == "observation.state" and len(self._state_specs) > 1:
                dict_key = f"{spec.key}_{spec.topic.replace('/', '_')}"
            else:
                dict_key = spec.key
            transport_ms = max(0.0, (recv_ts_ns - ts_ns) / 1_000_000)
            _trace.info(
                "[obs_receive] key=%s topic=%s source=%s source_ts_ns=%d recv_ts_ns=%d transport_ms=%.3f shape=%s",
                dict_key,
                spec.topic,
                "header" if use_header else "receive",
                ts_ns,
                recv_ts_ns,
                transport_ms,
                _trace_shape(val),
            )
            self._subs[dict_key].buf.push(ts_ns, val)

    def _sample_obs_frame(
        self,
        sample_t_ns: int | None = None,
        request_id: str = "",
    ) -> dict[str, Any]:
        """Sample observation frame at a given timestamp."""
        if sample_t_ns is None:
            sample_t_ns = self.get_clock().now().nanoseconds

        obs_frame: dict[str, Any] = {}
        ready_count = 0
        missing_count = 0

        if len(self._state_specs) > 1:
            parts = []
            for sv in self._state_specs:
                key = f"{sv.key}_{sv.topic.replace('/', '_')}"
                buf = self._subs[key].buf if key in self._subs else None
                v = buf.sample(sample_t_ns) if buf is not None else None
                source_ts_ns = buf.last_ts if buf is not None and buf.last_ts is not None else 0
                sampled = v if v is not None else self._obs_zero.get(key, np.zeros(1))
                parts.append(sampled)
                ready = int(v is not None)
                ready_count += ready
                missing_count += int(not ready)
                age_ms = max(0.0, (sample_t_ns - source_ts_ns) / 1_000_000) if source_ts_ns else -1.0
                _trace.info(
                    "[obs_sample] request_id=%s key=%s topic=%s ready=%d "
                    "age_ms=%.3f source_ts_ns=%d sample_ts_ns=%d shape=%s",
                    request_id,
                    key,
                    sv.topic,
                    ready,
                    age_ms,
                    source_ts_ns,
                    sample_t_ns,
                    _trace_shape(sampled),
                )
            obs_frame["observation.state"] = np.concatenate(parts)

        for key, st in self._subs.items():
            if key.startswith("observation.state_") and len(self._state_specs) > 1:
                continue
            v = st.buf.sample(sample_t_ns)
            sampled = v if v is not None else self._obs_zero.get(key, np.zeros(1))
            obs_frame[key] = sampled
            ready = int(v is not None)
            ready_count += ready
            missing_count += int(not ready)
            source_ts_ns = st.buf.last_ts if st.buf.last_ts is not None else 0
            age_ms = max(0.0, (sample_t_ns - source_ts_ns) / 1_000_000) if source_ts_ns else -1.0
            _trace.info(
                "[obs_sample] request_id=%s key=%s topic=%s ready=%d "
                "age_ms=%.3f source_ts_ns=%d sample_ts_ns=%d shape=%s",
                request_id,
                key,
                st.spec.topic,
                ready,
                age_ms,
                source_ts_ns,
                sample_t_ns,
                _trace_shape(sampled),
            )

        _trace.info(
            "[obs_frame] request_id=%s sample_ts_ns=%d ready=%d missing=%d total=%d",
            request_id,
            sample_t_ns,
            ready_count,
            missing_count,
            ready_count + missing_count,
        )

        return obs_frame

    # -- Unit conversion: ros2_control radians <-> LeRobot units ------------
    # LeRobot datasets store joint positions in normalised units that depend
    # on the motor norm mode selected during training:
    #   RANGE_M100_100:  arm [-100,+100], gripper [0,100]
    #   DEGREES:         centred degrees = (tick - mid) * 360 / 4095
    #   NONE:            no conversion (pass-through)
    # ros2_control publishes /joint_states in radians.
    #
    # Observation path (input):  rad  →  _rad_to_lerobot  →  model
    # Action path     (output):  model  →  _lerobot_to_rad  →  rad
    #
    # _joint_rad_limits is populated at runtime from the calibration JSON
    # in _load_contract() via build_joint_conversion_table().

    def _rad_to_lerobot(self, state: np.ndarray) -> np.ndarray:
        """Convert radians to LeRobot units (observation input path)."""
        if not self._joint_rad_limits:
            return state  # no calibration loaded, pass-through
        out = state.astype(np.float64).copy()
        for i, (rmin, rmax, span, offset) in enumerate(self._joint_rad_limits):
            if i < len(state):
                out[i] = (state[i] - rmin) / (rmax - rmin) * span + offset
        return out

    def _lerobot_to_rad(self, action: np.ndarray) -> np.ndarray:
        """Convert LeRobot units to radians (action output path)."""
        if not self._joint_rad_limits:
            return action  # no calibration loaded, pass-through
        out = action.astype(np.float64).copy()
        for i, (rmin, rmax, span, offset) in enumerate(self._joint_rad_limits):
            if i < action.shape[-1]:
                out[..., i] = (action[..., i] - offset) / span * (rmax - rmin) + rmin
        return out

    def _dispatch_infer_callback(self, goal_handle):
        """Execute inference requested by dispatcher."""
        goal = goal_handle.request
        obs_timestamp_ns = goal.obs_timestamp.sec * 10**9 + goal.obs_timestamp.nanosec
        request_id = goal.inference_id or ""

        try:
            obs_frame = self._sample_obs_frame(obs_timestamp_ns, request_id=request_id)

            # Convert observation.state from radians (ros2_control) to LeRobot
            # units to match the dataset statistics used for normalization.
            if "observation.state" in obs_frame:
                obs_frame["observation.state"] = self._rad_to_lerobot(obs_frame["observation.state"])

            if self._config.execution_mode == "distributed":
                result = self._execute_distributed(obs_frame, request_id)
            else:
                result = self._execute_monolithic(obs_frame, request_id)

            # Convert action chunk from LeRobot units back to radians
            # so the dispatcher receives Contract-declared units (radians).
            action_rad = result.action
            if self._joint_rad_limits:
                action_np = result.action.detach().cpu().numpy() if torch.is_tensor(result.action) else result.action
                action_rad = torch.from_numpy(self._lerobot_to_rad(action_np)).float()

            publish_start = time.perf_counter()
            action_msg = self._create_action_msg(action_rad)
            self._action_pub.publish(action_msg)
            publish_latency_ms = (time.perf_counter() - publish_start) * 1000.0
            _trace.info(
                "[action_chunk_publish] request_id=%s chunk_size=%d publish_ms=%.2f shape=%s",
                request_id,
                result.chunk_size,
                publish_latency_ms,
                _trace_shape(action_rad),
            )

            response = DispatchInfer.Result()
            response.action_chunk = action_msg
            response.chunk_size = result.chunk_size
            response.success = True
            response.message = "OK"
            response.inference_latency_ms = result.total_latency_ms

            goal_handle.succeed()
            self._last_inference_time = time.time()
            self._inference_count += 1

            if self._consecutive_timeouts > 0:
                self.get_logger().info(f"Cloud inference connected (after {self._consecutive_timeouts} timeouts)")
                self._consecutive_timeouts = 0
            self._cloud_connected = True

            if self._inference_count == 1:
                mode_tag = "distributed" if self._config.execution_mode == "distributed" else "monolithic"
                self.get_logger().info(
                    f"✓ First inference complete ({mode_tag}): "
                    f"total={result.total_latency_ms:.1f}ms "
                    f"(pre={result.preprocess_latency_ms:.1f}ms, "
                    f"inf={result.inference_latency_ms:.1f}ms, "
                    f"post={result.postprocess_latency_ms:.1f}ms)"
                )

            self.get_logger().debug(
                f"Inference #{self._inference_count}: {goal.inference_id}, "
                f"latency: {result.total_latency_ms:.1f}ms "
                f"(pre: {result.preprocess_latency_ms:.1f}ms, "
                f"inf: {result.inference_latency_ms:.1f}ms, "
                f"post: {result.postprocess_latency_ms:.1f}ms)"
            )

            return response

        except TimeoutError:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts == 1 and not self._cloud_connected:
                self.get_logger().warn(
                    "Waiting for cloud inference node... "
                    f"(timeout={self._config.request_timeout}s, "
                    f"topic=/preprocessed/batch → /inference/action)"
                )
            elif self._consecutive_timeouts == 1 and self._cloud_connected:
                self.get_logger().warn("Cloud inference disconnected, waiting for reconnect...")
                self._cloud_connected = False
            else:
                self.get_logger().debug(f"Cloud timeout #{self._consecutive_timeouts}")

            response = DispatchInfer.Result()
            response.action_chunk = VariantsList()
            response.chunk_size = 0
            response.success = False
            response.message = "Inference timeout - cloud did not respond"
            response.inference_latency_ms = self._config.request_timeout * 1000.0

            goal_handle.abort()
            self._health_status = DiagnosticStatus.WARN
            self._error_message = "Cloud inference timeout"

            return response

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}\n{traceback.format_exc()}")

            response = DispatchInfer.Result()
            response.action_chunk = VariantsList()
            response.chunk_size = 0
            response.success = False
            response.message = str(e)
            response.inference_latency_ms = 0.0

            goal_handle.abort()
            self._health_status = DiagnosticStatus.ERROR
            self._error_message = str(e)

            return response

    def _execute_monolithic(
        self,
        obs_frame: dict[str, Any],
        inference_id: str,
    ) -> CoordinatorResult:
        """Execute inference in monolithic mode (zero-copy)."""
        request_id = inference_id or ""
        _trace.info(
            "[inference_begin] request_id=%s model=%s device=%s",
            request_id,
            self._policy_type,
            self._device,
        )
        result = self._coordinator(obs_frame)
        _trace.info(
            "[inference_end] request_id=%s latency_ms=%.2f shape=%s",
            request_id,
            result.total_latency_ms,
            list(result.action.shape),
        )
        return result

    def _execute_distributed(
        self,
        obs_frame: dict[str, Any],
        inference_id: str,
    ) -> CoordinatorResult:
        """
        Execute inference in distributed mode.

        Flow:
        1. Preprocess locally (edge CPU)
        2. Publish to cloud with request_id
        3. Block thread and wait for cloud result (with timeout)
        4. Postprocess locally (edge CPU)
        """
        total_start = time.perf_counter()
        request_id = inference_id or str(uuid.uuid4())

        _trace.info("[preprocess_begin] request_id=%s", request_id)
        preprocess_start = time.perf_counter()
        batch = self._preprocessor(obs_frame)
        preprocess_latency = (time.perf_counter() - preprocess_start) * 1000.0
        _trace.info(
            "[preprocess_end] request_id=%s latency_ms=%.2f",
            request_id,
            preprocess_latency,
        )
        batch["task.request_id"] = [request_id]

        msg = TensorMsgConverter.to_variant(batch)

        event = threading.Event()
        self._pending_requests[request_id] = [event, None]
        _trace.info(
            "[edge_publish] request_id=%s topic=%s",
            request_id,
            self._config.cloud_inference_topic,
        )
        self._pub_batch.publish(msg)
        self.get_logger().debug(f"Published batch to cloud, request_id={request_id}")

        success = event.wait(timeout=self._config.request_timeout)

        if not success:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Inference timeout for request {request_id}")

        req_data = self._pending_requests.pop(request_id, None)
        if not req_data or req_data[1] is None:
            raise RuntimeError(f"Event set but no cloud result found for {request_id}")

        cloud_result = req_data[1]

        inference_latency = cloud_result.get("_latency_ms", 0.0)
        _trace.info(
            "[edge_receive] request_id=%s latency_ms=%.2f",
            request_id,
            inference_latency,
        )

        _trace.info("[postprocess_begin] request_id=%s", request_id)
        postprocess_start = time.perf_counter()
        action = self._postprocessor(cloud_result["action"])
        postprocess_latency = (time.perf_counter() - postprocess_start) * 1000.0
        _trace.info(
            "[postprocess_end] request_id=%s latency_ms=%.2f",
            request_id,
            postprocess_latency,
        )
        total_latency = (time.perf_counter() - total_start) * 1000.0

        return CoordinatorResult(
            action=action,
            chunk_size=self._chunk_size,
            total_latency_ms=total_latency,
            preprocess_latency_ms=preprocess_latency,
            inference_latency_ms=inference_latency,
            postprocess_latency_ms=postprocess_latency,
            policy_type=self._policy_type,
        )

    def _cloud_result_callback(self, msg: VariantsList):
        """
        Callback for cloud inference results.

        Matches the result to the pending request using action.request_id
        and completes the corresponding Event.
        """
        try:
            batch = TensorMsgConverter.from_variant(msg, self._device)

            req_list = batch.pop("action.request_id", None)
            request_id = req_list[0] if req_list and isinstance(req_list, list) else None

            if request_id is None:
                self.get_logger().warn("Received cloud result without action.request_id")
                return

            if request_id in self._pending_requests:
                req = self._pending_requests[request_id]
                req[1] = batch
                req[0].set()
                self.get_logger().debug(f"Cloud result received for request_id={request_id}")
            else:
                self.get_logger().warn(f"No pending request found for request_id={request_id}")

        except Exception as e:
            self.get_logger().error(f"Error processing cloud result: {e}")

    def _create_action_msg(self, action: torch.Tensor) -> VariantsList:
        """Create VariantsList message from action tensor."""
        from std_msgs.msg import MultiArrayDimension

        from ibrobot_msgs.msg import Variant

        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()

        msg = VariantsList()
        variant = Variant()
        variant.key = "action"
        variant.type = "float_32_array"

        array_msg = variant.float_32_array
        array_msg.layout.dim = []

        for i, dim in enumerate(action.shape):
            dim_msg = MultiArrayDimension()
            dim_msg.label = f"dim_{i}"
            dim_msg.size = int(dim)
            dim_msg.stride = 1 if i == len(action.shape) - 1 else int(np.prod(action.shape[i + 1 :]))
            array_msg.layout.dim.append(dim_msg)

        array_msg.data = action.flatten().tolist()
        msg.variants.append(variant)
        return msg

    def _health_callback(self):
        """Health monitoring timer callback."""
        if self._last_inference_time is None:
            self._health_status = DiagnosticStatus.OK
        elif time.time() - self._last_inference_time > self._config.frequency * 2:
            self._health_status = DiagnosticStatus.WARN
            self._error_message = f"No inference for {time.time() - self._last_inference_time:.1f}s"
        else:
            self._health_status = DiagnosticStatus.OK

        health_msg = DiagnosticStatus()
        health_msg.level = self._health_status
        health_msg.name = self._config.node_name
        health_msg.message = self._error_message or f"{self._config.name} operating normally"
        health_msg.hardware_id = self._config.node_name
        health_msg.values = [
            KeyValue(key="inference_count", value=str(self._inference_count)),
            KeyValue(key="model_type", value=self._config.model_type),
            KeyValue(key="policy_type", value=self._policy_type),
            KeyValue(key="chunk_size", value=str(self._chunk_size)),
            KeyValue(key="execution_mode", value=self._config.execution_mode),
        ]

        self._health_pub.publish(health_msg)


def main() -> None:
    """Main entry point for LeRobot policy node."""
    rclpy.init()

    try:
        temp_node = Node("_param_reader")
        for p in [
            "name",
            "node_name",
            "model_type",
            "repo_id",
            "checkpoint",
            "robot_config_path",
            "lerobot_norm_mode",
            "device",
            "frequency",
            "use_header_time",
            "execution_mode",
            "request_timeout",
            "cloud_inference_topic",
            "cloud_result_topic",
        ]:
            if p == "execution_mode":
                default = "monolithic"
            elif p == "request_timeout":
                default = 5.0
            elif p == "cloud_inference_topic":
                default = "/preprocessed/batch"
            elif p == "cloud_result_topic":
                default = "/inference/action"
            elif p in ["repo_id", "checkpoint", "robot_config_path"]:
                default = ""
            elif p == "lerobot_norm_mode":
                default = "range_m100_100"
            elif p == "device":
                default = "auto"
            elif p == "frequency":
                default = 10.0
            elif p == "use_header_time":
                default = True
            elif p == "node_name":
                default = "lerobot_policy_node"
            else:
                default = "lerobot_policy"
            temp_node.declare_parameter(p, default)

        config = {
            p: temp_node.get_parameter(p).value or None
            for p in [
                "name",
                "node_name",
                "model_type",
                "repo_id",
                "checkpoint",
                "robot_config_path",
            ]
        }
        config["device"] = temp_node.get_parameter("device").value
        config["lerobot_norm_mode"] = temp_node.get_parameter("lerobot_norm_mode").value
        config["frequency"] = temp_node.get_parameter("frequency").value
        config["use_header_time"] = temp_node.get_parameter("use_header_time").value
        config["execution_mode"] = temp_node.get_parameter("execution_mode").value
        config["request_timeout"] = temp_node.get_parameter("request_timeout").value
        config["cloud_inference_topic"] = temp_node.get_parameter("cloud_inference_topic").value
        config["cloud_result_topic"] = temp_node.get_parameter("cloud_result_topic").value
        temp_node.destroy_node()

        node = LeRobotPolicyNode(config)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        node.get_logger().info("LeRobot policy node started")

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
