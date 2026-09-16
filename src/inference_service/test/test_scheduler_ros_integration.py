"""Real ROS action closure for Global -> pipeline Open/Dispatch/Close."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

from action_dispatch.safe_stop import build_safe_stop_plan
from action_dispatch.scheduled_action_dispatcher_node import DispatcherState, ScheduledActionDispatcherNode
from ibrobot_msgs.action import (
    CloseInferenceSession,
    ClosePipelineBinding,
    DispatchInfer,
    DispatchPipelineBinding,
    OpenInferenceSession,
    OpenPipelineBinding,
    ScheduledDispatchInfer,
)
from ibrobot_msgs.msg import InferenceOutcome, InferenceServingStatus, InferenceWorkCapacity
from inference_service import pipeline_policy_node as pipeline_policy_module
from inference_service.backends import BackendCapabilities
from inference_service.global_inference_scheduler_node import GlobalInferenceSchedulerNode
from inference_service.pipeline_policy_node import PipelineNodeConfig, PipelinePolicyNode
from inference_service.runtime_composition import build_policy_runtime_dependencies
from robot_config.contract_utils import ActionSpec, Contract, ObservationSpec, iter_specs
from robot_config.inference_runtime_options import effective_latency_runtime_options


@pytest.fixture
def runtime_dependencies():
    dependencies = build_policy_runtime_dependencies()
    try:
        yield {"registry_set": dependencies.registry_set, "providers": dependencies.providers}
    finally:
        dependencies.providers.close()
        if rclpy.ok():
            rclpy.shutdown()


def _wait_future(future, timeout: float = 5.0):
    done = threading.Event()
    future.add_done_callback(lambda _future: done.set())
    assert done.wait(timeout), "ROS future timed out"
    return future.result()


class _PipelineServer(Node):
    def __init__(self, endpoints: dict[str, str], identity: dict[str, str]) -> None:
        super().__init__("scheduler_test_pipeline")
        self.identity = identity
        self.calls: list[str] = []
        group = ReentrantCallbackGroup()
        self.open_server = ActionServer(
            self,
            OpenPipelineBinding,
            endpoints["open"],
            execute_callback=self._open,
            callback_group=group,
        )
        self.dispatch_server = ActionServer(
            self,
            DispatchPipelineBinding,
            endpoints["dispatch"],
            execute_callback=self._dispatch,
            callback_group=group,
        )
        self.close_server = ActionServer(
            self,
            ClosePipelineBinding,
            endpoints["close"],
            execute_callback=self._close,
            callback_group=group,
        )
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(InferenceServingStatus, endpoints["status"], qos)

    def publish_status(self) -> None:
        status = InferenceServingStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.pipeline_id = "policy"
        status.boot_id = str(uuid.uuid4())
        status.sequence = 1
        status.state = InferenceServingStatus.IDLE
        status.deployment_fingerprint = self.identity["deployment"]
        status.runtime_policy_fingerprint = self.identity["runtime"]
        status.pipeline_compatibility_fingerprint = self.identity["compatibility"]
        status.configured_hardware_resource_id = "ascend:0"
        status.runtime_hardware_resource_id = "ascend:0"
        status.hardware_priority_levels = 8
        status.scheduling_capability_schema_version = 1
        status.stage_policy = "SEQUENTIAL"
        status.supports_priority_zero_deadline_admission = True
        status.max_supported_public_priority = 7
        for work_class in (InferenceWorkCapacity.SESSION_CONTROL, InferenceWorkCapacity.ACTION_GENERATION):
            capacity = InferenceWorkCapacity()
            capacity.work_class = work_class
            capacity.max_in_flight = 1
            capacity.accepting_requests = True
            status.capacities.append(capacity)
        self.status_publisher.publish(status)

    def _open(self, goal_handle):
        self.calls.append("open")
        goal = goal_handle.request
        result = OpenPipelineBinding.Result()
        result.success = True
        result.session_id = goal.session_id
        result.logical_generation = goal.logical_generation
        result.binding_id = goal.binding_id
        result.binding_incarnation = goal.binding_incarnation
        result.operation_id = goal.operation_id
        result.boot_id = goal.expected_boot_id
        result.pipeline_id = "policy"
        result.pipeline_generation = 11
        result.deployment_fingerprint = self.identity["deployment"]
        result.runtime_policy_fingerprint = self.identity["runtime"]
        result.outcome.value = InferenceOutcome.COMPLETED
        goal_handle.succeed()
        return result

    def _dispatch(self, goal_handle):
        self.calls.append("dispatch")
        goal = goal_handle.request
        result = DispatchPipelineBinding.Result()
        result.success = True
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.logical_generation = goal.logical_generation
        result.binding_id = goal.binding_id
        result.binding_incarnation = goal.binding_incarnation
        result.operation_id = goal.operation_id
        result.boot_id = goal.expected_boot_id
        result.pipeline_id = "policy"
        result.pipeline_generation = goal.expected_pipeline_generation
        result.deployment_fingerprint = self.identity["deployment"]
        result.runtime_policy_fingerprint = self.identity["runtime"]
        result.chunk_size = 1
        result.outcome.value = InferenceOutcome.COMPLETED
        goal_handle.succeed()
        return result

    def _close(self, goal_handle):
        self.calls.append("close")
        goal = goal_handle.request
        result = ClosePipelineBinding.Result()
        result.success = True
        result.session_id = goal.session_id
        result.pipeline_id = "policy"
        result.logical_generation = goal.logical_generation
        result.binding_id = goal.binding_id
        result.binding_incarnation = goal.binding_incarnation
        result.operation_id = goal.operation_id
        result.boot_id = goal.expected_boot_id
        result.closed_pipeline_generation = goal.expected_pipeline_generation
        result.drained_generation = goal.expected_pipeline_generation + 1
        result.outcome.value = InferenceOutcome.COMPLETED
        goal_handle.succeed()
        return result


def _parameters(candidate: dict, endpoints: dict[str, str]) -> list[Parameter]:
    values = {
        "readiness_endpoint": endpoints["readiness"],
        "open_session_endpoint": endpoints["global_open"],
        "dispatch_endpoint": endpoints["global_dispatch"],
        "close_session_endpoint": endpoints["global_close"],
        "default_target_pipeline_id": candidate["pipeline_id"],
        "pipelines_json": json.dumps([candidate]),
        "default_open_timeout_ns": 2_000_000_000,
        "default_request_timeout_ns": 5_000_000_000,
        "status_stale_timeout_ns": 5_000_000_000,
        "clock_skew_tolerance_ns": 1_000_000_000,
        "goal_acceptance_timeout_ns": 1_000_000_000,
        "session_idle_timeout_ns": 10_000_000_000,
        "terminal_session_retention_ns": 1_000_000_000,
        "max_duplicate_waiters_per_request": 2,
        "max_product_requests_per_session": 10,
        "terminal_result_cache_entries": 10,
        "max_session_records": 10,
        "max_fallback_pipelines": 4,
        "profile_min_samples": 1,
        "profile_max_age_days": 30,
        "goal_acceptance_safety_margin_ms": 1,
        "dispatch_safety_margin_ms": 1,
        "max_prompt_bytes": 4096,
        "max_error_message_bytes": 1024,
        "max_error_details_bytes": 8192,
        "default_priority": 0,
    }
    return [Parameter(name, value=value) for name, value in values.items()]


def test_global_to_pipeline_action_closure(tmp_path, runtime_dependencies) -> None:
    suffix = f"test_{uuid.uuid4().hex}"
    base = f"/scheduler_test/{suffix}"
    endpoints = {
        "open": f"{base}/pipeline/open",
        "dispatch": f"{base}/pipeline/dispatch",
        "close": f"{base}/pipeline/close",
        "status": f"{base}/pipeline/status",
        "readiness": f"{base}/ready",
        "global_open": f"{base}/open",
        "global_dispatch": f"{base}/dispatch",
        "global_close": f"{base}/close",
    }
    identity = {
        "deployment": "d" * 64,
        "runtime": "r" * 64,
        "compatibility": "c" * 64,
        "profile": "f" * 64,
    }
    now_ns = time.time_ns()
    common_profile = {
        "deployment_fingerprint": identity["deployment"],
        "hardware_fingerprint": "a" * 64,
        "profile_compatibility_fingerprint": identity["profile"],
        "scope": "global_proxy",
        "hardware_priority": 0,
        "goal_acceptance_p999_ms": 1.0,
        "profiled_at_ns": now_ns,
        "sample_count": 1,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "closure_profiles": [
                    {
                        **common_profile,
                        "work_class": 1,
                        "closure_key": "session_open",
                        "input_contract_fingerprint": "",
                        "prompt_bytes_max": 0,
                        "latency_p99_ms": 2.0,
                    },
                    {
                        **common_profile,
                        "work_class": 2,
                        "closure_key": "full_infer",
                        "input_contract_fingerprint": identity["compatibility"],
                        "prompt_bytes_max": 4096,
                        "latency_p99_ms": 3.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = {
        "pipeline_id": "policy",
        "compatibility_group": "test_group",
        "hardware_resource_id": "ascend:0",
        "hardware_profile_fingerprint": "a" * 64,
        "deployment_fingerprint": identity["deployment"],
        "runtime_policy_fingerprint": identity["runtime"],
        "profile_compatibility_fingerprint": identity["profile"],
        "open_session": endpoints["open"],
        "dispatch": endpoints["dispatch"],
        "close_session": endpoints["close"],
        "serving_status": endpoints["status"],
        "profile_path": str(profile_path),
        "required": True,
        "public_capacity": {
            "session_control": {"max_in_flight": 1},
            "action_generation": {"max_in_flight": 1},
        },
    }

    rclpy.init()
    pipeline = _PipelineServer(endpoints, identity)
    scheduler = GlobalInferenceSchedulerNode(
        parameter_overrides=_parameters(candidate, endpoints), **runtime_dependencies
    )
    client_node = Node("scheduler_test_client")
    executor = MultiThreadedExecutor(num_threads=8)
    for node in (pipeline, scheduler, client_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        pipeline.publish_status()
        readiness = client_node.create_client(Trigger, endpoints["readiness"])
        assert readiness.wait_for_service(timeout_sec=2.0)
        ready_response = _wait_future(readiness.call_async(Trigger.Request()))
        assert ready_response.success, ready_response.message

        session_id = str(uuid.uuid4())
        open_client = ActionClient(client_node, OpenInferenceSession, endpoints["global_open"])
        assert open_client.wait_for_server(timeout_sec=2.0)
        open_goal = OpenInferenceSession.Goal()
        open_goal.session_id = session_id
        open_result = _wait_future(_wait_future(open_client.send_goal_async(open_goal)).get_result_async()).result
        assert open_result.success
        assert pipeline.calls == []

        dispatch_client = ActionClient(client_node, ScheduledDispatchInfer, endpoints["global_dispatch"])
        assert dispatch_client.wait_for_server(timeout_sec=2.0)
        dispatch_goal = ScheduledDispatchInfer.Goal()
        dispatch_goal.request_id = str(uuid.uuid4())
        dispatch_goal.session_id = session_id
        dispatch_goal.session_generation = open_result.session_generation
        dispatch_goal.target_pipeline_id = "policy"
        dispatch_goal.priority = 0
        deadline_ns = time.time_ns() + 3_000_000_000
        dispatch_goal.deadline.sec, dispatch_goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
        dispatch_result = _wait_future(
            _wait_future(dispatch_client.send_goal_async(dispatch_goal)).get_result_async()
        ).result
        assert dispatch_result.success
        assert pipeline.calls == ["open", "dispatch"]

        close_client = ActionClient(client_node, CloseInferenceSession, endpoints["global_close"])
        assert close_client.wait_for_server(timeout_sec=2.0)
        close_goal = CloseInferenceSession.Goal()
        close_goal.session_id = session_id
        close_goal.session_generation = open_result.session_generation
        close_deadline_ns = time.time_ns() + 3_000_000_000
        close_goal.deadline.sec, close_goal.deadline.nanosec = divmod(close_deadline_ns, 1_000_000_000)
        close_result = _wait_future(_wait_future(close_client.send_goal_async(close_goal)).get_result_async()).result
        assert close_result.success
        assert pipeline.calls == ["open", "dispatch", "close"]
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        for node in (client_node, scheduler, pipeline):
            node.destroy_node()
        rclpy.shutdown()


class _PolicyFeature:
    def __init__(self, feature_type: str, shape: tuple[int, ...]) -> None:
        self.feature_type = feature_type
        self.shape = shape

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return {"type": self.feature_type, "shape": list(self.shape)}


class _FakePipelineManager:
    @staticmethod
    def supports_priority_zero_deadline_admission(_pipeline_id: str) -> bool:
        return False

    def __init__(self) -> None:
        self.priorities: list[int] = []
        self.reset_count = 0
        self.frames: list[object] = []

    @staticmethod
    def capabilities(_pipeline_id: str) -> BackendCapabilities:
        return BackendCapabilities(max_in_flight_per_instance=1, hardware_resource_id="ascend:0")

    def infer(self, _pipeline_id: str, request):
        self.priorities.append(request.priority)
        return SimpleNamespace(
            action=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=1.0,
            total_latency_ms=2.0,
            metadata={},
        )

    def reset(self, _pipeline_id: str, _deadline) -> None:
        self.reset_count += 1

    def submit_frame(self, _pipeline_id: str, request, *, deadline=None):
        del deadline
        from concurrent.futures import Future

        self.frames.append(request)
        future = Future()
        future.set_result(len(self.frames))
        return future

    @staticmethod
    def close() -> None:
        return None


def test_disabled_pipeline_materializes_only_legacy_ros_runtime(tmp_path, monkeypatch, runtime_dependencies) -> None:
    suffix = uuid.uuid4().hex
    base = f"/scheduler_disabled/test_{suffix}"
    contract = Contract(
        name="scheduler_disabled",
        version=1,
        rate_hz=20.0,
        max_duration_s=10.0,
        observations=[
            ObservationSpec(
                key="observation.state",
                topic=f"{base}/joint_states",
                type="sensor_msgs/msg/JointState",
                selector={"names": ["1", "2"]},
            )
        ],
        actions=[
            ActionSpec(
                key="action",
                publish_topic=f"{base}/command",
                type="std_msgs/msg/Float64MultiArray",
                selector={"names": ["1", "2"]},
                safety_behavior="hold",
            )
        ],
        tasks=[],
        recording={},
    )
    policy = SimpleNamespace(
        n_obs_steps=1,
        policy_type="act",
        input_features={"observation.state": _PolicyFeature("STATE", (2,))},
        output_features={"action": _PolicyFeature("ACTION", (2,))},
        max_action_dimension=2,
    )
    manifest = SimpleNamespace(
        policy=policy,
        fingerprint="d" * 64,
        manifest=SimpleNamespace(bundle=SimpleNamespace(name="scheduler-disabled")),
        deployment=SimpleNamespace(backend="fake"),
    )
    manager = _FakePipelineManager()

    def _load_contract(node, _path: str) -> None:
        node._contract = contract
        node._frequency = contract.rate_hz
        node._obs_specs = [spec for spec in iter_specs(contract) if not spec.is_action]
        node._state_specs = [spec for spec in node._obs_specs if spec.key == "observation.state"]
        node._topic_to_qos = {}
        node._joint_rad_limits = []

    monkeypatch.setattr(pipeline_policy_module, "load_inference_manifest", lambda *_args, **_kwargs: manifest)
    manager_options = {}

    def _create_manager(*_args, **kwargs):
        manager_options.update(kwargs)
        return manager

    monkeypatch.setattr(pipeline_policy_module, "create_pipeline_manager", _create_manager)
    monkeypatch.setattr(PipelinePolicyNode, "_load_contract", _load_contract)

    config = PipelineNodeConfig(
        pipeline_id="policy",
        model_path=str(tmp_path),
        deployment="fake",
        execution_mode="monolithic",
        request_timeout=2.0,
        default_task="",
        runtime_options_json="{}",
        robot_config_path=str(tmp_path / "unused.yaml"),
        use_sim=False,
        action_server=f"{base}/dispatch",
        reset_service=f"{base}/reset",
        health_topic=f"{base}/health",
        action_topic=f"{base}/action",
        request_topic="",
        result_topic="",
        heartbeat_topic="",
    )

    rclpy.init()
    pipeline = PipelinePolicyNode(config, node_name=f"pipeline_disabled_{suffix}", **runtime_dependencies)
    try:
        assert pipeline._action_pub is not None
        assert pipeline._action_server is not None
        assert pipeline._reset_server is not None
        for scheduled_state in (
            "_frame_trigger_lock",
            "_frame_trigger_future",
            "_visual_trigger_epoch",
            "_visual_trigger_enabled",
            "_pending_visual_sample_time_ns",
            "_frame_trigger_priority",
            "_scheduled_operation_slots",
            "_scheduled_operation_capacity",
            "_pipeline_compatibility_fingerprint",
            "_session_controller",
            "_pipeline_ledger",
            "_serving_status_pub",
            "_serving_status_timer",
            "_boot_id",
            "_serving_sequence",
        ):
            assert not hasattr(pipeline, scheduled_state)
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()


def test_disabled_pipeline_preserves_legacy_dispatch_action_closure(
    tmp_path, monkeypatch, runtime_dependencies
) -> None:
    suffix = uuid.uuid4().hex
    base = f"/scheduler_legacy/test_{suffix}"
    action_server = f"{base}/dispatch"
    action_topic = f"{base}/action"
    contract = Contract(
        name="scheduler_legacy",
        version=1,
        rate_hz=20.0,
        max_duration_s=10.0,
        observations=[
            ObservationSpec(
                key="observation.state",
                topic=f"{base}/joint_states",
                type="sensor_msgs/msg/JointState",
                selector={"names": ["1", "2"]},
            )
        ],
        actions=[
            ActionSpec(
                key="action",
                publish_topic=f"{base}/command",
                type="std_msgs/msg/Float64MultiArray",
                selector={"names": ["1", "2"]},
                safety_behavior="hold",
            )
        ],
        tasks=[],
        recording={},
    )
    policy = SimpleNamespace(
        n_obs_steps=1,
        policy_type="act",
        input_features={"observation.state": _PolicyFeature("STATE", (2,))},
        output_features={"action": _PolicyFeature("ACTION", (2,))},
        max_action_dimension=2,
    )
    manifest = SimpleNamespace(
        policy=policy,
        fingerprint="d" * 64,
        manifest=SimpleNamespace(bundle=SimpleNamespace(name="scheduler-legacy")),
        deployment=SimpleNamespace(backend="fake"),
    )
    manager = _FakePipelineManager()

    def _load_contract(node, _path: str) -> None:
        node._contract = contract
        node._frequency = contract.rate_hz
        node._obs_specs = [spec for spec in iter_specs(contract) if not spec.is_action]
        node._state_specs = [spec for spec in node._obs_specs if spec.key == "observation.state"]
        node._topic_to_qos = {}
        node._joint_rad_limits = []

    monkeypatch.setattr(pipeline_policy_module, "load_inference_manifest", lambda *_args, **_kwargs: manifest)
    manager_options = {}

    def _create_manager(*_args, **kwargs):
        manager_options.update(kwargs)
        return manager

    monkeypatch.setattr(pipeline_policy_module, "create_pipeline_manager", _create_manager)
    monkeypatch.setattr(PipelinePolicyNode, "_load_contract", _load_contract)
    config = PipelineNodeConfig(
        pipeline_id="policy",
        model_path=str(tmp_path),
        deployment="fake",
        execution_mode="monolithic",
        request_timeout=2.0,
        default_task="",
        runtime_options_json="{}",
        robot_config_path=str(tmp_path / "unused.yaml"),
        use_sim=False,
        action_server=action_server,
        reset_service=f"{base}/reset",
        health_topic=f"{base}/health",
        action_topic=action_topic,
        request_topic="",
        result_topic="",
        heartbeat_topic="",
    )

    rclpy.init()
    pipeline = PipelinePolicyNode(config, node_name=f"pipeline_legacy_{suffix}", **runtime_dependencies)
    pipeline._sample_observations = lambda _sample_time, **_kwargs: {
        "observation.state": np.asarray([0.0, 0.0], dtype=np.float32)
    }
    client_node = Node(f"legacy_client_{suffix}")
    published_actions = []
    client_node.create_subscription(
        pipeline_policy_module.VariantsList,
        action_topic,
        published_actions.append,
        10,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(pipeline)
    executor.add_node(client_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        client = ActionClient(client_node, DispatchInfer, action_server)
        assert client.wait_for_server(timeout_sec=2.0)
        goal = DispatchInfer.Goal()
        goal.inference_id = str(uuid.uuid4())
        goal.obs_timestamp = client_node.get_clock().now().to_msg()
        goal_handle = _wait_future(client.send_goal_async(goal))
        assert goal_handle.accepted
        result = _wait_future(goal_handle.get_result_async()).result
        assert result.success
        assert result.chunk_size == 2
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not published_actions:
            time.sleep(0.01)
        assert published_actions
        assert manager.priorities == [0]
        assert not hasattr(pipeline, "_boot_id")
        assert not hasattr(pipeline, "_session_controller")
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        client_node.destroy_node()
        pipeline.destroy_node()
        rclpy.shutdown()


def test_real_pipeline_global_dispatcher_reaches_command_topic(tmp_path, monkeypatch, runtime_dependencies) -> None:
    suffix = uuid.uuid4().hex
    base = f"/scheduler_real/test_{suffix}"
    endpoints = {
        "open": f"{base}/pipeline/open",
        "dispatch": f"{base}/pipeline/dispatch",
        "close": f"{base}/pipeline/close",
        "status": f"{base}/pipeline/status",
        "readiness": f"{base}/ready",
        "global_open": f"{base}/open",
        "global_dispatch": f"{base}/dispatch",
        "global_close": f"{base}/close",
    }
    joint_topic = f"{base}/joint_states"
    command_topic = f"{base}/command"
    contract = Contract(
        name="scheduler_real",
        version=1,
        rate_hz=20.0,
        max_duration_s=10.0,
        observations=[
            ObservationSpec(
                key="observation.state",
                topic=joint_topic,
                type="sensor_msgs/msg/JointState",
                selector={"names": ["1", "2"]},
            )
        ],
        actions=[
            ActionSpec(
                key="action",
                publish_topic=command_topic,
                type="std_msgs/msg/Float64MultiArray",
                selector={"names": ["1", "2"]},
                safety_behavior="hold",
            )
        ],
        tasks=[],
        recording={},
    )
    policy = SimpleNamespace(
        n_obs_steps=1,
        policy_type="act",
        input_features={"observation.state": _PolicyFeature("STATE", (2,))},
        output_features={"action": _PolicyFeature("ACTION", (2,))},
        max_action_dimension=2,
    )
    manifest = SimpleNamespace(
        policy=policy,
        fingerprint="d" * 64,
        manifest=SimpleNamespace(bundle=SimpleNamespace(name="scheduler-real")),
        deployment=SimpleNamespace(backend="fake"),
    )
    manager = _FakePipelineManager()

    def _load_contract(node, _path: str) -> None:
        node._contract = contract
        node._frequency = contract.rate_hz
        node._obs_specs = [spec for spec in iter_specs(contract) if not spec.is_action]
        node._state_specs = [spec for spec in node._obs_specs if spec.key == "observation.state"]
        node._topic_to_qos = {}
        node._joint_rad_limits = []

    def _load_dispatch_contract(node) -> None:
        node._action_specs = [spec for spec in iter_specs(contract) if spec.is_action]
        node._safe_stop_plan = build_safe_stop_plan(action_specs=node._action_specs, joint_order=["1", "2"])
        node._joint_max_age_ns = 1_000_000_000

    monkeypatch.setattr(pipeline_policy_module, "load_inference_manifest", lambda *_args, **_kwargs: manifest)
    manager_options = {}

    def _create_manager(*_args, **kwargs):
        manager_options.update(kwargs)
        return manager

    monkeypatch.setattr(pipeline_policy_module, "create_pipeline_manager", _create_manager)
    monkeypatch.setattr(PipelinePolicyNode, "_load_contract", _load_contract)
    monkeypatch.setattr(ScheduledActionDispatcherNode, "_load_contract_and_plan", _load_dispatch_contract)

    capacity = {
        "session_control": {"max_in_flight": 1},
        "action_generation": {"max_in_flight": 1},
    }
    runtime_policy = {
        "pipeline_id": "policy",
        "execution_mode": "monolithic",
        "hardware_resource_id": "ascend:0",
        "deployment_fingerprint": manifest.fingerprint,
        "public_capacity": capacity,
        "runtime_options": effective_latency_runtime_options({}),
        "transport": {
            "open_session": endpoints["open"],
            "dispatch": endpoints["dispatch"],
            "close_session": endpoints["close"],
            "serving_status": endpoints["status"],
            "health_topic": f"{base}/health",
        },
    }
    scheduling = {
        "stage_policy": "independent",
        "frame_base_priority": 0,
        "stages": {
            "vlm": {"priority_offset": 1},
            "action_expert": {"priority_offset": 2},
        },
    }
    runtime_policy["scheduling"] = scheduling
    runtime_policy_json = json.dumps(runtime_policy, sort_keys=True, separators=(",", ":"))
    pipeline_config = PipelineNodeConfig(
        pipeline_id="policy",
        model_path=str(tmp_path),
        deployment="fake",
        execution_mode="monolithic",
        request_timeout=2.0,
        default_task="",
        runtime_options_json="{}",
        robot_config_path=str(tmp_path / "unused.yaml"),
        use_sim=False,
        action_server=f"{base}/legacy_dispatch",
        reset_service=f"{base}/legacy_reset",
        health_topic=f"{base}/health",
        action_topic=f"{base}/legacy_action",
        request_topic="",
        result_topic="",
        heartbeat_topic="",
        scheduled_open_session=endpoints["open"],
        scheduled_dispatch=endpoints["dispatch"],
        scheduled_close_session=endpoints["close"],
        scheduled_serving_status=endpoints["status"],
        runtime_policy_json=runtime_policy_json,
        runtime_policy_fingerprint=hashlib.sha256(runtime_policy_json.encode()).hexdigest(),
        hardware_resource_id="ascend:0",
        session_idle_timeout_ns=10_000_000_000,
        public_capacity_json=json.dumps(capacity, sort_keys=True),
        max_session_records=4,
        terminal_result_cache_entries=4,
        max_duplicate_waiters_per_request=2,
        terminal_session_retention_ns=1_000_000_000,
        pipeline_stage_policy="independent",
        pipeline_scheduling_json=json.dumps(scheduling),
    )

    rclpy.init()
    pipeline = PipelinePolicyNode(pipeline_config, node_name=f"pipeline_{suffix}", **runtime_dependencies)
    assert manager_options["stage_policy"] == "independent"
    assert manager_options["stage_scheduling"]["vlm"]["priority_offset"] == 1
    pipeline._sample_observations = lambda _sample_time: {"observation.state": np.asarray([0.0, 0.0], dtype=np.float32)}
    compatibility = pipeline._pipeline_compatibility_fingerprint
    profile_identity = "f" * 64
    now_ns = time.time_ns()
    profile_path = tmp_path / "real_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "closure_profiles": [
                    {
                        "deployment_fingerprint": manifest.fingerprint,
                        "hardware_fingerprint": "a" * 64,
                        "profile_compatibility_fingerprint": profile_identity,
                        "scope": "global_proxy",
                        "work_class": 1,
                        "closure_key": "session_open",
                        "hardware_priority": 0,
                        "input_contract_fingerprint": "",
                        "prompt_bytes_max": 0,
                        "goal_acceptance_p999_ms": 1.0,
                        "latency_p99_ms": 2.0,
                        "profiled_at_ns": now_ns,
                        "sample_count": 1,
                    },
                    {
                        "deployment_fingerprint": manifest.fingerprint,
                        "hardware_fingerprint": "a" * 64,
                        "profile_compatibility_fingerprint": profile_identity,
                        "scope": "global_proxy",
                        "work_class": 2,
                        "closure_key": "full_infer",
                        "hardware_priority": 0,
                        "input_contract_fingerprint": compatibility,
                        "prompt_bytes_max": 4096,
                        "goal_acceptance_p999_ms": 1.0,
                        "latency_p99_ms": 2.0,
                        "profiled_at_ns": now_ns,
                        "sample_count": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = {
        "pipeline_id": "policy",
        "compatibility_group": "test_group",
        "hardware_resource_id": "ascend:0",
        "hardware_profile_fingerprint": "a" * 64,
        "deployment_fingerprint": manifest.fingerprint,
        "runtime_policy_fingerprint": pipeline_config.runtime_policy_fingerprint,
        "profile_compatibility_fingerprint": profile_identity,
        "open_session": endpoints["open"],
        "dispatch": endpoints["dispatch"],
        "close_session": endpoints["close"],
        "serving_status": endpoints["status"],
        "profile_path": str(profile_path),
        "required": True,
        "public_capacity": capacity,
    }
    scheduler = GlobalInferenceSchedulerNode(
        parameter_overrides=_parameters(candidate, endpoints), **runtime_dependencies
    )
    dispatcher = ScheduledActionDispatcherNode(
        parameter_overrides=[
            Parameter("joint_state_topic", value=joint_topic),
            Parameter("queue_size", value=4),
            Parameter("watermark_threshold", value=1),
            Parameter("control_frequency", value=20.0),
            Parameter("chunk_size", value=2),
            Parameter("scheduler_readiness_endpoint", value=endpoints["readiness"]),
            Parameter("open_session_endpoint", value=endpoints["global_open"]),
            Parameter("dispatch_endpoint", value=endpoints["global_dispatch"]),
            Parameter("close_session_endpoint", value=endpoints["global_close"]),
            Parameter("inference_pipeline", value="policy"),
            Parameter("inference_fallback_chain", value="[]"),
            Parameter("inference_retry_json", value="{}"),
            Parameter("startup_readiness_timeout_ns", value=5_000_000_000),
            Parameter("default_open_timeout_ns", value=2_000_000_000),
            Parameter("default_request_timeout_ns", value=2_000_000_000),
            Parameter("inference_priority", value=0),
        ]
    )
    io_node = Node(f"scheduler_real_io_{suffix}")
    joint_pub = io_node.create_publisher(JointState, joint_topic, 10)
    commands: list[list[float]] = []
    io_node.create_subscription(Float64MultiArray, command_topic, lambda msg: commands.append(list(msg.data)), 10)
    executor = MultiThreadedExecutor(num_threads=12)
    for node in (pipeline, scheduler, dispatcher, io_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not commands:
            message = JointState()
            message.header.stamp = io_node.get_clock().now().to_msg()
            message.name = ["1", "2"]
            message.position = [0.0, 0.0]
            joint_pub.publish(message)
            time.sleep(0.05)
        assert commands, f"dispatcher did not reach command topic; state={dispatcher._state.value}"
        assert dispatcher._state == DispatcherState.ACTIVE
        assert manager.priorities and set(manager.priorities) == {0}
        assert any(np.allclose(commands[-1], expected) for expected in ([0.1, 0.2], [0.3, 0.4]))
        stop_client = io_node.create_client(Trigger, "/action_dispatcher/stop_evaluate")
        assert stop_client.wait_for_service(timeout_sec=2.0)
        stop_response = _wait_future(stop_client.call_async(Trigger.Request()))
        assert stop_response.success
        assert dispatcher._state == DispatcherState.STOPPED
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        for node in (io_node, dispatcher, scheduler, pipeline):
            node.destroy_node()
        rclpy.shutdown()
