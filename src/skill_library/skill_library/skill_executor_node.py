"""Skill and primitive execution node for the embodied minimal closure."""

import copy
import json
import math
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from skill_catalog.compiler import SkillCatalogCompiler
from skill_catalog.models import DelegatedExecutorDescriptor, SkillCompileContext
from skill_catalog.source import AmentShareSkillSource, DevelopmentStagingSkillSource, DirectoryReleaseSkillSource
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from unique_identifier_msgs.msg import UUID

from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.dispatch_binding import (
    copy_binding,
    delegated_executor_identity,
    delegated_executor_identity_matches,
    fill_delegated_executor_identity,
    load_delegated_model_identity,
)
from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_V1, primitive_contract_for_version
from embodied_common.skill_request import derive_skill_task_id, validate_request_schema_version
from embodied_common.wire_contracts import validate_public_request_wire_contracts
from embodied_common.workflow_contracts import CanonicalWorkflowStep, compute_workflow_digest, normalize_workflow_steps
from ibrobot_msgs.action import (
    ExecuteNavigation,
    ExecuteTaskPlan,
    ImitateHumanMotion,
    PickObject,
    PlaceObject,
    PrimitiveCommand,
    SkillCommand,
)
from ibrobot_msgs.msg import SkillCapabilityStatus, SkillDiagnostic, SkillRegistryEvent, TaskStep
from ibrobot_msgs.srv import (
    BeginWorkflowExecution,
    FinalizeWorkflowExecution,
    GetSemanticObjects,
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    MoveToConfiguration,
    ReloadSkillCatalog,
    SetSoundFollowing,
    ValidatePrimitive,
    ValidateSkill,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from skill_library.gateway_policy import (
    GATEWAY_FINALIZATION_FAILED,
    SKILL_BUSY,
    TIMEOUT_EXCEEDS_POLICY,
    BoundedRequestLedger,
    ExecutionOwner,
    GatewayPolicy,
    GatewayRequest,
    RootExecutionLease,
    RuntimeSnapshot,
    SkillRequirements,
)
from skill_library.resolver import PrimitiveSpec, load_json_mapping, resolve_skill_primitives
from skill_library.runtime_coordinator import SkillRegistryOwner

EE_POSITION_TOLERANCE_M = 0.02
SKILL_CANCEL_TIMEOUT = "SKILL_CANCEL_TIMEOUT"
PRIMITIVE_CANCEL_CLEANUP_TIMEOUT = "CANCEL_CLEANUP_TIMEOUT"
WORKFLOW_STEP_PENDING = "pending"
WORKFLOW_STEP_ACTIVE = "active"
WORKFLOW_STEP_SUCCEEDED = "succeeded"
WORKFLOW_STEP_FAILED = "failed"
WORKFLOW_STEP_CANCELED = "canceled"

_NAVIGATION_ERROR_CODES = {
    ExecuteNavigation.Result.INVALID_GOAL: "NAVIGATION_INVALID_GOAL",
    ExecuteNavigation.Result.BUSY: "NAVIGATION_BUSY",
    ExecuteNavigation.Result.TF_UNAVAILABLE: "NAVIGATION_TF_UNAVAILABLE",
    ExecuteNavigation.Result.NAV2_UNAVAILABLE: "NAVIGATION_NAV2_UNAVAILABLE",
    ExecuteNavigation.Result.GOAL_REJECTED: "NAVIGATION_GOAL_REJECTED",
    ExecuteNavigation.Result.NAVIGATION_ABORTED: "NAVIGATION_ABORTED",
    ExecuteNavigation.Result.NAVIGATION_CANCELED: "NAVIGATION_CANCELED",
    ExecuteNavigation.Result.STOP_TIMEOUT: "NAVIGATION_STOP_TIMEOUT",
    ExecuteNavigation.Result.INTERNAL_ERROR: "NAVIGATION_INTERNAL_ERROR",
}


def _binding_task_id(value) -> str:
    return str(value.dispatch_binding.task_id).strip()


def _binding_key(binding) -> tuple:
    budget = binding.task_budget
    return (
        binding.schema_version,
        binding.task_id,
        binding.root_task_id,
        budget.schema_version,
        budget.started_at.sec,
        budget.started_at.nanosec,
        budget.deadline.sec,
        budget.deadline.nanosec,
        binding.expected_registry_epoch,
        binding.expected_registry_generation,
        binding.expected_registry_digest,
        binding.workflow_digest,
        binding.workflow_step_index,
        binding.root_lease_nonce,
        binding.dispatch_nonce,
    )


@dataclass
class _RetainedAdmissionCleanup:
    admission: object
    audit_context: dict[str, str]
    duration_sec: float = 0.0
    step_count: int = 0
    retained: bool = False
    parent_terminal: bool = False
    pending_goal_keys: set[bytes] = field(default_factory=set)
    confirmed_goal_keys: set[bytes] = field(default_factory=set)
    finalizing: bool = False


@dataclass(frozen=True)
class _EePoseSnapshot:
    pose: PoseStamped
    received_monotonic: float


class _EePoseSnapshotExpired(Exception):
    pass


@dataclass
class _WorkflowExecution:
    root_task_id: str
    workflow_digest: str
    root_lease_nonce: str
    lease_token: object
    owner: ExecutionOwner
    policy: GatewayPolicy
    bundle: object
    workflow_steps: tuple[CanonicalWorkflowStep, ...]
    task_budget_key: tuple[int, int, int, int]
    deadline_unix_sec: float
    completed_step_count: int = 0
    step_terminal_states: list[str] = field(default_factory=list)
    terminal_recorded: bool = False
    lease_released: bool = False
    runtime_generation_released: bool = False
    child_results: dict[int, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _TerminalWorkflow:
    workflow_digest: str
    root_lease_nonce: str
    registry_epoch: str
    registry_generation: int
    registry_digest: str
    task_budget_key: tuple[int, int, int, int]
    terminal_state: int
    completed_step_count: int
    workflow_steps: tuple[CanonicalWorkflowStep, ...]


@dataclass
class _ExternalPrimitiveAdmission:
    task_id: str
    payload_digest: str
    lease_token: object
    bundle: object
    generation_released: bool = False
    terminal_recorded: bool = False
    lease_released: bool = False


@dataclass(frozen=True)
class _WorkflowChildAdmission:
    workflow: _WorkflowExecution
    child_owner: ExecutionOwner


class _LateCleanupConfirmation:
    """Confirm one downstream cleanup terminal state and notify listeners once."""

    def __init__(self) -> None:
        self._confirmed = False
        self._callbacks: list[Callable[[], None]] = []
        self._lock = RLock()

    def add_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if not self._confirmed:
                self._callbacks.append(callback)
                return
        callback()

    def confirm(self) -> bool:
        with self._lock:
            if self._confirmed:
                return False
            self._confirmed = True
            callbacks = self._callbacks
            self._callbacks = []
        for callback in callbacks:
            callback()
        return True

    def watch_result_future(self, result_future) -> None:
        def confirm_terminal(future) -> None:
            try:
                wrapped_result = future.result()
            except Exception:
                return
            result = getattr(wrapped_result, "result", wrapped_result)
            if getattr(result, "error_code", "") != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                self.confirm()

        result_future.add_done_callback(confirm_terminal)

    def watch_goal_future(self, goal_future, cancel_goal: Callable[[object], None]) -> None:
        def confirm_late_goal(future) -> None:
            try:
                goal_handle = future.result()
            except Exception:
                return
            if goal_handle is None or not goal_handle.accepted:
                self.confirm()
                return
            cancel_goal(goal_handle)
            try:
                self.watch_result_future(goal_handle.get_result_async())
            except Exception:
                return

        goal_future.add_done_callback(confirm_late_goal)


class _DeferredTerminalGoalHandle:
    """Proxy child terminal requests until Gateway bookkeeping is complete."""

    def __init__(self, goal_handle) -> None:
        self._goal_handle = goal_handle
        self._terminal_intent = ""
        self._committed = False
        self._lock = RLock()
        self.late_cleanup_confirmation: _LateCleanupConfirmation | None = None

    @property
    def terminal_intent(self) -> str:
        return self._terminal_intent

    def _record_terminal_intent(self, intent: str) -> None:
        with self._lock:
            if not self._terminal_intent:
                self._terminal_intent = intent

    def succeed(self) -> None:
        self._record_terminal_intent("succeeded")

    def abort(self) -> None:
        self._record_terminal_intent("aborted")

    def canceled(self) -> None:
        self._record_terminal_intent("canceled")

    def commit(self) -> bool:
        """Commit the recorded child terminal state exactly once."""
        with self._lock:
            if self._committed:
                return False
            self._committed = True
            if self._terminal_intent == "succeeded":
                self._goal_handle.succeed()
            elif self._terminal_intent == "canceled":
                self._goal_handle.canceled()
            else:
                self._goal_handle.abort()
            return True

    def force_abort(self) -> bool:
        """Abort the real action exactly once when bookkeeping cannot complete."""
        with self._lock:
            if self._committed:
                return False
            self._committed = True
            self._goal_handle.abort()
            return True

    def __getattr__(self, name):
        return getattr(self._goal_handle, name)


def _duration_from_seconds(seconds: float) -> Duration:
    sec = int(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return DurationMsg(sec=sec, nanosec=nanosec)


def _build_joint_trajectory_goal(
    joint_names: list[str],
    joint_waypoints: list[list[float]],
    waypoint_duration_sec: float,
) -> FollowJointTrajectory.Goal:
    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = JointTrajectory()
    goal_msg.trajectory.joint_names = list(joint_names)

    for index, waypoint in enumerate(joint_waypoints, start=1):
        point = JointTrajectoryPoint()
        point.positions = [float(position) for position in waypoint]
        point.time_from_start = _duration_from_seconds(float(waypoint_duration_sec) * index)
        goal_msg.trajectory.points.append(point)
    return goal_msg


def _load_json_list(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    parsed = load_json_mapping(f'{{"items": {raw_value}}}').get("items", [])
    if not isinstance(parsed, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in parsed]


class SkillExecutorNode(Node):
    """Expose skill and primitive actions with explicit safety validation."""

    def __init__(self, parameter_overrides=None, node_name: str | None = None) -> None:
        super().__init__(node_name or "skill_executor_node", parameter_overrides=parameter_overrides)
        validate_public_request_wire_contracts()
        startup_descriptor = ParameterDescriptor(read_only=True)
        self.declare_parameter("skill_action_name", "/embodied/execute_skill")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("validate_primitive_service", "/embodied/validate_primitive")
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("relative_motion_step_m", 0.03)
        self.declare_parameter("relative_motion_reference_frame", "base")
        self.declare_parameter("relative_motion_direction_mapping_json", "{}")
        self.declare_parameter("rpc_timeout_sec", 5.0, descriptor=startup_descriptor)
        self.declare_parameter("gripper_settle_sec", 1.5, descriptor=startup_descriptor)
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("arm_trajectory_action_name", "/arm_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("task_executor_action_name", "/task_executor/execute_task_plan")
        self.declare_parameter("pick_action_name", "/manipulation/execute_pick")
        self.declare_parameter("place_action_name", "/manipulation/execute_place")
        self.declare_parameter("imitate_human_motion_action_name", "/hri/imitate_human_motion")
        self.declare_parameter("imitate_human_motion_enabled", False, descriptor=startup_descriptor)
        self.declare_parameter("sound_following_service", "/sound_orientation_node/set_following")
        self.declare_parameter("sound_following_enabled", False, descriptor=startup_descriptor)
        self.declare_parameter("grasp_execution_json", "{}")
        self.declare_parameter("placement_execution_json", "{}")
        self.declare_parameter("semantic_map_target_service", "")
        self.declare_parameter("semantic_map_stand_off_distance_m", 0.3)
        self.declare_parameter("move_configuration_service", "/moveit_gateway/move_to_configuration")
        self.declare_parameter("ee_pose_topic", "/robot_status/ee_pose")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cmd_pose_topic", "/cmd_pose")
        self.declare_parameter("debug_tracing", False)
        self.declare_parameter("motion_authorized", False, descriptor=startup_descriptor)
        self.declare_parameter("active_control_mode", "", descriptor=startup_descriptor)
        self.declare_parameter("skill_required_control_mode", "", descriptor=startup_descriptor)
        self.declare_parameter("supported_control_modes_json", "[]", descriptor=startup_descriptor)
        self.declare_parameter(
            "motion_mode_service", "/motion_mode/set_navigation_enabled", descriptor=startup_descriptor
        )
        self.declare_parameter(
            "skill_gateway_status_service",
            "/embodied/get_skill_gateway_status",
            descriptor=startup_descriptor,
        )
        self.declare_parameter("skill_catalog_reload_service", "/embodied/reload_skill_catalog")
        self.declare_parameter("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot")
        self.declare_parameter("begin_workflow_service", "/embodied/begin_workflow_execution")
        self.declare_parameter("finalize_workflow_service", "/embodied/finalize_workflow_execution")
        self.declare_parameter("skill_registry_event_topic", "/embodied/skill_registry_events")
        self.declare_parameter("skill_catalog_source_mode", "installed")
        self.declare_parameter("skill_catalog_source_root", "")
        self.declare_parameter("skill_catalog_profile", "")
        self.declare_parameter("robot_name", "unknown", descriptor=startup_descriptor)
        self.declare_parameter("context_schema_version", 1, descriptor=startup_descriptor)
        self.declare_parameter("default_skill_timeout_sec", 120.0, descriptor=startup_descriptor)
        self.declare_parameter("task_budget_sec", 180.0, descriptor=startup_descriptor)
        self.declare_parameter("robot_state_freshness_sec", 0.5, descriptor=startup_descriptor)
        self.declare_parameter("scene_freshness_sec", 0.5, descriptor=startup_descriptor)
        self.declare_parameter("model_idle_timeout_sec", 120.0, descriptor=startup_descriptor)
        self.declare_parameter("config_digest", "", descriptor=startup_descriptor)
        self.declare_parameter("ledger_terminal_capacity", 100, descriptor=startup_descriptor)

        self._skill_action_name = self.get_parameter("skill_action_name").get_parameter_value().string_value
        self._primitive_action_name = self.get_parameter("primitive_action_name").get_parameter_value().string_value
        self._validate_skill_service = self.get_parameter("validate_skill_service").get_parameter_value().string_value
        self._validate_primitive_service = (
            self.get_parameter("validate_primitive_service").get_parameter_value().string_value
        )
        self._named_poses = load_json_mapping(self.get_parameter("named_poses_json").get_parameter_value().string_value)
        self._named_targets = load_json_mapping(
            self.get_parameter("named_targets_json").get_parameter_value().string_value
        )
        self._skill_templates = {}
        self._relative_motion_step_m = self.get_parameter("relative_motion_step_m").get_parameter_value().double_value
        self._relative_motion_reference_frame = (
            self.get_parameter("relative_motion_reference_frame").get_parameter_value().string_value
        )
        self._relative_motion_direction_mapping = load_json_mapping(
            self.get_parameter("relative_motion_direction_mapping_json").get_parameter_value().string_value
        )
        self._rpc_timeout = self.get_parameter("rpc_timeout_sec").get_parameter_value().double_value
        self._gripper_settle_sec = self.get_parameter("gripper_settle_sec").get_parameter_value().double_value
        self._gripper_open = self.get_parameter("gripper_open_position").get_parameter_value().double_value
        self._gripper_closed = self.get_parameter("gripper_closed_position").get_parameter_value().double_value
        self._arm_joint_names = _load_json_list(
            self.get_parameter("arm_joint_names_json").get_parameter_value().string_value
        )
        self._joint_limits = load_json_mapping(
            self.get_parameter("joint_limits_json").get_parameter_value().string_value
        )
        self._workspace = load_json_mapping(self.get_parameter("workspace_json").get_parameter_value().string_value)
        self._arm_trajectory_action_name = (
            self.get_parameter("arm_trajectory_action_name").get_parameter_value().string_value
        )
        self._task_executor_action_name = (
            self.get_parameter("task_executor_action_name").get_parameter_value().string_value
        )
        self._pick_action_name = self.get_parameter("pick_action_name").get_parameter_value().string_value
        self._place_action_name = self.get_parameter("place_action_name").get_parameter_value().string_value
        self._imitate_human_motion_action_name = (
            self.get_parameter("imitate_human_motion_action_name").get_parameter_value().string_value
        )
        self._imitate_human_motion_enabled = self.get_parameter("imitate_human_motion_enabled").value
        self._sound_following_enabled = self.get_parameter("sound_following_enabled").value
        self._sound_following_service = self.get_parameter("sound_following_service").get_parameter_value().string_value
        self._grasp_execution = load_json_mapping(self.get_parameter("grasp_execution_json").value)
        self._placement_execution = load_json_mapping(self.get_parameter("placement_execution_json").value)
        self._semantic_map_target_service = self.get_parameter("semantic_map_target_service").value
        self._semantic_map_stand_off_distance_m = self.get_parameter("semantic_map_stand_off_distance_m").value
        self._move_configuration_service = (
            self.get_parameter("move_configuration_service").get_parameter_value().string_value
        )
        self._ee_pose_topic = self.get_parameter("ee_pose_topic").get_parameter_value().string_value
        self._joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self._cmd_pose_topic = self.get_parameter("cmd_pose_topic").get_parameter_value().string_value
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value
        self._motion_authorized = self.get_parameter("motion_authorized").get_parameter_value().bool_value
        self._active_control_mode = self.get_parameter("active_control_mode").get_parameter_value().string_value
        self._skill_required_control_mode = (
            self.get_parameter("skill_required_control_mode").get_parameter_value().string_value
        )
        self._supported_control_modes = tuple(
            str(mode).strip()
            for mode in json.loads(self.get_parameter("supported_control_modes_json").value)
            if str(mode).strip()
        )
        self._motion_mode_service = self.get_parameter("motion_mode_service").value
        self._motion_mode_switch_lock = RLock()
        self._skill_gateway_status_service = (
            self.get_parameter("skill_gateway_status_service").get_parameter_value().string_value
        )
        self._skill_catalog_reload_service = (
            self.get_parameter("skill_catalog_reload_service").get_parameter_value().string_value
        )
        self._skill_catalog_snapshot_service = (
            self.get_parameter("skill_catalog_snapshot_service").get_parameter_value().string_value
        )
        self._begin_workflow_service = self.get_parameter("begin_workflow_service").get_parameter_value().string_value
        self._finalize_workflow_service = (
            self.get_parameter("finalize_workflow_service").get_parameter_value().string_value
        )
        self._skill_registry_event_topic = (
            self.get_parameter("skill_registry_event_topic").get_parameter_value().string_value
        )
        self._skill_catalog_source_mode = (
            self.get_parameter("skill_catalog_source_mode").get_parameter_value().string_value.strip().lower()
        )
        self._skill_catalog_source_root = (
            self.get_parameter("skill_catalog_source_root").get_parameter_value().string_value.strip()
        )
        self._skill_catalog_profile = self.get_parameter("skill_catalog_profile").get_parameter_value().string_value
        if self._skill_catalog_source_mode not in {"installed", "development", "production"}:
            raise ValueError("skill_catalog_source_mode must be installed, development, or production")
        if self._skill_catalog_source_mode in {"development", "production"} and not self._skill_catalog_source_root:
            raise ValueError("skill_catalog_source_root is required in development and production modes")
        self._robot_name = self.get_parameter("robot_name").get_parameter_value().string_value
        self._context_schema_version = self.get_parameter("context_schema_version").get_parameter_value().integer_value
        if self._context_schema_version not in {1, 2, 3}:
            raise ValueError("context_schema_version must be 1, 2, or 3")
        self._primitive_contract = primitive_contract_for_version(self._context_schema_version)
        self._primitive_descriptors = self._primitive_contract.descriptors
        self._primitive_contract_digest = self._primitive_contract.digest
        if self._context_schema_version >= 2:
            self.declare_parameter("navigation_action_name", "")
            self._navigation_action_name = (
                self.get_parameter("navigation_action_name").get_parameter_value().string_value
            )
            if not self._navigation_action_name.strip():
                raise ValueError("navigation_action_name must be a non-empty projected endpoint")
        if not self._skill_catalog_profile:
            self._skill_catalog_profile = self._robot_name
        self._default_skill_timeout_sec = (
            self.get_parameter("default_skill_timeout_sec").get_parameter_value().double_value
        )
        self._task_budget_sec = self.get_parameter("task_budget_sec").get_parameter_value().double_value
        self._robot_state_freshness_sec = (
            self.get_parameter("robot_state_freshness_sec").get_parameter_value().double_value
        )
        self._scene_freshness_sec = self.get_parameter("scene_freshness_sec").get_parameter_value().double_value
        self._model_idle_timeout_sec = self.get_parameter("model_idle_timeout_sec").get_parameter_value().double_value
        self._ledger_terminal_capacity = (
            self.get_parameter("ledger_terminal_capacity").get_parameter_value().integer_value
        )
        self._skill_requirements = {}
        self._skill_parameter_schemas = {}
        self._normalized_gateway_config = {
            "name": self._robot_name,
            "embodied": {
                "named_poses": self._named_poses,
                "named_targets": self._named_targets,
                "timeouts": {
                    "default_skill_timeout_sec": self._default_skill_timeout_sec,
                    "task_budget_sec": self._task_budget_sec,
                    "robot_state_freshness_sec": self._robot_state_freshness_sec,
                    "scene_freshness_sec": self._scene_freshness_sec,
                    "model_idle_timeout_sec": self._model_idle_timeout_sec,
                    "rpc_timeout_sec": self._rpc_timeout,
                    "gripper_settle_sec": self._gripper_settle_sec,
                },
            },
        }
        self._gateway_timeout_policy = resolve_embodied_timeout_policy(self._normalized_gateway_config["embodied"])
        self._default_skill_timeout_sec = self._gateway_timeout_policy["default_skill_timeout_sec"]
        self._task_budget_sec = self._gateway_timeout_policy["task_budget_sec"]
        self._robot_state_freshness_sec = self._gateway_timeout_policy["robot_state_freshness_sec"]
        self._scene_freshness_sec = self._gateway_timeout_policy["scene_freshness_sec"]
        self._model_idle_timeout_sec = self._gateway_timeout_policy["model_idle_timeout_sec"]
        self._rpc_timeout = self._gateway_timeout_policy["rpc_timeout_sec"]
        self._gripper_settle_sec = self._gateway_timeout_policy["gripper_settle_sec"]
        self._normalized_gateway_config["embodied"]["timeouts"] = dict(self._gateway_timeout_policy)
        self._gateway_ledger = BoundedRequestLedger(self._ledger_terminal_capacity)
        self._gateway_lease = RootExecutionLease()
        self._gateway_policy = GatewayPolicy(
            self._gateway_timeout_policy,
            self._skill_requirements,
            parameter_schemas=self._skill_parameter_schemas,
            skill_timeout_caps={},
            ledger=self._gateway_ledger,
            lease=self._gateway_lease,
        )
        self._state_lock = RLock()
        self._active_skill_admission = None
        self._active_skill_owner = None
        self._active_audit_context = None
        self._active_runtime_generation_retained = False
        self._pending_internal_primitive_goals = {}
        self._active_internal_primitive_goals = {}
        self._active_delegated_dispatches = {}
        self._retained_admission_cleanup = {}
        self._active_workflow: _WorkflowExecution | None = None
        self._terminal_workflows: dict[str, _TerminalWorkflow] = {}
        self._external_admissions: dict[int, _ExternalPrimitiveAdmission] = {}
        self._robot_config_digest = self.get_parameter("config_digest").get_parameter_value().string_value
        self._runtime_coordinator = SkillRegistryOwner(
            self._compile_runtime_snapshot,
            state_lock=self._state_lock,
        )
        startup_reload = self._runtime_coordinator.reload("startup")
        if not startup_reload.success:
            raise ValueError(f"skill catalog startup failed: {startup_reload.error_code}: {startup_reload.message}")
        startup_bundle = self._runtime_coordinator.current
        startup_requirements, startup_parameter_schemas, startup_timeout_caps, startup_control_modes = (
            self._snapshot_policy_inputs(startup_bundle.snapshot)
        )
        self._gateway_policy.replace_catalog(
            startup_bundle.snapshot.robot_context.timeout_policy,
            startup_requirements,
            parameter_schemas=startup_parameter_schemas,
            skill_timeout_caps=startup_timeout_caps,
            skill_schema_versions=self._snapshot_skill_schema_versions(startup_bundle.snapshot),
            skill_control_modes=startup_control_modes,
        )
        self._skill_requirements = startup_requirements
        self._skill_parameter_schemas = startup_parameter_schemas
        self._skill_control_modes = startup_control_modes
        self._skill_templates = self._mutable_templates(startup_bundle.snapshot.templates)
        self._active_runtime_bundle = None
        self._latest_ee_pose = None
        self._latest_ee_pose_monotonic = None
        self._latest_joint_state = None
        self._skill_goal_lock = threading.Lock()
        self._skill_goal_active = False

        callback_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._pose_publisher = self.create_publisher(Pose, self._cmd_pose_topic, 10)
        self._task_executor_client = ActionClient(
            self, ExecuteTaskPlan, self._task_executor_action_name, callback_group=callback_group
        )
        self._arm_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._arm_trajectory_action_name,
            callback_group=callback_group,
        )
        if self._context_schema_version >= 2:
            self._navigation_client = ActionClient(
                self,
                ExecuteNavigation,
                self._navigation_action_name,
                callback_group=callback_group,
            )
        if self._semantic_map_target_service:
            self._semantic_map_target_client = self.create_client(
                GetSemanticObjects,
                self._semantic_map_target_service,
                callback_group=callback_group,
            )
        self.create_subscription(
            PoseStamped,
            self._ee_pose_topic,
            self._handle_ee_pose,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._handle_joint_state,
            10,
            callback_group=callback_group,
        )
        self._validate_skill_client = self.create_client(
            ValidateSkill, self._validate_skill_service, callback_group=callback_group
        )
        self._validate_primitive_client = self.create_client(
            ValidatePrimitive, self._validate_primitive_service, callback_group=callback_group
        )
        self._primitive_client = ActionClient(
            self, PrimitiveCommand, self._primitive_action_name, callback_group=callback_group
        )
        self._motion_mode_client = self.create_client(SetBool, self._motion_mode_service, callback_group=callback_group)
        self._pick_client = ActionClient(self, PickObject, self._pick_action_name, callback_group=callback_group)
        self._place_client = ActionClient(self, PlaceObject, self._place_action_name, callback_group=callback_group)
        self._imitate_human_motion_client = ActionClient(
            self,
            ImitateHumanMotion,
            self._imitate_human_motion_action_name,
            callback_group=callback_group,
        )
        self._sound_following_client = self.create_client(
            SetSoundFollowing, self._sound_following_service, callback_group=callback_group
        )
        self._move_configuration_client = self.create_client(
            MoveToConfiguration,
            self._move_configuration_service,
            callback_group=callback_group,
        )
        self._gateway_status_server = self.create_service(
            GetSkillGatewayStatus,
            self._skill_gateway_status_service,
            self._get_gateway_status,
            callback_group=callback_group,
        )
        registry_event_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._skill_registry_event_publisher = self.create_publisher(
            SkillRegistryEvent, self._skill_registry_event_topic, registry_event_qos
        )
        self._skill_catalog_reload_server = self.create_service(
            ReloadSkillCatalog,
            self._skill_catalog_reload_service,
            self._reload_skill_catalog,
            callback_group=callback_group,
        )
        self._skill_catalog_snapshot_server = self.create_service(
            GetSkillSnapshot,
            self._skill_catalog_snapshot_service,
            self._get_skill_snapshot,
            callback_group=callback_group,
        )
        self._begin_workflow_server = self.create_service(
            BeginWorkflowExecution,
            self._begin_workflow_service,
            self._begin_workflow_execution,
            callback_group=callback_group,
        )
        self._finalize_workflow_server = self.create_service(
            FinalizeWorkflowExecution,
            self._finalize_workflow_service,
            self._finalize_workflow_execution,
            callback_group=callback_group,
        )
        self._workflow_reaper_timer = self.create_timer(
            min(1.0, self._rpc_timeout), self._reap_expired_workflow, callback_group=callback_group
        )
        self._skill_server = ActionServer(
            self,
            SkillCommand,
            self._skill_action_name,
            execute_callback=self._execute_skill,
            goal_callback=self._handle_skill_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self._primitive_server = ActionServer(
            self,
            PrimitiveCommand,
            self._primitive_action_name,
            execute_callback=self._execute_primitive,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self._publish_registry_event(startup_reload)

        self.get_logger().info(
            "[embodied-debug] skill_executor ready: "
            f"skill_action={self._skill_action_name}, primitive_action={self._primitive_action_name}, "
            f"relative_frame={self._relative_motion_reference_frame}, "
            f"direction_mapping={self._relative_motion_direction_mapping or 'default'}, "
            f"ee_pose_topic={self._ee_pose_topic}, joint_state_topic={self._joint_state_topic}"
        )

    def _compile_runtime_snapshot(self):
        from skill_catalog.models import SkillRobotContext

        direction_mapping = {
            str(name): tuple(float(value) for value in values)
            for name, values in self._relative_motion_direction_mapping.items()
            if isinstance(values, list | tuple) and len(values) == 3
        }
        execution_endpoints = {
            "skill_action": self._skill_action_name,
            "primitive_action": self._primitive_action_name,
            "validate_skill_service": self._validate_skill_service,
            "validate_primitive_service": self._validate_primitive_service,
            "gateway_status_service": self._skill_gateway_status_service,
            "begin_workflow_service": self._begin_workflow_service,
            "finalize_workflow_service": self._finalize_workflow_service,
            "task_executor_action": self._task_executor_action_name,
            "arm_trajectory_action": self._arm_trajectory_action_name,
            "move_configuration_service": self._move_configuration_service,
        }
        if self._context_schema_version >= 2:
            execution_endpoints["navigation_action"] = self._navigation_action_name
        robot_context = SkillRobotContext(
            robot_name=self._robot_name,
            context_schema_version=self._context_schema_version,
            robot_config_digest=self._robot_config_digest,
            named_poses=self._named_poses,
            named_targets=self._named_targets,
            arm_joint_names=tuple(self._arm_joint_names),
            joint_limits=self._joint_limits,
            workspace_limits=self._workspace,
            required_control_mode=self._skill_required_control_mode,
            timeout_policy=self._gateway_timeout_policy,
            relative_motion_reference_frame=self._relative_motion_reference_frame,
            relative_motion_step_m=self._relative_motion_step_m,
            relative_motion_direction_mapping=direction_mapping,
            gripper_open_position=self._gripper_open,
            gripper_closed_position=self._gripper_closed,
            execution_endpoints=execution_endpoints,
            supported_control_modes=getattr(self, "_supported_control_modes", ()),
        )
        source_root = Path(self._skill_catalog_source_root)
        if self._skill_catalog_source_mode == "installed":
            source = AmentShareSkillSource()
        elif self._skill_catalog_source_mode == "development":
            source = DevelopmentStagingSkillSource(source_root)
        else:
            source = DirectoryReleaseSkillSource(source_root)
        context = SkillCompileContext(
            robot=robot_context,
            primitive_contracts=getattr(self, "_primitive_descriptors", PRIMITIVE_CONTRACT_V1.descriptors),
            primitive_contract_digest=getattr(self, "_primitive_contract_digest", PRIMITIVE_CONTRACT_V1.digest),
            delegated_executors=self._delegated_executor_descriptors(),
        )
        snapshot = SkillCatalogCompiler().compile(
            source,
            profile_name=self._skill_catalog_profile,
            context=context,
        )
        self._validate_hri_runtime_catalog_consistency(snapshot)
        return snapshot

    def _validate_hri_runtime_catalog_consistency(self, snapshot) -> None:
        runtime_enabled = bool(getattr(self, "_imitate_human_motion_enabled", False))
        catalog_enabled = "imitate_human_motion" in snapshot.enabled_skill_names
        if runtime_enabled != catalog_enabled:
            raise ValueError(
                "imitate_human_motion configuration mismatch: "
                f"runtime enabled={runtime_enabled}, catalog enabled={catalog_enabled}"
            )
        sound_runtime_enabled = bool(getattr(self, "_sound_following_enabled", False))
        sound_catalog_enabled = "sound_following" in snapshot.enabled_skill_names
        if sound_runtime_enabled != sound_catalog_enabled:
            raise ValueError(
                "sound_following configuration mismatch: "
                f"runtime enabled={sound_runtime_enabled}, catalog enabled={sound_catalog_enabled}"
            )

    def _delegated_executor_descriptors(self):
        descriptors = {}
        executor_configs = {
            "grasp_pipeline": (self._pick_action_name, self._grasp_execution),
            "placement_pipeline": (self._place_action_name, self._placement_execution),
            "imitate_human_motion": (
                getattr(self, "_imitate_human_motion_action_name", "/hri/imitate_human_motion"),
                {"implementation": "mock_v1"},
            ),
        }
        configured_executors = {
            str(template.get("executor", "")).strip()
            for template in self._skill_templates.values()
            if str(template.get("executor", "")).strip()
        }
        if self._grasp_execution.get("enabled", False):
            configured_executors.add("grasp_pipeline")
        if self._placement_execution.get("enabled", False):
            configured_executors.add("placement_pipeline")
        if getattr(self, "_imitate_human_motion_enabled", False):
            configured_executors.add("imitate_human_motion")
        if getattr(self, "_sound_following_enabled", False):
            configured_executors.add("sound_following")
        # Startup compilation runs before skill templates are cached. The
        # configured service is the SSOT signal that this executor is present.
        if self._semantic_map_target_service:
            configured_executors.add("semantic_map_query")
        for name in sorted(configured_executors):
            if name == "sound_following":
                endpoint_name = self._sound_following_service
                descriptor = DelegatedExecutorDescriptor(
                    **delegated_executor_identity(
                        name=name,
                        endpoint_name=endpoint_name,
                        endpoint_kind="ros_service",
                        configuration={},
                    )
                )
                descriptors[descriptor.name] = descriptor
                continue
            if name == "semantic_map_query":
                endpoint_name = self._semantic_map_target_service
                if not endpoint_name:
                    continue
                configuration = {
                    "query_service": endpoint_name,
                    "stand_off_distance_m": float(self._semantic_map_stand_off_distance_m),
                }
                descriptor = DelegatedExecutorDescriptor(
                    **delegated_executor_identity(
                        name=name,
                        endpoint_name=endpoint_name,
                        endpoint_kind="ros_service",
                        configuration=configuration,
                    )
                )
                descriptors[descriptor.name] = descriptor
                continue
            endpoint_name, configuration = executor_configs.get(name, ("", {}))
            if not endpoint_name:
                continue
            descriptor = DelegatedExecutorDescriptor(
                **delegated_executor_identity(
                    name=name,
                    endpoint_name=endpoint_name,
                    configuration=configuration,
                    **(
                        load_delegated_model_identity(configuration)
                        if name == "grasp_pipeline"
                        else {"model_deployment_name": "", "model_fingerprint": "", "model_bundle_digest": ""}
                    ),
                )
            )
            descriptors[descriptor.name] = descriptor
        return descriptors

    def _execute_sound_following_skill(self, goal_handle, *, effective_timeout_sec: float | None = None):
        result = SkillCommand.Result()
        request = goal_handle.request
        mode = str(request.motion_direction or "").strip().lower()
        mode = {"forward": "enable", "backward": "disable"}.get(mode, mode)
        if mode not in {"enable", "disable"}:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SOUND_FOLLOWING_INVALID_MODE",
                f"motion_direction must be enable or disable, got: {request.motion_direction!r}",
            )
        timeout_sec = float(effective_timeout_sec or request.timeout_sec or self._rpc_timeout)
        if not self._sound_following_client.wait_for_service(timeout_sec=min(timeout_sec, self._rpc_timeout)):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SOUND_FOLLOWING_SERVICE_UNAVAILABLE",
                f"sound following service unavailable: {self._sound_following_service}",
            )

        service_request = SetSoundFollowing.Request()
        service_request.schema_version = 1
        service_request.enable = mode == "enable"
        future = self._sound_following_client.call_async(service_request)
        if not self._wait_for_future(future, timeout_sec, cancel_requested=lambda: goal_handle.is_cancel_requested):
            if goal_handle.is_cancel_requested:
                return self._cancel_skill(result, goal_handle, [], request.skill_name)
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SOUND_FOLLOWING_TIMEOUT",
                "sound following toggle timed out",
            )
        try:
            response = future.result()
        except Exception as exc:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SOUND_FOLLOWING_FAILED",
                f"sound following toggle failed: {exc}",
            )
        if response is None or not response.success:
            message = str(getattr(response, "message", "")).strip()
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SOUND_FOLLOWING_FAILED",
                message or "sound following toggle failed",
            )

        result.success = True
        result.error_code = ""
        result.message = f"sound_following {mode}: state={response.current_state}"
        result.executed_primitives = [f"sound_following:{mode}"]
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        goal_handle.succeed()
        return result

    def _execute_semantic_map_query(self, goal_handle, *, effective_timeout_sec: float | None = None):
        result = SkillCommand.Result()
        if not hasattr(self, "_semantic_map_target_client"):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_SERVICE_UNAVAILABLE",
                "semantic map target service is not configured",
            )
        timeout_sec = float(effective_timeout_sec or goal_handle.request.timeout_sec or self._rpc_timeout)
        if not self._semantic_map_target_client.wait_for_service(timeout_sec=min(timeout_sec, self._rpc_timeout)):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_SERVICE_UNAVAILABLE",
                f"semantic map target service unavailable: {self._semantic_map_target_service}",
            )

        request = GetSemanticObjects.Request()
        request.label = goal_handle.request.target_name
        request.include_inactive = True
        request.max_age_sec = 0.0
        request.max_results = 1
        future = self._semantic_map_target_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec, cancel_requested=lambda: goal_handle.is_cancel_requested):
            if goal_handle.is_cancel_requested:
                return self._cancel_skill(result, goal_handle, [], goal_handle.request.skill_name)
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_QUERY_TIMEOUT",
                "semantic map target query timed out",
            )
        try:
            response = future.result()
        except Exception as exc:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_QUERY_FAILED",
                f"semantic map target query failed: {exc}",
            )
        if response is None or not response.success or not response.semantic_map.objects:
            message = str(getattr(response, "message", "")).strip()
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_QUERY_FAILED",
                message or "semantic map target query failed",
            )

        object_pose = response.semantic_map.objects[0].pose.pose.pose.position
        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_link", Time(), timeout=Duration(seconds=min(timeout_sec, self._rpc_timeout))
            )
        except TransformException as exc:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SEMANTIC_MAP_TF_UNAVAILABLE",
                f"map to base_link transform is unavailable: {exc}",
            )
        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y
        dx = robot_x - object_pose.x
        dy = robot_y - object_pose.y
        distance = math.hypot(dx, dy) or 1.0
        ux, uy = dx / distance, dy / distance
        stand_off_distance_m = (
            float(goal_handle.request.distance)
            if float(goal_handle.request.distance) > 0.0
            else float(self._semantic_map_stand_off_distance_m)
        )
        goal_x = object_pose.x + ux * stand_off_distance_m
        goal_y = object_pose.y + uy * stand_off_distance_m
        yaw_degrees = math.degrees(math.atan2(object_pose.y - goal_y, object_pose.x - goal_x))
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        result.success = True
        result.error_code = ""
        result.message = json.dumps(
            [float(goal_x), float(goal_y), yaw_degrees],
            separators=(",", ":"),
        )
        result.executed_primitives = []
        goal_handle.succeed()
        return result

    @classmethod
    def _snapshot_policy_inputs(cls, snapshot):
        requirements = {
            str(name): SkillRequirements(
                validate_skill="validate_skill" in values,
                task_executor="task_executor" in values,
                arm_trajectory="arm_trajectory" in values,
                fresh_ee_pose="fresh_ee_pose" in values,
                navigation="navigation" in values,
            )
            for name, values in snapshot.requirements.items()
        }
        parameter_schemas = {
            str(name): cls._mutable_template_value(schema) for name, schema in snapshot.parameter_schemas.items()
        }
        timeout_caps = {str(name): float(template["timeout_sec"]) for name, template in snapshot.templates.items()}
        control_modes = {
            str(name): str(capability.get("required_control_mode", ""))
            for name, capability in snapshot.capability_view.items()
        }
        return requirements, parameter_schemas, timeout_caps, control_modes

    @staticmethod
    def _snapshot_skill_schema_versions(snapshot) -> dict[str, int]:
        return {
            str(name): int(capability.get("schema_version", 1)) for name, capability in snapshot.capability_view.items()
        }

    def _primitive_descriptor(self, primitive_name: str):
        descriptors = getattr(self, "_primitive_descriptors", None)
        if descriptors is None:
            context_version = getattr(self, "_context_schema_version", None)
            if context_version not in {1, 2, 3}:
                context_version = 2 if primitive_name in {"nav_straight", "nav_turn", "nav_abs_coordinate"} else 1
            descriptors = primitive_contract_for_version(context_version).descriptors
        return descriptors.get(str(primitive_name))

    @staticmethod
    def _skill_schema_version(bundle, skill_name: str) -> int:
        if bundle is None:
            return 1
        capability = bundle.snapshot.capability_view.get(skill_name, {})
        return int(capability.get("schema_version", 1))

    def _validate_skill_goal_schema(self, goal, bundle) -> str:
        try:
            submitted_version = validate_request_schema_version(getattr(goal, "schema_version", 1))
        except ValueError:
            return "SKILL_SCHEMA_INVALID"
        if bundle is not None and submitted_version != self._skill_schema_version(bundle, str(goal.skill_name)):
            return "SKILL_SCHEMA_INVALID"
        return ""

    def _validate_primitive_goal_schema(self, goal) -> str:
        try:
            submitted_version = validate_request_schema_version(getattr(goal, "schema_version", 1))
        except ValueError:
            return "SKILL_SCHEMA_INVALID"
        descriptor = self._primitive_descriptor(goal.primitive_name)
        if descriptor is None or descriptor.schema_version != submitted_version:
            return "SKILL_SCHEMA_INVALID"
        return ""

    @staticmethod
    def _fill_catalog_diagnostics(response, diagnostics) -> None:
        response.diagnostics = []
        for diagnostic in diagnostics:
            item = SkillDiagnostic()
            item.schema_version = int(diagnostic.schema_version)
            item.severity = int(diagnostic.severity)
            item.error_code = str(diagnostic.error_code)
            item.source_relative_path = str(diagnostic.source_relative_path)
            item.field_path = str(diagnostic.field_path)
            item.message = str(diagnostic.message)
            response.diagnostics.append(item)

    def _publish_registry_event(self, result) -> None:
        if not result.success or result.generation == result.old_generation:
            return
        event = SkillRegistryEvent()
        bundle = self._runtime_coordinator.current
        snapshot = bundle.snapshot if bundle is not None else None
        event.schema_version = 1
        event.registry_epoch = result.registry_epoch
        event.old_generation = result.old_generation
        event.new_generation = result.generation
        event.registry_digest = result.registry_digest
        event.capability_digest = result.capability_digest
        event.source_release_digest = result.source_release_digest
        event.provenance_digest = result.provenance_digest
        event.profile_name = snapshot.profile_name if snapshot is not None else ""
        event.changed_skills = list(result.changed_skills)
        self._skill_registry_event_publisher.publish(event)

    def _reload_skill_catalog(self, request, response):
        if request.schema_version != 1:
            response.success = False
            response.error_code = "SKILL_SCHEMA_INVALID"
            response.message = "schema_version must be 1"
            return response
        result, prepared = self._runtime_coordinator.prepare_reload(
            request.request_id,
            force=bool(request.force),
        )
        if prepared is not None:
            requirements, parameter_schemas, timeout_caps, control_modes = self._snapshot_policy_inputs(
                prepared.snapshot
            )
            templates = self._mutable_templates(prepared.snapshot.templates)
        with self._state_guard():
            if prepared is not None:
                result = self._runtime_coordinator.activate_reload(prepared)
            if prepared is not None and result is not None and result.success:
                bundle = self._runtime_coordinator.current
                self._gateway_policy.replace_catalog(
                    bundle.snapshot.robot_context.timeout_policy,
                    requirements,
                    parameter_schemas=parameter_schemas,
                    skill_timeout_caps=timeout_caps,
                    skill_schema_versions=self._snapshot_skill_schema_versions(bundle.snapshot),
                    skill_control_modes=control_modes,
                )
                self._skill_requirements = requirements
                self._skill_parameter_schemas = parameter_schemas
                self._skill_control_modes = control_modes
                self._skill_templates = templates
                self._publish_registry_event(result)
        assert result is not None
        response.success = result.success
        response.registry_epoch = result.registry_epoch
        response.old_generation = result.old_generation
        response.generation = result.generation
        response.registry_digest = result.registry_digest
        response.capability_digest = result.capability_digest
        response.source_release_digest = result.source_release_digest
        response.provenance_digest = result.provenance_digest
        response.error_code = result.error_code
        response.message = result.message
        response.changed_skills = list(result.changed_skills)
        self._fill_catalog_diagnostics(response, result.diagnostics)
        return response

    def _get_skill_snapshot(self, request, response):
        response.success = False
        if request.schema_version != 1 or (request.generation > 0 and not request.registry_epoch):
            response.error_code = "SKILL_SCHEMA_INVALID"
            response.message = "schema_version must be 1 and exact generation queries require registry_epoch"
            return response
        try:
            with self._state_guard():
                bundle = self._runtime_coordinator.get_snapshot(
                    registry_epoch=request.registry_epoch,
                    generation=int(request.generation),
                )
        except Exception as exc:
            response.error_code = getattr(exc, "code", "SKILL_SNAPSHOT_NOT_RETAINED")
            response.message = str(exc)
            return response
        snapshot = bundle.snapshot
        response.success = True
        response.registry_epoch = bundle.registry_epoch
        response.generation = bundle.generation
        response.registry_digest = snapshot.registry_digest
        response.capability_digest = snapshot.capability_digest
        response.source_release_digest = str(snapshot.provenance.get("source_release_digest", ""))
        response.provenance_digest = snapshot.provenance_digest
        response.profile_name = snapshot.profile_name
        response.snapshot_json = snapshot.snapshot_json
        response.error_code = ""
        response.message = ""
        return response

    @classmethod
    def _mutable_templates(cls, templates):
        return {str(name): cls._mutable_template_value(template) for name, template in templates.items()}

    @classmethod
    def _mutable_template_value(cls, value):
        if isinstance(value, Mapping):
            return {key: cls._mutable_template_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [cls._mutable_template_value(item) for item in value]
        if isinstance(value, list):
            return [cls._mutable_template_value(item) for item in value]
        return value

    def _handle_ee_pose(self, msg: PoseStamped) -> None:
        with self._state_guard():
            self._latest_ee_pose = msg
            self._latest_ee_pose_monotonic = time.monotonic()

    def _handle_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _runtime_snapshot(self) -> RuntimeSnapshot:
        owner = self._gateway_lease.owner
        active_workflow = self._active_workflow
        return RuntimeSnapshot(
            motion_authorized=self._motion_authorized,
            active_control_mode=self._active_control_mode,
            required_control_mode=self._skill_required_control_mode,
            busy=owner is not None or active_workflow is not None,
            active_task_id=(
                owner.root_task_id
                if owner is not None
                else active_workflow.root_task_id
                if active_workflow is not None
                else ""
            ),
            validate_ready=self._validate_skill_client.service_is_ready(),
            task_executor_ready=self._task_executor_client.server_is_ready(),
            arm_trajectory_ready=self._arm_trajectory_client.server_is_ready(),
            ee_pose_fresh=self._ee_pose_is_fresh(),
            navigation_ready=(
                self._navigation_client.server_is_ready() if self._context_schema_version >= 2 else False
            ),
            control_mode_switching_enabled=(
                self._context_schema_version >= 3
                and bool(getattr(self, "_supported_control_modes", ()))
                and bool(getattr(self, "_motion_mode_service", ""))
            ),
        )

    def _ee_pose_is_fresh(self) -> bool:
        with self._state_guard():
            return self._ee_pose_is_fresh_unlocked()

    def _ee_pose_is_fresh_unlocked(self) -> bool:
        return (
            self._latest_ee_pose is not None
            and self._latest_ee_pose_monotonic is not None
            and self._ee_pose_receipt_is_fresh(self._latest_ee_pose_monotonic)
        )

    def _ee_pose_receipt_is_fresh(self, received_monotonic: float) -> bool:
        return time.monotonic() - received_monotonic <= self._robot_state_freshness_sec

    def _fresh_ee_pose_snapshot(self) -> _EePoseSnapshot | None:
        with self._state_guard():
            if not self._ee_pose_is_fresh_unlocked():
                return None
            return _EePoseSnapshot(
                pose=copy.deepcopy(self._latest_ee_pose),
                received_monotonic=self._latest_ee_pose_monotonic,
            )

    @staticmethod
    def _task_budget_key(binding) -> tuple[int, int, int, int]:
        budget = binding.task_budget
        return (
            int(budget.started_at.sec),
            int(budget.started_at.nanosec),
            int(budget.deadline.sec),
            int(budget.deadline.nanosec),
        )

    def _task_budget_duration_sec(self, binding) -> float:
        started_sec, started_nanosec, deadline_sec, deadline_nanosec = self._task_budget_key(binding)
        if (
            binding.task_budget.schema_version != 1
            or started_sec < 0
            or deadline_sec < 0
            or not 0 <= started_nanosec < 1_000_000_000
            or not 0 <= deadline_nanosec < 1_000_000_000
        ):
            raise ValueError("invalid task budget")
        started = started_sec + started_nanosec / 1_000_000_000
        deadline = deadline_sec + deadline_nanosec / 1_000_000_000
        duration = deadline - started
        if not math.isfinite(duration) or duration <= 0.0 or deadline <= self._ros_time_sec():
            raise ValueError("task budget is expired or invalid")
        return duration

    def _validate_task_budget_schema(self, binding) -> None:
        started_sec, started_nanosec, deadline_sec, deadline_nanosec = self._task_budget_key(binding)
        if (
            binding.task_budget.schema_version != 1
            or started_sec < 0
            or deadline_sec < 0
            or not 0 <= started_nanosec < 1_000_000_000
            or not 0 <= deadline_nanosec < 1_000_000_000
            or (deadline_sec, deadline_nanosec) <= (started_sec, started_nanosec)
        ):
            raise ValueError("invalid task budget")

    def _remaining_task_budget_sec(self, binding) -> float:
        self._task_budget_duration_sec(binding)
        deadline_sec = binding.task_budget.deadline.sec + binding.task_budget.deadline.nanosec / 1_000_000_000
        remaining = deadline_sec - self._ros_time_sec()
        if not math.isfinite(remaining) or remaining <= 0.0:
            raise ValueError("task budget is expired or invalid")
        return remaining

    @classmethod
    def _task_budget_is_zero(cls, binding) -> bool:
        return binding.task_budget.schema_version == 0 and cls._task_budget_key(binding) == (0, 0, 0, 0)

    def _set_task_budget(self, binding, timeout_sec: float) -> None:
        started = self._ros_time_sec()
        deadline = started + timeout_sec
        binding.task_budget.schema_version = 1
        for target, value in ((binding.task_budget.started_at, started), (binding.task_budget.deadline, deadline)):
            target.sec = int(value)
            target.nanosec = int((value - int(value)) * 1_000_000_000)

    def _ros_time_sec(self) -> float:
        if not hasattr(self, "_clock"):
            raise ValueError("ROS clock is unavailable")
        return self.get_clock().now().nanoseconds / 1_000_000_000

    def _prepare_root_binding(self, binding, bundle, *, allow_zero_budget: bool) -> str:
        task_id = str(binding.task_id).strip()
        root_task_id = str(binding.root_task_id).strip()
        if (
            binding.schema_version != 1
            or not task_id
            or root_task_id != task_id
            or binding.workflow_step_index != 0
            or binding.workflow_digest
            or binding.root_lease_nonce
            or binding.dispatch_nonce
            or not binding.expected_registry_epoch
            or binding.expected_registry_generation <= 0
            or not binding.expected_registry_digest
        ):
            return "SKILL_SCHEMA_INVALID"
        binding.task_id = task_id
        binding.root_task_id = task_id
        if bundle is None or (
            binding.expected_registry_epoch,
            int(binding.expected_registry_generation),
            binding.expected_registry_digest,
        ) != (bundle.registry_epoch, bundle.generation, bundle.snapshot.registry_digest):
            return "SKILL_REGISTRY_VERSION_MISMATCH"
        if self._task_budget_is_zero(binding):
            if not allow_zero_budget:
                return "SKILL_SCHEMA_INVALID"
            self._set_task_budget(binding, self._task_budget_sec)
        try:
            duration = self._task_budget_duration_sec(binding)
        except ValueError:
            return "SKILL_SCHEMA_INVALID"
        if duration > self._task_budget_sec:
            return "TIMEOUT_EXCEEDS_POLICY"
        return ""

    def _set_primitive_result_catalog_identity(self, result, bundle=None) -> None:
        if bundle is None:
            bundle = getattr(self, "_active_runtime_bundle", None)
        if bundle is None:
            coordinator = getattr(self, "_runtime_coordinator", None)
            bundle = coordinator.current if coordinator is not None else None
        result.actual_registry_epoch = bundle.registry_epoch if bundle is not None else ""
        result.actual_registry_generation = bundle.generation if bundle is not None else 0
        result.actual_registry_digest = bundle.snapshot.registry_digest if bundle is not None else ""

    def _set_primitive_replay_identity(self, result, task_id: str) -> None:
        ledger = getattr(self, "_gateway_ledger", None)
        record = ledger.get(task_id) if ledger is not None else None
        metadata = record.terminal_metadata if record is not None else {}
        if record is not None and record.state == "terminal" and metadata.get("registry_epoch"):
            result.actual_registry_epoch = str(metadata["registry_epoch"])
            result.actual_registry_generation = int(metadata["registry_generation"])
            result.actual_registry_digest = str(metadata["registry_digest"])
            return
        self._set_primitive_result_catalog_identity(result)

    @staticmethod
    def _primitive_payload_digest(goal) -> str:
        """Digest the complete canonical external primitive payload before admission mutation."""
        is_navigation = str(goal.primitive_name) in {"nav_straight", "nav_turn", "nav_abs_coordinate"}
        payload = {
            "schema_version": validate_request_schema_version(getattr(goal, "schema_version", None)),
            "binding": _binding_key(goal.dispatch_binding),
            "primitive_name": str(goal.primitive_name),
            "pose_name": str(goal.pose_name),
            "target_pose": [
                float(goal.target_pose.position.x),
                float(goal.target_pose.position.y),
                float(goal.target_pose.position.z),
                float(goal.target_pose.orientation.x),
                float(goal.target_pose.orientation.y),
                float(goal.target_pose.orientation.z),
                float(goal.target_pose.orientation.w),
            ],
            "relative": [float(goal.relative_dx), float(goal.relative_dy), float(goal.relative_dz)],
            "velocity_scaling": float(goal.velocity_scaling),
            "gripper_position": float(goal.gripper_position),
            "joint_names": list(goal.joint_names),
            "joint_positions": [float(value) for value in goal.joint_positions],
            "primitive_duration_sec": float(goal.primitive_duration_sec),
            "joint_waypoints": [float(value) for value in goal.joint_waypoints],
            "joint_waypoint_count": int(goal.joint_waypoint_count),
            "waypoint_duration_sec": float(goal.waypoint_duration_sec),
            "timeout_sec": float(goal.timeout_sec),
        }
        if is_navigation:
            target = getattr(goal, "navigation_target_pose", PoseStamped())
            payload["navigation"] = {
                "command_type": int(getattr(goal, "navigation_command_type", 0)),
                "target_pose": {
                    "frame_id": str(target.header.frame_id),
                    "stamp": [int(target.header.stamp.sec), int(target.header.stamp.nanosec)],
                    "position": [
                        float(target.pose.position.x),
                        float(target.pose.position.y),
                        float(target.pose.position.z),
                    ],
                    "orientation": [
                        float(target.pose.orientation.x),
                        float(target.pose.orientation.y),
                        float(target.pose.orientation.z),
                        float(target.pose.orientation.w),
                    ],
                },
                "value": float(getattr(goal, "navigation_value", 0.0)),
            }
        return sha256_text(to_canonical_json(payload))

    @staticmethod
    def _set_begin_failure(response, error_code: str, message: str = ""):
        response.success = False
        response.error_code = error_code
        response.message = message or error_code
        return response

    def _begin_workflow_execution(self, request, response):
        binding = request.dispatch_binding
        root_task_id = str(binding.root_task_id).strip()
        try:
            steps = normalize_workflow_steps(request.workflow_steps)
            duration_sec = self._task_budget_duration_sec(binding)
        except (TypeError, ValueError) as exc:
            return self._set_begin_failure(response, "SKILL_SCHEMA_INVALID", str(exc))
        if (
            binding.schema_version != 1
            or not root_task_id
            or binding.task_id != root_task_id
            or binding.workflow_step_index != 0
            or binding.root_lease_nonce
            or binding.dispatch_nonce
            or not binding.workflow_digest
            or not binding.expected_registry_epoch
            or binding.expected_registry_generation <= 0
            or not binding.expected_registry_digest
        ):
            return self._set_begin_failure(response, "SKILL_SCHEMA_INVALID")
        try:
            expected_digest = compute_workflow_digest(
                root_task_id=root_task_id,
                task_budget=binding.task_budget,
                expected_registry_epoch=binding.expected_registry_epoch,
                expected_registry_generation=binding.expected_registry_generation,
                expected_registry_digest=binding.expected_registry_digest,
                workflow_steps=steps,
            )
        except (TypeError, ValueError) as exc:
            return self._set_begin_failure(response, "SKILL_SCHEMA_INVALID", str(exc))
        if expected_digest != binding.workflow_digest:
            return self._set_begin_failure(response, "SKILL_WORKFLOW_DIGEST_MISMATCH")

        with self._state_guard():
            active = self._active_workflow
            if active is not None:
                if (
                    active.root_task_id == root_task_id
                    and active.workflow_digest == expected_digest
                    and active.workflow_steps == steps
                    and active.task_budget_key == self._task_budget_key(binding)
                ):
                    response.success = True
                    response.root_lease_nonce = active.root_lease_nonce
                    response.workflow_digest = active.workflow_digest
                    response.actual_registry_epoch = active.bundle.registry_epoch
                    response.actual_registry_generation = active.bundle.generation
                    response.actual_registry_digest = active.bundle.snapshot.registry_digest
                    return response
                return self._set_begin_failure(response, "SKILL_REQUEST_ID_CONFLICT")
            if root_task_id in self._terminal_workflows:
                return self._set_begin_failure(response, "SKILL_REQUEST_ID_CONFLICT")
            if self._gateway_ledger.query(root_task_id, expected_digest).state:
                return self._set_begin_failure(response, "SKILL_REQUEST_ID_CONFLICT")

            coordinator = getattr(self, "_runtime_coordinator", None)
            bundle = coordinator.current if coordinator is not None else None
            if bundle is None or (
                bundle.registry_epoch,
                bundle.generation,
                bundle.snapshot.registry_digest,
            ) != (
                binding.expected_registry_epoch,
                binding.expected_registry_generation,
                binding.expected_registry_digest,
            ):
                return self._set_begin_failure(response, "SKILL_REGISTRY_VERSION_MISMATCH")

            snapshot = self._runtime_snapshot()
            owner = ExecutionOwner.workflow(root_task_id)
            requirements, parameter_schemas, timeout_caps, control_modes = self._snapshot_policy_inputs(bundle.snapshot)
            workflow_policy = GatewayPolicy(
                bundle.snapshot.robot_context.timeout_policy,
                requirements,
                parameter_schemas=parameter_schemas,
                skill_timeout_caps=timeout_caps,
                skill_schema_versions=self._snapshot_skill_schema_versions(bundle.snapshot),
                skill_control_modes=control_modes,
                ledger=self._gateway_ledger,
                lease=self._gateway_lease,
            )
            for index, step in enumerate(steps):
                if step.timeout_sec > 0.0 and step.timeout_sec > duration_sec:
                    return self._set_begin_failure(
                        response,
                        "SKILL_TASK_BUDGET_MISMATCH",
                        f"workflow step {index} timeout exceeds root task budget",
                    )
                decision = workflow_policy.evaluate(
                    GatewayRequest(
                        task_id=derive_skill_task_id(root_task_id, index),
                        skill_name=step.skill_name,
                        schema_version=step.schema_version,
                        target_name=step.target_name,
                        container_name=step.container_name,
                        place_name=step.place_name,
                        motion_direction=step.motion_direction,
                        motion_distance=step.motion_distance,
                        arm_side=step.arm_side,
                        imitation_duration_sec=step.imitation_duration_sec,
                        direction=step.direction,
                        distance=step.distance,
                        degree=step.degree,
                        x=step.x,
                        y=step.y,
                        yaw=step.yaw,
                        timeout_sec=step.timeout_sec or None,
                    ),
                    snapshot,
                    owner=owner,
                )
                if not decision.admitted:
                    return self._set_begin_failure(response, decision.error_code, decision.message)

            try:
                retained_bundle = self._runtime_coordinator.retain(bundle.generation)
            except Exception as exc:
                return self._set_begin_failure(response, "SKILL_SNAPSHOT_NOT_RETAINED", str(exc))
            error_code, lease_token = workflow_policy.admit_workflow(owner, snapshot, timeout_sec=duration_sec)
            if error_code or lease_token is None:
                with suppress(Exception):
                    self._runtime_coordinator.release(bundle.generation)
                return self._set_begin_failure(response, error_code or "SKILL_BUSY")
            try:
                self._gateway_ledger.begin(root_task_id, expected_digest)
            except Exception as exc:
                workflow_policy.release_workflow(owner, lease_token)
                self._runtime_coordinator.release(bundle.generation)
                return self._set_begin_failure(response, getattr(exc, "code", "SKILL_REQUEST_ID_CONFLICT"), str(exc))
            active = _WorkflowExecution(
                root_task_id=root_task_id,
                workflow_digest=expected_digest,
                root_lease_nonce=uuid.uuid4().hex,
                lease_token=lease_token,
                owner=owner,
                policy=workflow_policy,
                bundle=retained_bundle,
                workflow_steps=steps,
                task_budget_key=self._task_budget_key(binding),
                deadline_unix_sec=binding.task_budget.deadline.sec
                + binding.task_budget.deadline.nanosec / 1_000_000_000,
                step_terminal_states=[WORKFLOW_STEP_PENDING] * len(steps),
            )
            self._active_workflow = active

        response.success = True
        response.root_lease_nonce = active.root_lease_nonce
        response.workflow_digest = active.workflow_digest
        response.actual_registry_epoch = active.bundle.registry_epoch
        response.actual_registry_generation = active.bundle.generation
        response.actual_registry_digest = active.bundle.snapshot.registry_digest
        return response

    def _reap_expired_workflow(self) -> None:
        with self._state_guard():
            active = self._active_workflow
            if active is None or active.deadline_unix_sec > self._ros_time_sec():
                return
            active_child = self._active_skill_admission
            if isinstance(active_child, _WorkflowChildAdmission) and active_child.workflow is active:
                return
            try:
                self._record_workflow_terminal(
                    active,
                    FinalizeWorkflowExecution.Request.FAILED,
                    error_code="SKILL_TASK_DEADLINE_EXPIRED",
                )
                if not self._cleanup_terminal_workflow(active):
                    return
            except Exception as exc:
                self.get_logger().error(f"failed to reap expired workflow {active.root_task_id}: {exc}")
                return
            self._active_workflow = None

    def _record_workflow_terminal(
        self,
        active: _WorkflowExecution,
        terminal_state: int,
        *,
        error_code: str,
    ) -> _TerminalWorkflow:
        """Record idempotency state before any operation can free the root lease."""
        terminal = _TerminalWorkflow(
            workflow_digest=active.workflow_digest,
            root_lease_nonce=active.root_lease_nonce,
            registry_epoch=active.bundle.registry_epoch,
            registry_generation=active.bundle.generation,
            registry_digest=active.bundle.snapshot.registry_digest,
            task_budget_key=active.task_budget_key,
            terminal_state=terminal_state,
            completed_step_count=active.completed_step_count,
            workflow_steps=active.workflow_steps,
        )
        existing = self._terminal_workflows.get(active.root_task_id)
        if existing is not None and existing != terminal:
            raise ValueError("workflow terminal record conflicts with active finalization")
        if not active.terminal_recorded:
            self._gateway_ledger.terminal(
                active.root_task_id,
                active.workflow_digest,
                error_code=error_code,
                terminal_metadata={"completed_step_count": active.completed_step_count},
            )
            self._terminal_workflows[active.root_task_id] = terminal
            active.terminal_recorded = True
        return terminal

    def _cleanup_terminal_workflow(self, active: _WorkflowExecution) -> bool:
        """Release retained runtime state before making the root lease available."""
        if not active.runtime_generation_released:
            self._runtime_coordinator.release(active.bundle.generation)
            active.runtime_generation_released = True
        if not active.lease_released:
            if not active.policy.release_workflow(active.owner, active.lease_token):
                return False
            active.lease_released = True
        return True

    @staticmethod
    def _set_finalize_failure(response, error_code: str, message: str = ""):
        response.success = False
        response.error_code = error_code
        response.message = message or error_code
        return response

    def _finalize_workflow_execution(self, request, response):
        binding = request.dispatch_binding
        root_task_id = str(binding.root_task_id).strip()
        try:
            self._validate_task_budget_schema(binding)
        except ValueError as exc:
            return self._set_finalize_failure(response, "SKILL_SCHEMA_INVALID", str(exc))
        if (
            binding.schema_version != 1
            or not root_task_id
            or binding.task_id != root_task_id
            or not binding.workflow_digest
            or not binding.root_lease_nonce
            or request.terminal_state
            not in {
                FinalizeWorkflowExecution.Request.SUCCEEDED,
                FinalizeWorkflowExecution.Request.FAILED,
                FinalizeWorkflowExecution.Request.CANCELED,
            }
        ):
            return self._set_finalize_failure(response, "SKILL_SCHEMA_INVALID")
        if binding.workflow_step_index != 0:
            return self._set_finalize_failure(response, "SKILL_WORKFLOW_STEP_MISMATCH")
        if binding.dispatch_nonce:
            return self._set_finalize_failure(response, "SKILL_DISPATCH_NOT_AUTHORIZED")
        with self._state_guard():
            active = self._active_workflow
            if active is None:
                terminal = self._terminal_workflows.get(root_task_id)
                if terminal is None:
                    return self._set_finalize_failure(response, "SKILL_WORKFLOW_LEASE_MISMATCH")
                try:
                    recomputed_digest = compute_workflow_digest(
                        root_task_id=root_task_id,
                        task_budget=binding.task_budget,
                        expected_registry_epoch=binding.expected_registry_epoch,
                        expected_registry_generation=binding.expected_registry_generation,
                        expected_registry_digest=binding.expected_registry_digest,
                        workflow_steps=terminal.workflow_steps,
                    )
                except (TypeError, ValueError) as exc:
                    return self._set_finalize_failure(response, "SKILL_SCHEMA_INVALID", str(exc))
                if terminal.registry_epoch != binding.expected_registry_epoch:
                    return self._set_finalize_failure(response, "SKILL_REGISTRY_EPOCH_MISMATCH")
                if (
                    terminal.registry_generation != binding.expected_registry_generation
                    or terminal.registry_digest != binding.expected_registry_digest
                ):
                    return self._set_finalize_failure(response, "SKILL_REGISTRY_VERSION_MISMATCH")
                if recomputed_digest != binding.workflow_digest or terminal.workflow_digest != binding.workflow_digest:
                    return self._set_finalize_failure(response, "SKILL_WORKFLOW_DIGEST_MISMATCH")
                if terminal.root_lease_nonce != binding.root_lease_nonce:
                    return self._set_finalize_failure(response, "SKILL_WORKFLOW_LEASE_MISMATCH")
                if terminal.task_budget_key != self._task_budget_key(binding):
                    return self._set_finalize_failure(response, "SKILL_TASK_BUDGET_MISMATCH")
                if (
                    terminal.terminal_state != request.terminal_state
                    or terminal.completed_step_count != request.completed_step_count
                ):
                    return self._set_finalize_failure(response, "SKILL_REQUEST_ID_CONFLICT")
                response.success = True
                response.actual_terminal_state = terminal.terminal_state
                response.actual_completed_step_count = terminal.completed_step_count
                return response
            try:
                recomputed_digest = compute_workflow_digest(
                    root_task_id=root_task_id,
                    task_budget=binding.task_budget,
                    expected_registry_epoch=binding.expected_registry_epoch,
                    expected_registry_generation=binding.expected_registry_generation,
                    expected_registry_digest=binding.expected_registry_digest,
                    workflow_steps=active.workflow_steps,
                )
            except (TypeError, ValueError) as exc:
                return self._set_finalize_failure(response, "SKILL_SCHEMA_INVALID", str(exc))
            if active.root_task_id != root_task_id:
                return self._set_finalize_failure(response, "SKILL_WORKFLOW_LEASE_MISMATCH")
            if binding.expected_registry_epoch != active.bundle.registry_epoch:
                return self._set_finalize_failure(response, "SKILL_REGISTRY_EPOCH_MISMATCH")
            if (
                binding.expected_registry_generation != active.bundle.generation
                or binding.expected_registry_digest != active.bundle.snapshot.registry_digest
            ):
                return self._set_finalize_failure(response, "SKILL_REGISTRY_VERSION_MISMATCH")
            if recomputed_digest != binding.workflow_digest or active.workflow_digest != binding.workflow_digest:
                return self._set_finalize_failure(response, "SKILL_WORKFLOW_DIGEST_MISMATCH")
            if active.root_lease_nonce != binding.root_lease_nonce:
                return self._set_finalize_failure(response, "SKILL_WORKFLOW_LEASE_MISMATCH")
            if active.task_budget_key != self._task_budget_key(binding):
                return self._set_finalize_failure(response, "SKILL_TASK_BUDGET_MISMATCH")
            active_child = self._active_skill_admission
            if isinstance(active_child, _WorkflowChildAdmission) and active_child.workflow is active:
                return self._set_finalize_failure(
                    response, "SKILL_WORKFLOW_LEASE_MISMATCH", "workflow child cleanup pending"
                )
            if request.completed_step_count != active.completed_step_count:
                return self._set_finalize_failure(response, "SKILL_REQUEST_ID_CONFLICT")
            expected_step_state = {
                FinalizeWorkflowExecution.Request.FAILED: WORKFLOW_STEP_FAILED,
                FinalizeWorkflowExecution.Request.CANCELED: WORKFLOW_STEP_CANCELED,
            }.get(request.terminal_state)
            if request.terminal_state == FinalizeWorkflowExecution.Request.SUCCEEDED and (
                active.completed_step_count != len(active.workflow_steps)
                or any(state != WORKFLOW_STEP_SUCCEEDED for state in active.step_terminal_states)
            ):
                return self._set_finalize_failure(
                    response,
                    "SKILL_REQUEST_ID_CONFLICT",
                    "successful workflow must complete every step",
                )
            if expected_step_state is not None and active.completed_step_count < len(active.step_terminal_states):
                current_state = active.step_terminal_states[active.completed_step_count]
                allowed_states = (
                    {WORKFLOW_STEP_PENDING, WORKFLOW_STEP_FAILED}
                    if request.terminal_state == FinalizeWorkflowExecution.Request.FAILED
                    else {WORKFLOW_STEP_PENDING, WORKFLOW_STEP_CANCELED}
                )
                if current_state not in allowed_states:
                    return self._set_finalize_failure(
                        response,
                        "SKILL_REQUEST_ID_CONFLICT",
                        f"workflow step terminal state {current_state} conflicts with root finalization",
                    )
            if expected_step_state is not None and active.completed_step_count >= len(active.step_terminal_states):
                return self._set_finalize_failure(
                    response,
                    "SKILL_REQUEST_ID_CONFLICT",
                    "completed workflow cannot be finalized as failed or canceled",
                )
            error_code = (
                ""
                if request.terminal_state == FinalizeWorkflowExecution.Request.SUCCEEDED
                else (
                    "SKILL_CANCELLED"
                    if request.terminal_state == FinalizeWorkflowExecution.Request.CANCELED
                    else "CAPABILITY_NOT_READY"
                )
            )
            try:
                terminal = self._record_workflow_terminal(active, request.terminal_state, error_code=error_code)
            except Exception as exc:
                return self._set_finalize_failure(response, GATEWAY_FINALIZATION_FAILED, str(exc))
            try:
                if not self._cleanup_terminal_workflow(active):
                    return self._set_finalize_failure(response, GATEWAY_FINALIZATION_FAILED)
            except Exception as exc:
                return self._set_finalize_failure(response, GATEWAY_FINALIZATION_FAILED, str(exc))
            self._active_workflow = None
        response.success = True
        response.actual_terminal_state = terminal.terminal_state
        response.actual_completed_step_count = terminal.completed_step_count
        return response

    def _get_gateway_status(self, request, response):
        with self._state_guard():
            snapshot = self._runtime_snapshot()
            query = self._gateway_ledger.query(request.task_id, request.payload_hash)
            coordinator = getattr(self, "_runtime_coordinator", None)
            bundle = coordinator.current if coordinator is not None else None
            catalog_snapshot = bundle.snapshot if bundle is not None else None
            capability_names = (
                catalog_snapshot.enabled_skill_names if catalog_snapshot else tuple(sorted(self._skill_requirements))
            )
            capabilities = [self._capability_status(skill_name, snapshot) for skill_name in capability_names]
            owner = self._gateway_lease.owner
            active_workflow = self._active_workflow
            coordinator_state = self._runtime_coordinator.state
            coordinator_error_code = self._runtime_coordinator.error_code
            retained_generations = list(self._runtime_coordinator.retained_generations)
        response.schema_version = 1
        response.robot_name = self._robot_name
        response.motion_authorized = snapshot.motion_authorized
        response.active_control_mode = snapshot.active_control_mode
        response.busy = snapshot.is_busy
        response.active_task_id = snapshot.active_task_id
        response.default_skill_timeout_sec = self._default_skill_timeout_sec
        response.task_budget_sec = self._task_budget_sec
        response.rpc_timeout_sec = self._rpc_timeout
        response.config_digest = catalog_snapshot.capability_digest if catalog_snapshot else ""
        response.capability_digest = catalog_snapshot.capability_digest if catalog_snapshot else ""
        response.registry_epoch = bundle.registry_epoch if bundle else ""
        response.registry_generation = bundle.generation if bundle else 0
        response.registry_digest = catalog_snapshot.registry_digest if catalog_snapshot else ""
        response.primitive_contract_digest = self._primitive_contract_digest
        response.source_release_digest = (
            str(catalog_snapshot.provenance.get("source_release_digest", "")) if catalog_snapshot else ""
        )
        response.provenance_digest = catalog_snapshot.provenance_digest if catalog_snapshot else ""
        response.retained_generations = retained_generations
        owner_kind = owner.kind if owner else ""
        response.active_owner_kind = (
            "workflow" if active_workflow is not None else "catalog_entry" if owner_kind else ""
        )
        response.active_workflow_digest = active_workflow.workflow_digest if active_workflow else ""
        response.active_workflow_step_index = active_workflow.completed_step_count if active_workflow else -1
        response.control_plane_ready = coordinator_state == "READY"
        response.control_plane_state = coordinator_state
        response.control_plane_error_code = coordinator_error_code
        response.request_state = query.state
        response.request_error_code = query.error_code
        response.capabilities = capabilities
        return response

    def _capability_status(self, skill_name: str, snapshot: RuntimeSnapshot) -> SkillCapabilityStatus:
        coordinator = getattr(self, "_runtime_coordinator", None)
        bundle = coordinator.current if coordinator is not None else None
        decision = self._gateway_policy.evaluate(
            GatewayRequest(
                task_id=f"status-{skill_name}",
                skill_name=skill_name,
                schema_version=self._skill_schema_version(bundle, skill_name),
                arm_side="",
                imitation_duration_sec=0.0,
            ),
            snapshot,
            validate_parameters=False,
        )
        capability = SkillCapabilityStatus()
        capability.schema_version = 1
        capability.name = skill_name
        catalog_snapshot = bundle.snapshot if bundle else None
        capability.semantic_level = (
            catalog_snapshot.semantic_levels.get(skill_name, "skill") if catalog_snapshot else "skill"
        )
        capability.planner_visible = (
            skill_name in catalog_snapshot.planner_visible_skill_names if catalog_snapshot else True
        )
        capability.ready = decision.admitted
        capability.reason = f"{decision.error_code}: {decision.message}" if decision.error_code else ""
        capability.required_control_mode = (
            self._gateway_policy.required_control_mode(skill_name) or self._skill_required_control_mode
        )
        return capability

    def _audit(
        self,
        event: str,
        *,
        task_id: str = "",
        payload_hash: str = "",
        skill: str = "",
        error_code: str = "",
        duration_sec: float | None = None,
        step_count: int | None = None,
    ) -> None:
        payload = {"event": event}
        for field_name, value in (
            ("task_id", task_id),
            ("payload_hash", payload_hash),
            ("skill", skill),
            ("error_code", error_code),
            ("duration_sec", duration_sec),
            ("step_count", step_count),
        ):
            if value not in ("", None):
                payload[field_name] = value
        self.get_logger().info(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _public_audit_context_copy(self) -> dict[str, str]:
        with self._state_guard():
            context = getattr(self, "_active_audit_context", None)
            if context is None:
                return {}
            return {
                field_name: str(context[field_name])
                for field_name in ("task_id", "payload_hash", "skill")
                if field_name in context and context[field_name] not in ("", None)
            }

    def _audit_cancel_propagated(self) -> None:
        context = self._public_audit_context_copy()
        if context:
            self._audit("cancel_propagated", **context)

    def _handle_cancel(self, _cancel_request):
        self._audit("cancel_requested", **self._public_audit_context_copy())
        return CancelResponse.ACCEPT

    def _handle_skill_goal(self, _goal_request):
        # GatewayPolicy owns admission decisions and must see concurrent roots so
        # it can return the documented busy reason to callers.
        if hasattr(self, "_gateway_policy"):
            return GoalResponse.ACCEPT
        with self._skill_goal_lock:
            if self._skill_goal_active:
                return GoalResponse.REJECT
            self._skill_goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _wait_for_future(
        future,
        timeout_sec: float,
        cancel_requested: Callable[[], bool] | None = None,
        cancel_callback: Callable[[], None] | None = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if cancel_requested is not None and cancel_requested():
                if cancel_callback is not None:
                    cancel_callback()
                return False
            time.sleep(0.05)
        return future.done()

    def _required_control_mode_for_skill(self, skill_name: str) -> str:
        """Read the immutable catalog capability selected for this execution."""
        if skill_name.startswith("__primitive__:"):
            return skill_name.partition(":")[2].strip()
        bundle = getattr(self, "_active_runtime_bundle", None)
        if bundle is not None:
            capability = bundle.snapshot.capability_view.get(str(skill_name), {})
            mode = str(capability.get("required_control_mode", "")).strip()
            if mode:
                return mode
        return str(getattr(self, "_skill_control_modes", {}).get(str(skill_name), "")).strip()

    def _ensure_skill_control_mode(self, skill_name: str, goal_handle) -> tuple[bool, str]:
        """Switch arm/base ownership through the existing motion-mode service.

        The switch is deliberately performed before any delegated executor or
        primitive is dispatched.  A failed transition therefore fails closed.
        """
        # Keep object-level/unit-test fixtures that predate the motion-mode
        # client on the legacy unchecked path.
        if not hasattr(self, "_motion_mode_client") and not hasattr(self, "_supported_control_modes"):
            return True, ""
        required_mode = self._required_control_mode_for_skill(skill_name)
        if not required_mode:
            required_mode = str(getattr(self, "_skill_required_control_mode", "")).strip()
        if not required_mode:
            return False, "skill control mode is not declared"
        active_mode = str(getattr(self, "_active_control_mode", "")).strip()
        supported_modes = tuple(getattr(self, "_supported_control_modes", ()))
        motion_mode_service = str(getattr(self, "_motion_mode_service", "")).strip()
        # A production node always creates the client.  If a hybrid fixture
        # supplies supported modes but omits the client, fail closed instead of
        # dispatching under an unverified controller ownership state.
        if not hasattr(self, "_motion_mode_client"):
            if not supported_modes:
                return True, ""
            return False, "motion-mode client is unavailable"
        switching_enabled = bool(supported_modes) and bool(motion_mode_service)
        if not switching_enabled:
            if active_mode == required_mode or ("moveit" in active_mode.lower() and "moveit" in required_mode.lower()):
                return True, ""
            return False, f"requires {required_mode}, active mode is {active_mode}"
        if required_mode not in supported_modes:
            return False, f"unsupported control mode: {required_mode}"
        if active_mode == required_mode:
            return True, ""
        navigation_enabled = required_mode == "base_navigation"
        if required_mode not in {"moveit_planning", "base_navigation"}:
            return False, f"unsupported runtime control mode: {required_mode}"
        with self._motion_mode_switch_lock:
            if goal_handle.is_cancel_requested:
                return False, "skill canceled before control-mode switch"
            current_mode = str(getattr(self, "_active_control_mode", "")).strip()
            if current_mode == required_mode:
                return True, ""
            if not self._motion_mode_client.wait_for_service(timeout_sec=self._rpc_timeout):
                return False, f"motion-mode service unavailable: {motion_mode_service}"
            request = SetBool.Request()
            request.data = navigation_enabled
            try:
                future = self._motion_mode_client.call_async(request)
            except Exception as exc:
                return False, f"motion-mode switch failed: {exc}"
            # From this point until a successful response, the hardware mode is
            # intentionally treated as unknown so a later skill cannot skip a
            # transition after a timed-out service call.
            self._active_control_mode = ""
            if not self._wait_for_future(
                future,
                timeout_sec=self._rpc_timeout,
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                return False, "motion-mode switch timed out"
            try:
                response = future.result()
            except Exception as exc:
                return False, f"motion-mode switch failed: {exc}"
            if response is None or not bool(response.success):
                message = str(getattr(response, "message", "")).strip()
                return False, message or "motion-mode switch rejected"
            self._active_control_mode = required_mode
            return True, ""

    def _set_result_catalog_identity(self, result, bundle=None) -> None:
        if bundle is None:
            bundle = getattr(self, "_active_runtime_bundle", None)
        if bundle is None:
            coordinator = getattr(self, "_runtime_coordinator", None)
            bundle = coordinator.current if coordinator is not None else None
        snapshot = bundle.snapshot if bundle is not None else None
        result.actual_registry_epoch = bundle.registry_epoch if bundle is not None else ""
        result.actual_registry_generation = bundle.generation if bundle is not None else 0
        result.actual_registry_digest = snapshot.registry_digest if snapshot is not None else ""
        result.source_release_digest = (
            str(snapshot.provenance.get("source_release_digest", "")) if snapshot is not None else ""
        )
        result.provenance_digest = snapshot.provenance_digest if snapshot is not None else ""

    def _abort_skill(self, result, goal_handle, executed_primitives, error_code: str, message: str):
        """Set failure fields on *result*, abort the goal, and return result."""
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        result.success = False
        result.error_code = error_code
        result.message = message
        result.executed_primitives = executed_primitives
        goal_handle.abort()
        return result

    def _cancel_skill(self, result, goal_handle, executed_primitives, skill_name: str):
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        result.success = False
        result.error_code = "SKILL_CANCELLED"
        result.message = f"skill cancelled: {skill_name}"
        result.executed_primitives = executed_primitives
        goal_handle.canceled()
        return result

    def _finish_primitive_failure(self, result, goal_handle, error_code: str, message: str, pose_name: str):
        self._set_primitive_result_catalog_identity(result)
        result.success = False
        result.error_code = (
            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT if message.startswith("cancel cleanup timed out") else error_code
        )
        result.message = message
        result.pose_name = pose_name
        if result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
            goal_handle.abort()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    @staticmethod
    def _late_cleanup_confirmation(goal_handle) -> _LateCleanupConfirmation | None:
        return getattr(goal_handle, "late_cleanup_confirmation", None)

    def _best_effort_cancel_goal(self, goal_handle, *, audit: bool = True) -> None:
        if audit:
            self._audit_cancel_propagated()
        with suppress(Exception):
            goal_handle.cancel_goal_async()

    def _cancel_goal(self, goal_handle, result_future=None, late_cleanup_confirmation=None) -> bool:
        if goal_handle is None:
            return True
        self._audit_cancel_propagated()
        if result_future is None:
            try:
                result_future = goal_handle.get_result_async()
            except Exception:
                self._best_effort_cancel_goal(goal_handle, audit=False)
                return False
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if not self._wait_for_future(cancel_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        try:
            cancel_response = cancel_future.result()
        except Exception:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if hasattr(cancel_response, "goals_canceling") and not cancel_response.goals_canceling:
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        if not self._wait_for_future(result_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_result_future(result_future)
            return False
        try:
            wrapped_result = result_future.result()
        except Exception:
            return False
        child_result = getattr(wrapped_result, "result", None)
        return getattr(child_result, "error_code", "") != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT

    def _cancel_goal_when_ready(self, goal_future, late_cleanup_confirmation=None) -> None:
        def cancel_accepted_goal(completed_future) -> None:
            try:
                goal_handle = completed_future.result()
            except Exception:
                return
            if goal_handle is not None and goal_handle.accepted:
                self._cancel_goal(goal_handle, late_cleanup_confirmation=late_cleanup_confirmation)

        goal_future.add_done_callback(cancel_accepted_goal)

    def _cancel_goal_future(self, goal_future, late_cleanup_confirmation=None) -> bool:
        if not self._wait_for_future(goal_future, timeout_sec=self._rpc_timeout):
            if late_cleanup_confirmation is not None:
                late_cleanup_confirmation.watch_goal_future(goal_future, self._best_effort_cancel_goal)
            else:
                self._cancel_goal_when_ready(goal_future)
            return False
        try:
            goal_handle = goal_future.result()
        except Exception:
            return False
        if goal_handle is None or not goal_handle.accepted:
            return True
        return self._cancel_goal(goal_handle, late_cleanup_confirmation=late_cleanup_confirmation)

    def _sleep_with_cancel(self, goal_handle, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(0.05)
        return True

    def _wait_for_pose_target(self, goal_handle, target_pose: Pose, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            ee_pose_snapshot = self._fresh_ee_pose_snapshot()
            if ee_pose_snapshot is not None:
                ee_pose = ee_pose_snapshot.pose
                dx = float(ee_pose.pose.position.x - target_pose.position.x)
                dy = float(ee_pose.pose.position.y - target_pose.position.y)
                dz = float(ee_pose.pose.position.z - target_pose.position.z)
                if (dx * dx + dy * dy + dz * dz) ** 0.5 <= EE_POSITION_TOLERANCE_M:
                    return True
            time.sleep(0.05)
        return False

    def _validate_skill(
        self,
        skill_name: str,
        target_name: str,
        place_name: str,
        container_name: str = "",
        motion_direction: str = "",
        motion_distance: float = 0.0,
        arm_side: str = "",
        imitation_duration_sec: float = 0.0,
        direction: str = "",
        distance: float = 0.0,
        degree: float = 0.0,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        schema_version: int | None = None,
        dispatch_binding=None,
    ) -> tuple[bool, str]:
        if not self._validate_skill_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, "validate_skill service unavailable"

        request = ValidateSkill.Request()
        request.schema_version = validate_request_schema_version(schema_version if schema_version is not None else 1)
        if dispatch_binding is not None:
            request.dispatch_binding = copy_binding(dispatch_binding)
        request.skill_name = skill_name
        request.target_name = target_name
        request.container_name = container_name
        request.place_name = place_name
        request.motion_direction = motion_direction
        request.motion_distance = float(motion_distance)
        request.arm_side = arm_side
        request.imitation_duration_sec = float(imitation_duration_sec)
        request.direction = direction
        request.distance = float(distance)
        request.degree = float(degree)
        request.has_x = x is not None
        request.has_y = y is not None
        request.has_yaw = yaw is not None
        request.x = float(x) if x is not None else 0.0
        request.y = float(y) if y is not None else 0.0
        request.yaw = float(yaw) if yaw is not None else 0.0
        future = self._validate_skill_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=self._rpc_timeout):
            return False, "validate_skill timeout"
        response = future.result()
        if response is None:
            return False, "validate_skill returned no response"
        expected_identity = (
            request.dispatch_binding.expected_registry_epoch,
            int(request.dispatch_binding.expected_registry_generation),
            request.dispatch_binding.expected_registry_digest,
        )
        actual_identity = (
            response.actual_registry_epoch,
            int(response.actual_registry_generation),
            response.actual_registry_digest,
        )
        if any(expected_identity) and actual_identity != expected_identity:
            return False, "SKILL_REGISTRY_VERSION_MISMATCH"
        return response.allowed, response.reason

    def _validate_primitive(
        self,
        primitive_name: str,
        pose_name: str,
        gripper_position: float,
        relative_dx: float = 0.0,
        relative_dy: float = 0.0,
        relative_dz: float = 0.0,
        target_pose: Pose | None = None,
        velocity_scaling: float = 0.0,
        joint_names: list[str] | None = None,
        joint_positions: list[float] | None = None,
        joint_waypoints: list[float] | None = None,
        joint_waypoint_count: int = 0,
        primitive_duration_sec: float = 0.0,
        waypoint_duration_sec: float = 0.0,
        navigation_command_type: int = 0,
        navigation_target_pose: PoseStamped | None = None,
        navigation_value: float = 0.0,
        dispatch_binding=None,
    ) -> tuple[bool, str]:
        if not self._validate_primitive_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, "validate_primitive service unavailable"

        request = ValidatePrimitive.Request()
        if dispatch_binding is not None:
            request.dispatch_binding = copy_binding(dispatch_binding)
        descriptor = self._primitive_descriptor(primitive_name)
        if descriptor is None:
            return False, f"unsupported primitive: {primitive_name}"
        if hasattr(request, "schema_version"):
            request.schema_version = descriptor.schema_version
        request.primitive_name = primitive_name
        request.pose_name = pose_name
        request.relative_dx = float(relative_dx)
        request.relative_dy = float(relative_dy)
        request.relative_dz = float(relative_dz)
        if target_pose is not None:
            request.target_x = float(target_pose.position.x)
            request.target_y = float(target_pose.position.y)
            request.target_z = float(target_pose.position.z)
            request.target_qx = float(target_pose.orientation.x)
            request.target_qy = float(target_pose.orientation.y)
            request.target_qz = float(target_pose.orientation.z)
            request.target_qw = float(target_pose.orientation.w)
        else:
            request.target_x = 0.0
            request.target_y = 0.0
            request.target_z = 0.0
            request.target_qx = 0.0
            request.target_qy = 0.0
            request.target_qz = 0.0
            request.target_qw = 0.0
        request.velocity_scaling = float(velocity_scaling)
        request.gripper_position = float(gripper_position)
        request.joint_names = list(joint_names or [])
        request.joint_positions = [float(position) for position in (joint_positions or [])]
        request.joint_waypoints = [float(position) for position in (joint_waypoints or [])]
        request.joint_waypoint_count = int(joint_waypoint_count)
        request.primitive_duration_sec = float(primitive_duration_sec)
        request.waypoint_duration_sec = float(waypoint_duration_sec)
        request.navigation_command_type = int(navigation_command_type)
        if navigation_target_pose is not None:
            request.navigation_target_pose = copy.deepcopy(navigation_target_pose)
        request.navigation_value = float(navigation_value)
        future = self._validate_primitive_client.call_async(request)
        if not self._wait_for_future(future, timeout_sec=self._rpc_timeout):
            return False, "validate_primitive timeout"
        response = future.result()
        if response is None:
            return False, "validate_primitive returned no response"
        expected_identity = (
            request.dispatch_binding.expected_registry_epoch,
            int(request.dispatch_binding.expected_registry_generation),
            request.dispatch_binding.expected_registry_digest,
        )
        actual_identity = (
            response.actual_registry_epoch,
            int(response.actual_registry_generation),
            response.actual_registry_digest,
        )
        if any(expected_identity) and actual_identity != expected_identity:
            return False, "SKILL_REGISTRY_VERSION_MISMATCH"
        return response.allowed, response.reason

    def _current_joint_positions(self) -> dict[str, float]:
        if self._latest_joint_state is None:
            return {}
        return {
            str(name): float(position)
            for name, position in zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=False)
        }

    def _pose_from_name(self, pose_name: str) -> Pose:
        pose_cfg = self._named_poses[pose_name]
        position = pose_cfg.get("position", {})
        orientation = pose_cfg.get("orientation", {})
        pose = Pose()
        pose.position.x = float(position.get("x", 0.0))
        pose.position.y = float(position.get("y", 0.0))
        pose.position.z = float(position.get("z", 0.0))
        pose.orientation.x = float(orientation.get("x", 0.0))
        pose.orientation.y = float(orientation.get("y", 0.0))
        pose.orientation.z = float(orientation.get("z", 0.0))
        pose.orientation.w = float(orientation.get("w", 1.0))
        return pose

    @staticmethod
    def _pose_from_relative_offset(ee_pose: PoseStamped, dx: float, dy: float, dz: float) -> Pose:
        pose = Pose()
        pose.position.x = float(ee_pose.pose.position.x + dx)
        pose.position.y = float(ee_pose.pose.position.y + dy)
        pose.position.z = float(ee_pose.pose.position.z + dz)
        # Keep current EE orientation unchanged during position-only moves
        pose.orientation.x = float(ee_pose.pose.orientation.x)
        pose.orientation.y = float(ee_pose.pose.orientation.y)
        pose.orientation.z = float(ee_pose.pose.orientation.z)
        pose.orientation.w = float(ee_pose.pose.orientation.w)
        return pose

    @staticmethod
    def _pose_from_gripper_rotation(ee_pose: PoseStamped, primitive_name: str, angle_deg: float) -> Pose:
        angle_rad = math.radians(abs(angle_deg))
        half = angle_rad / 2.0
        sign = -1.0 if primitive_name == "rotate_gripper_cw" else 1.0
        cur = ee_pose.pose.orientation
        qc = (cur.x, cur.y, cur.z, cur.w)
        qd = (0.0, 0.0, sign * math.sin(half), math.cos(half))

        # Right-multiply: q_target = q_current * q_delta (local-frame rotation)
        qx = qc[3] * qd[0] + qc[0] * qd[3] + qc[1] * qd[2] - qc[2] * qd[1]
        qy = qc[3] * qd[1] - qc[0] * qd[2] + qc[1] * qd[3] + qc[2] * qd[0]
        qz = qc[3] * qd[2] + qc[0] * qd[1] - qc[1] * qd[0] + qc[2] * qd[3]
        qw = qc[3] * qd[3] - qc[0] * qd[0] - qc[1] * qd[1] - qc[2] * qd[2]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm > 1e-9:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        target_pose = Pose()
        target_pose.position.x = float(ee_pose.pose.position.x)
        target_pose.position.y = float(ee_pose.pose.position.y)
        target_pose.position.z = float(ee_pose.pose.position.z)
        target_pose.orientation.x = qx
        target_pose.orientation.y = qy
        target_pose.orientation.z = qz
        target_pose.orientation.w = qw
        return target_pose

    @staticmethod
    def _goal_id_key(goal_id: UUID) -> bytes:
        return bytes(goal_id.uuid)

    def _state_guard(self):
        # _state_lock is initialized in __init__ before any callback can fire;
        # the getattr fallback only protects against test fixtures that skip __init__.
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = RLock()
            object.__setattr__(self, "_state_lock", lock)
        return lock

    def _register_internal_primitive_goal(self, goal_id: UUID, admission, binding) -> bytes:
        goal_key = self._goal_id_key(goal_id)
        with self._state_guard():
            self._pending_internal_primitive_goals[goal_key] = (admission, copy_binding(binding))
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context={}),
            )
            if cleanup.admission is admission:
                cleanup.pending_goal_keys.add(goal_key)
                cleanup.confirmed_goal_keys.discard(goal_key)
        return goal_key

    def _register_delegated_dispatch(self, nonce: str, admission, binding) -> bytes:
        cleanup_key = nonce.encode("utf-8")
        with self._state_guard():
            self._active_delegated_dispatches[nonce] = (admission, copy_binding(binding))
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context={}),
            )
            if cleanup.admission is admission:
                cleanup.pending_goal_keys.add(cleanup_key)
                cleanup.confirmed_goal_keys.discard(cleanup_key)
        return cleanup_key

    def _confirm_delegated_terminal(self, admission, nonce: str, cleanup_key: bytes) -> None:
        with self._state_guard():
            self._active_delegated_dispatches.pop(nonce, None)
        self._confirm_late_internal_cleanup(admission, cleanup_key)

    def _register_delegated_primitive_cleanup(self, admission, goal_key: bytes) -> None:
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context={}),
            )
            if cleanup.admission is admission:
                cleanup.pending_goal_keys.add(goal_key)
                cleanup.confirmed_goal_keys.discard(goal_key)

    def _activate_internal_primitive_goal(self, goal_handle):
        goal_id = getattr(goal_handle, "goal_id", None)
        if goal_id is None:
            return b"", None
        goal_key = self._goal_id_key(goal_id)
        with self._state_guard():
            pending = getattr(self, "_pending_internal_primitive_goals", {}).pop(goal_key, None)
            if pending is None or _binding_key(pending[1]) != _binding_key(goal_handle.request.dispatch_binding):
                return goal_key, None
            self._active_internal_primitive_goals[goal_key] = pending
            return goal_key, pending[0]

    def _forget_internal_primitive_goal(self, goal_key: bytes) -> None:
        with self._state_guard():
            getattr(self, "_pending_internal_primitive_goals", {}).pop(goal_key, None)
            getattr(self, "_active_internal_primitive_goals", {}).pop(goal_key, None)

    def _retain_admission_cleanup(
        self,
        admission,
        audit_context: dict[str, str],
        duration_sec: float,
        step_count: int,
    ) -> None:
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.setdefault(
                id(admission),
                _RetainedAdmissionCleanup(admission=admission, audit_context=dict(audit_context)),
            )
            cleanup.audit_context = dict(audit_context)
            cleanup.duration_sec = duration_sec
            cleanup.step_count = step_count
            cleanup.retained = True
            cleanup.parent_terminal = True
        self._converge_retained_admission(admission)

    def _confirm_late_internal_cleanup(self, admission, goal_key: bytes) -> None:
        self._forget_internal_primitive_goal(goal_key)
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.get(id(admission))
            if cleanup is not None and cleanup.admission is admission and goal_key in cleanup.pending_goal_keys:
                cleanup.confirmed_goal_keys.add(goal_key)
        self._converge_retained_admission(admission)

    def _converge_retained_admission(self, admission) -> None:
        with self._state_guard():
            cleanup = self._retained_admission_cleanup.get(id(admission))
            if (
                cleanup is None
                or cleanup.admission is not admission
                or not cleanup.retained
                or not cleanup.parent_terminal
                or not cleanup.pending_goal_keys
                or cleanup.confirmed_goal_keys != cleanup.pending_goal_keys
                or cleanup.finalizing
                or self._active_skill_admission is not admission
            ):
                return
            if isinstance(admission, _WorkflowChildAdmission):
                self._active_skill_admission = None
                self._active_skill_owner = None
                self._active_audit_context = None
                self._active_runtime_bundle = None
                del self._retained_admission_cleanup[id(admission)]
                return
            cleanup.finalizing = True
            try:
                self._gateway_policy.finalize(
                    admission,
                    error_code=SKILL_CANCEL_TIMEOUT,
                    terminal_metadata={"duration_sec": cleanup.duration_sec, "step_count": cleanup.step_count},
                )
            except Exception:
                cleanup.finalizing = False
                return
            self._active_skill_admission = None
            self._active_skill_owner = None
            self._active_audit_context = None
            active_bundle = getattr(self, "_active_runtime_bundle", None)
            if getattr(self, "_active_runtime_generation_retained", False) and active_bundle is not None:
                with suppress(Exception):
                    self._runtime_coordinator.release(active_bundle.generation)
            self._active_runtime_generation_retained = False
            self._active_runtime_bundle = None
            del self._retained_admission_cleanup[id(admission)]

    def _release_late_external_primitive(self, token) -> None:
        """Converge terminal ledger and retained generation before reopening the root lease."""
        admission = getattr(self, "_external_admissions", {}).get(id(token))
        if admission is None:
            with suppress(Exception):
                self._gateway_policy.release_external_primitive(token)
            return
        try:
            if not admission.terminal_recorded:
                self._gateway_policy.terminal_external_primitive(
                    admission.task_id,
                    admission.payload_digest,
                    token,
                    error_code=PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    terminal_metadata={
                        "registry_epoch": admission.bundle.registry_epoch,
                        "registry_generation": admission.bundle.generation,
                        "registry_digest": admission.bundle.snapshot.registry_digest,
                    },
                )
                admission.terminal_recorded = True
            if not admission.generation_released:
                self._runtime_coordinator.release(admission.bundle.generation)
                admission.generation_released = True
            if not admission.lease_released:
                admission.lease_released = self._gateway_policy.release_external_primitive(token)
            if admission.lease_released:
                self._external_admissions.pop(id(token), None)
        except Exception:
            return

    def _schedule_late_internal_cleanup(self, goal_future, admission, goal_key: bytes) -> None:
        confirmation = _LateCleanupConfirmation()
        confirmation.add_callback(lambda: self._confirm_late_internal_cleanup(admission, goal_key))
        confirmation.watch_goal_future(goal_future, self._best_effort_cancel_goal)

    def _admit_primitive(self, goal, internal_admission) -> tuple[str, object | None, object | None, bool]:
        schema_error = self._validate_primitive_goal_schema(goal)
        if schema_error:
            return schema_error, None, None, False
        snapshot = self._runtime_snapshot()
        delegated_primitive = False
        if internal_admission is None and goal.dispatch_binding.dispatch_nonce:
            with self._state_guard():
                delegated = self._active_delegated_dispatches.get(goal.dispatch_binding.dispatch_nonce)
            if delegated is None or _binding_key(goal.dispatch_binding) != _binding_key(delegated[1]):
                return SKILL_BUSY, None, None, False
            internal_admission = delegated[0]
            delegated_primitive = True
        if isinstance(internal_admission, _WorkflowChildAdmission):
            workflow = internal_admission.workflow
            try:
                remaining_budget = self._remaining_task_budget_sec(goal.dispatch_binding)
            except ValueError:
                return "SKILL_SCHEMA_INVALID", None, None, False
            if goal.timeout_sec > remaining_budget:
                return TIMEOUT_EXCEEDS_POLICY, None, None, False
            if goal.timeout_sec <= 0.0:
                goal.timeout_sec = remaining_budget
            borrow = workflow.policy.borrow_workflow_internal(
                workflow.owner,
                workflow.lease_token,
                _binding_task_id(goal),
                str(goal.primitive_name),
            )
            return (
                ("", None, internal_admission, delegated_primitive)
                if borrow is not None
                else (SKILL_BUSY, None, None, False)
            )
        if (
            internal_admission is not None
            and self._gateway_policy.borrow_internal(
                internal_admission,
                _binding_task_id(goal),
                str(goal.primitive_name),
            )
            is not None
        ):
            try:
                remaining_budget = self._remaining_task_budget_sec(goal.dispatch_binding)
            except ValueError:
                return "SKILL_SCHEMA_INVALID", None, None, False
            if goal.timeout_sec > remaining_budget:
                return TIMEOUT_EXCEEDS_POLICY, None, None, False
            if goal.timeout_sec <= 0.0:
                goal.timeout_sec = remaining_budget
            return "", None, internal_admission, delegated_primitive
        coordinator = getattr(self, "_runtime_coordinator", None)
        if coordinator is None:
            task_id = _binding_task_id(goal) or f"external-primitive-{uuid.uuid4()}"
            error_code, token = self._gateway_policy.admit_external_primitive(task_id, snapshot)
            return error_code, token, None, False
        with self._state_guard():
            if coordinator.state != "READY":
                return "SKILL_RELOAD_IN_PROGRESS", None, None, False
            bundle = coordinator.current
            payload_digest = self._primitive_payload_digest(goal)
            binding_error = self._prepare_root_binding(goal.dispatch_binding, bundle, allow_zero_budget=True)
            if binding_error:
                return binding_error, None, None, False
            try:
                remaining_budget = self._remaining_task_budget_sec(goal.dispatch_binding)
            except ValueError:
                return "SKILL_SCHEMA_INVALID", None, None, False
            if goal.timeout_sec > remaining_budget:
                return TIMEOUT_EXCEEDS_POLICY, None, None, False
            if goal.timeout_sec <= 0.0:
                goal.timeout_sec = remaining_budget
            if not goal.dispatch_binding.dispatch_nonce:
                goal.dispatch_binding.schema_version = 1
                goal.dispatch_binding.task_id = _binding_task_id(goal) or f"external-primitive-{uuid.uuid4()}"
                goal.dispatch_binding.root_task_id = goal.dispatch_binding.task_id
                goal.dispatch_binding.dispatch_nonce = uuid.uuid4().hex
            task_id = _binding_task_id(goal) or f"external-primitive-{uuid.uuid4()}"
            error_code, token = self._gateway_policy.admit_external_primitive(task_id, payload_digest, snapshot)
            if not error_code and token is not None:
                try:
                    retained_bundle = self._runtime_coordinator.retain(bundle.generation)
                    self._external_admissions[id(token)] = _ExternalPrimitiveAdmission(
                        task_id=task_id,
                        payload_digest=payload_digest,
                        lease_token=token,
                        bundle=retained_bundle,
                    )
                except Exception:
                    self._gateway_policy.release_external_primitive(token)
                    return "SKILL_SNAPSHOT_NOT_RETAINED", None, None, False
            return error_code, token, None, False

    def _execute_primitive(self, goal_handle):
        # See _execute_skill for the rationale of this test-fixture fallback.
        if not hasattr(self, "_gateway_policy"):
            return self._execute_primitive_unchecked(goal_handle)

        schema_error = self._validate_primitive_goal_schema(goal_handle.request)
        if schema_error:
            result = PrimitiveCommand.Result()
            result.success = False
            result.error_code = schema_error
            result.message = "primitive schema_version does not match its selected descriptor"
            result.pose_name = goal_handle.request.pose_name
            self._set_primitive_replay_identity(result, _binding_task_id(goal_handle.request))
            goal_handle.abort()
            return result

        goal_key, internal_admission = self._activate_internal_primitive_goal(goal_handle)
        error_code, token, internal_admission, delegated_primitive = self._admit_primitive(
            goal_handle.request, internal_admission
        )
        if error_code:
            if internal_admission is not None:
                self._forget_internal_primitive_goal(goal_key)
            result = PrimitiveCommand.Result()
            result.success = False
            result.error_code = error_code
            result.message = error_code
            result.pose_name = goal_handle.request.pose_name
            self._set_primitive_replay_identity(result, _binding_task_id(goal_handle.request))
            goal_handle.abort()
            return result
        if token is not None and hasattr(self, "_runtime_coordinator"):
            external_admission = getattr(self, "_external_admissions", {}).get(id(token))
            external_payload_digest = (
                external_admission.payload_digest
                if external_admission is not None
                else self._primitive_payload_digest(goal_handle.request)
            )
        else:
            external_payload_digest = ""
        deferred_goal_handle = _DeferredTerminalGoalHandle(goal_handle)
        late_cleanup_confirmation = _LateCleanupConfirmation()
        if internal_admission is not None:
            if delegated_primitive:
                self._register_delegated_primitive_cleanup(internal_admission, goal_key)
            late_cleanup_confirmation.add_callback(
                lambda: self._confirm_late_internal_cleanup(internal_admission, goal_key)
            )
        elif token is not None:
            late_cleanup_confirmation.add_callback(lambda: self._release_late_external_primitive(token))
        deferred_goal_handle.late_cleanup_confirmation = late_cleanup_confirmation
        result = None
        try:
            result = self._execute_primitive_unchecked(deferred_goal_handle)
        except Exception:
            self.get_logger().error(f"primitive execution failed:\n{traceback.format_exc()}")
            result = PrimitiveCommand.Result()
            result.success = False
            result.error_code = PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
            result.message = "cancel cleanup timed out: primitive execution state is unknown"
            result.pose_name = goal_handle.request.pose_name
            self._set_primitive_result_catalog_identity(result)
            deferred_goal_handle.force_abort()
            return result
        finally:
            if internal_admission is not None and not delegated_primitive:
                self._forget_internal_primitive_goal(goal_key)
        if token is not None and result.error_code != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
            try:
                external_admission = getattr(self, "_external_admissions", {}).get(id(token))
                if external_admission is not None:
                    self._set_primitive_result_catalog_identity(result, external_admission.bundle)
                terminal_external = getattr(self._gateway_policy, "terminal_external_primitive", None)
                if terminal_external is not None:
                    terminal_external(
                        _binding_task_id(goal_handle.request),
                        external_payload_digest,
                        token,
                        error_code=result.error_code,
                        terminal_metadata=(
                            {
                                "registry_epoch": external_admission.bundle.registry_epoch,
                                "registry_generation": external_admission.bundle.generation,
                                "registry_digest": external_admission.bundle.snapshot.registry_digest,
                            }
                            if external_admission is not None
                            else None
                        ),
                    )
                if external_admission is not None:
                    self._runtime_coordinator.release(external_admission.bundle.generation)
                    external_admission.generation_released = True
                released = self._gateway_policy.release_external_primitive(token)
                if external_admission is not None:
                    external_admission.lease_released = bool(released)
                    self._external_admissions.pop(id(token), None)
            except Exception:
                released = False
            if not released:
                result.success = False
                result.error_code = GATEWAY_FINALIZATION_FAILED
                result.message = "external primitive lease finalization failed"
                self._set_primitive_result_catalog_identity(result)
                deferred_goal_handle.force_abort()
                return result
        elif token is not None:
            external_admission = getattr(self, "_external_admissions", {}).get(id(token))
            if external_admission is not None:
                self._set_primitive_result_catalog_identity(result, external_admission.bundle)
        committed = deferred_goal_handle.commit()
        if committed and internal_admission is not None and result.error_code != PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
            late_cleanup_confirmation.confirm()
        return result

    def _execute_primitive_unchecked(self, goal_handle):
        goal = goal_handle.request
        result = PrimitiveCommand.Result()

        primitive_mode = (
            "base_navigation"
            if goal.primitive_name
            in {
                "nav_straight",
                "nav_turn",
                "nav_abs_coordinate",
            }
            else "moveit_planning"
        )
        mode_ready, mode_reason = self._ensure_skill_control_mode(f"__primitive__:{primitive_mode}", goal_handle)
        if not mode_ready:
            return self._finish_primitive_failure(
                result,
                goal_handle,
                "CONTROL_MODE_MISMATCH",
                mode_reason,
                goal.pose_name,
            )

        schema_error = self._validate_primitive_goal_schema(goal)
        if schema_error:
            result.error_code = schema_error
            result.message = "primitive schema_version does not match its selected descriptor"
            result.pose_name = goal.pose_name
            goal_handle.abort()
            return result

        target_pose = None
        ee_pose_snapshot = None
        if goal.primitive_name in {"move_relative_ee", "rotate_gripper_cw", "rotate_gripper_ccw"}:
            ee_pose_snapshot = self._fresh_ee_pose_snapshot()
            if ee_pose_snapshot is None:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
        if goal.primitive_name == "move_to_pose":
            target_pose = goal.target_pose
        elif goal.primitive_name == "move_relative_ee":
            target_pose = self._pose_from_relative_offset(
                ee_pose_snapshot.pose,
                goal.relative_dx,
                goal.relative_dy,
                goal.relative_dz,
            )
        elif goal.primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            target_pose = self._pose_from_gripper_rotation(
                ee_pose_snapshot.pose,
                goal.primitive_name,
                goal.relative_dz,
            )

        allowed, reason = self._validate_primitive(
            goal.primitive_name,
            goal.pose_name,
            goal.gripper_position,
            relative_dx=goal.relative_dx,
            relative_dy=goal.relative_dy,
            relative_dz=goal.relative_dz,
            target_pose=target_pose,
            velocity_scaling=goal.velocity_scaling,
            joint_names=list(goal.joint_names),
            joint_positions=list(goal.joint_positions),
            joint_waypoints=list(goal.joint_waypoints),
            joint_waypoint_count=goal.joint_waypoint_count,
            primitive_duration_sec=goal.primitive_duration_sec,
            waypoint_duration_sec=goal.waypoint_duration_sec,
            navigation_command_type=goal.navigation_command_type,
            navigation_target_pose=goal.navigation_target_pose,
            navigation_value=goal.navigation_value,
            dispatch_binding=goal.dispatch_binding,
        )
        if not allowed:
            result.success = False
            result.error_code = "SAFETY_REJECTED"
            result.message = reason
            result.pose_name = goal.pose_name
            goal_handle.abort()
            return result

        if goal.dispatch_binding.task_budget.schema_version == 1:
            try:
                remaining_budget = self._remaining_task_budget_sec(goal.dispatch_binding)
            except ValueError:
                return self._finish_primitive_failure(
                    result, goal_handle, "SKILL_TIMEOUT", "task budget expired or invalid", goal.pose_name
                )
            goal.timeout_sec = min(goal.timeout_sec, remaining_budget) if goal.timeout_sec > 0.0 else remaining_budget

        feedback = PrimitiveCommand.Feedback()
        feedback.state = "dispatching"
        feedback.detail = f"primitive={goal.primitive_name}"
        goal_handle.publish_feedback(feedback)

        if goal.primitive_name == "move_to_configuration":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            ok, err_msg = self._exec_move_configuration(
                goal_handle,
                list(goal.joint_names),
                list(goal.joint_positions),
                float(goal.velocity_scaling),
                move_timeout,
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name == "move_to_joint_positions":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            ok, err_msg = self._exec_arm_joint_trajectory(
                goal_handle,
                list(goal.joint_names),
                list(goal.joint_positions),
                _binding_task_id(goal),
                move_timeout,
                float(goal.primitive_duration_sec),
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name == "move_through_joint_positions":
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            joint_count = len(goal.joint_names)
            waypoints = [
                list(goal.joint_waypoints[index : index + joint_count])
                for index in range(0, len(goal.joint_waypoints), joint_count)
            ]
            ok, err_msg = self._exec_arm_joint_waypoint_trajectory(
                goal_handle,
                list(goal.joint_names),
                waypoints,
                _binding_task_id(goal),
                move_timeout,
                float(goal.waypoint_duration_sec),
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name in {"move_to_named_pose", "move_to_pose", "move_relative_ee"}:
            if goal.primitive_name == "move_to_named_pose":
                try:
                    pose = self._pose_from_name(goal.pose_name)
                except KeyError:
                    result.success = False
                    result.error_code = "UNKNOWN_POSE"
                    result.message = f"unknown named pose: {goal.pose_name!r}"
                    result.pose_name = goal.pose_name
                    goal_handle.abort()
                    return result
            elif goal.primitive_name == "move_to_pose":
                pose = goal.target_pose
            else:
                pose = target_pose
            if self._debug:
                self.get_logger().info(
                    f"[embodied-debug] primitive {goal.primitive_name} via task_dispatch "
                    f"task_id={_binding_task_id(goal)} pose={goal.pose_name or '-'} "
                    f"delta=({goal.relative_dx:.3f}, {goal.relative_dy:.3f}, {goal.relative_dz:.3f}) "
                    f"xyz=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
                )
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            try:
                ok, err_msg = self._exec_arm_via_task_dispatch(
                    goal_handle,
                    goal.primitive_name,
                    pose,
                    _binding_task_id(goal),
                    move_timeout,
                    velocity_scaling=float(goal.velocity_scaling),
                    ee_pose_receipt_monotonic=(
                        ee_pose_snapshot.received_monotonic if ee_pose_snapshot is not None else None
                    ),
                )
            except _EePoseSnapshotExpired:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
            if not ok:
                return self._finish_primitive_failure(
                    result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, goal.pose_name
                )
        elif goal.primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            move_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 30.0)
            try:
                ok, err_msg = self._exec_rotate_gripper_via_task_dispatch(
                    goal_handle,
                    goal.primitive_name,
                    target_pose,
                    _binding_task_id(goal),
                    move_timeout,
                    ee_pose_snapshot.received_monotonic,
                )
            except _EePoseSnapshotExpired:
                return self._finish_primitive_failure(
                    result,
                    goal_handle,
                    "CAPABILITY_NOT_READY",
                    "ee pose unavailable or stale",
                    goal.pose_name,
                )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, "PRIMITIVE_ARM_FAILED", err_msg, "")
        elif goal.primitive_name in {"nav_straight", "nav_turn", "nav_abs_coordinate"}:
            navigation_goal = ExecuteNavigation.Goal()
            navigation_goal.command_type = int(goal.navigation_command_type)
            navigation_goal.target_pose = copy.deepcopy(goal.navigation_target_pose)
            navigation_goal.value = float(goal.navigation_value)
            navigation_timeout = float(goal.timeout_sec if goal.timeout_sec > 0.0 else 120.0)
            ok, error_code, err_msg = self._exec_navigation(
                goal_handle,
                navigation_goal,
                navigation_timeout,
            )
            if not ok:
                return self._finish_primitive_failure(result, goal_handle, error_code, err_msg, "")
        else:
            # Delegate gripper control to task_dispatch via ExecuteTaskPlan action
            ok, err_msg = self._exec_gripper_via_task_dispatch(
                goal_handle, goal.primitive_name, goal.gripper_position, _binding_task_id(goal)
            )
            if not ok:
                return self._finish_primitive_failure(
                    result, goal_handle, "PRIMITIVE_GRIPPER_FAILED", err_msg, goal.pose_name
                )
        result.success = True
        result.error_code = ""
        result.message = f"primitive completed: {goal.primitive_name}"
        result.pose_name = goal.pose_name
        self._set_primitive_result_catalog_identity(result)
        goal_handle.succeed()
        return result

    def _exec_navigation(
        self,
        goal_handle,
        navigation_goal: ExecuteNavigation.Goal,
        timeout_sec: float,
    ) -> tuple[bool, str, str]:
        if self._context_schema_version < 2 or not hasattr(self, "_navigation_client"):
            return (
                False,
                "NAVIGATION_SERVER_UNAVAILABLE",
                "navigation action endpoint is unavailable in context schema v1",
            )
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        execution_deadline = time.monotonic() + timeout_sec
        server_wait_timeout = min(self._rpc_timeout, max(0.0, execution_deadline - time.monotonic()))
        if not self._navigation_client.wait_for_server(timeout_sec=server_wait_timeout):
            if time.monotonic() >= execution_deadline:
                return False, "NAVIGATION_TIMEOUT", "timeout waiting for navigation action server"
            message = f"navigation action server not available: {self._navigation_action_name}"
            self.get_logger().warning(f"[embodied-debug] {message}")
            return False, "NAVIGATION_SERVER_UNAVAILABLE", message
        if time.monotonic() >= execution_deadline:
            return False, "NAVIGATION_TIMEOUT", "timeout waiting for navigation action server"
        if goal_handle.is_cancel_requested:
            return False, "NAVIGATION_CANCELED", "navigation canceled before goal dispatch"

        def feedback_callback(feedback_message) -> None:
            navigation_feedback = getattr(feedback_message, "feedback", feedback_message)
            estimated = navigation_feedback.estimated_time_remaining
            estimated_sec = float(estimated.sec) + float(estimated.nanosec) / 1_000_000_000.0
            feedback = PrimitiveCommand.Feedback()
            feedback.state = str(navigation_feedback.state)
            feedback.detail = (
                f"distance_remaining={float(navigation_feedback.distance_remaining):.3f} "
                f"estimated_time_remaining_sec={estimated_sec:.3f} "
                f"number_of_recoveries={int(navigation_feedback.number_of_recoveries)}"
            )
            goal_handle.publish_feedback(feedback)

        try:
            send_future = self._navigation_client.send_goal_async(
                navigation_goal,
                feedback_callback=feedback_callback,
            )
        except Exception:
            return False, "NAVIGATION_INTERNAL_ERROR", "navigation goal send state is unknown"

        if goal_handle.is_cancel_requested:
            cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
            return (
                (False, "NAVIGATION_CANCELED", "navigation canceled")
                if cleaned
                else (False, PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, "cancel cleanup timed out while sending navigation goal")
            )
        acceptance_deadline = min(execution_deadline, time.monotonic() + self._rpc_timeout)
        while rclpy.ok() and not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    (False, "NAVIGATION_CANCELED", "navigation canceled")
                    if cleaned
                    else (
                        False,
                        PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                        "cancel cleanup timed out while sending navigation goal",
                    )
                )
            if time.monotonic() >= acceptance_deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    (False, "NAVIGATION_GOAL_TIMEOUT", "timeout waiting for navigation goal acceptance")
                    if cleaned
                    else (
                        False,
                        PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                        "cancel cleanup timed out while sending navigation goal",
                    )
                )
            time.sleep(0.05)
        if not send_future.done():
            self._cancel_goal_future(send_future, late_cleanup_confirmation)
            return False, "NAVIGATION_INTERNAL_ERROR", "navigation goal send interrupted"

        try:
            navigation_handle = send_future.result()
        except Exception:
            return False, "NAVIGATION_INTERNAL_ERROR", "navigation goal send state is unknown"
        if navigation_handle is None or not navigation_handle.accepted:
            return False, "NAVIGATION_GOAL_REJECTED", "navigation goal rejected"

        try:
            result_future = navigation_handle.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(navigation_handle)
            return (
                False,
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                "cancel cleanup timed out during navigation execution",
            )

        if goal_handle.is_cancel_requested:
            cleaned = self._cancel_goal(navigation_handle, result_future, late_cleanup_confirmation)
            return (
                (False, "NAVIGATION_CANCELED", "navigation canceled")
                if cleaned
                else (False, PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, "cancel cleanup timed out during navigation execution")
            )

        while rclpy.ok() and not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(navigation_handle, result_future, late_cleanup_confirmation)
                return (
                    (False, "NAVIGATION_CANCELED", "navigation canceled")
                    if cleaned
                    else (
                        False,
                        PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                        "cancel cleanup timed out during navigation execution",
                    )
                )
            if time.monotonic() >= execution_deadline:
                cleaned = self._cancel_goal(navigation_handle, result_future, late_cleanup_confirmation)
                return (
                    (False, "NAVIGATION_TIMEOUT", "timeout waiting for navigation result")
                    if cleaned
                    else (
                        False,
                        PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                        "cancel cleanup timed out during navigation execution",
                    )
                )
            time.sleep(0.05)
        if not result_future.done():
            cleaned = self._cancel_goal(navigation_handle, result_future, late_cleanup_confirmation)
            return (
                (False, "NAVIGATION_INTERNAL_ERROR", "navigation result wait interrupted")
                if cleaned
                else (False, PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, "cancel cleanup timed out during navigation execution")
            )

        try:
            wrapped_result = result_future.result()
        except Exception:
            self._best_effort_cancel_goal(navigation_handle)
            return False, "NAVIGATION_INTERNAL_ERROR", "navigation result state is unknown"
        navigation_result = getattr(wrapped_result, "result", None)
        if navigation_result is None:
            return False, "NAVIGATION_INTERNAL_ERROR", "navigation action returned no result"
        if navigation_result.success:
            return True, "", ""
        if int(navigation_result.error_code) == ExecuteNavigation.Result.STOP_TIMEOUT:
            return False, PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, f"cancel cleanup timed out: {navigation_result.message}"
        error_code = _NAVIGATION_ERROR_CODES.get(int(navigation_result.error_code), "NAVIGATION_FAILED")
        message = str(navigation_result.message) or error_code
        return False, error_code, message

    def _exec_rotate_gripper_via_task_dispatch(
        self,
        goal_handle,
        primitive_name: str,
        target_pose: Pose,
        task_id: str,
        timeout_sec: float,
        ee_pose_receipt_monotonic: float,
    ) -> tuple[bool, str]:
        """Dispatch a wrist rotation target derived from a fresh EE pose snapshot."""

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] primitive {primitive_name} via task_dispatch "
                f"task_id={task_id} target_quat=({target_pose.orientation.x:.3f},"
                f"{target_pose.orientation.y:.3f},{target_pose.orientation.z:.3f},"
                f"{target_pose.orientation.w:.3f})"
            )

        return self._exec_arm_via_task_dispatch(
            goal_handle,
            primitive_name,
            target_pose,
            task_id,
            timeout_sec,
            ee_pose_receipt_monotonic=ee_pose_receipt_monotonic,
        )

    def _exec_arm_via_task_dispatch(
        self,
        goal_handle,
        primitive_name: str,
        target_pose: Pose,
        task_id: str,
        timeout_sec: float,
        *,
        ee_pose_receipt_monotonic: float | None = None,
        velocity_scaling: float = 0.0,
    ) -> tuple[bool, str]:
        """Send a MOVE_TO_POSE TaskStep to task_dispatch ExecuteTaskPlan action server."""
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._task_executor_client.wait_for_server(timeout_sec=2.0):
            msg = f"task_executor action server not available: {self._task_executor_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        if ee_pose_receipt_monotonic is not None and not self._ee_pose_receipt_is_fresh(ee_pose_receipt_monotonic):
            raise _EePoseSnapshotExpired

        step = TaskStep()
        step.type = TaskStep.MOVE_TO_POSE
        step.label = primitive_name
        step.target_pose = target_pose
        step.velocity_scaling = float(velocity_scaling)

        goal_msg = ExecuteTaskPlan.Goal()
        goal_msg.steps = [step]
        goal_msg.task_id = task_id or str(uuid.uuid4())
        goal_msg.task_description = primitive_name

        if self._debug:
            self.get_logger().info(
                f"[embodied-debug] primitive arm command via task_dispatch "
                f"task_id={task_id} primitive={primitive_name} "
                f"action={self._task_executor_action_name}"
            )

        if ee_pose_receipt_monotonic is not None and not self._ee_pose_receipt_is_fresh(ee_pose_receipt_monotonic):
            raise _EePoseSnapshotExpired
        send_future = self._task_executor_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm goal rejected by task_executor"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm motion"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return False, "cancelled during arm motion" if cleaned else "cancel cleanup timed out during arm motion"
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm motion" if cleaned else "cancel cleanup timed out during arm motion",
                )
            time.sleep(0.05)

        result = result_future.result().result
        if not result.success:
            return False, f"arm motion failed: {result.message}"
        return True, ""

    def _exec_move_configuration(
        self,
        goal_handle,
        joint_names: list[str],
        joint_positions: list[float],
        velocity_scaling: float,
        timeout_sec: float,
    ) -> tuple[bool, str]:
        if not self._move_configuration_client.wait_for_service(timeout_sec=self._rpc_timeout):
            return False, f"move configuration service unavailable: {self._move_configuration_service}"

        request = MoveToConfiguration.Request()
        request.target_joint_state = JointState()
        request.target_joint_state.name = list(joint_names)
        request.target_joint_state.position = [float(position) for position in joint_positions]
        request.velocity_scaling = float(velocity_scaling)
        future = self._move_configuration_client.call_async(request)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if goal_handle.is_cancel_requested:
                future.cancel()
                return False, "cancelled during MoveIt configuration motion"
            if time.monotonic() >= deadline:
                future.cancel()
                return False, "timeout waiting for MoveIt configuration motion"
            time.sleep(0.05)

        response = future.result()
        if response is None:
            return False, "move configuration service returned no response"
        if not response.success:
            return False, response.message
        return True, ""

    def _exec_arm_joint_trajectory(
        self,
        goal_handle,
        joint_names: list[str],
        joint_positions: list[float],
        task_id: str,
        timeout_sec: float,
        duration_sec: float,
    ) -> tuple[bool, str]:
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._arm_trajectory_client.wait_for_server(timeout_sec=2.0):
            msg = f"arm trajectory action server not available: {self._arm_trajectory_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        goal_msg = _build_joint_trajectory_goal(
            list(joint_names),
            [[float(position) for position in joint_positions]],
            max(0.1, float(duration_sec)),
        )

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive arm joint trajectory "
                f"task_id={task_id} joints={joint_names} positions={joint_positions}"
            )

        send_future = self._arm_trajectory_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm trajectory goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm trajectory goal rejected"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm joint trajectory execution"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during arm joint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm joint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint trajectory execution",
                )
            time.sleep(0.05)

        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            error_code = result.result.error_code if result is not None else "unknown"
            return False, f"arm trajectory execution failed: {error_code}"
        return True, ""

    def _exec_arm_joint_waypoint_trajectory(
        self,
        goal_handle,
        joint_names: list[str],
        joint_waypoints: list[list[float]],
        task_id: str,
        timeout_sec: float,
        waypoint_duration_sec: float,
    ) -> tuple[bool, str]:
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if not self._arm_trajectory_client.wait_for_server(timeout_sec=2.0):
            msg = f"arm trajectory action server not available: {self._arm_trajectory_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        goal_msg = _build_joint_trajectory_goal(joint_names, joint_waypoints, waypoint_duration_sec)

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive arm joint waypoint trajectory "
                f"task_id={task_id} joints={joint_names} waypoints={len(joint_waypoints)} "
                f"waypoint_duration={waypoint_duration_sec:.3f}"
            )

        send_future = self._arm_trajectory_client.send_goal_async(goal_msg)
        accept_timeout = 5.0
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending arm trajectory goal"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm trajectory goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending arm trajectory goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "arm trajectory goal rejected"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during arm joint waypoint trajectory execution"
        deadline = time.monotonic() + timeout_sec
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during arm joint waypoint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint waypoint trajectory execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for arm joint waypoint trajectory execution"
                    if cleaned
                    else "cancel cleanup timed out during arm joint waypoint trajectory execution",
                )
            time.sleep(0.05)

        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            error_code = result.result.error_code if result is not None else "unknown"
            return False, f"arm trajectory execution failed: {error_code}"
        return True, ""

    def _exec_gripper_via_task_dispatch(
        self, goal_handle, primitive_name: str, gripper_position: float, task_id: str
    ) -> tuple[bool, str]:
        """Send a GRIPPER TaskStep to task_dispatch ExecuteTaskPlan action server."""
        late_cleanup_confirmation = self._late_cleanup_confirmation(goal_handle)
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] primitive gripper command via task_dispatch "
                f"task_id={task_id} primitive={primitive_name} value={gripper_position:.3f} "
                f"action={self._task_executor_action_name}"
            )
        if not self._task_executor_client.wait_for_server(timeout_sec=2.0):
            msg = f"task_executor action server not available: {self._task_executor_action_name}"
            self.get_logger().warning(f"[embodied-debug] {msg}")
            return False, msg

        step = TaskStep()
        step.type = TaskStep.GRIPPER
        step.label = primitive_name
        step.gripper_position = float(gripper_position)

        goal_msg = ExecuteTaskPlan.Goal()
        goal_msg.steps = [step]
        goal_msg.task_id = task_id or str(uuid.uuid4())
        goal_msg.task_description = primitive_name

        send_future = self._task_executor_client.send_goal_async(goal_msg)
        accept_timeout = self._gripper_settle_sec + 3.0
        exec_timeout = max(accept_timeout, 15.0)
        deadline = time.monotonic() + accept_timeout
        while not send_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled while sending gripper goal"
                    if cleaned
                    else "cancel cleanup timed out while sending gripper goal",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal_future(send_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for gripper goal acceptance"
                    if cleaned
                    else "cancel cleanup timed out while sending gripper goal",
                )
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            return False, "gripper goal rejected by task_executor"

        try:
            result_future = gh.get_result_async()
        except Exception:
            self._best_effort_cancel_goal(gh)
            return False, "cancel cleanup timed out during gripper execution"
        deadline = time.monotonic() + exec_timeout
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "cancelled during gripper execution"
                    if cleaned
                    else "cancel cleanup timed out during gripper execution",
                )
            if time.monotonic() > deadline:
                cleaned = self._cancel_goal(gh, result_future, late_cleanup_confirmation)
                return (
                    False,
                    "timeout waiting for gripper execution"
                    if cleaned
                    else "cancel cleanup timed out during gripper execution",
                )
            time.sleep(0.05)

        result = result_future.result().result
        if not result.success:
            return False, f"gripper execution failed: {result.message}"
        return True, ""

    def _finalize_active_admission(
        self,
        admission,
        *,
        error_code: str,
        duration_sec: float,
        step_count: int,
    ) -> bool:
        with self._state_guard():
            if self._active_skill_admission is not admission:
                return False
            try:
                self._gateway_policy.finalize(
                    admission,
                    error_code=error_code,
                    terminal_metadata={"duration_sec": duration_sec, "step_count": step_count},
                )
            except Exception:
                return False
            self._active_skill_admission = None
            self._active_skill_owner = None
            self._active_audit_context = None
            active_bundle = getattr(self, "_active_runtime_bundle", None)
            if getattr(self, "_active_runtime_generation_retained", False) and active_bundle is not None:
                with suppress(Exception):
                    self._runtime_coordinator.release(active_bundle.generation)
            self._active_runtime_generation_retained = False
            self._active_runtime_bundle = None
            self._retained_admission_cleanup.pop(id(admission), None)
            return True

    def _execute_pick_skill(
        self,
        goal_handle,
        template: dict,
        *,
        effective_timeout_sec: float | None = None,
    ) -> SkillCommand.Result:
        goal = goal_handle.request
        result = SkillCommand.Result()
        timeout_sec = float(effective_timeout_sec or goal.timeout_sec or template.get("timeout_sec", 180.0))

        if not self._pick_client.wait_for_server(timeout_sec=self._rpc_timeout):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "PICK_SERVER_UNAVAILABLE",
                f"pick action server unavailable: {self._pick_action_name}",
            )

        completed_phases: list[str] = []

        def _feedback_cb(feedback_msg) -> None:
            pick_feedback = feedback_msg.feedback
            phase = str(pick_feedback.phase)
            if phase and (not completed_phases or completed_phases[-1] != phase):
                completed_phases.append(phase)
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"pick:{phase}:{pick_feedback.detail}"
            goal_handle.publish_feedback(feedback)

        pick_goal = PickObject.Goal()
        pick_goal.dispatch_binding = copy_binding(goal.dispatch_binding)
        pick_goal.dispatch_binding.dispatch_nonce = uuid.uuid4().hex
        bundle = self._active_runtime_bundle
        executor = bundle.snapshot.delegated_executors.get("grasp_pipeline")
        if executor is None:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "pick executor missing",
            )
        expected_executor = {
            "name": executor.name,
            "contract_version": executor.contract_version,
            "endpoint_kind": executor.endpoint_kind,
            "endpoint_name": executor.endpoint_name,
            "configuration_digest": executor.configuration_digest,
            "model_deployment_name": executor.model_deployment_name,
            "model_fingerprint": executor.model_fingerprint,
            "model_bundle_digest": executor.model_bundle_digest,
        }
        fill_delegated_executor_identity(
            pick_goal.expected_executor,
            expected_executor,
        )
        pick_goal.target_query = goal.target_name
        pick_goal.timeout_sec = timeout_sec
        # SkillCommand is the only public motion boundary. The Gateway owns
        # delegated PickObject policy and deliberately exposes the canonical
        # execute path only; diagnostic modes and post-success release are not
        # caller-controlled fields in the v1 catalog contract.
        pick_goal.mode = PickObject.Goal.MODE_EXECUTE
        pick_goal.supervised_direct = False
        pick_goal.release_after_success = False
        delegated_admission = self._active_skill_admission
        delegated_nonce = pick_goal.dispatch_binding.dispatch_nonce
        cleanup_key = self._register_delegated_dispatch(
            delegated_nonce,
            delegated_admission,
            pick_goal.dispatch_binding,
        )
        delegated_terminal = False
        late_cleanup = _LateCleanupConfirmation()
        late_cleanup.add_callback(
            lambda: self._confirm_delegated_terminal(delegated_admission, delegated_nonce, cleanup_key)
        )
        try:
            send_future = self._pick_client.send_goal_async(pick_goal, feedback_callback=_feedback_cb)
            if not self._wait_for_future(send_future, timeout_sec=self._rpc_timeout):
                late_cleanup.watch_goal_future(send_future, self._best_effort_cancel_goal)
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "pick goal response timed out: delegated execution state is unknown",
                )

            pick_handle = send_future.result()
            if pick_handle is None or not pick_handle.accepted:
                delegated_terminal = True
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    "PICK_GOAL_REJECTED",
                    "pick executor rejected the goal",
                )

            result_future = pick_handle.get_result_async()
            deadline = time.monotonic() + timeout_sec
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    if not self._cancel_goal(pick_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "pick cancellation state is unknown",
                        )
                    delegated_terminal = True
                    result.success = False
                    result.error_code = "SKILL_CANCELLED"
                    result.message = "pick skill cancelled"
                    result.executed_primitives = [f"grasp_pipeline:{phase}" for phase in completed_phases]
                    goal_handle.canceled()
                    return result
                if time.monotonic() >= deadline:
                    if not self._cancel_goal(pick_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "pick timeout cleanup state is unknown",
                        )
                    delegated_terminal = True
                    return self._abort_skill(
                        result,
                        goal_handle,
                        [f"grasp_pipeline:{phase}" for phase in completed_phases],
                        "SKILL_TIMEOUT",
                        "pick skill deadline exceeded",
                    )
                time.sleep(0.05)
            if not result_future.done():
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "pick execution state is unknown",
                )
            delegated_terminal = True
            late_cleanup.confirm()
            action_result = result_future.result()
            pick_result = action_result.result if action_result is not None else None
        except Exception:
            self.get_logger().error(f"delegated pick execution failed:\n{traceback.format_exc()}")
            if (
                "pick_handle" in locals()
                and pick_handle is not None
                and "result_future" in locals()
                and self._cancel_goal(pick_handle, result_future, late_cleanup)
            ):
                delegated_terminal = True
            return self._abort_skill(
                result,
                goal_handle,
                completed_phases,
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                "delegated pick execution state is unknown",
            )
        finally:
            if delegated_terminal:
                late_cleanup.confirm()
        result.executed_primitives = [f"grasp_pipeline:{phase}" for phase in completed_phases]
        if pick_result is None:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "MISSING_PICK_RESULT",
                "pick executor returned no result",
            )
        if not delegated_executor_identity_matches(pick_result.actual_executor, expected_executor):
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "pick executor identity does not match the registry snapshot",
            )
        if not pick_result.success:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                pick_result.error_code or "PICK_FAILED",
                pick_result.message,
            )

        result.success = True
        result.error_code = ""
        result.message = pick_result.message
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        goal_handle.succeed()
        return result

    def _execute_imitate_human_motion_skill(
        self,
        goal_handle,
        template: dict,
        *,
        effective_timeout_sec: float | None = None,
    ) -> SkillCommand.Result:
        """Delegate one public request to the registered internal HRI Action."""
        goal = goal_handle.request
        result = SkillCommand.Result()
        timeout_sec = float(effective_timeout_sec or goal.timeout_sec or template.get("timeout_sec", 30.0))
        if not self._imitate_human_motion_client.wait_for_server(timeout_sec=self._rpc_timeout):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "IMITATE_HUMAN_MOTION_SERVER_UNAVAILABLE",
                f"imitate human motion action server unavailable: {self._imitate_human_motion_action_name}",
            )

        completed_phases: list[str] = []

        def feedback_cb(feedback_msg) -> None:
            hri_feedback = feedback_msg.feedback
            phase = str(hri_feedback.phase)
            if phase and (not completed_phases or completed_phases[-1] != phase):
                completed_phases.append(phase)
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"imitate_human_motion:{phase}:{hri_feedback.detail}"
            goal_handle.publish_feedback(feedback)

        hri_goal = ImitateHumanMotion.Goal()
        hri_goal.dispatch_binding = copy_binding(goal.dispatch_binding)
        hri_goal.dispatch_binding.dispatch_nonce = uuid.uuid4().hex
        bundle = self._active_runtime_bundle
        executor = bundle.snapshot.delegated_executors.get("imitate_human_motion")
        if executor is None:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "imitate human motion executor missing",
            )
        expected_executor = {
            "name": executor.name,
            "contract_version": executor.contract_version,
            "endpoint_kind": executor.endpoint_kind,
            "endpoint_name": executor.endpoint_name,
            "configuration_digest": executor.configuration_digest,
            "model_deployment_name": executor.model_deployment_name,
            "model_fingerprint": executor.model_fingerprint,
            "model_bundle_digest": executor.model_bundle_digest,
        }
        fill_delegated_executor_identity(hri_goal.expected_executor, expected_executor)
        hri_goal.arm_side = goal.arm_side
        hri_goal.imitation_duration_sec = goal.imitation_duration_sec
        hri_goal.timeout_sec = timeout_sec

        delegated_admission = self._active_skill_admission
        delegated_nonce = hri_goal.dispatch_binding.dispatch_nonce
        cleanup_key = self._register_delegated_dispatch(
            delegated_nonce,
            delegated_admission,
            hri_goal.dispatch_binding,
        )
        delegated_terminal = False
        late_cleanup = _LateCleanupConfirmation()
        late_cleanup.add_callback(
            lambda: self._confirm_delegated_terminal(delegated_admission, delegated_nonce, cleanup_key)
        )
        try:
            send_future = self._imitate_human_motion_client.send_goal_async(
                hri_goal,
                feedback_callback=feedback_cb,
            )
            if not self._wait_for_future(send_future, timeout_sec=self._rpc_timeout):
                late_cleanup.watch_goal_future(send_future, self._best_effort_cancel_goal)
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "imitate human motion goal response timed out: delegated execution state is unknown",
                )
            hri_handle = send_future.result()
            if hri_handle is None or not hri_handle.accepted:
                delegated_terminal = True
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    "IMITATE_HUMAN_MOTION_GOAL_REJECTED",
                    "imitate human motion executor rejected the goal",
                )

            result_future = hri_handle.get_result_async()
            deadline = time.monotonic() + timeout_sec
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    if not self._cancel_goal(hri_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "imitate human motion cancellation state is unknown",
                        )
                    delegated_terminal = True
                    return self._cancel_skill(
                        result,
                        goal_handle,
                        [f"imitate_human_motion:{phase}" for phase in completed_phases],
                        goal.skill_name,
                    )
                if time.monotonic() >= deadline:
                    if not self._cancel_goal(hri_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "imitate human motion timeout cleanup state is unknown",
                        )
                    delegated_terminal = True
                    return self._abort_skill(
                        result,
                        goal_handle,
                        [f"imitate_human_motion:{phase}" for phase in completed_phases],
                        "SKILL_TIMEOUT",
                        "imitate human motion skill deadline exceeded",
                    )
                time.sleep(0.05)
            if not result_future.done():
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "imitate human motion execution state is unknown",
                )
            delegated_terminal = True
            late_cleanup.confirm()
            action_result = result_future.result()
            hri_result = action_result.result if action_result is not None else None
        except Exception:
            self.get_logger().error(f"delegated imitate human motion execution failed:\n{traceback.format_exc()}")
            if (
                "hri_handle" in locals()
                and hri_handle is not None
                and "result_future" in locals()
                and self._cancel_goal(hri_handle, result_future, late_cleanup)
            ):
                delegated_terminal = True
            return self._abort_skill(
                result,
                goal_handle,
                completed_phases,
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                "delegated imitate human motion execution state is unknown",
            )
        finally:
            if delegated_terminal:
                late_cleanup.confirm()

        result.executed_primitives = [f"imitate_human_motion:{phase}" for phase in completed_phases]
        if hri_result is None:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "MISSING_IMITATE_HUMAN_MOTION_RESULT",
                "imitate human motion executor returned no result",
            )
        if not delegated_executor_identity_matches(hri_result.actual_executor, expected_executor):
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "imitate human motion executor identity does not match the registry snapshot",
            )
        if goal_handle.is_cancel_requested and hri_result.error_code == "CANCELED":
            return self._cancel_skill(result, goal_handle, result.executed_primitives, goal.skill_name)
        if not hri_result.success:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                hri_result.error_code or "IMITATE_HUMAN_MOTION_FAILED",
                hri_result.message,
            )

        result.success = True
        result.error_code = ""
        result.message = hri_result.message
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        goal_handle.succeed()
        return result

    @staticmethod
    def _place_public_error(place_result) -> str:
        """Preserve the irreversible release state in the public error code."""
        if str(place_result.error_code) in {
            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
            "PRIMITIVE_CANCEL_CLEANUP_TIMEOUT",
        }:
            return PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
        if int(place_result.release_status) == PlaceObject.Result.RELEASE_UNKNOWN:
            return "PLACE_RELEASE_STATE_UNKNOWN"
        if int(place_result.release_status) == PlaceObject.Result.RELEASE_RELEASED:
            if int(place_result.verification_status) == PlaceObject.Result.VERIFICATION_FAILED:
                return "PLACE_RELEASED_VERIFICATION_FAILED"
            if int(place_result.verification_status) == PlaceObject.Result.VERIFICATION_UNCERTAIN:
                return "PLACE_RELEASED_VERIFICATION_UNCERTAIN"
            if int(place_result.verification_status) == PlaceObject.Result.VERIFICATION_NOT_RUN:
                return "PLACE_RELEASED_VERIFICATION_NOT_RUN"
        return "PLACE_NOT_RELEASED"

    def _execute_place_skill(
        self,
        goal_handle,
        template: dict,
        *,
        effective_timeout_sec: float | None = None,
    ) -> SkillCommand.Result:
        """Delegate placement while retaining the Gateway lease through terminal cleanup."""
        goal = goal_handle.request
        result = SkillCommand.Result()
        timeout_sec = float(effective_timeout_sec or goal.timeout_sec or template.get("timeout_sec", 60.0))
        if not self._place_client.wait_for_server(timeout_sec=self._rpc_timeout):
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "PLACE_SERVER_UNAVAILABLE",
                f"placement action server unavailable: {self._place_action_name}",
            )

        completed_phases: list[str] = []

        def feedback_cb(feedback_msg) -> None:
            place_feedback = feedback_msg.feedback
            phase = str(place_feedback.phase)
            if phase and (not completed_phases or completed_phases[-1] != phase):
                completed_phases.append(phase)
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"place:{phase}:{place_feedback.detail}"
            goal_handle.publish_feedback(feedback)

        place_goal = PlaceObject.Goal()
        place_goal.dispatch_binding = copy_binding(goal.dispatch_binding)
        place_goal.dispatch_binding.dispatch_nonce = uuid.uuid4().hex
        bundle = self._active_runtime_bundle
        executor = bundle.snapshot.delegated_executors.get("placement_pipeline")
        if executor is None:
            return self._abort_skill(
                result,
                goal_handle,
                [],
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "placement executor missing",
            )
        expected_executor = {
            "name": executor.name,
            "contract_version": executor.contract_version,
            "endpoint_kind": executor.endpoint_kind,
            "endpoint_name": executor.endpoint_name,
            "configuration_digest": executor.configuration_digest,
            "model_deployment_name": executor.model_deployment_name,
            "model_fingerprint": executor.model_fingerprint,
            "model_bundle_digest": executor.model_bundle_digest,
        }
        fill_delegated_executor_identity(place_goal.expected_executor, expected_executor)
        place_goal.target_query = goal.target_name
        place_goal.container_query = goal.container_name
        place_goal.timeout_sec = timeout_sec

        delegated_admission = self._active_skill_admission
        delegated_nonce = place_goal.dispatch_binding.dispatch_nonce
        cleanup_key = self._register_delegated_dispatch(
            delegated_nonce,
            delegated_admission,
            place_goal.dispatch_binding,
        )
        delegated_terminal = False
        late_cleanup = _LateCleanupConfirmation()
        late_cleanup.add_callback(
            lambda: self._confirm_delegated_terminal(delegated_admission, delegated_nonce, cleanup_key)
        )
        try:
            send_future = self._place_client.send_goal_async(place_goal, feedback_callback=feedback_cb)
            if not self._wait_for_future(send_future, timeout_sec=self._rpc_timeout):
                late_cleanup.watch_goal_future(send_future, self._best_effort_cancel_goal)
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "placement goal response timed out: delegated execution state is unknown",
                )
            place_handle = send_future.result()
            if place_handle is None or not place_handle.accepted:
                delegated_terminal = True
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    "PLACE_GOAL_REJECTED",
                    "placement executor rejected the goal",
                )

            result_future = place_handle.get_result_async()
            deadline = time.monotonic() + timeout_sec
            while rclpy.ok() and not result_future.done():
                if goal_handle.is_cancel_requested:
                    if not self._cancel_goal(place_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "placement cancellation state is unknown",
                        )
                    delegated_terminal = True
                    break
                if time.monotonic() >= deadline:
                    if not self._cancel_goal(place_handle, result_future, late_cleanup):
                        return self._abort_skill(
                            result,
                            goal_handle,
                            completed_phases,
                            PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                            "placement timeout cleanup state is unknown",
                        )
                    delegated_terminal = True
                    break
                time.sleep(0.05)
            if not result_future.done():
                return self._abort_skill(
                    result,
                    goal_handle,
                    completed_phases,
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    "placement execution state is unknown",
                )
            delegated_terminal = True
            late_cleanup.confirm()
            action_result = result_future.result()
            place_result = action_result.result if action_result is not None else None
        except Exception:
            self.get_logger().error(f"delegated placement execution failed:\n{traceback.format_exc()}")
            if (
                "place_handle" in locals()
                and place_handle is not None
                and "result_future" in locals()
                and self._cancel_goal(place_handle, result_future, late_cleanup)
            ):
                delegated_terminal = True
            return self._abort_skill(
                result,
                goal_handle,
                completed_phases,
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                "delegated placement execution state is unknown",
            )
        finally:
            if delegated_terminal:
                late_cleanup.confirm()

        result.executed_primitives = [f"placement_pipeline:{phase}" for phase in completed_phases]
        if place_result is None:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "MISSING_PLACE_RESULT",
                "placement executor returned no result",
            )
        if not delegated_executor_identity_matches(place_result.actual_executor, expected_executor):
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                "SKILL_EXECUTOR_IDENTITY_MISMATCH",
                "placement executor identity does not match the registry snapshot",
            )
        result.debug_output_dir = str(getattr(place_result, "debug_output_dir", ""))
        if goal_handle.is_cancel_requested and place_result.error_code == "PLACE_CANCELLED":
            return self._cancel_skill(result, goal_handle, result.executed_primitives, goal.skill_name)
        if not place_result.success or not place_result.place_succeeded:
            return self._abort_skill(
                result,
                goal_handle,
                result.executed_primitives,
                self._place_public_error(place_result),
                place_result.message,
            )

        result.success = True
        result.error_code = ""
        result.message = place_result.message
        self._set_result_catalog_identity(result)
        result.diagnostics = []
        goal_handle.succeed()
        return result

    def _execute_skill(self, goal_handle):
        try:
            return self._execute_skill_gateway(goal_handle)
        finally:
            with self._skill_goal_lock:
                self._skill_goal_active = False

    def _execute_skill_gateway(self, goal_handle):
        # _gateway_policy is always assigned in __init__ on the production path;
        # this branch only triggers for unit-test fixtures built via object.__new__
        # that bypass __init__. A node with a half-initialized __init__ would have
        # raised before reaching the action server, so this fallback does not create
        # a production bypass; it keeps test fixtures that exercise _execute_skill_child
        # working without re-stubbing the full Gateway.
        if not hasattr(self, "_gateway_policy"):
            return self._execute_skill_child(goal_handle)

        goal = goal_handle.request
        if goal.dispatch_binding.root_lease_nonce:
            return self._execute_workflow_skill_gateway(goal_handle)
        request = GatewayRequest(
            task_id=_binding_task_id(goal),
            skill_name=goal.skill_name,
            schema_version=goal.schema_version,
            target_name=goal.target_name,
            container_name=goal.container_name,
            place_name=goal.place_name,
            motion_direction=goal.motion_direction,
            motion_distance=goal.motion_distance,
            arm_side=getattr(goal, "arm_side", ""),
            imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
            direction=goal.direction,
            distance=goal.distance,
            degree=goal.degree,
            x=goal.x,
            y=goal.y,
            yaw=goal.yaw,
            has_x=goal.has_x,
            has_y=goal.has_y,
            has_yaw=goal.has_yaw,
            timeout_sec=None if goal.timeout_sec == 0.0 else goal.timeout_sec,
        )
        owner = ExecutionOwner.skill_command(_binding_task_id(goal))
        with self._state_guard():
            coordinator = getattr(self, "_runtime_coordinator", None)
            if coordinator is not None and coordinator.state != "READY":
                result = SkillCommand.Result()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "SKILL_RELOAD_IN_PROGRESS",
                    "skill catalog reload is in progress",
                )
            bundle = coordinator.current if coordinator is not None else None
            schema_error = self._validate_skill_goal_schema(goal, bundle)
            if schema_error:
                result = SkillCommand.Result()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    schema_error,
                    "skill schema_version does not match the selected capability",
                )
            binding_error = (
                self._prepare_root_binding(goal.dispatch_binding, bundle, allow_zero_budget=True)
                if coordinator is not None
                else ""
            )
            if binding_error:
                result = SkillCommand.Result()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    binding_error,
                    "skill request binding is invalid or stale",
                )
            try:
                remaining_budget = (
                    self._remaining_task_budget_sec(goal.dispatch_binding)
                    if coordinator is not None
                    else getattr(self._gateway_policy, "_task_budget_sec", 1.0)
                )
            except ValueError:
                result = SkillCommand.Result()
                return self._abort_skill(result, goal_handle, [], "SKILL_SCHEMA_INVALID", "task budget is invalid")
            total_budget = (
                self._task_budget_duration_sec(goal.dispatch_binding)
                if coordinator is not None
                else getattr(self._gateway_policy, "_task_budget_sec", 1.0)
            )
            if goal.timeout_sec > 0.0 and goal.timeout_sec > total_budget:
                result = SkillCommand.Result()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "TIMEOUT_EXCEEDS_POLICY",
                    "skill timeout exceeds root task budget",
                )
            effective_timeout = goal.timeout_sec if goal.timeout_sec > 0.0 else self._default_skill_timeout_sec
            if effective_timeout > total_budget:
                result = SkillCommand.Result()
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "TIMEOUT_EXCEEDS_POLICY",
                    "default skill timeout exceeds root task budget",
                )
            admission = self._gateway_policy.admit(request, self._runtime_snapshot(), owner)
            prepared = admission.prepared_request
            audit_context = {
                "task_id": prepared.identity.task_id if prepared is not None else _binding_task_id(goal),
                "payload_hash": prepared.identity.payload_hash if prepared is not None else "",
                "skill": str(goal.skill_name).strip(),
            }
            if admission.admitted:
                self._active_skill_admission = admission
                self._active_skill_owner = owner
                self._active_audit_context = audit_context
                coordinator = getattr(self, "_runtime_coordinator", None)
                current_bundle = coordinator.current if coordinator is not None else None
                self._active_runtime_bundle = (
                    coordinator.retain(current_bundle.generation)
                    if coordinator is not None and current_bundle is not None
                    else None
                )
                self._active_runtime_generation_retained = self._active_runtime_bundle is not None
        self._audit("requested", **audit_context)
        if not admission.admitted:
            self._audit("precondition_rejected", error_code=admission.error_code, **audit_context)
            result = SkillCommand.Result()
            return self._abort_skill(
                result,
                goal_handle,
                [],
                admission.error_code,
                admission.message or admission.error_code,
            )

        started = time.monotonic()
        result = None
        deferred_goal_handle = _DeferredTerminalGoalHandle(goal_handle)
        try:
            allowed, reason = self._validate_skill(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                container_name=goal.container_name,
                motion_direction=goal.motion_direction,
                motion_distance=goal.motion_distance,
                arm_side=getattr(goal, "arm_side", ""),
                imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
                direction=goal.direction,
                distance=goal.distance,
                degree=goal.degree,
                x=goal.x if goal.has_x else None,
                y=goal.y if goal.has_y else None,
                yaw=goal.yaw if goal.has_yaw else None,
                schema_version=goal.schema_version,
                dispatch_binding=goal.dispatch_binding,
            )
            self._audit(
                "safety_validated",
                error_code="" if allowed else "SKILL_REJECTED",
                **audit_context,
            )
            if not allowed:
                result = SkillCommand.Result()
                result = self._abort_skill(result, deferred_goal_handle, [], "CAPABILITY_NOT_READY", reason)
            else:
                if coordinator is not None:
                    try:
                        remaining_budget = self._remaining_task_budget_sec(goal.dispatch_binding)
                    except ValueError:
                        result = SkillCommand.Result()
                        result = self._abort_skill(
                            result, deferred_goal_handle, [], "SKILL_TIMEOUT", "root task budget expired"
                        )
                if result is not None:
                    pass
                elif goal.timeout_sec > 0.0 and goal.timeout_sec > remaining_budget:
                    result = SkillCommand.Result()
                    result = self._abort_skill(
                        result,
                        deferred_goal_handle,
                        [],
                        "TIMEOUT_EXCEEDS_POLICY",
                        "skill timeout exceeds remaining root task budget",
                    )
                else:
                    result = self._execute_skill_child(
                        deferred_goal_handle,
                        validation_done=True,
                        effective_timeout_sec=min(admission.effective_timeout_sec, remaining_budget),
                        canonical_task_id=prepared.identity.task_id,
                    )
        except Exception as exc:
            self.get_logger().error(f"[skill_executor] gateway execution failed: {exc}")
            self._audit("gateway_exception", error_code="SKILL_REJECTED", **audit_context)
            result = SkillCommand.Result()
            result = self._abort_skill(
                result,
                deferred_goal_handle,
                [],
                "CAPABILITY_NOT_READY",
                f"gateway execution failed: {exc}",
            )
        finally:
            duration_sec = time.monotonic() - started
            error_code = result.error_code if result is not None else "SKILL_REJECTED"
            step_count = len(result.executed_primitives) if result is not None else 0
            cleanup_unknown = error_code in {PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SKILL_CANCEL_TIMEOUT}
            if cleanup_unknown:
                result.success = False
                result.error_code = SKILL_CANCEL_TIMEOUT
                result.message = "cancel cleanup timed out"
                self._audit(
                    "terminal",
                    error_code=result.error_code,
                    duration_sec=duration_sec,
                    step_count=step_count,
                    **audit_context,
                )
                deferred_goal_handle.force_abort()
                self._retain_admission_cleanup(admission, audit_context, duration_sec, step_count)
            else:
                if not self._finalize_active_admission(
                    admission,
                    error_code=error_code,
                    duration_sec=duration_sec,
                    step_count=step_count,
                ):
                    self.get_logger().error(GATEWAY_FINALIZATION_FAILED)
                    result.success = False
                    result.error_code = GATEWAY_FINALIZATION_FAILED
                    result.message = "gateway finalization failed"
                    self._audit(
                        "terminal",
                        error_code=result.error_code,
                        duration_sec=duration_sec,
                        step_count=step_count,
                        **audit_context,
                    )
                    deferred_goal_handle.force_abort()
                else:
                    self._audit(
                        "terminal",
                        error_code=error_code,
                        duration_sec=duration_sec,
                        step_count=step_count,
                        **audit_context,
                    )
                    deferred_goal_handle.commit()
        return result

    def _execute_workflow_skill_gateway(self, goal_handle):
        goal = goal_handle.request
        binding = goal.dispatch_binding
        result = SkillCommand.Result()

        with self._state_guard():
            workflow = self._active_workflow
            if workflow is None:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_LEASE_MISMATCH", "workflow is not active"
                )
            schema_error = self._validate_skill_goal_schema(goal, workflow.bundle)
            if schema_error:
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    schema_error,
                    "skill schema_version does not match the selected capability",
                )
            index = int(binding.workflow_step_index)
            if binding.schema_version != 1:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_SCHEMA_INVALID", "invalid dispatch binding schema"
                )
            if binding.dispatch_nonce:
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "SKILL_DISPATCH_NOT_AUTHORIZED",
                    "workflow child dispatch nonce must be empty",
                )
            if binding.root_task_id != workflow.root_task_id or not binding.root_task_id:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_LEASE_MISMATCH", "workflow root lease does not match"
                )
            if index < 0 or index >= len(workflow.workflow_steps):
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_STEP_MISMATCH", "workflow step index is out of range"
                )
            expected_step = workflow.workflow_steps[index]
            try:
                requested_step = CanonicalWorkflowStep(
                    schema_version=expected_step.schema_version,
                    skill_name=goal.skill_name,
                    target_name=goal.target_name,
                    container_name=goal.container_name,
                    place_name=goal.place_name,
                    motion_direction=goal.motion_direction,
                    motion_distance=goal.motion_distance,
                    arm_side=getattr(goal, "arm_side", ""),
                    imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
                    timeout_sec=goal.timeout_sec,
                    direction=goal.direction,
                    distance=goal.distance,
                    degree=goal.degree,
                    x=goal.x if goal.has_x else None,
                    y=goal.y if goal.has_y else None,
                    yaw=goal.yaw if goal.has_yaw else None,
                )
            except (TypeError, ValueError) as exc:
                return self._abort_skill(result, goal_handle, [], "SKILL_SCHEMA_INVALID", str(exc))
            expected_task_id = derive_skill_task_id(workflow.root_task_id, index)
            if binding.task_id != expected_task_id or binding.root_lease_nonce != workflow.root_lease_nonce:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_LEASE_MISMATCH", "workflow child lease does not match"
                )
            if binding.expected_registry_epoch != workflow.bundle.registry_epoch:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_REGISTRY_EPOCH_MISMATCH", "workflow registry epoch does not match"
                )
            if (
                binding.expected_registry_generation != workflow.bundle.generation
                or binding.expected_registry_digest != workflow.bundle.snapshot.registry_digest
            ):
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "SKILL_REGISTRY_VERSION_MISMATCH",
                    "workflow registry version does not match",
                )
            if self._task_budget_key(binding) != workflow.task_budget_key:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_TASK_BUDGET_MISMATCH", "workflow task budget does not match"
                )
            if binding.workflow_digest != workflow.workflow_digest:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_DIGEST_MISMATCH", "workflow digest does not match"
                )
            if requested_step != expected_step:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_STEP_MISMATCH", "workflow step does not match"
                )
            if workflow.step_terminal_states[index] != WORKFLOW_STEP_PENDING:
                replay = workflow.child_results.get(index)
                if replay is not None:
                    return replay
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    "SKILL_REQUEST_ID_CONFLICT",
                    "workflow child result is no longer available for replay",
                )
            if index != workflow.completed_step_count:
                return self._abort_skill(
                    result, goal_handle, [], "SKILL_WORKFLOW_STEP_MISMATCH", "workflow step is out of order"
                )
            child_owner = ExecutionOwner.internal_child(workflow.owner, goal.skill_name)
            decision = workflow.policy.evaluate(
                GatewayRequest(
                    task_id=binding.task_id,
                    skill_name=goal.skill_name,
                    schema_version=goal.schema_version,
                    target_name=goal.target_name,
                    container_name=goal.container_name,
                    place_name=goal.place_name,
                    motion_direction=goal.motion_direction,
                    motion_distance=goal.motion_distance,
                    arm_side=getattr(goal, "arm_side", ""),
                    imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
                    direction=goal.direction,
                    distance=goal.distance,
                    degree=goal.degree,
                    x=goal.x if goal.has_x else None,
                    y=goal.y if goal.has_y else None,
                    yaw=goal.yaw if goal.has_yaw else None,
                    timeout_sec=goal.timeout_sec or None,
                ),
                self._runtime_snapshot(),
                owner=child_owner,
            )
            if (
                not decision.admitted
                or workflow.policy.borrow_workflow_internal(
                    workflow.owner,
                    workflow.lease_token,
                    binding.task_id,
                    goal.skill_name,
                )
                is None
            ):
                return self._abort_skill(
                    result,
                    goal_handle,
                    [],
                    decision.error_code or SKILL_BUSY,
                    decision.message or "workflow child admission rejected",
                )
            admission = _WorkflowChildAdmission(workflow=workflow, child_owner=child_owner)
            audit_context = {
                "task_id": binding.task_id,
                "payload_hash": workflow.workflow_digest,
                "skill": goal.skill_name,
            }
            self._active_skill_admission = admission
            workflow.step_terminal_states[index] = WORKFLOW_STEP_ACTIVE
            self._active_skill_owner = child_owner
            self._active_audit_context = audit_context
            self._active_runtime_bundle = workflow.bundle

        self._audit("requested", **audit_context)
        deferred_goal_handle = _DeferredTerminalGoalHandle(goal_handle)
        started = time.monotonic()
        try:
            allowed, reason = self._validate_skill(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                container_name=goal.container_name,
                motion_direction=goal.motion_direction,
                motion_distance=goal.motion_distance,
                arm_side=getattr(goal, "arm_side", ""),
                imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
                direction=goal.direction,
                distance=goal.distance,
                degree=goal.degree,
                x=goal.x if goal.has_x else None,
                y=goal.y if goal.has_y else None,
                yaw=goal.yaw if goal.has_yaw else None,
                schema_version=goal.schema_version,
                dispatch_binding=goal.dispatch_binding,
            )
            if not allowed:
                result = self._abort_skill(result, deferred_goal_handle, [], "SKILL_REJECTED", reason)
            else:
                remaining_budget = workflow.deadline_unix_sec - self._ros_time_sec()
                if remaining_budget <= 0.0:
                    result = self._abort_skill(
                        result,
                        deferred_goal_handle,
                        [],
                        "SKILL_TASK_DEADLINE_EXPIRED",
                        "workflow task budget expired",
                    )
                elif goal_handle.request.timeout_sec > 0.0 and goal_handle.request.timeout_sec > remaining_budget:
                    result = self._abort_skill(
                        result,
                        deferred_goal_handle,
                        [],
                        "SKILL_TASK_BUDGET_MISMATCH",
                        "workflow step timeout exceeds remaining task budget",
                    )
                else:
                    result = self._execute_skill_child(
                        deferred_goal_handle,
                        validation_done=True,
                        effective_timeout_sec=min(decision.effective_timeout_sec, remaining_budget),
                        canonical_task_id=binding.task_id,
                    )
        except Exception as exc:
            result = self._abort_skill(
                result,
                deferred_goal_handle,
                [],
                "SKILL_REJECTED",
                f"workflow child execution failed: {exc}",
            )

        duration_sec = time.monotonic() - started
        cleanup_unknown = result.error_code in {PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SKILL_CANCEL_TIMEOUT}
        with self._state_guard():
            if self._active_skill_admission is admission:
                if result.success and self._active_workflow is workflow:
                    workflow.step_terminal_states[index] = WORKFLOW_STEP_SUCCEEDED
                    workflow.completed_step_count += 1
                elif self._active_workflow is workflow:
                    workflow.step_terminal_states[index] = (
                        WORKFLOW_STEP_CANCELED if result.error_code == "SKILL_CANCELLED" else WORKFLOW_STEP_FAILED
                    )
                if self._active_workflow is workflow:
                    workflow.child_results[index] = result
                if not cleanup_unknown:
                    self._active_skill_admission = None
                    self._active_skill_owner = None
                    self._active_audit_context = None
                    self._active_runtime_bundle = None
                    self._retained_admission_cleanup.pop(id(admission), None)
        self._audit(
            "terminal",
            error_code=result.error_code,
            duration_sec=duration_sec,
            step_count=len(result.executed_primitives),
            **audit_context,
        )
        if cleanup_unknown:
            result.success = False
            result.error_code = SKILL_CANCEL_TIMEOUT
            result.message = "cancel cleanup timed out"
            deferred_goal_handle.force_abort()
            self._retain_admission_cleanup(
                admission,
                audit_context,
                duration_sec,
                len(result.executed_primitives),
            )
        else:
            deferred_goal_handle.commit()
        self._set_result_catalog_identity(result, workflow.bundle)
        return result

    def _execute_skill_child(
        self,
        goal_handle,
        *,
        validation_done: bool = False,
        effective_timeout_sec: float | None = None,
        canonical_task_id: str | None = None,
    ):
        goal = goal_handle.request
        result = SkillCommand.Result()

        if not validation_done:
            allowed, reason = self._validate_skill(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                container_name=goal.container_name,
                motion_direction=goal.motion_direction,
                motion_distance=goal.motion_distance,
                arm_side=getattr(goal, "arm_side", ""),
                imitation_duration_sec=getattr(goal, "imitation_duration_sec", 0.0),
                direction=goal.direction,
                distance=goal.distance,
                degree=goal.degree,
                x=goal.x if goal.has_x else None,
                y=goal.y if goal.has_y else None,
                yaw=goal.yaw if goal.has_yaw else None,
                schema_version=goal.schema_version,
                dispatch_binding=goal.dispatch_binding,
            )
            if not allowed:
                return self._abort_skill(result, goal_handle, [], "SKILL_REJECTED", reason)

        control_mode_ready, control_mode_reason = self._ensure_skill_control_mode(goal.skill_name, goal_handle)
        if not control_mode_ready:
            return self._abort_skill(result, goal_handle, [], "CONTROL_MODE_MISMATCH", control_mode_reason)

        active_bundle = getattr(self, "_active_runtime_bundle", None)
        active_templates = (
            self._mutable_templates(active_bundle.snapshot.templates)
            if active_bundle is not None
            else self._skill_templates
        )
        template = active_templates.get(goal.skill_name, {})
        if str(template.get("executor", "")).strip() == "grasp_pipeline":
            return self._execute_pick_skill(
                goal_handle,
                template,
                effective_timeout_sec=effective_timeout_sec,
            )
        if str(template.get("executor", "")).strip() == "placement_pipeline":
            return self._execute_place_skill(
                goal_handle,
                template,
                effective_timeout_sec=effective_timeout_sec,
            )
        if str(template.get("executor", "")).strip() == "imitate_human_motion":
            return self._execute_imitate_human_motion_skill(
                goal_handle,
                template,
                effective_timeout_sec=effective_timeout_sec,
            )
        if str(template.get("executor", "")).strip() == "sound_following":
            return self._execute_sound_following_skill(
                goal_handle,
                effective_timeout_sec=effective_timeout_sec,
            )
        if str(template.get("executor", "")).strip() == "semantic_map_query":
            return self._execute_semantic_map_query(
                goal_handle,
                effective_timeout_sec=effective_timeout_sec,
            )

        try:
            primitives: list[PrimitiveSpec] = resolve_skill_primitives(
                goal.skill_name,
                goal.target_name,
                goal.place_name,
                goal.motion_direction,
                goal.motion_distance,
                self._named_targets,
                self._gripper_open,
                self._gripper_closed,
                active_templates,
                self._relative_motion_direction_mapping,
                current_joint_positions=self._current_joint_positions(),
                arm_joint_names=self._arm_joint_names,
                direction=goal.direction,
                distance=goal.distance,
                degree=goal.degree,
                x=goal.x if goal.has_x else None,
                y=goal.y if goal.has_y else None,
                yaw=goal.yaw if goal.has_yaw else None,
            )
        except Exception as exc:
            return self._abort_skill(result, goal_handle, [], "SKILL_RESOLUTION_FAILED", str(exc))

        with self._state_guard():
            audit_context = getattr(self, "_active_audit_context", None)
        if audit_context is not None:
            self._audit("accepted", **audit_context)

        skill_deadline = None
        if effective_timeout_sec is not None:
            skill_deadline = time.monotonic() + effective_timeout_sec
        elif goal.timeout_sec > 0.0:
            skill_deadline = time.monotonic() + float(goal.timeout_sec)

        if self._debug:
            self.get_logger().info(
                "[embodied-debug] skill_executor start "
                f"task_id={_binding_task_id(goal)} skill={goal.skill_name} primitives={primitives}"
            )

        if not self._primitive_client.wait_for_server(timeout_sec=self._rpc_timeout):
            result.success = False
            result.error_code = "PRIMITIVE_SERVER_UNAVAILABLE"
            result.message = "primitive action server unavailable"
            result.executed_primitives = []
            goal_handle.abort()
            return result

        executed_primitives: list[str] = []
        total_steps = len(primitives)
        for step_count, primitive in enumerate(primitives, start=1):
            if goal_handle.is_cancel_requested:
                self._audit_cancel_propagated()
                return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)

            remaining_timeout = None
            if skill_deadline is not None:
                remaining_timeout = skill_deadline - time.monotonic()
                if remaining_timeout <= 0.0:
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_TIMEOUT",
                        f"skill deadline exceeded before primitive {primitive.primitive_name}",
                    )

            primitive_name = primitive.primitive_name
            pose_name = primitive.pose_name
            gripper_position = primitive.gripper_position
            feedback = SkillCommand.Feedback()
            feedback.state = "executing"
            feedback.detail = f"step {step_count} of {total_steps}"
            goal_handle.publish_feedback(feedback)

            primitive_goal = PrimitiveCommand.Goal()
            primitive_goal.dispatch_binding = copy_binding(goal.dispatch_binding)
            primitive_goal.dispatch_binding.task_id = canonical_task_id or _binding_task_id(goal)
            primitive_goal.dispatch_binding.dispatch_nonce = uuid.uuid4().hex
            primitive_goal.primitive_name = primitive_name
            descriptor = self._primitive_descriptor(primitive_name)
            if descriptor is None:
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "SKILL_SCHEMA_INVALID",
                    f"unknown primitive descriptor: {primitive_name}",
                )
            primitive_goal.schema_version = descriptor.schema_version
            primitive_goal.pose_name = pose_name
            primitive_goal.velocity_scaling = 0.0
            primitive_goal.relative_dx = float(primitive.relative_dx)
            primitive_goal.relative_dy = float(primitive.relative_dy)
            primitive_goal.relative_dz = float(primitive.relative_dz)
            primitive_goal.gripper_position = float(gripper_position)
            primitive_goal.joint_names = list(primitive.joint_names)
            primitive_goal.joint_positions = [float(position) for position in primitive.joint_positions]
            primitive_goal.joint_waypoints = [
                float(position) for waypoint in primitive.joint_waypoints for position in waypoint
            ]
            primitive_goal.joint_waypoint_count = len(primitive.joint_waypoints)
            primitive_goal.primitive_duration_sec = float(primitive.duration_sec)
            primitive_goal.waypoint_duration_sec = float(primitive.waypoint_duration_sec)
            primitive_goal.timeout_sec = remaining_timeout if remaining_timeout is not None else 0.0
            if primitive.navigation_goal is not None:
                primitive_goal.navigation_command_type = int(primitive.navigation_goal.command_type)
                primitive_goal.navigation_target_pose = copy.deepcopy(primitive.navigation_goal.target_pose)
                primitive_goal.navigation_value = float(primitive.navigation_goal.value)

            if audit_context is not None:
                self._audit("primitive_started", step_count=step_count, **audit_context)

            internal_goal_key = None
            internal_late_cleanup_confirmation = None

            def primitive_feedback_callback(feedback_message, active_primitive_name=primitive_name) -> None:
                if active_primitive_name not in {"nav_straight", "nav_turn", "nav_abs_coordinate"}:
                    return
                primitive_feedback = getattr(feedback_message, "feedback", feedback_message)
                skill_feedback = SkillCommand.Feedback()
                skill_feedback.state = str(primitive_feedback.state)
                skill_feedback.detail = str(primitive_feedback.detail)
                goal_handle.publish_feedback(skill_feedback)

            with self._state_guard():
                admission = getattr(self, "_active_skill_admission", None)
            if admission is not None:
                internal_goal_id = UUID(uuid=list(uuid.uuid4().bytes))
                internal_goal_key = self._register_internal_primitive_goal(
                    internal_goal_id,
                    admission,
                    primitive_goal.dispatch_binding,
                )
                internal_late_cleanup_confirmation = _LateCleanupConfirmation()
                internal_late_cleanup_confirmation.add_callback(
                    lambda admission=admission, goal_key=internal_goal_key: self._confirm_late_internal_cleanup(
                        admission, goal_key
                    )
                )
                try:
                    send_goal_future = self._primitive_client.send_goal_async(
                        primitive_goal,
                        feedback_callback=primitive_feedback_callback,
                        goal_uuid=internal_goal_id,
                    )
                except Exception:
                    self._forget_internal_primitive_goal(internal_goal_key)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"primitive send state is unknown: {primitive_name}",
                    )
            else:
                try:
                    send_goal_future = self._primitive_client.send_goal_async(
                        primitive_goal,
                        feedback_callback=primitive_feedback_callback,
                    )
                except Exception:
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"primitive send state is unknown: {primitive_name}",
                    )
            send_goal_timeout = (
                self._rpc_timeout if remaining_timeout is None else min(self._rpc_timeout, remaining_timeout)
            )

            if not self._wait_for_future(
                send_goal_future,
                timeout_sec=max(0.1, send_goal_timeout),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                if internal_goal_key is not None:
                    if goal_handle.is_cancel_requested:
                        if self._cancel_goal_future(send_goal_future, internal_late_cleanup_confirmation):
                            self._forget_internal_primitive_goal(internal_goal_key)
                            return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                        return self._abort_skill(
                            result,
                            goal_handle,
                            executed_primitives,
                            SKILL_CANCEL_TIMEOUT,
                            f"cancel cleanup timed out for primitive {primitive_name}",
                        )
                    internal_late_cleanup_confirmation.watch_goal_future(
                        send_goal_future,
                        self._best_effort_cancel_goal,
                    )
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if goal_handle.is_cancel_requested:
                    if self._cancel_goal_future(send_goal_future):
                        if internal_goal_key is not None:
                            self._forget_internal_primitive_goal(internal_goal_key)
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if not self._cancel_goal_future(send_goal_future):
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_TIMEOUT",
                    f"timed out sending primitive {primitive_name}",
                )

            try:
                primitive_handle = send_goal_future.result()
            except Exception:
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive send state is unknown: {primitive_name}",
                )
            if primitive_handle is None or not primitive_handle.accepted:
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    "PRIMITIVE_GOAL_REJECTED",
                    f"primitive goal rejected: {primitive_name}",
                )

            try:
                result_future = primitive_handle.get_result_async()
            except Exception:
                self._best_effort_cancel_goal(primitive_handle)
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive result state is unknown: {primitive_name}",
                )
            remaining_timeout = None
            if skill_deadline is not None:
                remaining_timeout = skill_deadline - time.monotonic()
                if remaining_timeout <= 0.0:
                    if not self._cancel_goal(primitive_handle, result_future):
                        if internal_late_cleanup_confirmation is not None:
                            internal_late_cleanup_confirmation.watch_result_future(result_future)
                        return self._abort_skill(
                            result,
                            goal_handle,
                            executed_primitives,
                            SKILL_CANCEL_TIMEOUT,
                            f"cancel cleanup timed out for primitive {primitive_name}",
                        )
                    if internal_goal_key is not None:
                        self._forget_internal_primitive_goal(internal_goal_key)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        "SKILL_TIMEOUT",
                        f"primitive timed out: {primitive_name}",
                    )
            wait_timeout = remaining_timeout if remaining_timeout is not None else 30.0
            if not self._wait_for_future(
                result_future,
                timeout_sec=max(0.1, wait_timeout),
                cancel_requested=lambda: goal_handle.is_cancel_requested,
            ):
                if goal_handle.is_cancel_requested:
                    if self._cancel_goal(primitive_handle, result_future):
                        if internal_goal_key is not None:
                            self._forget_internal_primitive_goal(internal_goal_key)
                        return self._cancel_skill(result, goal_handle, executed_primitives, goal.skill_name)
                    if internal_late_cleanup_confirmation is not None:
                        internal_late_cleanup_confirmation.watch_result_future(result_future)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if not self._cancel_goal(primitive_handle, result_future):
                    if internal_late_cleanup_confirmation is not None:
                        internal_late_cleanup_confirmation.watch_result_future(result_future)
                    return self._abort_skill(
                        result,
                        goal_handle,
                        executed_primitives,
                        SKILL_CANCEL_TIMEOUT,
                        f"cancel cleanup timed out for primitive {primitive_name}",
                    )
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result, goal_handle, executed_primitives, "SKILL_TIMEOUT", f"primitive timed out: {primitive_name}"
                )

            try:
                action_result = result_future.result()
            except Exception:
                self._best_effort_cancel_goal(primitive_handle)
                if internal_goal_key is not None:
                    self._forget_internal_primitive_goal(internal_goal_key)
                return self._abort_skill(
                    result,
                    goal_handle,
                    executed_primitives,
                    SKILL_CANCEL_TIMEOUT,
                    f"primitive result state is unknown: {primitive_name}",
                )
            if internal_goal_key is not None:
                self._forget_internal_primitive_goal(internal_goal_key)
            primitive_result = action_result.result if action_result is not None else None
            if primitive_result is None or not primitive_result.success:
                error_code = primitive_result.error_code if primitive_result is not None else "MISSING_PRIMITIVE_RESULT"
                message = primitive_result.message if primitive_result is not None else "missing primitive result"
                if error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                    error_code = SKILL_CANCEL_TIMEOUT
                return self._abort_skill(result, goal_handle, executed_primitives, error_code, message)

            primitive_label = primitive_name if not pose_name else f"{primitive_name}:{pose_name}"
            if primitive_name == "move_relative_ee":
                primitive_label = (
                    f"{primitive_name}:{primitive.relative_dx:.3f},"
                    f"{primitive.relative_dy:.3f},{primitive.relative_dz:.3f}"
                )
            if primitive_name == "move_to_joint_positions":
                primitive_label = f"{primitive_name}:" + ",".join(
                    f"{joint_name}={joint_position:.3f}"
                    for joint_name, joint_position in zip(
                        primitive.joint_names, primitive.joint_positions, strict=False
                    )
                )
            if primitive_name == "move_through_joint_positions":
                primitive_label = f"{primitive_name}:{len(primitive.joint_waypoints)} waypoints"
            executed_primitives.append(primitive_label)

        result.success = True
        result.error_code = ""
        result.message = f"skill completed: {goal.skill_name}"
        result.executed_primitives = executed_primitives
        self._set_result_catalog_identity(result)
        goal_handle.succeed()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkillExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        # A client may disappear after its request times out.  Fast DDS then
        # raises from ``send_response``; do not let that transport condition
        # tear down the Gateway and all of its safety services.
        while rclpy.ok():
            try:
                executor.spin_once()
            except Exception as exc:
                node.get_logger().error(
                    "[skill_executor] ROS callback/response failed; keeping Gateway alive: "
                    f"{exc}\n{traceback.format_exc()}"
                )
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
