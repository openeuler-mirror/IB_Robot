"""Safety validation node for the embodied minimal closure."""

from collections.abc import Mapping
from threading import RLock

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from embodied_common.skill_request import validate_request_schema_version
from embodied_common.tracing import create_trace_logger, trace_scope, trace_stage
from embodied_common.wire_contracts import validate_public_request_wire_contracts
from ibrobot_msgs.msg import SkillRegistryEvent
from ibrobot_msgs.srv import GetSkillGatewayStatus, GetSkillSnapshot, ValidatePrimitive, ValidateSkill
from safety_guard.rules import (
    load_json_mapping,
    validate_primitive_request,
    validate_skill_request,
)
from safety_guard.snapshot_cache import SafetySnapshotCache, SnapshotCacheError, SnapshotIdentity

_trace = create_trace_logger("ib_trace.safety")


class SafetyGuardNode(Node):
    """Provide explicit safety validation services for skills and primitives."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("safety_guard_node", parameter_overrides=parameter_overrides)
        validate_public_request_wire_contracts()
        self.declare_parameter("validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("validate_primitive_service", "/embodied/validate_primitive")
        self.declare_parameter("skill_gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("skill_catalog_snapshot_service", "/embodied/get_skill_snapshot")
        self.declare_parameter("skill_registry_event_topic", "/embodied/skill_registry_events")
        self.declare_parameter("snapshot_sync_period_sec", 1.0)
        self.declare_parameter("named_poses_json", "{}")
        self.declare_parameter("named_targets_json", "{}")
        self.declare_parameter("workspace_json", "{}")
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("debug_tracing", False)

        self._validate_skill_service = self.get_parameter("validate_skill_service").get_parameter_value().string_value
        self._validate_primitive_service = (
            self.get_parameter("validate_primitive_service").get_parameter_value().string_value
        )
        self._skill_gateway_status_service = (
            self.get_parameter("skill_gateway_status_service").get_parameter_value().string_value
        )
        self._skill_catalog_snapshot_service = (
            self.get_parameter("skill_catalog_snapshot_service").get_parameter_value().string_value
        )
        self._skill_registry_event_topic = (
            self.get_parameter("skill_registry_event_topic").get_parameter_value().string_value
        )
        self._snapshot_sync_period_sec = (
            self.get_parameter("snapshot_sync_period_sec").get_parameter_value().double_value
        )
        if self._snapshot_sync_period_sec <= 0.0:
            raise ValueError("snapshot_sync_period_sec must be positive")
        self._named_poses = load_json_mapping(self.get_parameter("named_poses_json").get_parameter_value().string_value)
        self._named_targets = load_json_mapping(
            self.get_parameter("named_targets_json").get_parameter_value().string_value
        )
        self._workspace = load_json_mapping(self.get_parameter("workspace_json").get_parameter_value().string_value)
        arm_joint_names = load_json_mapping(
            '{"items": ' + self.get_parameter("arm_joint_names_json").get_parameter_value().string_value + "}"
        ).get("items", [])
        self._arm_joint_names = (
            [str(joint_name) for joint_name in arm_joint_names] if isinstance(arm_joint_names, list) else []
        )
        self._joint_limits = load_json_mapping(
            self.get_parameter("joint_limits_json").get_parameter_value().string_value
        )
        self._debug = self.get_parameter("debug_tracing").get_parameter_value().bool_value

        self._snapshot_cache = SafetySnapshotCache()
        self._sync_lock = RLock()
        self._status_in_flight = False
        self._snapshot_in_flight: set[tuple[str, int]] = set()
        self._desired_current: SnapshotIdentity | None = None
        self._retained_generations: set[int] = set()
        self._known_identities: dict[tuple[str, int], SnapshotIdentity] = {}
        self._gateway_status_client = self.create_client(GetSkillGatewayStatus, self._skill_gateway_status_service)
        self._snapshot_client = self.create_client(GetSkillSnapshot, self._skill_catalog_snapshot_service)
        registry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            SkillRegistryEvent,
            self._skill_registry_event_topic,
            self._handle_registry_event,
            registry_qos,
        )
        self.create_timer(self._snapshot_sync_period_sec, self._sync_gateway_status)

        self.create_service(ValidateSkill, self._validate_skill_service, self._handle_validate_skill)
        self.create_service(ValidatePrimitive, self._validate_primitive_service, self._handle_validate_primitive)

        self.get_logger().info(
            "[embodied-debug] safety_guard ready: "
            f"validate_skill_service={self._validate_skill_service}, "
            f"validate_primitive_service={self._validate_primitive_service}"
        )

    @staticmethod
    def _thaw(value):
        if isinstance(value, Mapping):
            return {str(key): SafetyGuardNode._thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [SafetyGuardNode._thaw(item) for item in value]
        return value

    def _sync_gateway_status(self) -> None:
        with self._sync_lock:
            if self._status_in_flight or not self._gateway_status_client.service_is_ready():
                return
            self._status_in_flight = True
        request = GetSkillGatewayStatus.Request()
        request.schema_version = 1
        try:
            future = self._gateway_status_client.call_async(request)
        except Exception:
            with self._sync_lock:
                self._status_in_flight = False
            return
        future.add_done_callback(self._handle_status_future)

    def _handle_status_future(self, future) -> None:
        with self._sync_lock:
            self._status_in_flight = False
        try:
            response = future.result()
        except Exception:
            return
        if (
            response is None
            or response.schema_version != 1
            or not response.registry_epoch
            or response.registry_generation <= 0
            or not response.registry_digest
        ):
            return
        current = SnapshotIdentity(
            response.registry_epoch,
            int(response.registry_generation),
            response.registry_digest,
        )
        retained = {int(value) for value in response.retained_generations if int(value) > 0}
        retained.add(current.generation)
        with self._sync_lock:
            desired = self._desired_current
            if (
                desired is not None
                and current.registry_epoch == desired.registry_epoch
                and current.generation < desired.generation
            ):
                return
            if desired is not None and current.registry_epoch != desired.registry_epoch:
                self._snapshot_cache.reconcile(current.registry_epoch, retained)
                self._known_identities.clear()
            self._desired_current = current
            self._retained_generations = retained
            self._known_identities[(current.registry_epoch, current.generation)] = current
        for generation in sorted(retained):
            identity = self._known_identities.get((current.registry_epoch, generation))
            if identity is not None:
                self._request_snapshot(identity)
        self._snapshot_cache.reconcile(current.registry_epoch, retained)

    def _handle_registry_event(self, event: SkillRegistryEvent) -> None:
        if (
            event.schema_version != 1
            or not event.registry_epoch
            or event.new_generation <= 0
            or not event.registry_digest
        ):
            return
        identity = SnapshotIdentity(event.registry_epoch, int(event.new_generation), event.registry_digest)
        with self._sync_lock:
            desired = self._desired_current
            if (
                desired is not None
                and identity.registry_epoch == desired.registry_epoch
                and identity.generation <= desired.generation
            ):
                return
            if desired is not None and identity.registry_epoch != desired.registry_epoch:
                self._snapshot_cache.reconcile(identity.registry_epoch, {identity.generation})
                self._retained_generations.clear()
                self._known_identities.clear()
            self._desired_current = identity
            self._retained_generations.add(identity.generation)
            self._known_identities[(identity.registry_epoch, identity.generation)] = identity
        self._request_snapshot(identity)

    def _request_snapshot(self, identity: SnapshotIdentity) -> None:
        if identity.registry_digest:
            try:
                self._snapshot_cache.get(identity)
                with self._sync_lock:
                    if self._desired_current == identity:
                        self._snapshot_cache.mark_current(identity)
                return
            except SnapshotCacheError:
                pass
        key = (identity.registry_epoch, identity.generation)
        with self._sync_lock:
            if key in self._snapshot_in_flight or not self._snapshot_client.service_is_ready():
                return
            self._snapshot_in_flight.add(key)
        request = GetSkillSnapshot.Request()
        request.schema_version = 1
        request.registry_epoch = identity.registry_epoch
        request.generation = identity.generation
        try:
            future = self._snapshot_client.call_async(request)
        except Exception:
            with self._sync_lock:
                self._snapshot_in_flight.discard(key)
            return
        future.add_done_callback(
            lambda completed, request_key=key: self._handle_snapshot_future(request_key, completed)
        )

    def _handle_snapshot_future(self, key: tuple[str, int], future) -> None:
        with self._sync_lock:
            self._snapshot_in_flight.discard(key)
        try:
            response = future.result()
        except Exception:
            return
        if response is None or not response.success or (response.registry_epoch, int(response.generation)) != key:
            return
        response_identity = SnapshotIdentity(
            response.registry_epoch,
            int(response.generation),
            response.registry_digest,
        )
        with self._sync_lock:
            retained = set(self._retained_generations)
            desired = self._desired_current
            expected = self._known_identities.get(key)
            if expected is None or response_identity != expected:
                return
            if response.generation not in retained and response_identity != desired:
                return
            try:
                self._snapshot_cache.activate(
                    registry_epoch=response.registry_epoch,
                    generation=int(response.generation),
                    registry_digest=response.registry_digest,
                    capability_digest=response.capability_digest,
                    provenance_digest=response.provenance_digest,
                    snapshot_json=response.snapshot_json,
                    make_current=response_identity == desired,
                )
            except SnapshotCacheError as exc:
                self.get_logger().error(f"safety snapshot rejected: {exc.code}: {exc}")
                return
            self._snapshot_cache.reconcile(response.registry_epoch, retained)

    @staticmethod
    def _set_actual_identity(response, identity: SnapshotIdentity | None) -> None:
        response.actual_registry_epoch = identity.registry_epoch if identity else ""
        response.actual_registry_generation = identity.generation if identity else 0
        response.actual_registry_digest = identity.registry_digest if identity else ""

    def _handle_validate_skill(self, request, response):
        task_id = str(request.dispatch_binding.task_id)
        with (
            trace_scope(str(getattr(request.dispatch_binding, "trace_id", "") or task_id)),
            trace_stage(
                _trace,
                "safety.skill",
                task_id=task_id,
                skill=str(request.skill_name),
            ),
        ):
            return self._handle_validate_skill_traced(request, response)

    def _handle_validate_skill_traced(self, request, response):
        expected = SnapshotIdentity(
            request.dispatch_binding.expected_registry_epoch,
            int(request.dispatch_binding.expected_registry_generation),
            request.dispatch_binding.expected_registry_digest,
        )
        if (
            request.dispatch_binding.schema_version != 1
            or not all((expected.registry_epoch, expected.generation > 0, expected.registry_digest))
            or request.dispatch_binding.dispatch_nonce
        ):
            response.allowed = False
            response.reason = "planned validation requires exact identity and an empty dispatch nonce"
            response.error_code = "SKILL_SCHEMA_INVALID"
            self._set_actual_identity(response, self._snapshot_cache.current_identity)
            return response
        try:
            snapshot = self._snapshot_cache.get(expected)
        except SnapshotCacheError as exc:
            response.allowed = False
            response.reason = str(exc)
            response.error_code = exc.code
            self._set_actual_identity(response, self._snapshot_cache.current_identity)
            return response
        robot_context = self._thaw(snapshot.robot_context)
        skill_templates = self._thaw(snapshot.templates)
        try:
            request_schema_version = validate_request_schema_version(request.schema_version)
        except ValueError as exc:
            response.allowed = False
            response.reason = str(exc)
            response.error_code = "SKILL_SCHEMA_INVALID"
            self._set_actual_identity(response, snapshot.identity)
            return response
        try:
            allowed, reason = validate_skill_request(
                request.skill_name,
                request.target_name,
                request.place_name,
                request.motion_direction,
                request.motion_distance,
                robot_context.get("named_poses", {}),
                robot_context.get("named_targets", {}),
                skill_templates,
                robot_context.get("arm_joint_names", []),
                robot_context.get("joint_limits", {}),
                container_name=request.container_name,
                arm_side=request.arm_side,
                imitation_duration_sec=request.imitation_duration_sec,
                direction=request.direction,
                distance=request.distance,
                degree=request.degree,
                x=request.x if request.has_x else None,
                y=request.y if request.has_y else None,
                yaw=request.yaw if request.has_yaw else None,
                schema_version=request_schema_version,
                primitive_descriptors=snapshot.primitive_descriptors,
            )
        except Exception as exc:
            self.get_logger().error(f"[safety_guard] uncaught exception in skill validation: {exc}")
            response.allowed = False
            response.reason = "safety validation is temporarily unavailable"
            response.error_code = "CAPABILITY_NOT_READY"
            self._set_actual_identity(response, snapshot.identity)
            response.diagnostics = []
            return response
        response.allowed = allowed
        response.reason = reason
        response.error_code = "" if allowed else "SKILL_LIMIT_VIOLATION"
        self._set_actual_identity(response, snapshot.identity)
        response.diagnostics = []
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] safety_guard skill_check "
                f"skill={request.skill_name} target={request.target_name or '-'} "
                f"container={request.container_name or '-'} "
                f"place={request.place_name or '-'} motion_direction={request.motion_direction or '-'} "
                f"motion_distance={request.motion_distance:.3f} allowed={allowed} reason='{reason}'"
            )
        return response

    def _handle_validate_primitive(self, request, response):
        task_id = str(request.dispatch_binding.task_id)
        with (
            trace_scope(str(getattr(request.dispatch_binding, "trace_id", "") or task_id)),
            trace_stage(
                _trace,
                "safety.primitive",
                task_id=task_id,
                primitive=str(request.primitive_name),
            ),
        ):
            return self._handle_validate_primitive_traced(request, response)

    def _handle_validate_primitive_traced(self, request, response):
        expected = SnapshotIdentity(
            request.dispatch_binding.expected_registry_epoch,
            int(request.dispatch_binding.expected_registry_generation),
            request.dispatch_binding.expected_registry_digest,
        )
        if (
            request.dispatch_binding.schema_version != 1
            or not expected.registry_epoch
            or expected.generation <= 0
            or not expected.registry_digest
            or not request.dispatch_binding.dispatch_nonce
        ):
            response.allowed = False
            response.reason = "primitive validation requires exact identity and dispatch nonce"
            response.error_code = "SKILL_SCHEMA_INVALID"
            self._set_actual_identity(response, self._snapshot_cache.current_identity)
            return response
        try:
            snapshot = self._snapshot_cache.get(expected)
        except SnapshotCacheError as exc:
            response.allowed = False
            response.reason = str(exc)
            response.error_code = exc.code
            self._set_actual_identity(response, self._snapshot_cache.current_identity)
            return response
        robot_context = self._thaw(snapshot.robot_context)
        try:
            request_schema_version = validate_request_schema_version(request.schema_version)
        except ValueError as exc:
            response.allowed = False
            response.reason = str(exc)
            response.error_code = "SKILL_SCHEMA_INVALID"
            self._set_actual_identity(response, snapshot.identity)
            return response
        try:
            allowed, reason = validate_primitive_request(
                request.primitive_name,
                request.pose_name,
                request.relative_dx,
                request.relative_dy,
                request.relative_dz,
                request.target_x,
                request.target_y,
                request.target_z,
                request.gripper_position,
                robot_context.get("named_poses", {}),
                robot_context.get("workspace_limits", {}),
                list(request.joint_names),
                list(request.joint_positions),
                list(request.joint_waypoints),
                request.joint_waypoint_count,
                robot_context.get("arm_joint_names", []),
                robot_context.get("joint_limits", {}),
                request.primitive_duration_sec,
                request.waypoint_duration_sec,
                request.target_qx,
                request.target_qy,
                request.target_qz,
                request.target_qw,
                request.velocity_scaling,
                request.navigation_command_type,
                request.navigation_target_pose.header.frame_id,
                request.navigation_target_pose.pose.position.x,
                request.navigation_target_pose.pose.position.y,
                request.navigation_target_pose.pose.position.z,
                request.navigation_target_pose.pose.orientation.x,
                request.navigation_target_pose.pose.orientation.y,
                request.navigation_target_pose.pose.orientation.z,
                request.navigation_target_pose.pose.orientation.w,
                request.navigation_value,
                schema_version=request_schema_version,
                primitive_descriptors=snapshot.primitive_descriptors,
            )
        except Exception as exc:
            self.get_logger().error(f"[safety_guard] uncaught exception in primitive validation: {exc}")
            response.allowed = False
            response.reason = "safety validation is temporarily unavailable"
            response.error_code = "CAPABILITY_NOT_READY"
            self._set_actual_identity(response, snapshot.identity)
            response.diagnostics = []
            return response
        response.allowed = allowed
        response.reason = reason
        response.error_code = "" if allowed else "SKILL_LIMIT_VIOLATION"
        self._set_actual_identity(response, snapshot.identity)
        response.diagnostics = []
        if self._debug:
            self.get_logger().info(
                "[embodied-debug] safety_guard primitive_check "
                f"primitive={request.primitive_name} pose={request.pose_name or '-'} "
                f"delta=({request.relative_dx:.3f}, {request.relative_dy:.3f}, {request.relative_dz:.3f}) "
                f"target=({request.target_x:.3f}, {request.target_y:.3f}, {request.target_z:.3f}) "
                f"gripper={request.gripper_position:.3f} allowed={allowed} reason='{reason}'"
            )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyGuardNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
