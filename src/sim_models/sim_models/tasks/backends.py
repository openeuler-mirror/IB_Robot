"""Platform backends for the pick_banana scene task (so101-mujoco-runtime-platform D9).

A backend owns everything platform-specific: placing the objects (the shared
randomisation index produces identical placements on both platforms) and
reporting live object positions for evaluation. Platform message packages
(``mujoco_ros2_control_msgs``, ``ros_gz_interfaces``, ``tf2_msgs``) are
imported lazily inside the backend classes, so importing this module — and
running the node on the other platform — never requires them.

Readiness is revalidated on every call (:meth:`PickBananaBackend.check_ready`)
rather than latched at startup: the task refuses to move anything unless the
backend can place right now. Gazebo uses only the bridged ``SetEntityPose``
service (the T0 spike chose the bridge; there is no CLI fallback); MuJoCo
places through one atomic ``ResetWorld`` call with state overrides (issue
#195) and observes the scene through the FreeJointStatePublisherPlugin
stream — no shadow model, the observed poses are the live simulation's.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from sim_models.scene_compiler import (
    _layout_dynamic_objects,
    _quat_from_extrinsic_xyz,
    free_joint_states_topic,
    randomization_placements,
)

# Evaluation treats an object pose older than this as unavailable (D9).
POSE_STALE_S = 1.0


def _wait_done(done: threading.Event, timeout_s: float, stop_event=None) -> bool:
    """Wait for an RPC completion event; False on timeout or stop (sliced)."""
    if stop_event is None:
        return done.wait(timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if done.wait(timeout=min(remaining, 0.1)):
            return True
        if stop_event.is_set():
            return False


class PickBananaBackend:
    """Shared placement/observation contract over the scene layout (D9).

    Placements come from one generator — ``scene_compiler.randomization_placements``
    — so index *k* is the same banana pose on every platform: the MuJoCo
    backend applies it as ``ResetWorld`` state overrides and the Gazebo
    backend sets the entity pose from ``placements[k]`` directly.
    ``index=None`` means the layout default placement.
    """

    #: Objects the backend must observe freshly to be ready.
    REQUIRED_OBJECTS: tuple[str, ...] = ()

    def __init__(self, node, layout: dict, *, callback_group) -> None:
        self._node = node
        self._cbg = callback_group
        layout = layout or {}
        self._placements = randomization_placements(layout)
        self._dynamic = {name: (position, euler) for name, position, euler in _layout_dynamic_objects(layout)}
        self._startup_ready = False

    @property
    def grid_size(self) -> int:
        return len(self._placements)

    def check_ready(self) -> tuple[bool, str]:
        """Live readiness revalidation; called before every task operation."""
        raise NotImplementedError

    def wait_ready(self, timeout_s: float) -> tuple[bool, str]:
        """Bounded startup initialisation: poll the live check until ready."""
        deadline = time.monotonic() + timeout_s
        result = (False, "initialisation timed out")
        while time.monotonic() < deadline:
            result = self.check_ready()
            if result[0]:
                self._startup_ready = True
                return result
            time.sleep(0.1)
        return result

    def start_readiness(self, timeout_s: float = 10.0) -> None:
        """Run the bounded initialisation on a daemon thread and log it (D9)."""

        def _wait() -> None:
            ok, message = self.wait_ready(timeout_s)
            state = "ready" if ok else f"NOT ready ({message})"
            self._node.get_logger().info(f"{type(self).__name__}: {state}")

        threading.Thread(target=_wait, daemon=True, name=f"{type(self).__name__}-readiness").start()

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        raise NotImplementedError

    def object_position(self, name: str) -> np.ndarray | None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def _banana_pose(self, index: int | None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Banana placement (position, w-x-y-z quaternion) for a placement index or the default."""
        if index is None:
            position, euler = self._dynamic["banana"]
            return tuple(position[:3]), _quat_from_extrinsic_xyz(*euler)  # type: ignore[return-value]
        x, y, z, yaw = self._placements[index]
        return (x, y, z), _quat_from_extrinsic_xyz(0.0, 0.0, yaw)

    def _plate_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        position, euler = self._dynamic["plate"]
        return tuple(position[:3]), _quat_from_extrinsic_xyz(*euler)  # type: ignore[return-value]


# ─── MuJoCo ───────────────────────────────────────────────────────────────────


class MujocoBackend(PickBananaBackend):
    """Places the scene through one atomic ``ResetWorld`` call (issue #195).

    ``ResetWorld(keyframe="", state_overrides=...)`` restores the simulator's
    captured startup state and then applies overrides atomically: the arm
    joints are pinned to the layout ``arm_rest_qpos`` (the startup capture
    holds zero arm joints, so the override is what puts the arm at rest) with
    zero velocity, and the free-joint bodies (banana at the placement index's
    pose or the layout default, plate at its layout pose) are teleported by
    body name with zero twist. Nothing in between is observable by the task.

    Observation runs through the simulator's FreeJointStatePublisherPlugin:
    the runtime provider publishes the scene's dynamic bodies on
    ``free_joint_states_topic(scene)`` and this backend subscribes, keeping
    per-object samples fresh by the monotonic receipt time (``POSE_STALE_S``).
    The observed poses are the live simulation's — pushing or grasping the
    banana is visible immediately, which the retired shadow model could not
    do.
    """

    REQUIRED_OBJECTS = ("banana", "plate")

    def __init__(
        self,
        node,
        layout: dict,
        *,
        scene: str,
        reset_world_service: str = "",
        callback_group,
    ) -> None:
        from mujoco_ros2_control_msgs.msg import FreeJointStateArray  # lazy: mujoco-only dependency
        from mujoco_ros2_control_msgs.srv import ResetWorld  # lazy: mujoco-only dependency

        super().__init__(node, layout, callback_group=callback_group)
        self._reset_world_name = reset_world_service or "/mujoco_ros2_control_node/reset_world"
        self._reset_world = node.create_client(ResetWorld, self._reset_world_name, callback_group=callback_group)
        self._arm_joint_names = [str(j) for j in (layout or {}).get("randomization", {}).get("arm_joint_names", [])]
        self._arm_rest_qpos = [float(q) for q in (layout or {}).get("randomization", {}).get("arm_rest_qpos", [])]
        self._poses: dict[str, tuple[np.ndarray, float]] = {}
        node.create_subscription(
            FreeJointStateArray,
            free_joint_states_topic(scene),
            self._on_free_joint_states,
            10,
            callback_group=callback_group,
        )

    def check_ready(self) -> tuple[bool, str]:
        if not self._reset_world.service_is_ready():
            return False, f"reset_world service not available ({self._reset_world_name})"
        stale = [name for name in self.REQUIRED_OBJECTS if not self._fresh_sample(name)]
        if stale:
            return False, f"no fresh free_joint_states sample for {', '.join(stale)}"
        return True, "reset_world service and free_joint_states stream present"

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        banana_position, banana_quat = self._banana_pose(index)
        plate_position, plate_quat = self._plate_pose()
        return self._call_reset_world(
            (("banana", banana_position, banana_quat), ("plate", plate_position, plate_quat)),
            stop_event=stop_event,
        )

    def object_position(self, name: str) -> np.ndarray | None:
        sample = self._poses.get(name)
        if sample is None:
            return None
        position, received_at = sample
        if time.monotonic() - received_at > POSE_STALE_S:
            return None
        return position.copy()

    def _fresh_sample(self, name: str) -> bool:
        sample = self._poses.get(name)
        return sample is not None and time.monotonic() - sample[1] <= POSE_STALE_S

    def _on_free_joint_states(self, msg) -> None:
        received_at = time.monotonic()
        for entry in msg.free_joints:
            position = entry.pose.pose.position
            self._poses[str(entry.name)] = (
                np.array([position.x, position.y, position.z]),
                received_at,
            )

    def _call_reset_world(
        self,
        object_poses: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float, float]], ...],
        timeout_s: float = 15.0,
        stop_event=None,
    ) -> tuple[bool, str]:
        """One atomic ``ResetWorld`` call: startup state + arm rest + object poses."""
        from mujoco_ros2_control_msgs.msg import FreeJointState  # lazy: mujoco-only dependency

        if not self._reset_world.wait_for_service(timeout_sec=5.0):
            return False, f"reset_world service not available ({self._reset_world_name})"
        request = self._reset_world.srv_type.Request()
        # keyframe "" restores the startup capture (arm at zero); the joint
        # override then pins the arm to the layout rest pose so the placement
        # and the trajectory controller's hold setpoint agree. Velocity is
        # set explicitly to zero — self-describing and immune to whatever the
        # startup capture might hold.
        request.keyframe = ""
        joints = request.state_overrides.joint_states
        joints.name = list(self._arm_joint_names)
        joints.position = list(self._arm_rest_qpos)
        joints.velocity = [0.0] * len(joints.name)
        for name, position, quat in object_poses:
            entry = FreeJointState()
            entry.name = name
            entry.pose.pose.position.x, entry.pose.pose.position.y, entry.pose.pose.position.z = position
            w, x, y, z = quat
            entry.pose.pose.orientation.w, entry.pose.pose.orientation.x = w, x
            entry.pose.pose.orientation.y, entry.pose.pose.orientation.z = y, z
            request.state_overrides.free_joints.append(entry)
        future = self._reset_world.call_async(request)
        done = threading.Event()
        result_box: list[tuple[bool, str]] = []

        def _cb(f):
            try:
                response = f.result()
                result_box.append((bool(response.success), str(response.message)))
            except Exception as exc:  # noqa: BLE001
                result_box.append((False, str(exc)))
            done.set()

        future.add_done_callback(_cb)
        if not _wait_done(done, timeout_s, stop_event):
            if stop_event is not None and stop_event.is_set():
                return False, "reset_world interrupted (stop)"
            return False, "reset_world timeout"
        return result_box[0]


# ─── Gazebo ───────────────────────────────────────────────────────────────────


class GazeboBackend(PickBananaBackend):
    """Places objects through the bridged ``SetEntityPose`` service (D9).

    Only the bridged transport is used (the T0 spike verified service
    bridging works on Fortress); readiness requires the bridge and fresh
    samples of every scene object the task places.
    """

    REQUIRED_OBJECTS = ("banana", "plate")

    def __init__(
        self,
        node,
        layout: dict,
        *,
        world_name: str,
        scene: str,
        callback_group,
    ) -> None:
        from ros_gz_interfaces.msg import Entity  # lazy: gazebo-only dependency
        from ros_gz_interfaces.srv import SetEntityPose  # lazy: gazebo-only dependency
        from tf2_msgs.msg import TFMessage  # lazy: gazebo-only dependency

        super().__init__(node, layout, callback_group=callback_group)
        self._world_name = world_name
        self._entity_type = Entity.MODEL
        self._set_pose_service = f"/world/{world_name}/set_pose"
        self._set_pose = node.create_client(SetEntityPose, self._set_pose_service, callback_group=callback_group)
        self._poses: dict[str, tuple[np.ndarray, float]] = {}
        node.create_subscription(
            TFMessage, f"/sim/{scene}/object_poses", self._on_object_poses, 10, callback_group=callback_group
        )

    def check_ready(self) -> tuple[bool, str]:
        if not self._set_pose.service_is_ready():
            return False, f"set_pose bridge not available ({self._set_pose_service})"
        stale = [name for name in self.REQUIRED_OBJECTS if not self._fresh_sample(name)]
        if stale:
            return False, f"no fresh object_poses sample for {', '.join(stale)} on /sim"
        return True, "set_pose bridge and object_poses stream present"

    def place(self, index: int | None, stop_event=None) -> tuple[bool, str]:
        banana_position, banana_quat = self._banana_pose(index)
        plate_position, plate_quat = self._plate_pose()
        for name, position, quat in (("banana", banana_position, banana_quat), ("plate", plate_position, plate_quat)):
            ok, message = self._set_entity_pose(name, position, quat, stop_event=stop_event)
            if not ok:
                return False, f"set pose of {name} failed: {message}"
        return True, "placed"

    def object_position(self, name: str) -> np.ndarray | None:
        sample = self._poses.get(name)
        if sample is None:
            return None
        position, received_at = sample
        if time.monotonic() - received_at > POSE_STALE_S:
            return None
        return position.copy()

    def _fresh_sample(self, name: str) -> bool:
        sample = self._poses.get(name)
        return sample is not None and time.monotonic() - sample[1] <= POSE_STALE_S

    def _on_object_poses(self, msg) -> None:
        received_at = time.monotonic()
        for transform in msg.transforms:
            translation = transform.transform.translation
            self._poses[transform.child_frame_id] = (
                np.array([translation.x, translation.y, translation.z]),
                received_at,
            )

    def _set_entity_pose(
        self,
        name: str,
        position: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
        stop_event=None,
    ) -> tuple[bool, str]:
        if not self._set_pose.service_is_ready():
            return False, "set_pose bridge not available"
        request = self._set_pose.srv_type.Request()
        request.entity.name = name
        request.entity.type = self._entity_type
        request.pose.position.x, request.pose.position.y, request.pose.position.z = position
        w, x, y, z = quat_wxyz
        request.pose.orientation.w, request.pose.orientation.x = w, x
        request.pose.orientation.y, request.pose.orientation.z = y, z
        future = self._set_pose.call_async(request)
        done = threading.Event()

        def _cb(_future):
            done.set()

        future.add_done_callback(_cb)
        if not _wait_done(done, 5.0, stop_event):
            if stop_event is not None and stop_event.is_set():
                return False, "set_pose interrupted (stop)"
            return False, "set_pose timeout"
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        return bool(response.success), "ok"
