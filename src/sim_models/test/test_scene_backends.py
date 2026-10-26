"""Platform backends for the pick_banana scene task (D9, issue #195).

Unit tests with fake clients: the shared randomisation (index k → identical
banana placement on both platforms), the atomic MuJoCo ``ResetWorld`` request
content (startup-state keyframe, arm rest joint override with zero velocity,
banana/plate free-joint overrides), default placement, stale-pose rejection,
and bounded readiness (a delayed Gazebo bridge, missing pose streams, the
MuJoCo reset_world service and free_joint_states stream).
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
from mujoco_ros2_control_msgs.srv import ResetWorld

from sim_models.scene_compiler import _quat_from_extrinsic_xyz, get_scene_layout, randomization_placements
from sim_models.tasks.backends import GazeboBackend, MujocoBackend


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class _Request:
    pass


def _set_pose_request():
    return SimpleNamespace(
        entity=SimpleNamespace(name="", type=0, id=0),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


class FakeServiceClient:
    def __init__(self, *, available=True, available_after=None, response=None, request_factory=_Request):
        self.available = available
        self.available_after = available_after
        self.response = response or SimpleNamespace(success=True, message="ok")
        self.request_factory = request_factory
        self.requests = []

    @property
    def srv_type(self):
        return SimpleNamespace(Request=self.request_factory)

    def wait_for_service(self, timeout_sec=None):
        if self.available_after is not None and time.monotonic() >= self.available_after:
            self.available = True
        return self.available

    def service_is_ready(self):
        return self.wait_for_service(0)

    def call_async(self, request):
        self.requests.append(request)
        return FakeFuture(self.response)


class FakeNode:
    """Records clients/subscriptions; scripted per service name."""

    def __init__(self, clients=None):
        self.logger = FakeLogger()
        self.clients = clients or {}
        self.subscriptions = []

    def get_logger(self):
        return self.logger

    def create_client(self, srv_type, name, callback_group=None):
        return self.clients.get(name, FakeServiceClient())

    def create_subscription(self, msg_type, topic, callback, qos, callback_group=None):
        self.subscriptions.append((topic, callback))


def _layout():
    return get_scene_layout("pick_banana")


def make_mujoco_backend(*, reset_client=None):
    node = FakeNode()
    if reset_client is not None:
        reset_client.request_factory = ResetWorld.Request
        node.clients["/mujoco_ros2_control_node/reset_world"] = reset_client
    else:
        node.clients["/mujoco_ros2_control_node/reset_world"] = FakeServiceClient(request_factory=ResetWorld.Request)
    backend = MujocoBackend(node, _layout(), scene="pick_banana", callback_group=None)
    return backend, node


def make_gazebo_backend(*, set_pose_client=None):
    node = FakeNode()
    if set_pose_client is None:
        set_pose_client = FakeServiceClient(request_factory=_set_pose_request)
    else:
        set_pose_client.request_factory = _set_pose_request
    node.clients["/world/demo/set_pose"] = set_pose_client
    backend = GazeboBackend(node, _layout(), world_name="demo", scene="pick_banana", callback_group=None)
    return backend, node


def _pose_message(name, x, y, z):
    return SimpleNamespace(
        transforms=[
            SimpleNamespace(
                child_frame_id=name,
                transform=SimpleNamespace(translation=SimpleNamespace(x=x, y=y, z=z)),
            )
        ]
    )


def _free_joint_message(*entries):
    return SimpleNamespace(
        free_joints=[
            SimpleNamespace(
                name=name,
                pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z))),
            )
            for name, x, y, z in entries
        ]
    )


def _feed_fresh_samples(node):
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/free_joint_states")
    callback(_free_joint_message(("banana", 0.0, 0.0, 0.05), ("plate", 0.22, 0.02, 0.02)))


# ── shared randomisation ──────────────────────────────────────────────────────


def test_index_k_places_the_same_banana_pose_on_both_backends():
    placements = randomization_placements(_layout())
    k = 137
    x, y, z, yaw = placements[k]

    mujoco, _ = make_mujoco_backend()
    ok, _ = mujoco.place(k)
    assert ok
    request = mujoco._reset_world.requests[0]
    assert request.keyframe == "", "startup-state reset; the placement rides in state_overrides"
    overrides = {entry.name: entry for entry in request.state_overrides.free_joints}
    banana = overrides["banana"]
    assert (banana.pose.pose.position.x, banana.pose.pose.position.y, banana.pose.pose.position.z) == (x, y, z)
    quat_wxyz = _quat_from_extrinsic_xyz(0.0, 0.0, yaw)
    assert (
        banana.pose.pose.orientation.w,
        banana.pose.pose.orientation.x,
        banana.pose.pose.orientation.y,
        banana.pose.pose.orientation.z,
    ) == (quat_wxyz)

    gazebo, _ = make_gazebo_backend()
    ok, _ = gazebo.place(k)
    assert ok
    requests = gazebo._set_pose.requests
    banana = next(r for r in requests if r.entity.name == "banana")
    assert (banana.pose.position.x, banana.pose.position.y, banana.pose.position.z) == (x, y, z)
    assert (
        banana.pose.orientation.w,
        banana.pose.orientation.x,
        banana.pose.orientation.y,
        banana.pose.orientation.z,
    ) == (quat_wxyz)


def test_mujoco_place_is_one_atomic_reset_world_with_arm_rest_and_both_objects():
    layout = _layout()
    backend, _ = make_mujoco_backend()

    ok, _ = backend.place(4)

    assert ok
    assert len(backend._reset_world.requests) == 1, "one atomic ResetWorld call places the whole scene"
    request = backend._reset_world.requests[0]
    assert request.keyframe == "", "never a named keyframe: the startup capture has zero arm joints"
    joints = request.state_overrides.joint_states
    rand = layout["randomization"]
    assert list(joints.name) == [str(j) for j in rand["arm_joint_names"]]
    assert list(joints.position) == [float(q) for q in rand["arm_rest_qpos"]]
    assert list(joints.velocity) == [0.0] * len(joints.name), "arm velocity pinned to zero explicitly"
    assert list(joints.effort) == [], "effort is not supported and must stay empty"
    overrides = {entry.name: entry for entry in request.state_overrides.free_joints}
    assert set(overrides) == {"banana", "plate"}
    x, y, z, yaw = randomization_placements(layout)[4]
    banana = overrides["banana"]
    assert (banana.pose.pose.position.x, banana.pose.pose.position.y, banana.pose.pose.position.z) == (x, y, z)
    plate = overrides["plate"]
    plate_default = next(o for o in layout["objects"] if o["name"] == "plate")
    assert (plate.pose.pose.position.x, plate.pose.pose.position.y, plate.pose.pose.position.z) == tuple(
        plate_default["position"]
    )


def test_default_placement_uses_the_layout_poses_on_both_backends():
    layout = _layout()
    banana_default = next(o for o in layout["objects"] if o["name"] == "banana")["position"]
    plate_default = next(o for o in layout["objects"] if o["name"] == "plate")["position"]

    mujoco, _ = make_mujoco_backend()
    ok, _ = mujoco.place(None)
    assert ok
    request = mujoco._reset_world.requests[0]
    assert request.keyframe == ""
    overrides = {entry.name: entry for entry in request.state_overrides.free_joints}
    assert (
        overrides["banana"].pose.pose.position.x,
        overrides["banana"].pose.pose.position.y,
        overrides["banana"].pose.pose.position.z,
    ) == tuple(banana_default)

    gazebo, _ = make_gazebo_backend()
    ok, _ = gazebo.place(None)
    assert ok
    requests = gazebo._set_pose.requests
    banana = next(r for r in requests if r.entity.name == "banana")
    plate = next(r for r in requests if r.entity.name == "plate")
    assert (banana.pose.position.x, banana.pose.position.y, banana.pose.position.z) == tuple(banana_default)
    assert (plate.pose.position.x, plate.pose.position.y, plate.pose.position.z) == tuple(plate_default)


# ── object position freshness ─────────────────────────────────────────────────


def test_gazebo_object_position_rejects_stale_samples():
    gazebo, _ = make_gazebo_backend()
    gazebo._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic() - 2.0)
    assert gazebo.object_position("banana") is None

    gazebo._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic())
    assert gazebo.object_position("banana") is not None


def test_mujoco_object_position_rejects_stale_samples():
    backend, node = make_mujoco_backend()
    _feed_fresh_samples(node)
    backend._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic() - 2.0)
    assert backend.object_position("banana") is None

    backend._poses["banana"] = (np.array([0.05, 0.04, 0.05]), time.monotonic())
    assert backend.object_position("banana") is not None


def test_gazebo_object_position_selects_by_child_frame_id():
    gazebo, node = make_gazebo_backend()
    topic, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.05, 0.04, 0.05))

    assert np.allclose(gazebo.object_position("banana"), [0.05, 0.04, 0.05])
    assert gazebo.object_position("plate") is None


def test_mujoco_object_position_selects_by_free_joint_entry_name():
    backend, node = make_mujoco_backend()
    topic, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/free_joint_states")
    callback(_free_joint_message(("banana", 0.05, 0.04, 0.05)))

    assert np.allclose(backend.object_position("banana"), [0.05, 0.04, 0.05])
    assert backend.object_position("plate") is None


# ── bounded readiness ─────────────────────────────────────────────────────────


def test_gazebo_waits_for_a_delayed_bridge_and_fresh_scene_objects():
    delayed = FakeServiceClient(available=False, available_after=time.monotonic() + 0.3)
    backend, node = make_gazebo_backend(set_pose_client=delayed)
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    ok, message = backend.wait_ready(3.0)

    assert ok, message
    assert "set_pose bridge" in message
    assert delayed.wait_for_service(), "the wait covered the bridge coming up"


def test_gazebo_not_ready_without_the_pose_stream():
    backend, _ = make_gazebo_backend()

    ok, message = backend.wait_ready(0.4)

    assert not ok and "object_poses" in message


def test_gazebo_check_ready_requires_fresh_samples_of_every_placed_object():
    backend, node = make_gazebo_backend()
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))

    ok, message = backend.check_ready()

    assert not ok and "plate" in message, "a fresh banana alone is not readiness"


def test_gazebo_check_ready_requires_the_bridge():
    backend, node = make_gazebo_backend(set_pose_client=FakeServiceClient(available=False))
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    ok, message = backend.check_ready()

    assert not ok and "set_pose bridge not available" in message


def test_gazebo_place_without_the_bridge_fails_without_a_fallback():
    backend, _ = make_gazebo_backend(set_pose_client=FakeServiceClient(available=False))

    ok, message = backend.place(0)

    assert not ok and "set_pose bridge not available" in message
    assert not backend._set_pose.requests, "nothing is sent when the bridge is down"


def test_mujoco_readiness_needs_the_service_and_fresh_object_samples():
    backend, node = make_mujoco_backend()
    ok, message = backend.check_ready()
    assert not ok and "free_joint_states" in message, "no pose stream is not readiness"

    _feed_fresh_samples(node)
    ok, message = backend.check_ready()
    assert ok, message

    missing_service, node2 = make_mujoco_backend(reset_client=FakeServiceClient(available=False))
    _feed_fresh_samples(node2)
    ok, message = missing_service.check_ready()
    assert not ok and "reset_world" in message


def test_mujoco_check_ready_requires_fresh_samples_of_every_placed_object():
    backend, node = make_mujoco_backend()
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/free_joint_states")
    callback(_free_joint_message(("banana", 0.0, 0.0, 0.05)))

    ok, message = backend.check_ready()

    assert not ok and "plate" in message, "a fresh banana alone is not readiness"


def test_mujoco_readiness_is_revalidated_not_latched():
    backend, node = make_mujoco_backend()
    _feed_fresh_samples(node)
    assert backend.check_ready()[0]

    backend._reset_world.available = False  # the service dropped between calls
    ok, message = backend.check_ready()
    assert not ok and "reset_world" in message

    backend._reset_world.available = True
    backend._poses["banana"] = (np.array([0.0, 0.0, 0.05]), time.monotonic() - 2.0)  # the stream went stale
    ok, message = backend.check_ready()
    assert not ok and "banana" in message


def test_mujoco_not_ready_without_the_pose_stream():
    backend, _ = make_mujoco_backend()

    ok, message = backend.wait_ready(0.4)

    assert not ok and "free_joint_states" in message


def test_mujoco_place_without_the_service_fails_without_a_fallback():
    backend, _ = make_mujoco_backend(reset_client=FakeServiceClient(available=False))

    ok, message = backend.place(0)

    assert not ok and "reset_world service not available" in message
    assert not backend._reset_world.requests, "nothing is sent when the service is down"


# ── startup readiness thread ──────────────────────────────────────────────────


def test_backend_start_readiness_logs_the_outcome():
    backend, node = make_gazebo_backend()
    _, callback = next(sub for sub in node.subscriptions if sub[0] == "/sim/pick_banana/object_poses")
    callback(_pose_message("banana", 0.0, 0.0, 0.05))
    callback(_pose_message("plate", 0.22, 0.02, 0.02))

    backend.start_readiness(timeout_s=2.0)
    deadline = time.monotonic() + 3.0
    while not any("ready" in text for _level, text in node.logger.messages) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert any("GazeboBackend: ready" in text for _level, text in node.logger.messages)


def test_concurrent_place_requests_stay_thread_safe():
    gazebo, _ = make_gazebo_backend()
    errors = []

    def worker(index):
        try:
            gazebo.place(index % gazebo.grid_size)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(gazebo._set_pose.requests) == 16, "8 workers × (banana + plate)"

    mujoco, _ = make_mujoco_backend()
    mujoco_errors = []

    def mujoco_worker(index):
        try:
            mujoco.place(index % mujoco.grid_size)
        except Exception as exc:  # noqa: BLE001
            mujoco_errors.append(exc)

    threads = [threading.Thread(target=mujoco_worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not mujoco_errors
    assert len(mujoco._reset_world.requests) == 8, "8 workers × one atomic ResetWorld each"
