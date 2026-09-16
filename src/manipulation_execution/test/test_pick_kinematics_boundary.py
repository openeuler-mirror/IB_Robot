"""Public kinematics binding and the explicitly selected legacy transport."""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState

from ibrobot_msgs.srv import ComputeFk, ComputeIk
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError
from robot_runtime.interface_description import build_description, description_digest
from robot_runtime.model_metadata import joint_conversions_fingerprint


@pytest.fixture
def description():
    return build_description(
        {
            "runtime": {"name": "test_robot", "version": "1.0", "instance_id": "test"},
            "joints": ["elbow"],
            "capabilities": {
                "joint.state": {},
                "motion.ik": {"endpoints": ["/custom/ik", "/custom/ik_one", "/custom/ik_two"]},
                "motion.fk": {"endpoints": ["/custom/fk", "/custom/fk_one", "/custom/fk_two"]},
                "motion.move_to_joint": {"endpoints": ["/custom/move"]},
            },
        },
        simulated=True,
    )


def _harness():
    return SimpleNamespace(
        _config={"base_frame": "root", "ee_frame": "tool"},
        _arm_joint_names=["elbow"],
        _home_joint_positions={},
    )


def _refresh_digest(description):
    description["digest"] = description_digest(description)


def test_public_endpoints_and_workers_use_bound_snapshot(description):
    harness = _harness()
    description["interfaces"] = dict(reversed(list(description["interfaces"].items())))
    ik, fk = PickExecutorNode._bind_kinematics_description(harness, description)
    assert ik == ["/custom/ik_one", "/custom/ik_two"]
    assert fk == ["/custom/fk_one", "/custom/fk_two"]
    assert harness._ik_service == "/custom/ik"
    assert harness._fk_service == "/custom/fk"
    assert harness._move_configuration_service == "/custom/move"
    assert harness._base_frame == "root"
    assert harness._ee_frame == "tool"


def test_public_workers_share_primary_fk_when_no_matching_fk_worker(description):
    del description["interfaces"]["motion.compute_fk.worker_1"]
    _refresh_digest(description)
    _, fk = PickExecutorNode._bind_kinematics_description(_harness(), description)
    assert fk == ["/custom/fk", "/custom/fk_two"]


@pytest.mark.parametrize("field", ["ik_service", "fk_service", "move_configuration_service", "joint_state_topic"])
def test_public_endpoint_conflicts_fail_closed(description, field):
    harness = _harness()
    harness._config[field] = "/stale/private/endpoint"
    with pytest.raises(ValueError, match="differs from the bound public interface"):
        PickExecutorNode._bind_kinematics_description(harness, description)


def test_moveit_service_in_public_snapshot_is_rejected(description):
    description["interfaces"]["motion.compute_ik"]["message_type"] = "moveit_msgs/srv/GetPositionIK"
    _refresh_digest(description)
    with pytest.raises(ValueError, match="ComputeIk"):
        PickExecutorNode._bind_kinematics_description(_harness(), description)


def test_runtime_snapshot_is_required():
    with pytest.raises(ValueError):
        PickExecutorNode._bind_kinematics_description(_harness(), {})


def test_public_model_supplies_arm_home_and_frames(description):
    conversions = {
        "schema_version": 1,
        "joint_names": ["elbow"],
        "quantities": {"elbow": "position"},
        "modes": {
            "none": {},
            "degrees": {"elbow": {"min": -1.0, "max": 1.0, "span": 180.0, "offset": -90.0}},
            "range_m100_100": {"elbow": {"min": -1.0, "max": 1.0, "span": 200.0, "offset": -100.0}},
        },
    }
    description["model"] = {
        "schema_version": 1,
        "authority": "urdf",
        "joint_groups": {"all": ["elbow"], "arm": ["elbow"], "gripper": [], "base": []},
        "joint_limits": {"elbow": {"min": -1.0, "max": 1.0}},
        "home_positions": {"elbow": 0.25},
        "frames": {"base_link": "public_root", "ee_link": "public_tool"},
        "joint_conversions": conversions,
        "joint_conversions_fingerprint": joint_conversions_fingerprint(conversions),
    }
    _refresh_digest(description)
    harness = _harness()
    harness._config = {}
    harness._arm_joint_names = []
    PickExecutorNode._bind_kinematics_description(harness, description)
    assert harness._arm_joint_names == ["elbow"]
    assert harness._home_joint_positions == {"elbow": 0.25}
    assert (harness._base_frame, harness._ee_frame) == ("public_root", "public_tool")
    harness._home_joint_positions = {"elbow": 0.5}
    with pytest.raises(ValueError, match="public home positions"):
        PickExecutorNode._bind_kinematics_description(harness, description)


@pytest.mark.parametrize("backend", ["runtime", "legacy_moveit"])
def test_ik_fk_requests_and_response_transport(backend):
    requests = []
    state = JointState(name=["elbow"], position=[0.2])
    pose = Pose()
    pose.orientation.w = 1.0
    achieved = PoseStamped(pose=pose)
    achieved.header.frame_id = "root"
    if backend == "runtime":
        responses = [
            ComputeIk.Response(success=True, solution=state),
            ComputeFk.Response(success=True, poses=[achieved]),
        ]
    else:
        from moveit_msgs.srv import GetPositionFK, GetPositionIK

        ik_response, fk_response = GetPositionIK.Response(), GetPositionFK.Response()
        ik_response.error_code.val = fk_response.error_code.val = 1
        ik_response.solution.joint_state = state
        fk_response.pose_stamped = [achieved]
        responses = [ik_response, fk_response]
    client = SimpleNamespace(call_async=lambda request: requests.append(request))
    harness = _harness()
    harness._kinematics_backend = backend
    harness._ik_client = harness._fk_client = client
    harness._base_frame, harness._ee_frame = "root", "tool"
    harness._rpc_timeout = 3.0
    harness._kinematics_joint_state = lambda value: PickExecutorNode._kinematics_joint_state(harness, value)
    harness._wait_future = lambda *args, **kwargs: responses.pop(0)
    assert PickExecutorNode._solve_ik(harness, pose, None, 100.0, state) == state
    assert PickExecutorNode._compute_fk(harness, state, None, 100.0) == pose
    if backend == "runtime":
        assert isinstance(requests[0], ComputeIk.Request)
        assert requests[0].target.header.frame_id == "root"
        assert requests[0].seed == state
        assert requests[0].orientation_tolerance == math.pi
        assert requests[0].timeout == 0.2
        assert requests[1].link_names == ["tool"]
    else:
        assert requests[0].ik_request.robot_state.joint_state == state
        assert requests[0].ik_request.ik_link_name == "tool"
        assert requests[1].fk_link_names == ["tool"]


def test_runtime_fk_rejects_wrong_frame():
    harness = _harness()
    harness._kinematics_backend = "runtime"
    harness._base_frame, harness._ee_frame = "root", "tool"
    harness._rpc_timeout = 3.0
    harness._fk_client = SimpleNamespace(call_async=lambda request: None)
    harness._kinematics_joint_state = lambda state: state
    harness._wait_future = lambda *args, **kwargs: ComputeFk.Response(success=True, poses=[PoseStamped()])
    with pytest.raises(PickFlowError, match="base frame"):
        PickExecutorNode._compute_fk(harness, JointState(), None, 100.0)


def test_generic_import_does_not_load_moveit_or_robot_specific_modules():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class RejectRobotImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('moveit_msgs', 'so101_suite', 'manipulation_execution.so101_')):
            raise AssertionError('unexpected generic import: ' + fullname)
sys.meta_path.insert(0, RejectRobotImports())
import manipulation_execution.pick_executor_node
import manipulation_execution.imitate_human_motion_executor
import manipulation_execution.imitate_human_motion_executor_node
import manipulation_execution.placement_executor_node
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("backend", ["runtime", "legacy_moveit"])
def test_node_initializes_without_status_discovery_and_preserves_legacy_workers(description, backend):
    config = {"base_frame": "root", "ee_frame": "tool"}
    if backend == "legacy_moveit":
        config["ik"] = {"worker_count": 2, "worker_namespace_prefix": "/legacy_worker"}
    parameters = {
        "kinematics_backend": backend,
        "interface_description_json": json.dumps(description) if backend == "runtime" else "{}",
        "grasp_execution_json": json.dumps(config),
        "arm_joint_names_json": '["elbow"]',
    }
    rclpy.init()
    node = None
    try:
        node = PickExecutorNode(
            parameter_overrides=[Parameter(name, value=value) for name, value in parameters.items()]
        )
        assert node._ik_worker_count == 2
        if backend == "runtime":
            assert node._grasp_geometry is node._wrist_guard is None
            assert node._ik_client.srv_name == "/custom/ik"
        else:
            assert node._grasp_geometry.__name__ == "manipulation_execution.so101_geometry"
            assert node._wrist_guard.__name__ == "manipulation_execution.so101_kinematics_guard"
            assert node._ik_worker_endpoints == ["/legacy_worker_0/compute_ik", "/legacy_worker_1/compute_ik"]
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
