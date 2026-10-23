#!/usr/bin/env python3
"""Offline tests for the radian-native SO-101 Placo wrapper.

These tests are **offline** (no ROS runtime, no real robot) and validate
that :class:`SO101PlacoKinematics` is correctly adapted to IB-Robot
conventions, using the IB-Robot URDF (NOT lerobot's). They confirm the
items most likely to silently break a Placo integration:
radian units, joint order (``1..5``), target frame (``gripper``) and the
xacro→urdf expansion.

Because the wrapper uses the *IB-Robot* URDF while lerobot's
``RobotKinematics`` ships its own URDF/SRDF, a numeric joint-by-joint parity
against lerobot is **not** meaningful (different models → different IK). So
instead of cross-library numeric parity, these tests validate:

1. **FK→IK→FK round-trip self-consistency** on the IB-Robot URDF — the
   authoritative correctness check under our constraint of using our own URDF.
2. **Solving-posture alignment with lerobot** — structural, not numeric:
   same Placo calls (``RobotWrapper`` / ``KinematicsSolver`` / ``mask_fbase`` /
   ``add_frame_task`` / ``set_joint`` seed / ``configure(soft, w_p, w_o)`` /
   ``solve``) and same default weights (1.0 / 0.01).

Run (workspace sourced so ``so101_description`` resolves):

    source .shrc_local
    python3 -m pytest src/robots/so101/so101_motion/test/test_so101_placo_kinematics.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

# Make the wrapper importable both from a sourced install and directly from
# the source tree (offline run before colcon install).
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Under colcon test (CMakeLists sets IBROBOT_REQUIRE_PLACO=1) a missing placo,
# import error or xacro failure must FAIL: a module-level skip would report the
# whole suite as green while running nothing.
_REQUIRE_PLACO = os.environ.get("IBROBOT_REQUIRE_PLACO") == "1"

if _REQUIRE_PLACO:
    import placo  # noqa: F401
else:
    placo = pytest.importorskip("placo", reason="placo not installed (kinematics extra)")

try:
    from so101_placo_kinematics import (
        DEFAULT_ARM_JOINT_NAMES,
        DEFAULT_ORIENTATION_WEIGHT,
        DEFAULT_POSITION_WEIGHT,
        DEFAULT_TARGET_FRAME,
        SO101PlacoDiffIK,
        SO101PlacoKinematics,
        expand_so101_xacro,
    )
except ImportError as exc:  # pragma: no cover - import diagnostics
    if _REQUIRE_PLACO:
        raise
    pytest.skip(f"cannot import so101_placo_kinematics: {exc}", allow_module_level=True)


# IK convergence settings. Placo's KinematicsSolver is a *differential* solver:
# a single solve() takes one step toward the target, so reaching a far target
# from a cold seed needs several iterations. (In live teleop the target only
# advances incrementally per tick, so one solve per tick is enough.)
_IK_ITERS = 80
_POS_TOL_M = 1e-4
_ORI_TOL = 1e-3
_JOINT_TOL_RAD = 1e-3


# SO-101 joint limits (rad) from the URDF, used to sample reachable configs.
_JOINT_LIMITS = {
    "1": (-2.0693, 2.0709),
    "2": (-1.92, 1.92),
    "3": (-1.6813, 1.6828),
    "4": (-1.65806, 1.65806),
    "5": (-2.9115, 2.9115),
}


@pytest.fixture(scope="module")
def urdf_xml() -> str:
    """Expand the IB-Robot SO-101 xacro once for the whole module."""
    try:
        return expand_so101_xacro()
    except Exception as exc:  # noqa: BLE001
        if _REQUIRE_PLACO:
            raise
        pytest.skip(f"xacro expansion needs a built/sourced so101_description: {exc}")


@pytest.fixture(scope="module")
def kin(urdf_xml: str) -> SO101PlacoKinematics:
    return SO101PlacoKinematics(urdf_xml=urdf_xml)


@pytest.fixture(scope="module")
def kin_tcp(urdf_xml: str) -> SO101PlacoKinematics:
    """Kinematics wrapper targeting the virtual TCP frame instead of gripper."""
    return SO101PlacoKinematics(urdf_xml=urdf_xml, target_frame="tcp")


def _converge_ik(kin: SO101PlacoKinematics, q_seed, target_pose, iters: int = _IK_ITERS, orientation_weight=None):
    q = np.asarray(q_seed, dtype=np.float64)
    for _ in range(iters):
        if orientation_weight is None:
            q = kin.inverse_kinematics(q, target_pose)
        else:
            q = kin.inverse_kinematics(q, target_pose, 1.0, orientation_weight)
    return q


# Representative arm configurations (rad), ordered [1,2,3,4,5]:
# neutral, typical, near elbow-straight singular neighbourhood, mixed.
_TEST_CONFIGS = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.1, -0.3, 0.4, 0.2, -0.1]),
    np.array([0.5, -0.8, 0.9, -0.4, 0.6]),
    np.array([-0.7, 0.5, -0.6, 0.8, -0.9]),
    np.array([0.3, 1.2, -1.0, 0.5, 1.5]),
]


# --------------------------------------------------------------------------
# 1. Conventions: units, joint order, target frame, xacro expansion
# --------------------------------------------------------------------------


def test_xacro_expands_to_plain_urdf(urdf_xml: str) -> None:
    assert "<robot" in urdf_xml
    # No unexpanded xacro constructs should remain. (The autogen header comment
    # legitimately mentions the source .xacro filename, so we check for actual
    # xacro tags/macros, not the substring "xacro".)
    assert "xacro:" not in urdf_xml, "xacro: tags/macros must be fully expanded"
    assert "${" not in urdf_xml, "xacro ${...} substitutions must be resolved"
    assert "$(find" not in urdf_xml, "xacro $(find ...) must be resolved"
    assert 'name="gripper"' in urdf_xml, "target frame link 'gripper' must exist"


def test_arm_joint_order_and_target_frame(kin: SO101PlacoKinematics) -> None:
    assert kin.arm_joint_names == list(DEFAULT_ARM_JOINT_NAMES) == ["1", "2", "3", "4", "5"]
    assert kin.target_frame == DEFAULT_TARGET_FRAME == "gripper"
    # The gripper joint "6" must NOT be part of the Cartesian IK joint set.
    assert "6" not in kin.arm_joint_names


def test_wrapper_matches_lerobot_solving_posture(kin: SO101PlacoKinematics) -> None:
    """Structural alignment with lerobot RobotKinematics (not numeric)."""
    # Same Placo objects/calls as lerobot kinematics.py.
    assert isinstance(kin.robot, placo.RobotWrapper)
    assert isinstance(kin.solver, placo.KinematicsSolver)
    assert kin.tip_task is not None  # add_frame_task result
    # Same default weights as lerobot ("position hard, orientation best-effort").
    assert DEFAULT_POSITION_WEIGHT == 1.0
    assert DEFAULT_ORIENTATION_WEIGHT == 0.01


# --------------------------------------------------------------------------
# 2. FK basic sanity (radian-native)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", _TEST_CONFIGS)
def test_fk_returns_valid_transform(kin: SO101PlacoKinematics, q) -> None:
    T = kin.forward_kinematics(q)
    assert T.shape == (4, 4)
    # Bottom row of a homogeneous transform.
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-9)
    # Rotation block is orthonormal.
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_fk_is_deterministic(kin: SO101PlacoKinematics) -> None:
    q = _TEST_CONFIGS[1]
    np.testing.assert_allclose(kin.forward_kinematics(q), kin.forward_kinematics(q), atol=1e-12)


# --------------------------------------------------------------------------
# 3. FK→IK→FK round-trip self-consistency (authoritative correctness check)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q_true", _TEST_CONFIGS)
def test_roundtrip_pose_recovered(kin: SO101PlacoKinematics, q_true) -> None:
    """FK(q)->pose; IK(pose) seeded near q; FK(IK) must recover the pose."""
    target = kin.forward_kinematics(q_true)
    # Seed slightly perturbed from the truth (closed-loop re-seed scenario).
    q_seed = q_true + 0.05
    q_sol = _converge_ik(kin, q_seed, target)
    achieved = kin.forward_kinematics(q_sol)

    pos_err = np.linalg.norm(target[:3, 3] - achieved[:3, 3])
    ori_err = np.linalg.norm(target[:3, :3] - achieved[:3, :3])
    assert pos_err < _POS_TOL_M, f"position round-trip {pos_err:.2e} m exceeds {_POS_TOL_M}"
    assert ori_err < _ORI_TOL, f"orientation round-trip {ori_err:.2e} exceeds {_ORI_TOL}"


def test_roundtrip_from_cold_zero_seed(kin: SO101PlacoKinematics) -> None:
    """Even from a cold zero seed the differential solver reaches the target."""
    q_true = _TEST_CONFIGS[1]
    target = kin.forward_kinematics(q_true)
    q_sol = _converge_ik(kin, np.zeros(5), target)
    achieved = kin.forward_kinematics(q_sol)
    pos_err = np.linalg.norm(target[:3, 3] - achieved[:3, 3])
    assert pos_err < _POS_TOL_M, f"cold-seed position round-trip {pos_err:.2e} m exceeds {_POS_TOL_M}"


def test_ik_recovers_seed_when_target_is_seed_pose(kin: SO101PlacoKinematics) -> None:
    """IK at the seed's own pose must not move the joints (closed-loop hold)."""
    q = _TEST_CONFIGS[1]
    target = kin.forward_kinematics(q)
    q_sol = kin.inverse_kinematics(q, target)
    np.testing.assert_allclose(q_sol, q, atol=_JOINT_TOL_RAD)


# --------------------------------------------------------------------------
# 4. Radian-native guard: confirm we are NOT in degrees
# --------------------------------------------------------------------------


def test_units_are_radians_not_degrees(kin: SO101PlacoKinematics) -> None:
    """A 1.0 input must move the arm like 1 rad (~57 deg), not 1 deg.

    FK at q=1 rad on joint 1 must differ substantially from FK at q=0; if the
    wrapper silently treated inputs as degrees, the pose delta would be ~57x
    smaller.
    """
    T0 = kin.forward_kinematics(np.zeros(5))
    T1 = kin.forward_kinematics(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    delta = np.linalg.norm(T0[:3, 3] - T1[:3, 3])
    # 1 rad of the base joint sweeps the EE by a clearly measurable arc.
    assert delta > 0.01, f"1.0 on joint 1 barely moved EE ({delta:.4e} m) — units may be degrees"


# --------------------------------------------------------------------------
# 5. Virtual TCP frame: drop-in equivalence + correct offset
# --------------------------------------------------------------------------

# TCP is a fixed link 9.5 cm down the gripper -Z finger line plus a small
# -X re-center onto the jaw mid-line, orientation identical to gripper
# (see so101-tcp-tool-frame-proposal.md).
_TCP_OFFSET_GRIPPER = np.array([-0.005, 0.0, -0.095])


def test_tcp_link_exists_in_urdf(urdf_xml: str) -> None:
    assert 'name="tcp"' in urdf_xml, "virtual TCP link must exist in the URDF"
    assert "gripper_to_tcp" in urdf_xml, "fixed gripper->tcp joint must exist"


def test_tcp_target_frame_selected(kin_tcp: SO101PlacoKinematics) -> None:
    assert kin_tcp.target_frame == "tcp"
    # Same 5-DOF arm joint set; the fixed joint adds no DOF.
    assert kin_tcp.arm_joint_names == ["1", "2", "3", "4", "5"]


@pytest.mark.parametrize("q", _TEST_CONFIGS)
def test_tcp_is_pure_translation_of_gripper(kin: SO101PlacoKinematics, kin_tcp: SO101PlacoKinematics, q) -> None:
    """TCP pose == gripper pose with a fixed -Y 9cm offset, same orientation.

    This is the formal "no relative rotation, pure translation" guarantee the
    TCP design depends on: rotations commanded at the TCP pivot about the tool
    point, while orientation tracks the gripper exactly.
    """
    T_grip = kin.forward_kinematics(q)
    T_tcp = kin_tcp.forward_kinematics(q)

    # Orientation must be identical (no relative rotation).
    np.testing.assert_allclose(T_tcp[:3, :3], T_grip[:3, :3], atol=1e-9)

    # TCP origin = gripper origin shifted by the offset expressed in the
    # gripper frame (R_grip @ offset).
    expected = T_grip[:3, 3] + T_grip[:3, :3] @ _TCP_OFFSET_GRIPPER
    np.testing.assert_allclose(T_tcp[:3, 3], expected, atol=1e-6)


def test_tcp_roundtrip_pose_recovered(kin_tcp: SO101PlacoKinematics) -> None:
    """FK/IK round-trip self-consistency holds for the TCP tip frame too."""
    q_true = _TEST_CONFIGS[1]
    target = kin_tcp.forward_kinematics(q_true)
    q_sol = _converge_ik(kin_tcp, q_true + 0.05, target)
    achieved = kin_tcp.forward_kinematics(q_sol)
    pos_err = np.linalg.norm(target[:3, 3] - achieved[:3, 3])
    ori_err = np.linalg.norm(target[:3, :3] - achieved[:3, :3])
    assert pos_err < _POS_TOL_M, f"tcp position round-trip {pos_err:.2e} m exceeds {_POS_TOL_M}"
    assert ori_err < _ORI_TOL, f"tcp orientation round-trip {ori_err:.2e} exceeds {_ORI_TOL}"


# --------------------------------------------------------------------------
# 6. Position-only solve: orientation fully free (orientation_weight == 0)
# --------------------------------------------------------------------------


def test_position_only_tracks_position_and_frees_orientation(
    kin: SO101PlacoKinematics,
) -> None:
    """With orientation_weight=0 the QP nails position and ignores orientation.

    This is the user's first evaluation mode ("solve position only, like
    lerobot, don't care about orientation"). The position target must be hit
    to sub-millimetre accuracy while the achieved orientation is allowed to
    differ from the seed pose's orientation — proving orientation is truly
    free, not just lightly weighted.
    """
    q_seed = _TEST_CONFIGS[1]
    T_seed = kin.forward_kinematics(q_seed)

    # Drive a pure position target a few cm away, keeping the seed orientation
    # in the target frame (which the solver is free to ignore).
    target = T_seed.copy()
    target[:3, 3] = target[:3, 3] + np.array([0.0, 0.0, 0.06])

    # Position-only converged solve (the servo node drives 4 mm/tick, here we
    # converge to assess the steady-state position-only solution quality).
    q_sol = _converge_ik(kin, q_seed, target, orientation_weight=0.0)
    achieved = kin.forward_kinematics(q_sol)

    pos_err = np.linalg.norm(target[:3, 3] - achieved[:3, 3])
    assert pos_err < _POS_TOL_M, f"position-only solve missed position by {pos_err:.2e} m"


def test_position_only_beats_weighted_orientation_on_position(
    kin: SO101PlacoKinematics,
) -> None:
    """Position-only solve tracks position at least as well as ori_w=0.01.

    For the under-actuated 5-DOF arm, spending DOF on orientation steals them
    from position. Freeing orientation entirely must not make position worse;
    in the reachable interior it tracks position to <= the weighted case.
    """
    q_seed = _TEST_CONFIGS[2]
    T_seed = kin.forward_kinematics(q_seed)
    target = T_seed.copy()
    target[:3, 3] = target[:3, 3] + np.array([0.04, 0.0, 0.0])

    q_pos = _converge_ik(kin, q_seed, target, orientation_weight=0.0)
    q_wgt = _converge_ik(kin, q_seed, target, orientation_weight=0.01)

    err_pos = np.linalg.norm(target[:3, 3] - kin.forward_kinematics(q_pos)[:3, 3])
    err_wgt = np.linalg.norm(target[:3, 3] - kin.forward_kinematics(q_wgt)[:3, 3])

    assert err_pos <= err_wgt + 1e-6, (
        f"position-only ({err_pos:.2e} m) should track position no worse than weighted-orientation ({err_wgt:.2e} m)"
    )


# --------------------------------------------------------------------------
# Differential-IK (Jacobian + QP velocity-level servo)
# --------------------------------------------------------------------------
#
# These validate the NEW servo core (SO101PlacoDiffIK): a velocity step per tick
# that solves Δq with a PositionTask + hard joint/velocity limits, instead of
# re-solving an absolute pose (which branch-flips at the reachable edge). The
# regression target is the previously branch-flipping directions (gripper -y,
# tcp +y): they must now be smooth, and the arm must yield (not snap) at the
# reachable boundary for both control points.

_DIFFIK_STEP_M = 0.004  # 4 mm/tick (config-equivalent)
_DIFFIK_DT = 0.02  # 50 Hz
_DIFFIK_DRIVE_M = 0.18  # 180 mm drive distance
_DIFFIK_JUMP_RAD = 0.3  # per-tick maxdq above this == branch flip
_DIFFIK_HOME = np.array([0.0, -0.3, 0.3, 1.0, 0.0])

# Axis directions to sweep for both control points.
_DIFFIK_AXES = [
    ("+x", np.array([1.0, 0.0, 0.0])),
    ("-x", np.array([-1.0, 0.0, 0.0])),
    ("+y", np.array([0.0, 1.0, 0.0])),
    ("-y", np.array([0.0, -1.0, 0.0])),
    ("+z", np.array([0.0, 0.0, 1.0])),
    ("-z", np.array([0.0, 0.0, -1.0])),
]


@pytest.fixture(scope="module")
def diffik(urdf_xml: str) -> SO101PlacoDiffIK:
    return SO101PlacoDiffIK(urdf_xml=urdf_xml, target_frame="gripper")


@pytest.fixture(scope="module")
def diffik_tcp(urdf_xml: str) -> SO101PlacoDiffIK:
    return SO101PlacoDiffIK(urdf_xml=urdf_xml, target_frame="tcp")


def _drive_diffik(solver: SO101PlacoDiffIK, axis: np.ndarray):
    """Drive the diff-IK solver along ``axis`` from home; return worst maxdq."""
    q = _DIFFIK_HOME.copy()
    v = axis * (_DIFFIK_STEP_M / _DIFFIK_DT)  # m/s so v*dt == step
    worst = 0.0
    jumps = 0
    for _ in range(int(_DIFFIK_DRIVE_M / _DIFFIK_STEP_M)):
        qn = solver.step(q, v, _DIFFIK_DT)
        mdq = float(np.max(np.abs(qn - q)))
        worst = max(worst, mdq)
        jumps += int(mdq > _DIFFIK_JUMP_RAD)
        q = qn
    return worst, jumps, q


def test_diffik_holds_still_on_zero_velocity(diffik: SO101PlacoDiffIK) -> None:
    """Zero commanded velocity must not move the joints (closed-loop hold)."""
    q = _DIFFIK_HOME.copy()
    qn = diffik.step(q, np.zeros(3), _DIFFIK_DT)
    np.testing.assert_allclose(qn, q, atol=1e-4)


@pytest.mark.parametrize("name,axis", _DIFFIK_AXES)
def test_diffik_gripper_no_branch_flip(diffik: SO101PlacoDiffIK, name, axis) -> None:
    """gripper: every axis (incl. the old 1.73 rad -y flip) stays smooth."""
    worst, jumps, _ = _drive_diffik(diffik, axis)
    assert jumps == 0, f"gripper {name}: {jumps} branch flips (worst maxdq {worst:.2f} rad)"


@pytest.mark.parametrize("name,axis", _DIFFIK_AXES)
def test_diffik_tcp_no_branch_flip(diffik_tcp: SO101PlacoDiffIK, name, axis) -> None:
    """tcp: every axis (incl. the old 5.65 rad +y flip) stays smooth.

    Proves the fixed 95 mm gripper->tcp lever arm is handled structurally by
    the frame Jacobian — no tcp special case in the solver.
    """
    worst, jumps, _ = _drive_diffik(diffik_tcp, axis)
    assert jumps == 0, f"tcp {name}: {jumps} branch flips (worst maxdq {worst:.2f} rad)"


def test_diffik_velocity_limit_is_respected(urdf_xml: str) -> None:
    """Per-tick Δq must never exceed max_joint_speed * dt (QP hard limit).

    Even commanding an absurd velocity, the in-QP velocity limit caps the step,
    so the unreachable component is projected away rather than snapping.
    """
    max_speed = 2.0
    solver = SO101PlacoDiffIK(urdf_xml=urdf_xml, target_frame="gripper", max_joint_speed=max_speed)
    q = _DIFFIK_HOME.copy()
    huge_v = np.array([100.0, 0.0, 0.0])  # 100 m/s — far beyond reach
    ceiling = max_speed * _DIFFIK_DT
    for _ in range(20):
        qn = solver.step(q, huge_v, _DIFFIK_DT)
        mdq = float(np.max(np.abs(qn - q)))
        # Small numeric tolerance over the hard ceiling.
        assert mdq <= ceiling + 1e-6, f"per-tick Δq {mdq:.4f} exceeded ceiling {ceiling:.4f}"
        q = qn


def test_diffik_yields_at_reachable_edge(diffik: SO101PlacoDiffIK) -> None:
    """Driving toward an unreachable point must yield (stay finite, no snap).

    Push -y far past the reachable edge; the EE must advance then stall
    smoothly, never producing a multi-radian jump or non-finite joints.
    """
    q = _DIFFIK_HOME.copy()
    v = np.array([0.0, -1.0, 0.0]) * (_DIFFIK_STEP_M / _DIFFIK_DT)
    worst = 0.0
    for _ in range(80):  # drive 320 mm -y, well past reach
        qn = diffik.step(q, v, _DIFFIK_DT)
        assert np.all(np.isfinite(qn))
        worst = max(worst, float(np.max(np.abs(qn - q))))
        q = qn
    assert worst < _DIFFIK_JUMP_RAD, f"edge-yield produced a {worst:.2f} rad jump"


def test_diffik_follows_reachable_direction(diffik: SO101PlacoDiffIK) -> None:
    """A reachable +z command must actually move the EE +z (direction correct)."""
    q = _DIFFIK_HOME.copy()
    p0 = diffik.forward_kinematics(q)[:3, 3]
    v = np.array([0.0, 0.0, 1.0]) * (_DIFFIK_STEP_M / _DIFFIK_DT)
    for _ in range(10):
        q = diffik.step(q, v, _DIFFIK_DT)
    p1 = diffik.forward_kinematics(q)[:3, 3]
    dz = p1[2] - p0[2]
    assert dz > 0.5 * 10 * _DIFFIK_STEP_M, f"+z command only moved EE {dz * 1000:.1f} mm in z"


# --- absolute-reference path (solve_to_position) — the gravity-ratchet fix ---
# The node holds a command-side reference position and asks the QP to reach it
# (not "v*dt from the measured pose"). These tests pin the behaviour that makes
# the hardware sag-hold work: a FIXED reference must pull a drifting (sagging)
# measured pose back to the held Cartesian point instead of welding in the sag.


def test_ee_position_matches_fk(diffik: SO101PlacoDiffIK) -> None:
    """ee_position must equal the FK translation for the same joints."""
    q = _DIFFIK_HOME.copy()
    np.testing.assert_allclose(diffik.ee_position(q), diffik.forward_kinematics(q)[:3, 3], atol=1e-9)


def test_solve_to_position_holds_fixed_reference(diffik: SO101PlacoDiffIK) -> None:
    """A fixed reference + exact seed must not move the joints (clean hold)."""
    q = _DIFFIK_HOME.copy()
    p_ref = diffik.ee_position(q)
    qn = diffik.solve_to_position(q, p_ref, _DIFFIK_DT)
    np.testing.assert_allclose(qn, q, atol=1e-4)


def test_solve_to_position_corrects_gravity_sag(diffik: SO101PlacoDiffIK) -> None:
    """The ratchet fix: a FIXED reference must cancel a per-tick measured sag.

    Simulates gravity by injecting a small sag into the measured seed every
    tick (the real arm's joints drift away from the last command). With the OLD
    step(v=0) path the command follows the sag down (ratchet); with the absolute
    reference the QP holds the EE at p_ref, so the position error stays bounded
    and small instead of growing every tick.
    """
    q = _DIFFIK_HOME.copy()
    p_ref = diffik.ee_position(q)
    sag = 0.01  # rad per tick injected into J2/J3 (gravity proxy)
    errs = []
    for _ in range(30):
        q_meas = q.copy()
        q_meas[1] += sag
        q_meas[2] += sag
        q = diffik.solve_to_position(q_meas, p_ref, _DIFFIK_DT)
        errs.append(float(np.linalg.norm(diffik.ee_position(q) - p_ref)))
    # Bounded (no ratchet): the held-position error must stay sub-millimetre and
    # must NOT grow monotonically the way the old step(v=0) sag did.
    assert max(errs) < 2e-3, f"hold error grew to {max(errs) * 1000:.2f} mm (ratchet not cancelled)"
    assert errs[-1] <= errs[4] + 1e-4, "hold error is still growing (reference not holding)"


def test_solve_to_position_follows_moving_reference(diffik: SO101PlacoDiffIK) -> None:
    """Advancing the reference by v*dt must move the EE along, even with sag."""
    q = _DIFFIK_HOME.copy()
    p_ref = diffik.ee_position(q)
    v = np.array([0.0, 0.0, 1.0]) * (_DIFFIK_STEP_M / _DIFFIK_DT)
    sag = 0.005
    z0 = p_ref[2]
    for _ in range(10):
        p_ref = p_ref + v * _DIFFIK_DT
        q_meas = q.copy()
        q_meas[1] += sag
        q = diffik.solve_to_position(q_meas, p_ref, _DIFFIK_DT)
    reached = diffik.ee_position(q)
    assert reached[2] > z0 + 0.5 * 10 * _DIFFIK_STEP_M, "moving reference did not carry the EE +z"
    # The EE must track the reference it was given (sub-mm), proving the sag did
    # not detach the command from the reference.
    assert np.linalg.norm(reached - p_ref) < 2e-3


# --- #156: an unreachable absolute reference must park, not oscillate ---------
# A VR tracking jump put the absolute reference ~0.66 m outside the workspace.
# Fed straight to the QP, the edge equilibrium loses stability (period-2 flip):
# every tick hit the joint speed ceiling and alternated between two joint sets
# until the trigger was released. These tests drive the solver exactly like
# so101_placo_servo_node: seeded from the previous command, fixed reference,
# post-QP clamp to the YAML joint limits.

_SERVO_YAML = os.path.join(os.path.dirname(_SCRIPTS_DIR), "config", "so101_placo_servo.yaml")
_EP10_DELTA = np.array([-0.056, -0.134, 0.645])  # issue #156 ep10 reference offset (m)
_BENT_Q0 = np.radians([15.0, 12.0, -72.0, 5.0, -15.0])
_BENT_DELTA = np.array([0.0, -0.6, 0.3])
_FLIP_FLOOR_DEG = 0.2  # reversals of smaller steps are numerical nullspace drift
_TAIL_TICKS = 50  # last 1 s at 50 Hz


def _servo_cfg() -> dict:
    with open(_SERVO_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _yaml_limits(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.array(cfg["joint_limits_lower"], dtype=np.float64), np.array(cfg["joint_limits_upper"], dtype=np.float64)


def _deployed_diffik(urdf_xml: str, frame: str, ori: float, **overrides) -> SO101PlacoDiffIK:
    """Solver with the deployed YAML gains; the reference-step bound is the class default."""
    cfg = _servo_cfg()
    kwargs = {
        "damping": cfg["diffik_damping"],
        "control_period": _DIFFIK_DT,
        "max_joint_speed": cfg["max_joint_speed"],
        "orientation_weight": ori,
    }
    kwargs.update(overrides)
    return SO101PlacoDiffIK(urdf_xml=urdf_xml, target_frame=frame, **kwargs)


def _servo_loop(solver, q0, delta, ticks, rotvec=None):
    """Run the node's hot path; return published joints and raw (pre-clamp) QP steps."""
    lo, hi = _yaml_limits(_servo_cfg())
    q = np.asarray(q0, dtype=np.float64).copy()
    p_ref = solver.ee_position(q) + np.asarray(delta, dtype=np.float64)
    r_ref = solver.ee_rotation(q)
    if rotvec is not None:
        r_ref = Rotation.from_rotvec(rotvec).as_matrix() @ r_ref
    position_only = solver.orientation_weight == 0.0
    published, raw_steps = [q.copy()], []
    for _ in range(ticks):
        if position_only:
            q_raw = solver.solve_to_position(q, p_ref, _DIFFIK_DT)
        else:
            q_raw = solver.solve_to_pose(q, p_ref, r_ref, _DIFFIK_DT)
        raw_steps.append(q_raw - q)
        q = np.clip(q_raw, lo, hi)
        published.append(q.copy())
    return np.array(published), np.array(raw_steps), p_ref, r_ref


def _tail_flips(published: np.ndarray) -> tuple[int, float]:
    """Count significant same-joint direction reversals in the last second."""
    steps = np.diff(published[-(_TAIL_TICKS + 1) :], axis=0)
    big = np.abs(steps) > np.radians(_FLIP_FLOOR_DEG)
    flips = (np.sign(steps[1:]) != np.sign(steps[:-1])) & big[1:] & big[:-1]
    return int(flips.sum()), float(np.degrees(np.abs(steps).max()))


_UNREACHABLE_CASES = [
    pytest.param("gripper", 0.01, np.zeros(5), _EP10_DELTA, id="gripper-pose-ep10"),
    pytest.param("gripper", 0.0, np.zeros(5), _EP10_DELTA, id="gripper-position-ep10"),
    pytest.param("tcp", 0.01, np.zeros(5), _EP10_DELTA, id="tcp-pose-ep10"),
    pytest.param("tcp", 0.0, np.zeros(5), _EP10_DELTA, id="tcp-position-ep10"),
    pytest.param("gripper", 0.01, _BENT_Q0, _BENT_DELTA, id="gripper-pose-bent"),
    pytest.param("tcp", 0.0, _BENT_Q0, _BENT_DELTA, id="tcp-position-bent"),
]


@pytest.mark.parametrize("frame,ori,q0,delta", _UNREACHABLE_CASES)
def test_unreachable_reference_parks_without_oscillation(urdf_xml: str, frame, ori, q0, delta) -> None:
    """The arm must come to rest at the reachable edge instead of flipping every tick."""
    solver = _deployed_diffik(urdf_xml, frame, ori)
    published, _, _, _ = _servo_loop(solver, q0, delta, ticks=300)
    flips, tail_max = _tail_flips(published)
    assert flips == 0, f"{flips} direction reversals > {_FLIP_FLOOR_DEG} deg in the last second"
    assert tail_max < _FLIP_FLOOR_DEG, f"arm still moving {tail_max:.2f} deg/tick after 6 s"


@pytest.mark.parametrize("frame,ori,q0,delta", _UNREACHABLE_CASES)
def test_unreachable_reference_does_not_saturate_speed_limit(urdf_xml: str, frame, ori, q0, delta) -> None:
    """Neither the raw QP step nor the published step may sit at the speed ceiling."""
    solver = _deployed_diffik(urdf_xml, frame, ori)
    published, raw_steps, _, _ = _servo_loop(solver, q0, delta, ticks=300)
    ceiling = 0.95 * _servo_cfg()["max_joint_speed"] * _DIFFIK_DT
    raw_tail = float(np.abs(raw_steps[-_TAIL_TICKS:]).max())
    pub_tail = float(np.abs(np.diff(published[-(_TAIL_TICKS + 1) :], axis=0)).max())
    assert raw_tail < ceiling, f"raw QP step {raw_tail:.4f} rad at the speed ceiling"
    assert pub_tail < ceiling, f"published step {pub_tail:.4f} rad at the speed ceiling"


def test_unreachable_reference_parks_at_stable_edge(urdf_xml: str) -> None:
    """Parking must match the stable solution of the same objective (heavy damping, no flip)."""
    published, _, p_ref, _ = _servo_loop(_deployed_diffik(urdf_xml, "gripper", 0.01), np.zeros(5), _EP10_DELTA, 600)
    reference_solver = _deployed_diffik(urdf_xml, "gripper", 0.01, damping=0.1)
    stable, _, _, _ = _servo_loop(reference_solver, np.zeros(5), _EP10_DELTA, 600)
    err = np.linalg.norm(reference_solver.ee_position(published[-1]) - p_ref)
    err_stable = np.linalg.norm(reference_solver.ee_position(stable[-1]) - p_ref)
    assert _tail_flips(published)[0] == 0
    assert err <= err_stable + 2e-3, f"parked {err * 1000:.1f} mm away vs stable {err_stable * 1000:.1f} mm"


def test_unreachable_reference_stable_at_half_speed(urdf_xml: str) -> None:
    """The flip does not come from speed saturation: a slower ceiling must not reintroduce it."""
    solver = _deployed_diffik(urdf_xml, "tcp", 0.0, max_joint_speed=2.0)
    published, _, _, _ = _servo_loop(solver, np.zeros(5), _EP10_DELTA, ticks=300)
    assert _tail_flips(published)[0] == 0


def test_reachable_reference_still_converges_quickly(urdf_xml: str) -> None:
    """Guard against over-damping: a reachable 5 cm pose target must converge within 15 ticks."""
    solver = _deployed_diffik(urdf_xml, "gripper", 0.01)
    q0 = _DIFFIK_HOME.copy()
    q_goal = q0 + np.array([0.25, -0.2, 0.2, 0.0, 0.0])  # ~50 mm, 4 ticks unbounded
    p_goal, r_goal = solver.ee_position(q_goal), solver.ee_rotation(q_goal)
    assert 0.04 < np.linalg.norm(p_goal - solver.ee_position(q0)) < 0.07
    q = q0.copy()
    for _ in range(15):
        q = solver.solve_to_pose(q, p_goal, r_goal, _DIFFIK_DT)
        if np.linalg.norm(solver.ee_position(q) - p_goal) < 1e-3:
            return
    pytest.fail(f"still {np.linalg.norm(solver.ee_position(q) - p_goal) * 1000:.1f} mm away after 15 ticks")


@pytest.mark.parametrize("frame", ["gripper", "tcp"])
@pytest.mark.parametrize("ori", [0.01, 0.0])
def test_deployed_yaml_parameters_park_without_oscillation(urdf_xml: str, frame, ori) -> None:
    """The committed YAML (damping, speed, step bound) must not flip in any frame/mode.

    The flip onset scales with damping, so lowering diffik_damping without also
    lowering diffik_max_task_step must make this test fail.
    """
    cfg = _servo_cfg()
    assert "diffik_max_task_step" in cfg, "so101_placo_servo.yaml must set diffik_max_task_step (#156)"
    solver = _deployed_diffik(urdf_xml, frame, ori, max_task_step=cfg["diffik_max_task_step"] or None)
    published, _, _, _ = _servo_loop(solver, np.zeros(5), _EP10_DELTA, ticks=300)
    assert _tail_flips(published)[0] == 0


def test_max_task_step_bounds_each_tick(urdf_xml: str) -> None:
    """One tick toward a far target may move the EE at most max_task_step (plus solver slack)."""
    solver = _deployed_diffik(urdf_xml, "gripper", 0.0, max_task_step=0.01)
    q0 = _DIFFIK_HOME.copy()
    p0 = solver.ee_position(q0)
    q1 = solver.solve_to_position(q0, p0 + np.array([0.0, 0.0, 0.3]), _DIFFIK_DT)
    assert np.linalg.norm(solver.ee_position(q1) - p0) <= 0.0105


@pytest.mark.parametrize("bad", [0.0, -0.01, float("nan"), float("inf")])
def test_max_task_step_rejects_invalid_values(urdf_xml: str, bad) -> None:
    with pytest.raises(ValueError):
        SO101PlacoDiffIK(urdf_xml=urdf_xml, max_task_step=bad)


def test_max_task_step_none_disables_bound(urdf_xml: str) -> None:
    """None restores the unbounded behaviour (the old flip on ep10) — the opt-out must work."""
    solver = _deployed_diffik(urdf_xml, "gripper", 0.01, max_task_step=None)
    published, _, _, _ = _servo_loop(solver, np.zeros(5), _EP10_DELTA, ticks=300)
    assert _tail_flips(published)[0] > 0


def test_step_velocity_api_unaffected_by_task_step_bound(urdf_xml: str) -> None:
    """step() already targets p + v*dt; the bound must not change its output."""
    bounded = _deployed_diffik(urdf_xml, "gripper", 0.0, max_task_step=0.01)
    unbounded = _deployed_diffik(urdf_xml, "gripper", 0.0, max_task_step=None)
    v = np.array([0.0, 0.0, 0.2])
    q = _DIFFIK_HOME.copy()
    np.testing.assert_allclose(bounded.step(q, v, _DIFFIK_DT), unbounded.step(q, v, _DIFFIK_DT), atol=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
