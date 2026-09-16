#!/usr/bin/env python3
"""Radian-native Placo kinematics wrapper for the SO-101 arm.

This module mirrors lerobot's ``RobotKinematics`` (``libs/lerobot/src/
lerobot/model/kinematics.py``) solving *posture* — ``RobotWrapper`` +
``KinematicsSolver`` + ``mask_fbase(True)`` + ``add_frame_task`` + per-frame
``set_joint`` seed → ``T_world_frame`` → ``configure(soft, pos_w, ori_w)`` →
``solve(True)`` — but adapts it to IB-Robot conventions:

* **Radian-native.** lerobot's public API is in *degrees*; IB-Robot's
  ``/joint_states`` and ``/arm_position_controller/commands`` are in
  *radians*. This wrapper takes and returns radians end to end.
* **IB-Robot URDF.** The SO-101 description lives as ``so101.urdf.xacro``
  (``$(find so101_description)/urdf/lerobot/so101/so101.urdf.xacro``). Placo
  needs plain URDF, so callers either pass an already-expanded URDF path or
  use :func:`expand_so101_xacro` to render one in-memory at runtime.
* **Target frame ``gripper`` or ``tcp``.** Matches ``moveit.ee_link`` in the
  robot SSOT. NOT lerobot's default ``gripper_frame_link``.
* **Explicit joint order.** The arm joints are ``["1".."5"]``; joint ``6``
  is the gripper and is excluded from Cartesian IK. Output is
  ordered to match ``arm_joint_names`` so it can be written straight to
  ``/arm_position_controller/commands``.

The wrapper has **no ROS runtime dependency** beyond the optional xacro
expansion helper, so it can be exercised by unit tests without a live robot.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import numpy as np

# Default location of the SO-101 description, relative to the ROS package
# ``so101_description``. Resolved lazily so importing this module never
# requires a sourced ROS workspace.
_SO101_XACRO_REL = "urdf/lerobot/so101/so101.urdf.xacro"

# IB-Robot conventions (SSOT: so101_single_arm.yaml / so101_placo_servo.yaml).
DEFAULT_TARGET_FRAME = "gripper"
DEFAULT_ARM_JOINT_NAMES: tuple[str, ...] = ("1", "2", "3", "4", "5")

# lerobot's "position hard, orientation best-effort" weights for the
# under-actuated 5-DOF arm (kinematics.py defaults).
DEFAULT_POSITION_WEIGHT = 1.0
DEFAULT_ORIENTATION_WEIGHT = 0.01


def expand_so101_xacro(xacro_path: str | None = None) -> str:
    """Expand the SO-101 xacro into a plain URDF string (in-memory).

    Args:
        xacro_path: Explicit path to ``so101.urdf.xacro``. If ``None``, the
            file is located via the ``so101_description`` ROS package share
            directory (requires a sourced/built workspace).

    Returns:
        The expanded URDF as an XML string.

    Raises:
        ImportError: if ``xacro`` is not importable.
        LookupError: if ``so101_description`` cannot be resolved and no
            explicit ``xacro_path`` is provided.
    """
    import xacro  # local import: keeps module import ROS-free

    if xacro_path is None:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("so101_description")
        xacro_path = os.path.join(share, _SO101_XACRO_REL)

    doc = xacro.process_file(xacro_path)
    return doc.toxml()


def _write_temp_urdf(urdf_xml: str) -> str:
    """Write a URDF string to a temp file and return its path.

    Placo's ``RobotWrapper`` reads URDF from a file path, so an in-memory
    string must be materialised. The caller owns deletion.
    """
    fd, path = tempfile.mkstemp(suffix=".urdf", prefix="so101_placo_")
    with os.fdopen(fd, "w") as f:
        f.write(urdf_xml)
    return path


def _as_joint_vector(values: np.ndarray | list[float], joint_names: list[str]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(joint_names),):
        raise ValueError(
            f"Joint vector length mismatch: expected {len(joint_names)} values "
            f"for {joint_names}, got shape {vector.shape}"
        )
    return vector


def _cleanup_temp_urdf(owner) -> None:
    if getattr(owner, "_owns_temp_urdf", False) and getattr(owner, "_temp_urdf_path", None):
        with contextlib.suppress(OSError):
            os.unlink(owner._temp_urdf_path)
        owner._temp_urdf_path = None
        owner._owns_temp_urdf = False


class SO101PlacoKinematics:
    """Radian-native Placo FK/IK for the SO-101 arm.

    Mirrors lerobot's ``RobotKinematics`` solving posture, adapted to
    IB-Robot units (rad), URDF, target frame (``gripper``) and arm joint
    order (``1..5``).
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        urdf_xml: str | None = None,
        target_frame: str = DEFAULT_TARGET_FRAME,
        arm_joint_names: tuple[str, ...] | list[str] = DEFAULT_ARM_JOINT_NAMES,
    ) -> None:
        """Initialise the Placo solver.

        Args:
            urdf_path: Path to a plain URDF file. Mutually exclusive with
                ``urdf_xml``.
            urdf_xml: Pre-expanded URDF XML string. If neither ``urdf_path``
                nor ``urdf_xml`` is given, the SO-101 xacro is expanded via
                :func:`expand_so101_xacro` (requires ROS workspace).
            target_frame: End-effector frame name in the URDF (default
                ``gripper``).
            arm_joint_names: Ordered arm joint names driven by Cartesian IK
                (default ``("1".."5")``). The gripper joint is intentionally
                excluded.
        """
        import placo  # local import: heavy native lib, keep optional at import time

        self._owns_temp_urdf = False
        self._temp_urdf_path: str | None = None

        if urdf_path is None and urdf_xml is None:
            urdf_xml = expand_so101_xacro()
        if urdf_xml is not None:
            if urdf_path is not None:
                raise ValueError("Pass only one of urdf_path / urdf_xml")
            urdf_path = _write_temp_urdf(urdf_xml)
            self._owns_temp_urdf = True
            self._temp_urdf_path = urdf_path

        self.target_frame = target_frame
        self.arm_joint_names: list[str] = list(arm_joint_names)

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # fixed base (lerobot parity)

        # Frame task with identity target; reconfigured each IK call.
        self.tip_task = self.solver.add_frame_task(self.target_frame, np.eye(4))

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Remove any materialised temporary URDF owned by this solver."""
        _cleanup_temp_urdf(self)

    # ------------------------------------------------------------------ FK
    def forward_kinematics(self, q_rad: np.ndarray | list[float]) -> np.ndarray:
        """Forward kinematics for the target frame.

        Args:
            q_rad: Arm joint positions in **radians**, ordered to match
                ``arm_joint_names``.

        Returns:
            4x4 homogeneous transform of ``target_frame`` in the base frame.
        """
        q = _as_joint_vector(q_rad, self.arm_joint_names)
        for name, value in zip(self.arm_joint_names, q, strict=True):
            self.robot.set_joint(name, float(value))
        self.robot.update_kinematics()
        return np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=np.float64)

    # ------------------------------------------------------------------ IK
    def inverse_kinematics(
        self,
        q_current_rad: np.ndarray | list[float],
        target_pose: np.ndarray,
        position_weight: float = DEFAULT_POSITION_WEIGHT,
        orientation_weight: float = DEFAULT_ORIENTATION_WEIGHT,
    ) -> np.ndarray:
        """Inverse kinematics for a desired target-frame pose.

        Uses the current joint positions as the initial guess (closed-loop
        re-seed, lerobot ``initial_guess_current_joints=True`` parity).

        Args:
            q_current_rad: Current arm joint positions in **radians** (IK
                seed), ordered to match ``arm_joint_names``.
            target_pose: Desired 4x4 base-frame pose of ``target_frame``.
            position_weight: Soft-task position weight.
            orientation_weight: Soft-task orientation weight (small for the
                under-actuated SO-101). Pass ``0.0`` for a position-only solve:
                the QP then minimises position error alone and orientation is
                left fully free.

        Returns:
            Arm joint positions in **radians**, ordered to match
            ``arm_joint_names``.
        """
        q = _as_joint_vector(q_current_rad, self.arm_joint_names)
        for name, value in zip(self.arm_joint_names, q, strict=True):
            self.robot.set_joint(name, float(value))

        self.tip_task.T_world_frame = np.asarray(target_pose, dtype=np.float64)
        self.tip_task.configure(self.target_frame, "soft", position_weight, orientation_weight)

        self.solver.solve(True)
        self.robot.update_kinematics()

        return np.array(
            [self.robot.get_joint(name) for name in self.arm_joint_names],
            dtype=np.float64,
        )


# Differential-IK defaults (velocity-level Jacobian + QP servo).
DEFAULT_DIFFIK_DAMPING = 1e-3  # regularization weight (DLS-like smooth yield)
DEFAULT_CONTROL_PERIOD = 0.02  # s (50 Hz) — solver.dt for the velocity step
# Per-joint velocity ceiling (rad/s) for the QP velocity-limit constraint. The
# SO-101 URDF declares 10 rad/s which is far too fast for hand teleop; this caps
# how aggressively the QP may step per tick so the arm yields smoothly. None ->
# keep the URDF limits unchanged.
DEFAULT_DIFFIK_MAX_JOINT_SPEED: float | None = 2.0
DEFAULT_DIFFIK_ORIENTATION_WEIGHT = 0.01


class SO101PlacoDiffIK:
    """Radian-native Placo **differential** (velocity-level) IK for SO-101.

    This is the Jacobian + QP Cartesian servo core. Unlike
    :class:`SO101PlacoKinematics` — which sets an *absolute* pose
    ``FrameTask`` and re-solves it globally (``solve(True)``), branch-flipping
    several radians when the absolute goal nears/leaves the reachable set — this
    class solves a **velocity** step per tick:

    * **Position primary, orientation optional.** The position task constrains
      the 3-DOF target-frame position. A low-weight orientation task can be
      added for angular teleop without making full 6D pose tracking mandatory
      on the under-actuated 5-DOF arm.
    * **Velocity goal, not accumulated absolute target.** Each :meth:`step`
      seeds from the *measured* joints (closed-loop) and asks the QP to move the
      EE by ``v * dt`` from where it currently is. There is no growing absolute
      target to overshoot, so no branch flip.
    * **Hard joint + velocity limits in the QP.** ``enable_joint_limits`` and
      ``enable_velocity_limits`` make the URDF limits real QP constraints, so
      unreachable direction components are projected onto the feasible set
      (correct "energy decomposition"): reachable components are followed, the
      arm slows and yields at the reachable edge instead of snapping. No
      post-hoc ``np.clip`` that would break direction consistency.
    * **Regularization damping.** A small regularization task gives DLS-like
      smoothness through singularities / boundaries.

    The ``target_frame`` is ``ik_link_name`` (== ``moveit.ee_link``), so the
    *same* code path serves both ``gripper`` and ``tcp``. tcp is a *fixed* child
    of gripper (pure 95 mm translation, identity rotation); Placo's
    ``frame_jacobian(target_frame)`` propagates that lever arm exactly, so tcp
    needs **no special case**.

    Offline-only (no ROS runtime beyond the optional xacro helper), so tests can
    exercise it without a live robot.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        urdf_xml: str | None = None,
        target_frame: str = DEFAULT_TARGET_FRAME,
        arm_joint_names: tuple[str, ...] | list[str] = DEFAULT_ARM_JOINT_NAMES,
        damping: float = DEFAULT_DIFFIK_DAMPING,
        control_period: float = DEFAULT_CONTROL_PERIOD,
        max_joint_speed: float | None = DEFAULT_DIFFIK_MAX_JOINT_SPEED,
        orientation_weight: float = 0.0,
    ) -> None:
        """Build the differential-IK solver.

        Args:
            urdf_path: Path to a plain URDF file. Mutually exclusive with
                ``urdf_xml``.
            urdf_xml: Pre-expanded URDF XML string. If neither is given, the
                SO-101 xacro is expanded via :func:`expand_so101_xacro`.
            target_frame: End-effector frame name in the URDF (``gripper`` or
                ``tcp``).
            arm_joint_names: Ordered arm joint names driven by IK (``1..5``).
            damping: Regularization weight (DLS-like smooth yield).
            control_period: Default solver ``dt`` (s) for the velocity step;
                :meth:`step` may override it per call.
            max_joint_speed: Per-joint velocity ceiling (rad/s) applied to every
                arm joint as the QP velocity-limit constraint. ``None`` keeps the
                URDF velocity limits (10 rad/s on the SO-101, too fast for hand
                teleop).
            orientation_weight: Optional soft orientation-task weight. ``0.0``
                keeps position-only behaviour; a small positive value enables
                low-priority Cartesian orientation tracking.
        """
        import placo  # local import: heavy native lib

        self._owns_temp_urdf = False
        self._temp_urdf_path: str | None = None

        if urdf_path is None and urdf_xml is None:
            urdf_xml = expand_so101_xacro()
        if urdf_xml is not None:
            if urdf_path is not None:
                raise ValueError("Pass only one of urdf_path / urdf_xml")
            urdf_path = _write_temp_urdf(urdf_xml)
            self._owns_temp_urdf = True
            self._temp_urdf_path = urdf_path

        self.target_frame = target_frame
        self.arm_joint_names: list[str] = list(arm_joint_names)
        self.control_period = float(control_period)
        self.orientation_weight = float(orientation_weight)

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # fixed base (lerobot parity)

        # Position-only task (no orientation goal). Target is reset every step.
        p0 = np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=np.float64)[:3, 3]
        self.pos_task = self.solver.add_position_task(self.target_frame, p0.copy())
        self.pos_task.configure(f"{self.target_frame}_diffik_pos", "soft", 1.0)

        self.orientation_task = None
        if self.orientation_weight > 0.0:
            r0 = np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=np.float64)[:3, :3]
            self.orientation_task = self.solver.add_orientation_task(self.target_frame, r0.copy())
            self.orientation_task.configure(
                f"{self.target_frame}_diffik_ori",
                "soft",
                self.orientation_weight,
            )

        # DLS-like damping for smooth yielding at singularities / boundaries.
        self.solver.add_regularization_task(damping)

        # Hard QP constraints: URDF joint limits + velocity limits. The velocity
        # limit uses solver.dt to bound per-tick Δq, so unreachable directions
        # are projected onto the feasible set rather than snapping a branch.
        # Tighten the per-joint velocity ceiling below the URDF's 10 rad/s for
        # smooth hand teleop (None -> keep URDF limits).
        if max_joint_speed is not None:
            for name in self.arm_joint_names:
                self.robot.set_velocity_limit(name, float(max_joint_speed))
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(True)
        self.solver.dt = self.control_period

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        """Remove any materialised temporary URDF owned by this solver."""
        _cleanup_temp_urdf(self)

    def _seed(self, q_rad: np.ndarray) -> None:
        q = _as_joint_vector(q_rad, self.arm_joint_names)
        for name, value in zip(self.arm_joint_names, q, strict=True):
            self.robot.set_joint(name, float(value))
        self.robot.update_kinematics()

    def forward_kinematics(self, q_rad: np.ndarray | list[float]) -> np.ndarray:
        """Forward kinematics for the target frame (4x4 base-frame transform)."""
        self._seed(np.asarray(q_rad, dtype=np.float64))
        return np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=np.float64)

    def ee_position(self, q_rad: np.ndarray | list[float]) -> np.ndarray:
        """Target-frame position (base frame, 3-vector) for the given joints."""
        return self.forward_kinematics(q_rad)[:3, 3].copy()

    def ee_rotation(self, q_rad: np.ndarray | list[float]) -> np.ndarray:
        """Target-frame rotation (base frame, 3x3) for the given joints."""
        return self.forward_kinematics(q_rad)[:3, :3].copy()

    def solve_to_position(
        self,
        q_seed_rad: np.ndarray | list[float],
        target_position: np.ndarray | list[float],
        dt: float | None = None,
    ) -> np.ndarray:
        """One QP step toward an **absolute** base-frame target position.

        The caller supplies an absolute command-side reference position. In the
        servo hot path, the QP is seeded from the last command rather than the
        latest measured joints so hardware sag or servo lag is not accepted as a
        new target.

        The per-tick joint motion is still bounded by the in-QP velocity limit,
        so a large reference error cannot snap the arm; it converges over a few
        ticks. Holding (zero user velocity) keeps the reference fixed, so the
        only motion the QP produces is the small correction that cancels sag.

        Args:
            q_seed_rad: Seed arm joints (rad), ordered by ``arm_joint_names``.
            target_position: Absolute desired base-frame position (3-vector) of
                ``target_frame`` — the command-side reference, advanced by the
                caller with ``ref += v*dt`` only while there is user input.
            dt: Control period (s); defaults to ``control_period``.

        Returns:
            Next arm joint positions (rad), ordered by ``arm_joint_names``.
        """
        step_dt = self.control_period if dt is None else float(dt)
        self._seed(np.asarray(q_seed_rad, dtype=np.float64))

        self.pos_task.target_world = np.asarray(target_position, dtype=np.float64)

        self.solver.dt = step_dt
        self.solver.solve(True)
        self.robot.update_kinematics()

        return np.array(
            [self.robot.get_joint(name) for name in self.arm_joint_names],
            dtype=np.float64,
        )

    def solve_to_pose(
        self,
        q_seed_rad: np.ndarray | list[float],
        target_position: np.ndarray | list[float],
        target_rotation: np.ndarray | list[list[float]],
        dt: float | None = None,
    ) -> np.ndarray:
        """One QP step toward an absolute position + soft orientation target."""
        step_dt = self.control_period if dt is None else float(dt)
        self._seed(np.asarray(q_seed_rad, dtype=np.float64))

        self.pos_task.target_world = np.asarray(target_position, dtype=np.float64)
        if self.orientation_task is not None:
            self.orientation_task.R_world_frame = np.asarray(target_rotation, dtype=np.float64)

        self.solver.dt = step_dt
        self.solver.solve(True)
        self.robot.update_kinematics()

        return np.array(
            [self.robot.get_joint(name) for name in self.arm_joint_names],
            dtype=np.float64,
        )

    def step(
        self,
        q_seed_rad: np.ndarray | list[float],
        v_base: np.ndarray | list[float],
        dt: float | None = None,
    ) -> np.ndarray:
        """One velocity-level differential-IK step.

        Seeds from ``q_seed_rad``, asks the QP to move the target frame by
        ``v_base * dt`` from the seeded position, and returns the integrated next
        joint positions. Joint and velocity limits are enforced *inside* the QP,
        so the output never violates them and the unreachable velocity component
        is gracefully projected away.

        Args:
            q_seed_rad: Seed arm joints (rad), ordered by ``arm_joint_names``.
            v_base: Desired EE linear velocity in the base frame (m/s).
            dt: Control period (s) for this step; defaults to
                ``control_period``.

        Returns:
            Next arm joint positions (rad), ordered by ``arm_joint_names``.
        """
        step_dt = self.control_period if dt is None else float(dt)
        self._seed(np.asarray(q_seed_rad, dtype=np.float64))

        p_cur = np.asarray(self.robot.get_T_world_frame(self.target_frame), dtype=np.float64)[:3, 3]
        self.pos_task.target_world = p_cur + np.asarray(v_base, dtype=np.float64) * step_dt

        self.solver.dt = step_dt
        self.solver.solve(True)
        self.robot.update_kinematics()

        return np.array(
            [self.robot.get_joint(name) for name in self.arm_joint_names],
            dtype=np.float64,
        )
