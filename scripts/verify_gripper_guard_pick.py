#!/usr/bin/env python3
"""Grasp-pipeline-integrated gripper stall-protection verification.

This script drives a minimal pick sequence and concurrently monitors the
GripperGuardNode (Layer 2 ROS-level stall protection from PR #196) to verify
it behaves correctly when the gripper closes on a physical target.

Why a dedicated script
----------------------
The normal grasp pipeline (task_dispatch ExecuteTaskPlan) commands the gripper
through ``gripper_trajectory_controller/follow_joint_trajectory``, which
**bypasses** GripperGuardNode entirely. The guard is only in the path when the
gripper is driven via ``/gripper/target`` (the proxy input), which is the
JointGroupPositionController path active in **teleop / model_inference modes**.

To verify the guard during a grasp, this script therefore:
  * Moves the arm via ``/motion/move_to_pose`` (same service MoveIt
    pipeline uses).
  * Commands the gripper by publishing to ``/gripper/target`` so the guard
    proxy is exercised.

What it verifies
----------------
  1. FORWARDING  - when the gripper can reach the target (no object), the guard
    passes the target through unchanged.
  2. STALL HOLD   - when the gripper is blocked by an object, the guard holds
    the current position instead of forcing the unreachable target.
  3. RELEASE      - when the gripper is reopened (target changes), the guard
    stops holding and forwards the new target again.

Layer 1 (hardware overcurrent in so101_hardware) is NOT verifiable from a ROS
node; watch for the ``Motor ID X overload!`` log on real hardware.

Prerequisites
-------------
Run the robot in teleop mode so ``gripper_position_controller`` is active and
the ``gripper_guard`` node is running::

    source .shrc_local
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop

Then in another terminal::

    python3 scripts/verify_gripper_guard_pick.py

Place a target object between the gripper fingers at the grasp pose so the
close step physically cannot reach 0.0 (this is the stall condition).
"""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ibrobot_msgs.srv import MoveToPose
from robot_runtime import contract as RUNTIME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify GripperGuardNode stall protection during a grasp sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Topics ---
    p.add_argument("--gripper-target-topic", default="/gripper/target")
    p.add_argument("--gripper-cmd-topic", default="/gripper_position_controller/commands")
    p.add_argument("--joint-states-topic", default="/joint_states")
    p.add_argument("--move-service", default=RUNTIME.MOVE_TO_POSE_SERVICE)
    p.add_argument("--gripper-joint", default="6", help="Gripper joint name in /joint_states")

    # --- Guard thresholds (read from /gripper_guard by default) ---
    p.add_argument(
        "--stall-threshold", type=float, default=None, help="Override guard stall_threshold (else read from node param)"
    )
    p.add_argument(
        "--stall-timeout", type=float, default=None, help="Override guard stall_timeout (else read from node param)"
    )
    p.add_argument(
        "--goal-tolerance", type=float, default=None, help="Override guard goal_tolerance (else read from node param)"
    )
    p.add_argument(
        "--release-delta", type=float, default=None, help="Override guard release_delta (else read from node param)"
    )
    p.add_argument("--guard-node", default="/gripper_guard", help="GripperGuardNode name for param read")

    # --- Gripper targets (SO-101 normalized [0=closed, 1=open]) ---
    p.add_argument("--open-target", type=float, default=1.0)
    p.add_argument("--close-target", type=float, default=0.0)

    # --- Timing ---
    p.add_argument("--open-settle-s", type=float, default=1.5)
    p.add_argument("--monitor-s", type=float, default=2.5, help="Duration to monitor guard during close+hold")
    p.add_argument("--reopen-monitor-s", type=float, default=1.5, help="Duration to monitor guard release after reopen")

    # --- Arm motion (optional, to integrate with a real grasp) ---
    p.add_argument("--with-arm-motion", action="store_true", help="Move arm to grasp poses via /motion/move_to_pose")
    p.add_argument("--approach-x", type=float, default=0.20)
    p.add_argument("--approach-y", type=float, default=0.0)
    p.add_argument("--approach-z", type=float, default=0.28)
    p.add_argument("--grasp-z", type=float, default=0.12, help="Descend height for grasp")
    p.add_argument("--lift-z", type=float, default=0.30)
    p.add_argument("--quat-x", type=float, default=1.0, help="Grasp orientation qx (top-down default)")
    p.add_argument("--quat-y", type=float, default=0.0)
    p.add_argument("--quat-z", type=float, default=0.0)
    p.add_argument("--quat-w", type=float, default=0.0)
    p.add_argument("--velocity-scaling", type=float, default=0.3)
    p.add_argument("--move-timeout-s", type=float, default=30.0)

    # --- Options ---
    p.add_argument("--no-release-check", action="store_true", help="Skip the reopen release verification")
    p.add_argument("--require-block", action="store_true", help="FAIL if the gripper closes fully (no object detected)")
    return p.parse_args()


class GuardSample:
    """One timestamped snapshot of the guard data path."""

    __slots__ = ("t", "target", "guard_out", "actual")

    def __init__(self, t: float, target: float | None, guard_out: float | None, actual: float | None):
        self.t = t
        self.target = target
        self.guard_out = guard_out
        self.actual = actual


class GripperGuardPickTest(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("gripper_guard_pick_test")
        self.args = args

        # Latest values (latched by callbacks)
        self._target: float | None = None
        self._guard_out: float | None = None
        self._actual: float | None = None

        # Recorded timeline during a monitoring window
        self._samples: list[GuardSample] = []
        self._recording = False

        self.create_subscription(Float64MultiArray, args.gripper_target_topic, self._on_target, 10)
        self.create_subscription(Float64MultiArray, args.gripper_cmd_topic, self._on_guard_out, 10)
        self.create_subscription(JointState, args.joint_states_topic, self._on_state, 10)

        self._gripper_pub = self.create_publisher(Float64MultiArray, args.gripper_target_topic, 10)
        self._move_client = self.create_client(MoveToPose, args.move_service)

        # Resolve guard thresholds: CLI override > node param > built-in default
        self.stall_threshold = self._resolve_threshold("stall_threshold", args.stall_threshold, 0.05)
        self.stall_timeout = self._resolve_threshold("stall_timeout", args.stall_timeout, 0.5)
        self.goal_tolerance = self._resolve_threshold("goal_tolerance", args.goal_tolerance, 0.01)
        self.release_delta = self._resolve_threshold("release_delta", args.release_delta, 0.05)

        self.get_logger().info(
            f"Guard thresholds: stall>{self.stall_threshold} timeout>{self.stall_timeout}s "
            f"goal_tol>{self.goal_tolerance} release_delta>{self.release_delta}"
        )

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #
    def _on_target(self, msg: Float64MultiArray) -> None:
        if msg.data:
            self._target = float(msg.data[0])
        if self._recording:
            self._samples.append(GuardSample(time.monotonic(), self._target, self._guard_out, self._actual))

    def _on_guard_out(self, msg: Float64MultiArray) -> None:
        if msg.data:
            self._guard_out = float(msg.data[0])

    def _on_state(self, msg: JointState) -> None:
        if self.args.gripper_joint in msg.name:
            self._actual = float(msg.position[msg.name.index(self.args.gripper_joint)])

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _resolve_threshold(self, name: str, cli_value: float | None, default: float) -> float:
        if cli_value is not None:
            return cli_value
        try:
            from rcl_interfaces.srv import GetParameters

            cli = self.create_client(GetParameters, f"{self.args.guard_node}/get_parameters")
            if cli.wait_for_service(timeout_sec=1.0):
                req = GetParameters.Request()
                req.names = [name]
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
                if fut.done() and fut.result() is not None and fut.result().values:
                    v = fut.result().values[0]
                    if v.type > 0:
                        return float(v.double_value)
        except Exception as e:
            self.get_logger().warn(f"Could not read '{name}' from {self.args.guard_node}: {e}")
        self.get_logger().warn(f"Using built-in default {name}={default}")
        return default

    def spin_wait(self, duration_s: float) -> None:
        """Spin the node for a wall-clock duration."""
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_actual(self, timeout_s: float = 3.0) -> float:
        """Wait until a gripper joint state arrives; return the position."""
        deadline = time.monotonic() + timeout_s
        while self._actual is None and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        if self._actual is None:
            raise RuntimeError(
                f"No /joint_states for joint '{self.args.gripper_joint}' after {timeout_s:.1f}s. Is the robot launched?"
            )
        return self._actual

    def publish_gripper(self, value: float) -> None:
        msg = Float64MultiArray()
        msg.data = [float(value)]
        self._gripper_pub.publish(msg)
        # Keep publishing at ~50 Hz so the guard proxy keeps a fresh target.
        # (Single publish is usually enough, but repeated publish mirrors teleop.)

    def hold_gripper(self, value: float, duration_s: float) -> None:
        """Publish a gripper target continuously for duration_s (like teleop does)."""
        rate_period = 0.02
        steps = max(1, int(duration_s / rate_period))
        for _ in range(steps):
            if not rclpy.ok():
                break
            self.publish_gripper(value)
            self.spin_wait(rate_period)

    def move_arm(self, label: str, x: float, y: float, z: float) -> None:
        """Move arm to a Cartesian pose via /motion/move_to_pose."""
        if not self._move_client.service_is_ready():
            self.get_logger().info(f"Waiting for {self.args.move_service} ...")
            if not self._move_client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{self.args.move_service} not available")
        req = MoveToPose.Request()
        req.target_pose = self._make_pose(x, y, z)
        req.velocity_scaling = float(self.args.velocity_scaling)
        self.get_logger().info(f"[arm] {label}: ({x:.3f},{y:.3f},{z:.3f})")
        fut = self._move_client.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < self.args.move_timeout_s and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not fut.done():
            raise RuntimeError(f"Arm move '{label}' timed out")
        resp = fut.result()
        if resp is None or not resp.success:
            raise RuntimeError(f"Arm move '{label}' failed: {resp.message if resp else 'no response'}")

    def _make_pose(self, x: float, y: float, z: float) -> Pose:
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(self.args.quat_x)
        pose.orientation.y = float(self.args.quat_y)
        pose.orientation.z = float(self.args.quat_z)
        pose.orientation.w = float(self.args.quat_w)
        return pose

    # ------------------------------------------------------------------ #
    #  Monitoring window
    # ------------------------------------------------------------------ #
    def start_recording(self) -> None:
        self._samples.clear()
        self._recording = True

    def stop_recording(self) -> list[GuardSample]:
        self._recording = False
        return list(self._samples)

    # ------------------------------------------------------------------ #
    #  Verdict
    # ------------------------------------------------------------------ #
    def evaluate_hold(self, samples: list[GuardSample]) -> dict:
        """Evaluate the close+hold window.

        Returns a dict with the verdict and supporting evidence.
        """
        verdict = {"blocked": None, "stall_engaged": None, "ok": None, "detail": ""}

        if not samples:
            verdict["detail"] = "No samples recorded (topics not publishing?)"
            return verdict

        # Steady-state = last 0.5s of the window (well past stall_timeout)
        end_t = samples[-1].t
        steady = [s for s in samples if s.t >= end_t - 0.5] or samples[-3:]

        target = self._last_known(steady, "target")
        guard_out = self._last_known(steady, "guard_out")
        actual = self._last_known(steady, "actual")

        if target is None or guard_out is None or actual is None:
            verdict["detail"] = f"Incomplete data: target={target} guard_out={guard_out} actual={actual}"
            return verdict

        verdict["target"] = target
        verdict["guard_out"] = guard_out
        verdict["actual"] = actual

        blocked = abs(target - actual) > self.stall_threshold
        # Guard holds actual position (not forwarding the unreachable target)
        guard_holds = abs(guard_out - actual) <= max(self.goal_tolerance, 0.02)
        guard_diverged = abs(guard_out - target) > self.release_delta
        stall_engaged = blocked and guard_holds and guard_diverged

        verdict["blocked"] = blocked
        verdict["stall_engaged"] = stall_engaged

        if blocked:
            # An object prevented full close -> protection MUST engage
            verdict["ok"] = stall_engaged
            verdict["detail"] = (
                "STALL_PROTECTION_ENGAGED (correct)"
                if stall_engaged
                else "PROTECTION_FAILED: blocked but guard kept forcing target"
            )
        else:
            # Gripper reached the close target -> no stall needed
            forwarding = abs(guard_out - target) <= max(self.goal_tolerance, 0.02)
            verdict["ok"] = forwarding
            verdict["detail"] = (
                "FORWARDING (no object, no false stall)"
                if forwarding
                else "FALSE_STALL: guard held despite reachable target"
            )
        return verdict

    def evaluate_release(self, samples: list[GuardSample]) -> dict:
        """Evaluate the reopen window: guard should forward the new target."""
        verdict = {"ok": None, "detail": ""}
        if not samples:
            verdict["detail"] = "No samples recorded during reopen"
            return verdict
        end_t = samples[-1].t
        steady = [s for s in samples if s.t >= end_t - 0.5] or samples[-3:]
        target = self._last_known(steady, "target")
        guard_out = self._last_known(steady, "guard_out")
        if target is None or guard_out is None:
            verdict["detail"] = "Incomplete data during reopen"
            return verdict
        ok = abs(guard_out - target) <= max(self.goal_tolerance, 0.03)
        verdict["ok"] = ok
        verdict["target"] = target
        verdict["guard_out"] = guard_out
        verdict["detail"] = "RELEASED (guard forwarding again)" if ok else "STILL_HOLDING (release failed)"
        return verdict

    @staticmethod
    def _last_known(samples: list[GuardSample], attr: str) -> float | None:
        for s in reversed(samples):
            v = getattr(s, attr)
            if v is not None:
                return v
        return None

    # ------------------------------------------------------------------ #
    #  Sequence
    # ------------------------------------------------------------------ #
    def run(self) -> int:
        self.wait_for_actual()
        open_actual = self._actual
        self.get_logger().info(f"Initial gripper position: {open_actual:.3f}")

        results = {}

        # 1. Open gripper (baseline / forwarding check)
        self.get_logger().info(f"=== STEP open gripper -> {self.args.open_target} ===")
        self.hold_gripper(self.args.open_target, self.args.open_settle_s)

        # 2. [optional] Arm approach + descend
        if self.args.with_arm_motion:
            self.move_arm("approach", self.args.approach_x, self.args.approach_y, self.args.approach_z)
            self.move_arm("descend", self.args.approach_x, self.args.approach_y, self.args.grasp_z)

        # 3. Close gripper + MONITOR (the core stall-protection test)
        self.get_logger().info(f"=== STEP close gripper -> {self.args.close_target} (monitoring) ===")
        self.get_logger().info(
            ">>> Make sure an object is between the fingers so the gripper CANNOT "
            "fully close (this is the stall condition)."
        )
        self.start_recording()
        self.hold_gripper(self.args.close_target, self.args.monitor_s)
        close_samples = self.stop_recording()
        results["hold"] = self.evaluate_hold(close_samples)

        # 4. [optional] Lift
        if self.args.with_arm_motion:
            try:
                self.move_arm("lift", self.args.approach_x, self.args.approach_y, self.args.lift_z)
            except Exception as e:
                self.get_logger().warn(f"Lift failed (continuing): {e}")

        # 5. Reopen + MONITOR RELEASE
        if not self.args.no_release_check:
            self.get_logger().info(f"=== STEP reopen gripper -> {self.args.open_target} (release check) ===")
            self.start_recording()
            self.hold_gripper(self.args.open_target, self.args.reopen_monitor_s)
            reopen_samples = self.stop_recording()
            results["release"] = self.evaluate_release(reopen_samples)

        self._print_report(results)

        overall = self._overall(results)
        return 0 if overall else 1

    def _overall(self, results: dict) -> bool:
        ok = True
        hold = results.get("hold", {})
        if hold.get("ok") is False:
            ok = False
        if self.args.require_block and not hold.get("blocked"):
            # User insisted on a real blocked-close scenario
            ok = False
        rel = results.get("release")
        if rel is not None and rel.get("ok") is False:
            ok = False
        return ok

    def _print_report(self, results: dict) -> None:
        print("\n" + "=" * 64, flush=True)
        print("GRIPPER GUARD VERIFICATION REPORT", flush=True)
        print("=" * 64, flush=True)
        hold = results.get("hold", {})
        print(
            f"[HOLD]   blocked={hold.get('blocked')}  stall_engaged={hold.get('stall_engaged')}  ok={hold.get('ok')}",
            flush=True,
        )
        if "target" in hold:
            print(
                f"         target={hold.get('target'):.3f} actual={hold.get('actual'):.3f} "
                f"guard_out={hold.get('guard_out'):.3f}",
                flush=True,
            )
        print(f"         {hold.get('detail')}", flush=True)

        rel = results.get("release")
        if rel is not None:
            print(f"[RELEASE] ok={rel.get('ok')}  {rel.get('detail')}", flush=True)
            if "target" in rel:
                print(f"          target={rel.get('target'):.3f} guard_out={rel.get('guard_out'):.3f}", flush=True)

        if self.args.require_block and not hold.get("blocked"):
            print("[NOTE] --require-block set but gripper closed fully (no object?).", flush=True)

        overall = self._overall(results)
        print("-" * 64, flush=True)
        print(f"OVERALL: {'PASS' if overall else 'FAIL'}", flush=True)
        print("=" * 64 + "\n", flush=True)


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.open_target <= 1.0 and 0.0 <= args.close_target <= 1.0):
        raise SystemExit("open_target/close_target must be in [0,1] for SO-101 gripper")
    if math.isclose(args.open_target, args.close_target, abs_tol=args.release_delta):
        raise SystemExit("open_target and close_target too close; release check is meaningless")

    rclpy.init()
    node = GripperGuardPickTest(args)
    try:
        code = node.run()
    except Exception as e:
        node.get_logger().error(f"Test aborted: {e}")
        code = 2
    finally:
        # Safety: leave the gripper open
        try:
            node.publish_gripper(args.open_target)
            node.spin_wait(0.3)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
