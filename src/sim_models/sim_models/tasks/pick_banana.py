"""PickBananaTask — scene task for the pick_banana scene.

Exposes three ROS2 services:
  /sim/pick_banana/random_set     (Trigger) — apply a pre-baked random keyframe
  /sim/pick_banana/reset          (Trigger) — reset scene to default
  /sim/pick_banana/autotest/start (Trigger) — run full AutoTest loop (non-blocking)

Position-based evaluation is performed via a ShadowSim that mirrors the main
mujoco_ros2_control simulation by tracking arm /joint_states and stepping a
local mujoco Python instance in a background thread.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Empty, Trigger

try:
    import mujoco as _mujoco

    _MUJOCO_OK = True
except ImportError:
    _mujoco = None  # type: ignore[assignment]
    _MUJOCO_OK = False

from mujoco_ros2_control_msgs.srv import ResetWorld

from sim_models.scene_compiler import get_mujoco_scene_path, get_scene_layout
from sim_models.tasks.base import EvalResult, SceneTask

# Fixed plate centre (from layout.yaml); used for eval distance check.
_PLATE_XY = np.array([0.22, 0.02])
_SUCCESS_RADIUS_M = 0.15
_FLEW_OUT_RADIUS_M = 0.40
_FELL_OFF_Z = -0.05


# ─── Shadow Simulation ───────────────────────────────────────────────────────


class ShadowSim:
    """Runs a local MuJoCo instance in a background thread.

    Receives arm joint positions from /joint_states and applies them each step
    so that free-body (banana) positions roughly track the real simulation.
    mujoco_ros2_control does not publish free-body poses, so this is the only
    way to observe banana position without modifying the third-party package.
    """

    def __init__(self, mjb_path: str) -> None:
        self._model = _mujoco.MjModel.from_binary_path(mjb_path)
        self._data = _mujoco.MjData(self._model)
        self._lock = threading.Lock()
        self._running = True
        self._dt = float(self._model.opt.timestep)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def reset_to_keyframe(self, name: str) -> bool:
        kid = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            return False
        with self._lock:
            _mujoco.mj_resetDataKeyframe(self._model, self._data, kid)
        return True

    def reset_default(self) -> None:
        with self._lock:
            _mujoco.mj_resetData(self._model, self._data)

    def set_joint_pos(self, joint_name: str, value: float) -> None:
        jid = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return
        with self._lock:
            self._data.qpos[self._model.jnt_qposadr[jid]] = value

    def get_body_pos(self, body_name: str) -> np.ndarray | None:
        bid = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None
        with self._lock:
            return self._data.xpos[bid].copy()

    def shutdown(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                _mujoco.mj_step(self._model, self._data)
            time.sleep(max(self._dt, 0.02))


# ─── PickBananaTask ───────────────────────────────────────────────────────────


class PickBananaTask(SceneTask):
    scene_name = "pick_banana"

    def __init__(self, node: Node) -> None:
        self._node = node
        self._cbg = ReentrantCallbackGroup()

        layout = get_scene_layout("pick_banana")
        rand = layout.get("randomization", {})
        banana_cfg = rand.get("banana", {})
        self._arm_joint_names: list[str] = rand.get("arm_joint_names", ["1", "2", "3", "4", "5", "6"])
        cols = int(banana_cfg.get("grid_cols", 5))
        rows = int(banana_cfg.get("grid_rows", 7))
        yaws = int(banana_cfg.get("yaw_steps", 6))
        self._n_keyframes = cols * rows * yaws  # 210
        self._kf_idx = -1

        # Shadow sim
        self._shadow: ShadowSim | None = None
        if _MUJOCO_OK:
            try:
                mjb = self._resolve_scene_mjb()
                self._shadow = ShadowSim(str(mjb))
                node.get_logger().info(f"ShadowSim running from {mjb}")
            except Exception as exc:
                node.get_logger().warning(f"ShadowSim unavailable: {exc}")

        # AutoTest state
        self._autotest_running = False
        self._autotest_lock = threading.Lock()

        # Eval output dir (configurable via ROS param eval_output_dir)
        default_out = str(Path.home() / "ib_robot_eval")
        out_dir = node.declare_parameter("eval_output_dir", default_out).value
        self._eval_root = Path(out_dir) / "pick_banana"

        # Subscriptions & service clients
        node.create_subscription(
            JointState,
            "/joint_states",
            self._on_js,
            10,
            callback_group=self._cbg,
        )
        # mujoco_ros2_control advertises ~/reset_world relative to its node, so
        # the fully-qualified name is /<mujoco_node>/reset_world — NOT the bare
        # /reset_world. The default matches the installed plugin's node name;
        # override via the reset_world_service param, and we also auto-discover
        # the real name at first use to stay robust across plugin versions
        # (e.g. /ros2_control_node/reset_world).
        configured_rw = node.declare_parameter("reset_world_service", "").value
        self._reset_world_name = configured_rw or "/mujoco_ros2_control_node/reset_world"
        self._reset_world = node.create_client(ResetWorld, self._reset_world_name, callback_group=self._cbg)

        # Dispatcher control clients (used by AutoTest to drive clean episodes).
        self._disp_start = node.create_client(Trigger, "/action_dispatcher/start_evaluate", callback_group=self._cbg)
        self._disp_stop = node.create_client(Trigger, "/action_dispatcher/stop_evaluate", callback_group=self._cbg)
        self._disp_reset = node.create_client(Empty, "/action_dispatcher/reset", callback_group=self._cbg)

        # Arm command publishers + rest pose. Re-commanding the rest pose right
        # before a world reset aligns the controller setpoint with the keyframe,
        # so the arm does not snap back to the last inference target while the
        # controllers stay active during reset_world.
        rest_qpos = list(rand.get("arm_rest_qpos", [0.0, -1.5854, 1.5708, 0.7854, 0.0, 0.0]))
        self._rest_arm = rest_qpos[:5]
        self._rest_gripper = float(rest_qpos[5]) if len(rest_qpos) > 5 else 0.0
        arm_topic = node.declare_parameter("arm_command_topic", "/arm_position_controller/commands").value
        gripper_topic = node.declare_parameter("gripper_command_topic", "/gripper_position_controller/commands").value
        self._arm_pub = node.create_publisher(Float64MultiArray, arm_topic, 10)
        self._grip_pub = node.create_publisher(Float64MultiArray, gripper_topic, 10)

        # Services exposed by this task
        node.create_service(
            Trigger,
            "/sim/pick_banana/random_set",
            self._svc_random_set,
            callback_group=self._cbg,
        )
        node.create_service(
            Trigger,
            "/sim/pick_banana/reset",
            self._svc_reset,
            callback_group=self._cbg,
        )
        node.create_service(
            Trigger,
            "/sim/pick_banana/autotest/start",
            self._svc_autotest,
            callback_group=self._cbg,
        )

    # ── SceneTask interface ────────────────────────────────────────────────────

    def randomize(self) -> tuple[bool, str]:
        self._kf_idx = (self._kf_idx + 1) % self._n_keyframes
        kname = f"random_{self._kf_idx:03d}"
        ok, msg = self._call_reset_world_sync(kname)
        if ok and self._shadow is not None:
            self._shadow.reset_to_keyframe(kname)
        return ok, msg

    def reset(self) -> tuple[bool, str]:
        ok, msg = self._call_reset_world_sync("")
        if ok and self._shadow is not None:
            self._shadow.reset_default()
        return ok, msg

    def evaluate(self, duration_s: float = 30.0) -> EvalResult:
        t0 = time.monotonic()
        banana_final = [0.0, 0.0, 0.0]
        start_dist_xy: float | None = None

        while (time.monotonic() - t0) < duration_s:
            if self._shadow is not None:
                pos = self._shadow.get_body_pos("banana")
                if pos is not None:
                    banana_final = pos.tolist()
                    dist_xy = float(np.linalg.norm(pos[:2] - _PLATE_XY))
                    if start_dist_xy is None:
                        start_dist_xy = dist_xy
                    if pos[2] < _FELL_OFF_Z or dist_xy > _FLEW_OUT_RADIUS_M:
                        return EvalResult(
                            success=False,
                            reason="flew_out",
                            banana_final_pos=banana_final,
                            duration_s=time.monotonic() - t0,
                        )
                    if start_dist_xy >= _SUCCESS_RADIUS_M and dist_xy < _SUCCESS_RADIUS_M:
                        return EvalResult(
                            success=True,
                            reason="placed_on_plate",
                            banana_final_pos=banana_final,
                            duration_s=time.monotonic() - t0,
                        )
            time.sleep(0.1)

        return EvalResult(
            success=False,
            reason="timeout",
            banana_final_pos=banana_final,
            duration_s=duration_s,
        )

    # ── service handlers ──────────────────────────────────────────────────────

    def _svc_random_set(self, _req: Trigger.Request, resp: Trigger.Response):
        ok, msg = self._clean_world_reset(randomize=True, resume=True)
        resp.success = ok
        resp.message = msg
        return resp

    def _svc_reset(self, _req: Trigger.Request, resp: Trigger.Response):
        ok, msg = self._clean_world_reset(randomize=False, resume=True)
        resp.success = ok
        resp.message = msg
        return resp

    def _clean_world_reset(self, randomize: bool, resume: bool, settle_s: float = 0.8) -> tuple[bool, str]:
        """Pause inference → reset the world → (optionally) resume inference.

        Stop, publish the rest pose, reset the world and policy episode, then
        resume when requested. This task uses the legacy dispatcher only.
        """
        self._call_service_sync(
            self._disp_stop,
            Trigger.Request(),
            "dispatcher/stop",
            check_success=True,
        )
        self._publish_rest_pose()

        ok, msg = self.randomize() if randomize else self.reset()

        if ok:
            time.sleep(settle_s)

        if resume:
            if ok:
                self._call_service_sync(self._disp_reset, Empty.Request(), "dispatcher/reset")
            # Preserve the legacy resume behavior even when the world reset fails.
            self._call_service_sync(self._disp_start, Trigger.Request(), "dispatcher/start")
        return ok, msg

    def _publish_rest_pose(self, count: int = 6, period: float = 0.05) -> None:
        for _ in range(count):
            self._arm_pub.publish(Float64MultiArray(data=list(self._rest_arm)))
            self._grip_pub.publish(Float64MultiArray(data=[self._rest_gripper]))
            time.sleep(period)

    def _svc_autotest(self, _req: Trigger.Request, resp: Trigger.Response):
        with self._autotest_lock:
            if self._autotest_running:
                resp.success = False
                resp.message = "AutoTest already running"
                return resp
            self._autotest_running = True
        threading.Thread(target=self._run_autotest, daemon=True).start()
        resp.success = True
        resp.message = f"AutoTest started ({self._n_keyframes} iterations)"
        return resp

    # ── AutoTest loop ─────────────────────────────────────────────────────────

    def _run_autotest(self) -> None:
        log = self._node.get_logger()
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self._eval_root / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            results_path = run_dir / "results.jsonl"
            summary_path = run_dir / "summary.yaml"

            log.info(f"AutoTest: saving to {run_dir}")

            passed = 0
            total = self._n_keyframes

            with results_path.open("w") as fh:
                for i in range(total):
                    # Clean episode: pause → reposition (no snap) → settle → resume.
                    ok, _ = self._clean_world_reset(randomize=True, resume=True, settle_s=1.5)
                    if not ok:
                        log.warning(f"AutoTest {i}: random_set failed, skipping")
                        continue

                    result = self.evaluate(duration_s=30.0)
                    if result.success:
                        passed += 1

                    record = {
                        "idx": i,
                        "keyframe": f"random_{self._kf_idx:03d}",
                        "success": result.success,
                        "reason": result.reason,
                        "banana_final_pos": result.banana_final_pos,
                        "duration_s": round(result.duration_s, 2),
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()

                    sym = "✓" if result.success else "✗"
                    log.info(f"AutoTest {i + 1}/{total} {sym} {result.reason} ({result.duration_s:.1f}s)")

            # Leave the robot idle after the batch completes.
            self._call_service_sync(self._disp_stop, Trigger.Request(), "dispatcher/stop")

            rate = passed / total if total > 0 else 0.0
            summary_path.write_text(
                f"scene: pick_banana\n"
                f'date: "{datetime.now().isoformat()}"\n'
                f"total: {total}\n"
                f"passed: {passed}\n"
                f"failed: {total - passed}\n"
                f"success_rate: {rate:.3f}\n"
            )
            log.info(f"AutoTest complete: {passed}/{total} ({rate:.1%})")
        except Exception as exc:  # noqa: BLE001
            log.error(f"AutoTest failed: {exc!r}")
            self._call_service_sync(self._disp_stop, Trigger.Request(), "dispatcher/stop")
        finally:
            with self._autotest_lock:
                self._autotest_running = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_scene_mjb(self) -> Path:
        """Return the compiled scene model that mirrors the live simulation.

        The launch's MuJoCo backend compiles the full robot+scene (with the
        injected ``random_KKK`` keyframes) to ``/tmp/sim_models_<scene>.mjb``.
        Reusing that exact file is essential: a robot-less recompile would have
        no arm, so the ShadowSim could never observe the gripper pushing the
        banana, and the keyframe qpos layout would not match.
        """
        compiled = Path(f"/tmp/sim_models_{self.scene_name}.mjb")
        if compiled.exists():
            return compiled
        self._node.get_logger().warning(
            f"{compiled} not found; falling back to a standalone scene compile "
            "(no robot arm — banana contact eval will be unreliable)"
        )
        return get_mujoco_scene_path(self.scene_name)

    def _discover_reset_world_service(self) -> str | None:
        """Find the live reset_world service by scanning the ROS graph."""
        try:
            names_and_types = self._node.get_service_names_and_types()
        except Exception:  # noqa: BLE001
            return None
        for name, types in names_and_types:
            if name.endswith("/reset_world") and any("ResetWorld" in t for t in types):
                return name
        return None

    def _on_js(self, msg: JointState) -> None:
        if self._shadow is None:
            return
        for name, pos in zip(msg.name, msg.position, strict=False):
            if name in self._arm_joint_names:
                self._shadow.set_joint_pos(name, pos)

    def _call_reset_world_sync(self, keyframe: str) -> tuple[bool, str]:
        """Call /reset_world from a background thread (NOT from a ROS callback).

        Uses a threading.Event so the calling thread blocks while the ROS
        executor (running in another thread) processes the response callback.
        This avoids the deadlock that occurs when you spin-wait inside a
        callback thread — the executor can't deliver the response because the
        callback thread is blocked polling future.done().
        """
        if not self._reset_world.wait_for_service(timeout_sec=5.0):
            # The configured name may be wrong for this plugin version; try to
            # locate the real service on the graph and rebind the client.
            discovered = self._discover_reset_world_service()
            if discovered and discovered != self._reset_world_name:
                self._node.get_logger().info(f"reset_world: using discovered service '{discovered}'")
                self._reset_world_name = discovered
                self._reset_world = self._node.create_client(ResetWorld, discovered, callback_group=self._cbg)
            if not self._reset_world.wait_for_service(timeout_sec=5.0):
                return False, f"reset_world service not available ({self._reset_world_name})"
        req = ResetWorld.Request()
        req.keyframe = keyframe
        future = self._reset_world.call_async(req)

        done_event = threading.Event()
        result_box: list = []

        def _cb(f):
            try:
                r = f.result()
                result_box.append((r.success, r.message))
            except Exception as exc:
                result_box.append((False, str(exc)))
            done_event.set()

        future.add_done_callback(_cb)
        if not done_event.wait(timeout=15.0):
            return False, "reset_world timeout"
        return result_box[0]

    def _call_service_sync(
        self, client, request, label: str, timeout: float = 5.0, *, check_success: bool = False
    ) -> bool:
        """Call a service and block (safe only from a background thread).

        Uses the same Event-based pattern as _call_reset_world_sync so the
        executor (running in other threads) can deliver the response. When
        check_success is True the Trigger.success field is inspected so
        stop/start/restart must be confirmed, not just completed.
        """
        if not client.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().warning(f"{label}: service not available")
            return False
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=timeout + 5.0):
            self._node.get_logger().warning(f"{label}: timed out")
            return False
        if not check_success:
            return True
        try:
            response = future.result()
        except Exception as e:  # noqa: BLE001
            self._node.get_logger().warning(f"{label}: result error {e}")
            return False
        if response is None or not getattr(response, "success", False):
            self._node.get_logger().warning(f"{label}: service returned failure")
            return False
        return True


# ─── Node entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    rclpy.init()
    node = Node("pick_banana_task_node")
    task = PickBananaTask(node)  # noqa: F841 — registers services on node
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if task._shadow is not None:
            task._shadow.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
