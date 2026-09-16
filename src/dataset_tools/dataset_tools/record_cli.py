#!/usr/bin/env python3

"""
Command Line Interface for triggering episodic recordings.
Sends Action goals to the EpisodeRecorderServer.
"""

import json
import signal
import sys
import threading
import time
import traceback

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Empty, Trigger

# Import the global interface
from ibrobot_msgs.action import RecordEpisode
from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode


class RecordCLI(Node):
    def __init__(self):
        super().__init__("record_cli")

        self.declare_parameter("dispatcher_reset_service", "/action_dispatcher/reset")
        self.declare_parameter("policy_reset_service", "/inference/policy/reset")
        self.declare_parameter("restart_session_service", "")
        self.declare_parameter("reset_timeout_sec", 2.0)
        self.declare_parameter("control_mode", "teleop")
        self.declare_parameter("reset_before_episode", "auto")
        self.declare_parameter("runtime_set_mode_service", "")
        self.declare_parameter("teleop_rearm_service", "/robot_teleop_node/rearm")
        self.declare_parameter("teleop_stop_service", "")
        self.declare_parameter("runtime_status_topic", "/runtime_status")
        self.declare_parameter("admission_timeout_sec", 5.0)
        self.declare_parameter("admission_attempts", 3)
        self.declare_parameter("shutdown_timeout_sec", 15.0)
        self._reset_timeout_s = float(self.get_parameter("reset_timeout_sec").value)
        self._episode_finished_evt = threading.Event()
        self._goal_started_evt = threading.Event()
        self._goal_rejected_evt = threading.Event()
        self._last_result_success = False
        self._last_result_message = ""
        self._goal_lock = threading.RLock()
        self._owned_goal = None
        self._send_goal_future = None
        self._goal_pending = False
        self._cancel_future = None
        self._cancel_requested = False
        self._closing = False
        self._admission_future = None
        self._admission_client = None
        self._admission_dirty = False
        self._teleop_session_owned = False
        self._runtime_status = None
        self._runtime_status_received = 0.0
        self._runtime_status_event = threading.Event()
        mode_service = str(self.get_parameter("runtime_set_mode_service").value)
        self._mode_client = self.create_client(SetRuntimeMode, mode_service) if mode_service else None
        self._rearm_client = (
            self.create_client(Trigger, str(self.get_parameter("teleop_rearm_service").value)) if mode_service else None
        )
        stop_service = str(self.get_parameter("teleop_stop_service").value)
        self._teleop_stop_client = self.create_client(Trigger, stop_service) if mode_service and stop_service else None
        if mode_service:
            self._status_sub = self.create_subscription(
                RuntimeStatus, str(self.get_parameter("runtime_status_topic").value), self._status_callback, 1
            )

        # Action client to start recording
        self._action_client = ActionClient(self, RecordEpisode, "record_episode")

        self._dispatcher_reset_client = self.create_client(
            Empty,
            self.get_parameter("dispatcher_reset_service").value,
        )
        self._policy_reset_client = self.create_client(
            Trigger,
            self.get_parameter("policy_reset_service").value,
        )
        restart_service = str(self.get_parameter("restart_session_service").value)
        self._restart_session_client = self.create_client(Trigger, restart_service) if restart_service else None
        # Service client to get dataset info
        self._info_client = self.create_client(Trigger, "record_episode/get_info")
        self._last_episode_client = self.create_client(Trigger, "record_episode/get_last_episode")
        self._delete_last_client = self.create_client(Trigger, "record_episode/delete_last")

        self.get_logger().info("Record CLI started. Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Connected to Episode Recorder Server!")

    def send_goal(self, prompt_text: str):
        with self._goal_lock:
            if self._closing or self._owned_goal is not None or self._goal_pending:
                self.get_logger().error("Previous goal is still unresolved; no new recording submitted.")
                return False
            self._cancel_requested = False
            self._cancel_future = None
        self._episode_finished_evt.clear()
        self._goal_started_evt.clear()
        self._goal_rejected_evt.clear()
        self._last_result_success = False
        self._last_result_message = ""
        if (
            self._mode_client is not None
            and str(self.get_parameter("control_mode").value) == "teleop"
            and not self._admit_teleop_episode()
        ):
            self._last_result_message = "Managed teleop admission failed; enter a new Prompt to try again."
            self.get_logger().error(self._last_result_message)
            self._goal_rejected_evt.set()
            self._episode_finished_evt.set()
            return False
        if self._should_reset_before_episode() and not self.prepare_new_episode():
            self._last_result_message = "inference session restart failed"
            self._goal_rejected_evt.set()
            self._episode_finished_evt.set()
            return False

        goal_msg = RecordEpisode.Goal()
        goal_msg.prompt = prompt_text

        self.get_logger().info(f"Sending goal with prompt: '{prompt_text}'")

        # We don't block here so the user can cancel it
        with self._goal_lock:
            if self._closing:
                return False
            self._goal_pending = True
            try:
                self._send_goal_future = self._action_client.send_goal_async(
                    goal_msg, feedback_callback=self.feedback_callback
                )
                self._admission_dirty = False
            except Exception:
                self._goal_pending = False
                raise
            self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        with self._goal_lock:
            self._goal_pending = False
            try:
                goal_handle = future.result()
            except Exception as exc:
                self.get_logger().error(f"Goal submission failed: {exc}")
                self._goal_rejected_evt.set()
                self._episode_finished_evt.set()
                return
            if not goal_handle.accepted:
                self.get_logger().warning("Goal rejected by server (Is it already recording?)")
                self._goal_rejected_evt.set()
                self._episode_finished_evt.set()
                return
            self._owned_goal = goal_handle
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)
            if self._cancel_requested:
                self.cancel_recording()

        self._goal_started_evt.set()
        self.get_logger().info("🔴 RECORDING STARTED.")
        print("Controls while recording:")
        print("  Enter       stop and review")
        print("  d + Enter   stop, discard, then return to Prompt")
        print("  r + Enter   stop, discard, then retry same prompt")

    def get_result_callback(self, future):
        try:
            result_wrapper = future.result()
            result = result_wrapper.result
            self._last_result_success = bool(result.success)
            self._last_result_message = str(result.message)
            if result.success:
                self.get_logger().info(f"✅ RECORDING FINALIZED: {result.message}")
            else:
                self.get_logger().info(f"⚠️  RECORDING CANCELLED/ENDED: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Action failed to get result: {e}")

        print("\n----------------------------------------")
        print("Episode finalized! Ready for next episode.")
        with self._goal_lock:
            self._owned_goal = None
            self._episode_finished_evt.set()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Optional: Print progress on the same line
        sys.stdout.write(f"\r[Time Left: {feedback.seconds_remaining}s] {feedback.feedback_message}   ")
        sys.stdout.flush()

    def cancel_recording(self):
        with self._goal_lock:
            # Retain intent while send_goal_async is awaiting acceptance.
            self._cancel_requested = True
            if self._owned_goal is not None and self._cancel_future is None:
                self._cancel_future = self._owned_goal.cancel_goal_async()
                self._cancel_future.add_done_callback(self._cancel_response_callback)

    def _cancel_response_callback(self, future):
        try:
            response = future.result()
            if response.goals_canceling:
                self.get_logger().info("Recording stop accepted; waiting for bag finalization.")
            else:
                self.get_logger().warning("Owned goal cancel was not accepted; waiting for its result.")
        except Exception as e:
            self.get_logger().error(f"Owned goal cancel failed: {e}")

    def finish_before_shutdown(self):
        """Stop our teleop session promptly while the recorder finalizes, within a bound."""
        deadline = time.monotonic() + float(self.get_parameter("shutdown_timeout_sec").value)
        with self._goal_lock:
            self._closing = True
            pending = self._owned_goal is not None or self._goal_pending
            self.cancel_recording()
        # Bag flush/metadata work can outlast the entire shutdown budget. It
        # must never prevent sending HOLD for the session this CLI armed.
        if self._admission_dirty or self._teleop_session_owned:
            self._release_owned_teleop(deadline)
        if pending and not self._episode_finished_evt.wait(max(0.0, deadline - time.monotonic())):
            self.get_logger().error(
                "Shutdown timed out awaiting owned goal result; bag finalization is unconfirmed. "
                "The recorder may still be flushing."
            )

    def _release_owned_teleop(self, deadline: float) -> None:
        """Stop the managed teleop session this CLI admitted, then restore idle.

        Only the session whose rearm this CLI requested is stopped; a CLI that
        merely records an externally armed session leaves it untouched.
        """
        # A service future cannot cancel a remote transition. Drain it before
        # requesting stop/idle, otherwise a late rearm can overtake cleanup.
        if self._admission_future is not None:
            if not self._wait_for_future(self._admission_future, max(0.0, deadline - time.monotonic())):
                self.get_logger().error(
                    "Shutdown timed out draining admission; runtime mode is unknown and needs operator stop."
                )
                return
            try:
                response = self._admission_future.result()
            except Exception:
                response = None
            # A rearm that completed after our timeout may have armed late;
            # an explicit late refusal proves no session exists.
            if (
                response is not None
                and self._admission_client is not None
                and self._admission_client is self._rearm_client
            ):
                self._teleop_session_owned = bool(response.success)
        if self._teleop_session_owned:
            if self._teleop_stop_client is None:
                self.get_logger().error(
                    "Owned teleop session has no configured stop service; runtime needs operator stop."
                )
                return
            stop_confirmed = False
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                if not self._teleop_stop_client.wait_for_service(timeout_sec=min(0.5, remaining)):
                    break
                try:
                    future = self._teleop_stop_client.call_async(Trigger.Request())
                    if self._wait_for_future(future, remaining):
                        response = future.result()
                        if response is not None and response.success:
                            stop_confirmed = True
                            break
                except Exception as exc:
                    self.get_logger().warning(f"Managed teleop stop request failed: {exc}")
                    break
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            if not stop_confirmed:
                self.get_logger().error("Managed teleop HOLD was not confirmed; runtime needs operator stop.")
                return
            self._teleop_session_owned = False
            self.get_logger().info("Managed teleop stopped (HOLD confirmed).")
        # HOLD leaves the runtime latched. Admission only ever armed from a
        # freshly observed ACTIVE/idle/unlatched state, so requesting idle
        # restores exactly that; a pre-existing stop or fault is never cleared
        # because admission could not have succeeded from such a state.
        try:
            remaining = max(0.0, deadline - time.monotonic())
            future = self._mode_client.call_async(SetRuntimeMode.Request(mode="idle"))
            if not self._wait_for_future(future, remaining) or future.result() is None or not future.result().success:
                self.get_logger().error("Admission cleanup could not restore idle; runtime needs operator stop.")
                return
        except Exception as exc:
            self.get_logger().error(f"Admission idle cleanup failed: {exc}")
            return
        self._admission_dirty = False

    def _status_callback(self, status):
        self._runtime_status = status
        self._runtime_status_received = time.monotonic()
        self._runtime_status_event.set()

    def configure_from_recorder_info(self, info):
        """Discover managed admission only when the recorder explicitly advertises it."""
        config = info.get("teleop_admission")
        if self._mode_client is not None or not isinstance(config, dict) or not config.get("runtime_set_mode_service"):
            return
        names = (
            "runtime_set_mode_service",
            "teleop_rearm_service",
            "teleop_stop_service",
            "runtime_status_topic",
            "admission_timeout_sec",
            "admission_attempts",
        )
        values = dict(config)
        # Older recorders may not advertise a stop endpoint; an empty value
        # leaves the owned-session stop path unconfigured.
        values.setdefault("teleop_stop_service", "")
        self.set_parameters([Parameter(name, value=values[name]) for name in names])
        self.set_parameters([Parameter("control_mode", value="teleop")])
        self._mode_client = self.create_client(SetRuntimeMode, config["runtime_set_mode_service"])
        self._rearm_client = self.create_client(Trigger, config["teleop_rearm_service"])
        stop_service = str(config.get("teleop_stop_service", ""))
        self._teleop_stop_client = self.create_client(Trigger, stop_service) if stop_service else None
        self._status_sub = self.create_subscription(
            RuntimeStatus, config["runtime_status_topic"], self._status_callback, 1
        )
        self.get_logger().info("Discovered managed teleop episode admission from recorder.")

    def _admit_teleop_episode(self) -> bool:
        """An explicit Prompt owns one bounded idle/rearm/status admission sequence."""
        if self._admission_future is not None and not self._admission_future.done():
            self.get_logger().error("Previous mode/rearm request is unresolved; admission is blocked.")
            return False
        timeout = max(0.01, float(self.get_parameter("admission_timeout_sec").value))
        attempts = min(3, max(1, int(self.get_parameter("admission_attempts").value)))
        self._admission_dirty = True
        for attempt in range(attempts):
            self.get_logger().info(f"Teleop admission {attempt + 1}/{attempts}: idle then rearm")
            admitted = True
            for client, request in (
                (self._mode_client, SetRuntimeMode.Request(mode="idle")),
                (self._rearm_client, Trigger.Request()),
            ):
                if not client.wait_for_service(timeout_sec=timeout):
                    admitted = False
                    break
                try:
                    if client is self._rearm_client:
                        # A service request cannot be recalled: claim ownership
                        # before sending, so an ambiguous timeout still stops
                        # the session on shutdown. An observed refusal clears
                        # it again because no session was armed.
                        self._teleop_session_owned = True
                    self._admission_future = client.call_async(request)
                    self._admission_client = client
                    if not self._wait_for_future(self._admission_future, timeout):
                        self.get_logger().error(
                            "Mode/rearm timed out; runtime mode is unknown. "
                            "No retry until the request settles; verify runtime state or stop the robot."
                        )
                        return False
                    response = self._admission_future.result()
                    if response is None or not response.success:
                        if client is self._rearm_client and response is not None:
                            # The mapper explicitly refused; no session was armed.
                            self._teleop_session_owned = False
                        self.get_logger().warning(
                            f"Admission request failed: {getattr(response, 'message', 'no response')}"
                        )
                        admitted = False
                        break
                except Exception as exc:
                    self.get_logger().error(f"Mode/rearm failed: {exc}")
                    return False
                expected_mode = "idle" if client is self._mode_client else "stream"
                if not self._wait_runtime_mode(expected_mode, timeout):
                    self.get_logger().warning(
                        f"No fresh ACTIVE {expected_mode} unlatched status after admission request."
                    )
                    admitted = False
                    break
            if not admitted:
                continue
            return True
        return False

    def _wait_runtime_mode(self, mode, timeout):
        after = time.monotonic()
        after_ros_ns = self.get_clock().now().nanoseconds
        deadline = after + timeout
        while time.monotonic() < deadline:
            self._runtime_status_event.clear()
            status = self._runtime_status
            now_ros_ns = self.get_clock().now().nanoseconds
            if status is not None:
                stamp_ns = status.stamp.sec * 1_000_000_000 + status.stamp.nanosec
                if (
                    self._runtime_status_received > after
                    and after_ros_ns <= stamp_ns <= now_ros_ns
                    and now_ros_ns - stamp_ns <= int(timeout * 1e9)
                    and status.lifecycle == "ACTIVE"
                    and status.active_mode == mode
                    and not status.stop_latched
                ):
                    return True
            self._runtime_status_event.wait(max(0.0, deadline - time.monotonic()))
        return False

    def prepare_new_episode(self) -> bool:
        """Best-effort reset so a new episode reuses clean inference/dispatcher state.

        When the inference scheduler is enabled, record_cli must call the
        ScheduledActionDispatcher restart_session service (safe-stop + Close
        old session + Open new UUID) and must NOT call the direct policy reset
        fallback, so it cannot bypass the Close barrier. When the scheduler is
        disabled/absent, the legacy dispatcher-reset -> policy-reset chain is kept.
        The scheduled endpoint must be passed explicitly by a scheduler-enabled
        launch. Its empty default preserves the legacy call sequence and timing.
        """
        if self._restart_session_client is not None:
            if self._restart_session():
                return True
            # restart_session failed: do not fall through to direct policy reset
            # on the scheduled path (would bypass Close). Surface the failure.
            self.get_logger().error("restart_session failed; not falling back to direct policy reset on scheduled path")
            return False
        # legacy/disabled branch
        if self._reset_dispatcher_state():
            return True
        self._reset_policy_state()
        return True

    def _restart_session(self) -> bool:
        assert self._restart_session_client is not None
        if not self._restart_session_client.wait_for_service(timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("restart_session service disappeared mid-call")
            return False
        try:
            future = self._restart_session_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"restart_session call failed: {e}")
            return False
        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("restart_session call timed out.")
            return False
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"restart_session call failed: {e}")
            return False
        if response is not None and response.success:
            self.get_logger().info("Inference session restarted for new episode.")
            return True
        message = response.message if response is not None else ""
        self.get_logger().warning(f"restart_session failed. {message}")
        return False

    def _should_reset_before_episode(self) -> bool:
        override = str(self.get_parameter("reset_before_episode").value).strip().lower()
        if override in {"true", "1", "yes", "on"}:
            return True
        if override in {"false", "0", "no", "off"}:
            return False
        return str(self.get_parameter("control_mode").value).strip() == "model_inference"

    def _reset_dispatcher_state(self) -> bool:
        service_name = self.get_parameter("dispatcher_reset_service").value
        if not service_name:
            return False
        if not self._dispatcher_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(f"Dispatcher reset service unavailable: {service_name}")
            return False

        try:
            future = self._dispatcher_reset_client.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Dispatcher reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if response is not None:
            self.get_logger().info("Dispatcher state reset for new episode.")
            return True

        self.get_logger().warning("Dispatcher reset call failed.")
        return False

    def _reset_policy_state(self) -> bool:
        service_name = self.get_parameter("policy_reset_service").value
        if not service_name:
            return False
        if not self._policy_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(f"Policy reset service unavailable: {service_name}")
            return False

        try:
            future = self._policy_reset_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Policy reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if response is not None and response.success:
            self.get_logger().info("Policy runtime state reset for new episode.")
            return True

        message = response.message if response is not None else ""
        self.get_logger().warning(f"Policy reset call failed. {message}")
        return False

    def delete_last_episode(self, timeout_sec: float = 5.0) -> bool:
        """Ask the recorder server to delete the last finalized episode."""
        if not self._delete_last_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Delete service unavailable: record_episode/delete_last")
            return False
        try:
            future = self._delete_last_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Delete service call failed: {e}")
            return False
        if not self._wait_for_future(future, timeout_sec=timeout_sec):
            self.get_logger().warning("Delete service call timed out.")
            return False
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Delete service call failed: {e}")
            return False
        if response is None:
            self.get_logger().warning("Delete service returned no response.")
            return False
        if response.success:
            info = _parse_json(response.message) or {}
            episode_dir = info.get("episode_dir", response.message)
            dataset_root = info.get("dataset_root", "")
            print(f"🗑️  DISCARDED {episode_dir}")
            if dataset_root:
                print(f"📁 Dataset remains: {dataset_root}")
            return True
        self.get_logger().warning(f"Delete service refused: {response.message}")
        return False

    def get_last_episode_info(self, timeout_sec: float = 3.0) -> dict | None:
        """Return the last finalized episode info from the recorder server."""
        if not self._last_episode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Last episode service unavailable: record_episode/get_last_episode")
            return None
        try:
            future = self._last_episode_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Last episode service call failed: {e}")
            return None
        if not self._wait_for_future(future, timeout_sec=timeout_sec):
            self.get_logger().warning("Last episode service call timed out.")
            return None
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Last episode service call failed: {e}")
            return None
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            self.get_logger().warning(f"Last episode unavailable: {message}")
            return None
        info = _parse_json(response.message)
        if info is None:
            self.get_logger().warning(f"Last episode response was not valid JSON: {response.message}")
            return None
        return info

    @staticmethod
    def _wait_for_future(future, timeout_sec: float) -> bool:
        """Wait for a future while the node is already spun by the executor thread."""
        if future.done():
            return True
        done_event = threading.Event()
        future.add_done_callback(lambda _future: done_event.set())
        if future.done():
            return True
        return done_event.wait(timeout=timeout_sec)


def _parse_json(value: str) -> dict | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _recording_command(raw: str) -> str:
    cmd = raw.strip().lower()
    if cmd in {"", "q", "quit"}:
        return "review"
    if cmd in {"d", "discard"}:
        return "discard"
    if cmd in {"r", "retry"}:
        return "retry"
    print(f"⚠️  Unknown recording command '{raw}'. Stopping for review.")
    return "review"


def _review_choice(raw: str) -> str | None:
    cmd = raw.strip().lower()
    if cmd in {"", "s", "save"}:
        return "save"
    if cmd in {"d", "discard"}:
        return "discard"
    if cmd in {"r", "retry"}:
        return "retry"
    if cmd in {"q", "quit"}:
        return "quit_keep"
    return None


def _episode_label(info: dict | None) -> str:
    if not info:
        return "episode"
    return str(info.get("episode_name") or info.get("episode_dir") or "episode")


def _print_episode_review(info: dict | None, prompt: str) -> None:
    print("\n========================================")
    print("Episode stopped and finalized.")
    if info:
        print(f"Dataset: {info.get('dataset_root', 'Unknown')}")
        print(f"Episode: {info.get('episode_dir', 'Unknown')}")
        print(f"Messages written: {info.get('messages', 'Unknown')}")
        print(f"Prompt: {info.get('prompt') or prompt}")
    else:
        print("⚠️  Episode path unavailable from recorder server.")
        print(f"Prompt: {prompt}")
    print("")
    print("[s] save / [d] discard / [r] discard and retry / [q] quit and keep")


def _print_kept_episode(info: dict | None) -> None:
    print(f"✅ KEPT {_episode_label(info)}")
    if info and info.get("episode_dir"):
        print(f"📁 {info['episode_dir']}")


def _wait_for_recording_to_start(node: RecordCLI, timeout_sec: float = 5.0) -> bool:
    if node._goal_started_evt.wait(timeout=timeout_sec):
        return True
    if node._goal_rejected_evt.is_set():
        return False
    node.get_logger().warning("Timed out waiting for recorder goal acceptance.")
    node.cancel_recording()
    return False


def _finish_episode_or_warn(node: RecordCLI) -> bool:
    node.cancel_recording()
    if node._episode_finished_evt.wait(timeout=15.0):
        return True
    node.get_logger().warning(
        "Timed out waiting for recorder to finish. The recording server may have crashed — check its logs."
    )
    print(
        "⚠️  Recorder finalization did not complete in time. "
        "Discard/retry is disabled for this episode because writer state is unknown."
    )
    return False


def cli_loop(node):
    """Run the interactive prompt in a separate thread."""

    print("\nFetching dataset configuration from server...")
    if node._info_client.wait_for_service(timeout_sec=3.0):
        future = node._info_client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        if future.done() and future.result() is not None:
            try:
                import json

                info = json.loads(future.result().message)
                node.configure_from_recorder_info(info)
                path = info.get("path", "Unknown")
                count = info.get("episodes", 0)

                print("\n========================================")
                print("📊 DATASET TARGET INFO")
                print("========================================")
                print(f"📁 Path: {path}")
                if count > 0:
                    print(f"⚠️  Found {count} existing episodes in this directory.")
                    print("   New recordings will be APPENDED to this dataset.")
                else:
                    print("✨ New dataset directory. No existing data found.")
                print("")
                print("💡 Tip: To change the dataset name, restart the launch server with:")
                print("   ros2 launch ... dataset_name:=<new_name> bag_base_dir:=<custom_dir>")
                print("========================================")

                ans = input("\nPress Enter to CONFIRM and continue, or 'q' to quit > ")
                if ans.strip().lower() in ["q", "quit"]:
                    return
            except Exception as e:
                print(f"Failed to parse server info: {e}")
    else:
        print("⚠️  Warning: Could not fetch dataset info from server (timeout).")

    last_prompt = "default_task"

    while rclpy.ok():
        node._episode_finished_evt.clear()

        print("\n========================================")
        print("Dataset Collection CLI")
        print(f"Enter prompt text to start recording. (Press Enter to reuse: '{last_prompt}')")
        print("Type 'q' or 'quit' to exit.")
        print("========================================")

        try:
            prompt = input("Prompt > ")
            if prompt.strip().lower() in ["q", "quit"]:
                print("Exiting...")
                break

            if not prompt.strip():
                prompt = last_prompt
            else:
                last_prompt = prompt.strip()

            while rclpy.ok():
                node._episode_finished_evt.clear()
                if node.send_goal(prompt) is False:
                    break
                if not _wait_for_recording_to_start(node):
                    break

                recording_action = _recording_command(input())
                if not _finish_episode_or_warn(node):
                    break

                last_info = node.get_last_episode_info()
                can_modify_episode = node._last_result_success and last_info is not None
                if node._last_result_success and last_info is None:
                    print("⚠️  Recorder finalized, but episode details are unavailable; discard/retry is disabled.")
                elif not node._last_result_success:
                    print(
                        "⚠️  Recording did not finalize successfully; discard/retry is disabled "
                        "to avoid deleting an earlier episode."
                    )

                if recording_action == "discard":
                    if can_modify_episode:
                        node.delete_last_episode()
                    else:
                        print("⚠️  No current finalized episode is available to discard.")
                    break
                if recording_action == "retry":
                    if can_modify_episode and node.delete_last_episode():
                        print(f"🔁 Retrying with same prompt: {prompt}")
                        continue
                    if not can_modify_episode:
                        print("⚠️  No current finalized episode is available to discard before retry.")
                    break

                _print_episode_review(last_info, prompt)
                while rclpy.ok():
                    choice = _review_choice(input("Choice [s] > "))
                    if choice is None:
                        print("Unknown choice. Use Enter/s=save, d=discard, r=retry, q=quit and keep.")
                        continue
                    if choice == "save":
                        _print_kept_episode(last_info)
                        break
                    if choice == "discard":
                        if can_modify_episode:
                            node.delete_last_episode()
                        else:
                            print("⚠️  No current finalized episode is available to discard.")
                        break
                    if choice == "retry":
                        if can_modify_episode and node.delete_last_episode():
                            print(f"🔁 Retrying with same prompt: {prompt}")
                            break
                        if not can_modify_episode:
                            print("⚠️  No current finalized episode is available to discard before retry.")
                        choice = "discard_failed"
                        break
                    if choice == "quit_keep":
                        _print_kept_episode(last_info)
                        print("Exiting...")
                        return

                if choice == "retry":
                    continue
                break

        except EOFError:
            break
        except Exception:
            traceback.print_exc()


def main(args=None):
    # Python's SIGINT handler raises KeyboardInterrupt without invalidating ROS.
    previous_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RecordCLI()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Run ROS spinning in a background thread so input() doesn't block callbacks
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()

    try:
        # Run the interactive CLI in the main thread
        cli_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl+C must not interrupt writer cleanup halfway through.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.finish_before_shutdown()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
