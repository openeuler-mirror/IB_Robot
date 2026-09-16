"""VR dual-arm teleoperation node.

Receives left/right controller data from a Unity XR application via TCP and
drives one of two output profiles: ``so101`` (single-arm placo Cartesian
servo, relative-pose (clutch) passthrough) or ``humanoid`` (dual-arm differential
velocities published as ``Vector3Stamped`` to the ``/humanoid_teleop/*``
topics). The active profile is selected by the ``output_profile`` parameter.

Coordinate conversion
---------------------
Unity uses a left-handed coordinate system (X=right, Y=up, Z=forward).
ROS uses a right-handed coordinate system (X=forward, Y=left, Z=up).

Position uses a MIRROR basis (det -1) by design: face-to-face teleop makes the
intuitive translation mapping a reflection. Measured on sim (hand -> EE),
implemented by ``_POS_UNITY_TO_ROS``::

    ROS_X (forward) = -Unity_X
    ROS_Y (left)    = -Unity_Z
    ROS_Z (up)      =  Unity_Y

    -> [x, y, z]_unity  ->  [-x, -z, y]_ros   (mirror, det -1)

Orientation uses its OWN proper basis (det +1) — a rotation cannot be a mirror
or wrist attitude scrambles. Implemented by ``_ORI_UNITY_TO_ROS`` as a
change-of-basis on the quaternion vector part (w unchanged)::

    q_unity = (qx, qy, qz, qw)
    q_ros   = (qx, -qz, qy, qw)

See ``_POS_UNITY_TO_ROS`` / ``_ORI_UNITY_TO_ROS`` for the exact matrices and the
rationale for the position/orientation split.

Velocity output convention
--------------------------
Linear (base frame, ``base``)::

    x = forward/backward
    y = left/right
    z = up/down

Angular is computed in the tool frame. For the ``so101`` output profile the
angular channel is converted tool→base via :class:`ToolAngularAdapter` before
publishing, because the ``placo_servo`` QP orientation task expects a
base-frame angular velocity (it controls a true Cartesian orientation). The
``humanoid`` profile keeps publishing raw tool-frame angular to its dedicated
``angular_cmd_tool`` topics.

Typical speed magnitudes: linear ~0.02 m/s, angular ~1.0 rad/s.
"""

import contextlib
import json
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from ibrobot_msgs.action import ArmReturnHome

from .cartesian_backend.frame_adapter import ToolAngularAdapter
from .vr_rotation import compute_base_rotation_delta, remap_base_rotation

logger = logging.getLogger(__name__)


# POSITION mirror basis (det -1). Face-to-face teleop makes the intuitive
# translation mapping a mirror by nature. Measured on sim (hand -> EE):
# hand-forward -> EE-back, hand-right -> EE-left, hand-up -> EE-up. So x and y
# both flip, z stays: ros = [-ux, -uz, uy]. This is a reflection (det -1) no
# proper rotation can produce, which is expected for position; orientation gets
# its OWN proper basis below (_ORI_UNITY_TO_ROS).
_POS_UNITY_TO_ROS = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # det -1 (mirror, by design)

# ORIENTATION proper basis (det +1). Must map through a proper basis or wrist
# attitude scrambles. Aligned to the SAME face-to-face orientation as position
# (position's mirror = this rotation composed with a reflection), so wrist turns
# feel consistent with translation. ros_quat_vec = [qx, -qz, qy], w unchanged.
_ORI_UNITY_TO_ROS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # det +1


def _unity_pos_to_ros(pos_unity: np.ndarray) -> np.ndarray:
    #   ros_x = -unity_x (forward, mirrored)
    #   ros_y = -unity_z (left/right, mirrored)
    #   ros_z = +unity_y (up)
    return _POS_UNITY_TO_ROS @ np.asarray(pos_unity, dtype=float)


def _unity_quat_to_ros(rot_unity: Rotation) -> Rotation:
    # quaternion vector part transforms as M @ [x,y,z], w unchanged.
    xyzw = rot_unity.as_quat()
    v = _ORI_UNITY_TO_ROS @ xyzw[:3]
    return Rotation.from_quat([v[0], v[1], v[2], xyzw[3]])


@dataclass
class _ControllerData:
    position: np.ndarray
    rotation: Rotation
    grip_value: float = 0.0
    trigger_value: float = 0.0
    thumbstick: np.ndarray = field(default_factory=lambda: np.zeros(2))
    primary_button: bool = False
    secondary_button: bool = False
    enabled: bool = False


@dataclass
class _DualArmVRData:
    timestamp: float = 0.0
    left: _ControllerData | None = None
    right: _ControllerData | None = None
    headset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    headset_rotation: Rotation = field(default_factory=Rotation.identity)
    config_mode: bool = False


class VRDualArmTcpServer:
    """TCP server that receives JSON packets from Unity VR dual-controller sender.

    Packet format (newline-delimited JSON)::

        {
          "timestamp": 12.345,
          "left_controller":  { "position": [...], "rotation": [...], ... },
          "right_controller": { "position": [...], "rotation": [...], ... },
          "headset":          { "position": [...], "rotation": [...] },
          "config_mode": false
        }

    Runs a background thread to accept one client at a time.
    """

    _logged_ctrl_keys = False  # one-shot probe of controller JSON keys (TEMP)

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._latest: _DualArmVRData | None = None
        # Monotonic receive time of _latest. None until the first packet. Used by
        # the control loop to detect a stalled-but-connected client (TCP alive,
        # no fresh frames) so it does not republish one frozen frame forever.
        self._latest_recv_monotonic: float | None = None
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._connected_addr = ""

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self._host, self._port))
            sock.listen(1)
            sock.settimeout(1.0)
        except OSError:
            sock.close()
            self._server_sock = None
            self._running = False
            raise

        self._server_sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            with contextlib.suppress(OSError):
                self._server_sock.close()
            self._server_sock = None
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_latest(self) -> _DualArmVRData | None:
        with self._lock:
            return self._latest

    def get_latest_age_s(self) -> float | None:
        """Seconds since the last packet was stored, or None if none yet / the
        client disconnected. Lets the control loop distinguish a live stream
        from a frozen-but-connected client that stopped sending frames."""
        with self._lock:
            if self._latest_recv_monotonic is None:
                return None
            return time.monotonic() - self._latest_recv_monotonic

    @property
    def is_connected(self) -> bool:
        return self._connected

    def consume_connection_event(self) -> str | None:
        """Return and clear pending connection change event."""
        if self._connected and self._connected_addr:
            addr = self._connected_addr
            self._connected_addr = ""
            return f"connected {addr}"
        if not self._connected and self._connected_addr:
            self._connected_addr = ""
            return "disconnected"
        return None

    def _log(self, msg: str, level: str = "info") -> None:
        getattr(logger, level)(msg)

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
                self._connected = True
                self._connected_addr = f"{addr[0]}:{addr[1]}"
                self._log(f"\n{'=' * 60}\n  VR CLIENT CONNECTED from {addr[0]}:{addr[1]}\n{'=' * 60}")
                conn.settimeout(0.5)
                # _receive_loop owns per-connection cleanup in its own finally
                # (clears _latest + connected state) so an EXCEPTION path here —
                # e.g. a malformed-but-parseable packet raising AttributeError —
                # cannot leave the node believing a dead client is still online
                # and reusing the last control frame.
                self._receive_loop(conn, addr)
                self._log(
                    f"\n{'=' * 60}\n"
                    f"  VR CLIENT DISCONNECTED ({addr[0]}:{addr[1]})\n"
                    f"  Waiting for new connection...\n"
                    f"{'=' * 60}"
                )
            except TimeoutError:
                continue
            except OSError:
                break
            except Exception as exc:  # noqa: BLE001
                # Never let an unexpected error kill the accept loop — the
                # server would silently stop accepting reconnections. Log and
                # keep serving.
                self._log(f"VRTcpServer: accept loop error: {exc}", "error")
                continue

    # Drop the receive buffer if it grows past this without a newline. A
    # well-formed VR packet is well under 4 KiB; a buffer larger than this
    # with no delimiter means a misbehaving/hostile client streaming
    # newline-less data, which would otherwise grow memory unbounded.
    _MAX_BUFFER_BYTES = 1 << 20  # 1 MiB

    def _receive_loop(self, conn: socket.socket, addr) -> None:
        buf = ""
        try:
            while self._running:
                try:
                    chunk = conn.recv(4096)
                except TimeoutError:
                    continue
                if not chunk:
                    self._log("VR client disconnected", "warning")
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._parse_packet(line.encode("utf-8"))
                if len(buf) > self._MAX_BUFFER_BYTES:
                    self._log(
                        "VRTcpServer: receive buffer exceeded "
                        f"{self._MAX_BUFFER_BYTES} bytes without a newline; "
                        "dropping buffered data",
                        "warning",
                    )
                    buf = ""
        finally:
            # Single owner of per-connection teardown. Runs on EVERY exit path —
            # clean disconnect, socket error, or an unexpected exception from
            # packet parsing — so the node never keeps serving a stale _latest
            # for a client that is really gone.
            with self._lock:
                self._latest = None
                self._latest_recv_monotonic = None
            self._connected = False
            self._connected_addr = f"{addr[0]}:{addr[1]}"
            with contextlib.suppress(OSError):
                conn.close()

    def _parse_controller(self, ctrl: dict) -> _ControllerData:
        pos_unity = np.array(ctrl.get("position", [0.0, 0.0, 0.0]), dtype=float)
        rot_list = ctrl.get("rotation", [0.0, 0.0, 0.0, 1.0])
        rot_unity = Rotation.from_quat(rot_list)
        pos_ros = _unity_pos_to_ros(pos_unity)
        rot_ros = _unity_quat_to_ros(rot_unity)
        # B-button field-name compatibility: different VR clients send the
        # secondary (B/Y) button under different keys. Accept any of them so
        # go-home works regardless of the client's naming. TEMP probe below
        # logs the actual controller keys once so we can confirm the name.
        secondary = bool(
            ctrl.get(
                "secondaryButton",
                ctrl.get("buttonB", ctrl.get("button_b", ctrl.get("secondary_button", ctrl.get("bButton", False)))),
            )
        )
        if not VRDualArmTcpServer._logged_ctrl_keys:
            VRDualArmTcpServer._logged_ctrl_keys = True
            logger.warning(f"CTRL KEYS PROBE: {sorted(ctrl.keys())}")
        return _ControllerData(
            position=pos_ros,
            rotation=rot_ros,
            grip_value=float(ctrl.get("grip_value", 0.0)),
            trigger_value=float(ctrl.get("trigger_value", 0.0)),
            thumbstick=np.array(ctrl.get("thumbstick", [0.0, 0.0]), dtype=float),
            primary_button=bool(ctrl.get("primaryButton", False)),
            secondary_button=secondary,
            enabled=bool(ctrl.get("enabled", False)),
        )

    def _parse_packet(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._log(f"VRTcpServer: bad JSON packet: {e}", "debug")
            return

        # A syntactically valid JSON value that is not an object (e.g. "[]",
        # "5", "\"x\"") would sail past json.loads and then raise AttributeError
        # at msg.get(...) below — which is NOT a (ValueError, TypeError). Drop it
        # here so one odd packet is a no-op instead of tearing down the socket.
        if not isinstance(msg, dict):
            self._log("VRTcpServer: packet is not a JSON object; dropping", "debug")
            return

        # A single malformed packet (bad quaternion, wrong array length,
        # non-numeric field) raises ValueError/TypeError from numpy or
        # scipy. Catch it here so one bad packet is dropped rather than
        # killing the receive thread (which never reconnects).
        try:
            left = None
            right = None
            # A controller field that is present but not a JSON object (e.g.
            # right_controller=[]) would raise AttributeError inside
            # _parse_controller (list has no .get) — which is NOT a
            # (ValueError, TypeError). Treat a non-dict controller as absent so
            # one odd field is a no-op, not a torn-down socket.
            lc = msg.get("left_controller")
            if isinstance(lc, dict):
                left = self._parse_controller(lc)
            rc = msg.get("right_controller")
            if isinstance(rc, dict):
                right = self._parse_controller(rc)

            # headset=[] would likewise raise AttributeError at hs.get below.
            # Fall back to an empty dict (default pose) for any non-dict headset.
            hs = msg.get("headset")
            if not isinstance(hs, dict):
                hs = {}
            hs_pos_unity = np.array(hs.get("position", [0.0, 0.0, 0.0]), dtype=float)
            hs_rot_list = hs.get("rotation", [0.0, 0.0, 0.0, 1.0])
            hs_pos_ros = _unity_pos_to_ros(hs_pos_unity)
            hs_rot_ros = _unity_quat_to_ros(Rotation.from_quat(hs_rot_list))

            data = _DualArmVRData(
                timestamp=float(msg.get("timestamp", 0.0)),
                left=left,
                right=right,
                headset_position=hs_pos_ros,
                headset_rotation=hs_rot_ros,
                config_mode=bool(msg.get("config_mode", False)),
            )
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self._log(f"VRTcpServer: malformed packet fields: {e}", "debug")
            return
        with self._lock:
            self._latest = data
            self._latest_recv_monotonic = time.monotonic()


@dataclass
class _ArmState:
    calib_pos: np.ndarray | None = None
    calib_rot: Rotation | None = None
    prev_pos: np.ndarray | None = None
    prev_rot: Rotation | None = None
    enabled_prev: bool = False
    ema_linear: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ema_angular: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grip_prev: float = 0.0


class VRTeleopNode(Node):
    """ROS 2 node for VR dual-arm teleoperation."""

    def __init__(self):
        super().__init__("vr_teleop")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8889)
        self.declare_parameter("managed_teleop", False)
        self.declare_parameter("command_lease_topic", "")
        self.declare_parameter("gripper_joint_name", "")
        self.declare_parameter("joint_limits", "{}", ParameterDescriptor(read_only=True))
        self.declare_parameter("runtime_status_topic", "/runtime/status")
        self._managed = bool(self.get_parameter("managed_teleop").value)
        self._managed_release_required = True
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("linear_speed_scale", 1.0)
        self.declare_parameter("angular_speed_scale", 1.0)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 6.0)
        self.declare_parameter("velocity_ema_alpha", 1.0)
        self.declare_parameter("linear_deadzone", 0.005)
        self.declare_parameter("angular_deadzone", 0.05)
        self.declare_parameter("output_profile", "humanoid")
        self.declare_parameter("controller_side", "right")
        self.declare_parameter("linear_frame_id", "base_link_0")
        self.declare_parameter("angular_frame_id", "L5_origin")
        # so101 profile downstream wiring: default to the placo servo node.
        self.declare_parameter("so101_linear_topic", "/so101_placo_servo_node/linear_cmd_base")
        self.declare_parameter("so101_angular_topic", "/so101_placo_servo_node/angular_cmd_base")
        self.declare_parameter("so101_gripper_topic", "/gripper_position_controller/commands")
        self.declare_parameter("so101_start_service", "/so101_placo_servo_node/start")
        self.declare_parameter("so101_stop_service", "/so101_placo_servo_node/stop")
        self.declare_parameter("so101_home_action", "/so101_placo_servo_node/return_home")
        self.declare_parameter("estop_topic", "/emergency_stop")
        self.declare_parameter("so101_gripper_open", 1.0)
        self.declare_parameter("so101_gripper_closed", 0.0)
        # so101 input mode. "velocity" (default): publish tool-frame differential
        # twist to linear/angular topics (placo integrates). "pose": publish a
        # RELATIVE (clutch) pose command to so101_pose_topic — placo adds it onto
        # the EE pose it latched at enable (no velocity integration), giving 1:1
        # hand tracking and stop=stop with zero drift. position carries the
        # relative hand displacement scaled by position_scale; orientation
        # carries the relative base-frame rotation delta (identity at press).
        self.declare_parameter("so101_input_mode", "velocity")
        self.declare_parameter("so101_pose_topic", "/so101_placo_servo_node/pose_cmd_base")
        # Stale-frame watchdog (seconds). If the client stays TCP-connected but
        # stops sending frames (frozen app, stalled render thread), the server
        # keeps returning the same _latest forever and the control loop would
        # republish that frozen pose every tick — placo's own
        # incoming_command_timeout never fires because each republish refreshes
        # its receive time. When the newest frame is older than this, treat it as
        # no data: drop the clutch and call the placo stop service on the edge so
        # the arm actually stops instead of chasing a dead target. ~0.2 s = 10
        # missed frames at 50 Hz.
        self.declare_parameter("so101_command_stale_s", 0.2)
        # Hand→EE position gain for pose mode. 0.4 matches the SO-101 reachable
        # radius (~24 cm) to a comfortable single-arm teleop range (~60 cm).
        self.declare_parameter("position_scale", 0.4)
        # Pose-mode rotation gate. When True the arm holds its clutch-baseline
        # attitude and only position is teleoperated (ΔR sent as identity). The
        # base-frame correction matrix in vr_rotation was fitted under the old
        # body-frame contract; keep this False only once it is validated on the
        # current base-frame contract (default False = rotation ON, verified
        # usable). See vr_rotation.R_ROBOT_BASE_FROM_VR_BASE.
        self.declare_parameter("so101_position_only", False)
        # tool→base angular conversion (so101 profile). base_link/tool_frame
        # name the TF pair used by ToolAngularAdapter; the raw VR angular is in
        # the tool frame and placo expects a base-frame angular velocity.
        self.declare_parameter("base_link_name", "base")
        self.declare_parameter("tool_frame", "gripper")
        self.declare_parameter("tf_stale_threshold_s", 0.2)

        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)
        self._output_profile = str(self.get_parameter("output_profile").value)
        self._controller_side = str(self.get_parameter("controller_side").value)
        self._so101_input_mode = str(self.get_parameter("so101_input_mode").value).lower()
        self._so101_position_only = bool(self.get_parameter("so101_position_only").value)
        self._so101_command_stale_s = float(self.get_parameter("so101_command_stale_s").value)
        # Edge tracker for the stale watchdog: True once we have handled a stall
        # (dropped clutch + issued stop), so the stop service is called once per
        # stall rather than every tick.
        self._so101_stalled = False
        if self._output_profile not in ("humanoid", "so101"):
            raise ValueError("output_profile must be 'humanoid' or 'so101'")
        if self._controller_side not in ("left", "right"):
            raise ValueError("controller_side must be 'left' or 'right'")
        if self._so101_input_mode not in ("velocity", "pose"):
            raise ValueError("so101_input_mode must be 'velocity' or 'pose'")

        self._joint_limits = json.loads(self.get_parameter("joint_limits").value)
        if not isinstance(self._joint_limits, dict):
            raise ValueError("joint_limits must be a JSON object")
        for name, limit in self._joint_limits.items():
            try:
                valid = (
                    bool(name.strip())
                    and isinstance(limit, dict)
                    and all(
                        isinstance(limit.get(key), int | float)
                        and not isinstance(limit[key], bool)
                        and math.isfinite(limit[key])
                        for key in ("min", "max")
                    )
                    and limit["min"] < limit["max"]
                )
            except (OverflowError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"joint_limits[{name!r}] requires finite numeric min < max")
        if self._managed:
            if self._output_profile != "so101":
                raise ValueError("managed_teleop requires output_profile=so101")
            if self.get_parameter("gripper_joint_name").value not in self._joint_limits:
                raise ValueError("joint_limits must include the managed gripper joint")

        self._tcp_server = VRDualArmTcpServer(host, port)
        self._tcp_server.start()

        self._right_linear_pub = None
        self._left_linear_pub = None
        self._right_angular_pub = None
        self._left_angular_pub = None
        self._right_gripper_pub = None
        self._left_gripper_pub = None
        self._so101_linear_pub = None
        self._so101_angular_pub = None
        self._so101_pose_pub = None
        self._so101_gripper_pub = None
        self._so101_start_cli = None
        self._so101_stop_cli = None
        self._so101_home_cli = None
        self._so101_start_inflight = False
        self._so101_stop_inflight = False
        self._so101_recalib_inflight = False
        self._so101_home_inflight = False
        self._so101_home_goal_handle = None
        self._so101_home_cancel_pending = False
        # Latched "the arm MUST be stopped and has not confirmed it" intent.
        # Set whenever a stop is needed but could not be confirmed (service not
        # ready, request failed/rejected, or a start/recalib/home was inflight
        # and would race the stop). While true, every start path is blocked and
        # the watchdog keeps re-issuing stop until the servo acknowledges. Only a
        # confirmed stop response clears it. This is what makes the deadman
        # closed-loop: without it, a single failed stop left the software
        # believing the arm was stopped while placo kept tracking the last target.
        self._so101_stop_pending = False
        # Set by any safety stop or Home dispatch. It blocks the 0.5 s automatic
        # start path until a live, held trigger explicitly re-latches Placo.
        self._so101_reengage_required = False
        self._estop_active = False
        # Latches once the start service has been acknowledged; gates the 0.5 s
        # keepalive/deadman timer (see _so101_deadman_tick). Must exist before that
        # timer first fires: with fresh data and the trigger released, no stop path
        # runs to create it, so initialize it here.
        self._so101_started = False
        self._secondary_prev = False  # B-button edge detect (go-home)
        # Home blocks all arm input until the ArmReturnHome action reaches a terminal
        # result and a real trigger release has been observed. The next press then
        # re-latches both VR and Placo baselines.
        self._homing = False
        self._home_terminal = False
        self._home_deadman_held = False
        self._tf_buffer = None
        self._tf_listener = None
        self._angular_adapter = None

        if self._output_profile == "humanoid":
            self._init_humanoid_publishers()
        else:
            self._init_so101_publishers()

        self._arm_state: dict[str, _ArmState] = {
            "left": _ArmState(),
            "right": _ArmState(),
        }
        # pose-mode clutch baseline (base frame, axis-swapped). Latched on the
        # trigger-press rising edge, cleared on release; None => re-baseline next
        # frame.
        self._pose_calib_pos: np.ndarray | None = None
        self._pose_calib_rot: Rotation | None = None

        freq = self.get_parameter("control_frequency").value
        self._control_dt = 1.0 / freq
        self._timer = self.create_timer(self._control_dt, self._control_callback)

        self.get_logger().info(
            "\n"
            + "=" * 60
            + "\n"
            + "  VR Dual-Arm Teleop Node READY\n"
            + f"  TCP server listening on {host}:{port}\n"
            + f"  Control frequency: {freq} Hz\n"
            + f"  Output profile: {self._output_profile}\n"
            + (f"  so101 input mode: {self._so101_input_mode}\n" if self._output_profile == "so101" else "")
            + "  Waiting for Unity VR client connection...\n"
            + "=" * 60
        )

    def _init_humanoid_publishers(self) -> None:
        self._right_linear_pub = self.create_publisher(Vector3Stamped, "/humanoid_teleop/right_arm/linear_cmd_base", 10)
        self._left_linear_pub = self.create_publisher(Vector3Stamped, "/humanoid_teleop/left_arm/linear_cmd_base", 10)
        self._right_angular_pub = self.create_publisher(
            Vector3Stamped, "/humanoid_teleop/right_arm/angular_cmd_tool", 10
        )
        self._left_angular_pub = self.create_publisher(Vector3Stamped, "/humanoid_teleop/left_arm/angular_cmd_tool", 10)
        self._right_gripper_pub = self.create_publisher(Float64MultiArray, "/right_gripper_controller/commands", 10)
        self._left_gripper_pub = self.create_publisher(Float64MultiArray, "/left_gripper_controller/commands", 10)

    def _init_so101_publishers(self) -> None:
        if self._so101_input_mode == "pose":
            # Pose passthrough: publish a RELATIVE (clutch) pose command. No
            # tool→base angular conversion is needed (orientation is a base-frame
            # rotation delta, not a tool-frame twist), so no TF listener /
            # ToolAngularAdapter here.
            self._so101_pose_pub = self.create_publisher(PoseStamped, self.get_parameter("so101_pose_topic").value, 10)
        else:
            self._so101_linear_pub = self.create_publisher(
                Vector3Stamped, self.get_parameter("so101_linear_topic").value, 10
            )
            self._so101_angular_pub = self.create_publisher(
                Vector3Stamped, self.get_parameter("so101_angular_topic").value, 10
            )
        if self._managed:
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Empty

            self._so101_gripper_pub = self.create_publisher(
                JointState, self.get_parameter("so101_gripper_topic").value, 1
            )
            self._managed_lease_pub = self.create_publisher(Empty, self.get_parameter("command_lease_topic").value, 1)
            from ibrobot_msgs.msg import RuntimeStatus

            self.create_subscription(
                RuntimeStatus, self.get_parameter("runtime_status_topic").value, self._managed_runtime_status, 1
            )
        else:
            self._so101_gripper_pub = self.create_publisher(
                Float64MultiArray, self.get_parameter("so101_gripper_topic").value, 10
            )
        self._so101_start_cli = self.create_client(Trigger, self.get_parameter("so101_start_service").value)
        self._so101_stop_cli = self.create_client(Trigger, self.get_parameter("so101_stop_service").value)
        self._so101_home_cli = ActionClient(self, ArmReturnHome, self.get_parameter("so101_home_action").value)
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 10)
        if self._so101_input_mode == "velocity":
            # tool→base angular converter. placo consumes angular as a base-frame
            # velocity and does NOT transform it, so the raw tool-frame VR angular
            # must be rotated into base here (same path xbox/phone use via
            # PlacoServoBackend). Needs a live TF tree (robot_state_publisher).
            base_link = str(self.get_parameter("base_link_name").value)
            tool_frame = str(self.get_parameter("tool_frame").value)
            stale_s = float(self.get_parameter("tf_stale_threshold_s").value)
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._angular_adapter = ToolAngularAdapter(
                node=self,
                tf_buffer=self._tf_buffer,
                base_link=base_link,
                tool_frame=tool_frame,
                stale_threshold_s=stale_s,
            )
        self.create_timer(0.5, self._ensure_so101_started)

    def destroy_node(self) -> None:
        self._cancel_so101_home("VR teleop node shutdown")
        if self._output_profile == "so101" and self._so101_stop_cli and self._so101_stop_cli.service_is_ready():
            self._so101_stop_cli.call_async(Trigger.Request())
        if self._tcp_server:
            self._tcp_server.stop()
        super().destroy_node()

    def _ensure_so101_started(self) -> None:
        if getattr(self, "_managed", False):
            if self._so101_stop_pending:
                self._stop_so101_servo("retry managed stop")
            return
        if self._output_profile != "so101" or self._estop_active:
            return
        # If a stop is pending but never confirmed (e.g. it was rejected while the
        # stream was still fresh, so no watchdog tick retries it), drive the retry
        # from this always-on 0.5s timer. Until the servo confirms stopped, the
        # arm must stay disabled and re-engage stays blocked — but the latch must
        # be able to clear, or the user could never re-teleop. Retry, then return:
        # a start must not race the same tick.
        if self._so101_stop_pending and not self._so101_stop_inflight:
            self._stop_so101_servo("retry pending stop from watchdog timer")
            return
        if (
            self._so101_started
            or self._so101_start_inflight
            or self._homing
            or self._so101_home_inflight
            or self._so101_reengage_required
        ):
            return
        # Do NOT auto-restart while the command stream is stalled/disconnected, a
        # stop is still inflight, or a stop is pending (requested but not yet
        # confirmed): re-enabling here would undo the watchdog stop and let the
        # arm resume tracking. The trigger rising edge re-latches and restarts
        # once fresh frames return AND the user re-grips.
        if self._so101_stalled or self._so101_stop_inflight or self._so101_stop_pending:
            return
        if self._so101_start_cli is None or not self._so101_start_cli.service_is_ready():
            self.get_logger().warn("SO101 servo start service not ready; will retry")
            return
        self._so101_start_inflight = True
        future = self._so101_start_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_so101_start_response)

    def _on_so101_start_response(self, future) -> None:
        self._so101_start_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"SO101 servo start request failed: {exc}")
            return
        if not response.success:
            self.get_logger().error(response.message or "SO101 servo rejected start request")
            return
        # A stall/disconnect that landed while this start was inflight left the
        # node's intent at "stopped" but could not issue the stop (start was
        # inflight, so _stop_so101_servo latched _so101_stop_pending). Honor that
        # intent: the servo is now running, so issue the deferred stop instead of
        # flipping _so101_started back on and resuming tracking.
        if self._so101_stalled or self._so101_stop_pending:
            self.get_logger().warn("SO101 servo start completed but a stop is pending; stopping")
            self._stop_so101_servo("start completed while stop pending")
            return
        self._so101_started = True
        self.get_logger().info(response.message or "SO101 servo enabled")

    def _stop_so101_servo(self, reason: str) -> None:
        """Stop the placo servo so the arm stops tracking its last pose target,
        and mark the node stopped so the 0.5s auto-start timer does NOT re-enable
        it until the trigger rising edge re-runs the start/re-latch handshake.

        Closed-loop deadman: a stop that cannot be confirmed here (service not
        ready, or a start/recalib/home is inflight and would race it) latches
        _so101_stop_pending instead of being silently dropped. While pending, the
        watchdog paths keep calling this until the servo acknowledges, and every
        start path is blocked (_ensure_so101_started / _recalibrate / _go_home /
        their success callbacks all check the flag). Only _on_so101_stop_response
        clears it on a successful response.
        """
        # _so101_started is cleared unconditionally: whatever raced, the node's
        # intent is now "stopped", and the auto-start timer keys off this flag.
        self._so101_started = False
        self._so101_reengage_required = True
        self._cancel_so101_home(reason)
        if self._so101_stop_inflight:
            # A stop is already on the wire; its response will resolve pending.
            return
        # A start/recalib is inflight. Issuing stop now would race their
        # success callbacks (which set _so101_started=True). Latch intent; the
        # callback re-issues stop once it lands, or the watchdog retries next tick.
        if self._so101_start_inflight or self._so101_recalib_inflight:
            self._so101_stop_pending = True
            return
        if self._so101_stop_cli is None or not self._so101_stop_cli.service_is_ready():
            self._so101_stop_pending = True
            self.get_logger().warn(
                f"SO101 servo stop service not ready ({reason}); "
                "latched stop-pending, will retry until the arm confirms stopped"
            )
            return
        self.get_logger().warn(f"SO101 servo stopping: {reason}")
        self._so101_stop_pending = True
        self._so101_stop_inflight = True
        future = self._so101_stop_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_so101_stop_response)

    def _on_so101_stop_response(self, future) -> None:
        self._so101_stop_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            # Stop never landed. Keep _so101_stop_pending latched so the watchdog
            # retries; the arm is NOT confirmed stopped.
            self.get_logger().error(f"SO101 servo stop request failed: {exc}; will retry")
            return
        if not response.success:
            # Servo rejected the stop; stay pending and retry.
            self.get_logger().error(response.message or "SO101 servo rejected stop request; will retry")
            return
        # Confirmed stopped. Release the deadman latch. _so101_stalled stays set
        # (input still untrusted) until a trigger release clears it.
        self._so101_stop_pending = False

    def _recalibrate_so101_baseline(self) -> bool:
        """Re-latch placo's clutch baseline (_ee0_p/_ee0_R) to the current
        measured EE by re-calling the start service on the trigger rising edge.

        Pose mode zeroes on trigger press: the hand pose at press becomes the
        origin. VR relatches its own _pose_calib on this edge; calling start
        makes placo relatch its EE baseline to the same instant, so the single
        zero point is consistent across both nodes and clutch never jumps.

        Returns True if the re-latch call was actually dispatched (inflight now
        set). Returns False when the start service is not yet ready — the caller
        MUST then abort this rising edge (roll back its calib) rather than let
        the next frame emit a real displacement with no inflight gate, which
        jumps the arm on the very first press after startup.
        """
        if self._estop_active:
            return False
        if self._so101_start_cli is None or not self._so101_start_cli.service_is_ready():
            self.get_logger().warning(
                "SO101 start service not ready on trigger press; deferring "
                "clutch engage until it is discovered (arm holds this frame)"
            )
            return False
        # A stop is pending (deadman not yet confirmed): do not re-latch/start.
        # The stall must be resolved (stop confirmed + trigger released) before a
        # new grip may re-engage the arm.
        if self._so101_stop_pending or self._so101_stalled:
            self.get_logger().warning(
                "SO101 re-latch refused: stop pending / stream stalled; "
                "release the trigger and re-grip after the stream recovers"
            )
            return False
        if self._so101_recalib_inflight:
            return True
        self._so101_recalib_inflight = True
        future = self._so101_start_cli.call_async(Trigger.Request())
        future.add_done_callback(self._on_so101_recalib_response)
        return True

    def _on_so101_recalib_response(self, future) -> None:
        self._so101_recalib_inflight = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"SO101 servo re-latch failed: {exc}")
            # Re-latch never landed: placo's EE baseline is still its stale
            # pre-trigger pose. Clear the clutch baseline so the next held
            # frame re-runs the rising-edge path (re-latch + zero displacement)
            # instead of adding a real displacement onto the stale baseline
            # and jumping the arm.
            self._pose_calib_pos = None
            self._pose_calib_rot = None
            return
        if response.success:
            # A stop that latched while this re-latch was inflight must win: the
            # servo is now enabled, so honor the pending stop instead of marking
            # started and resuming tracking.
            if self._so101_stop_pending or self._so101_stalled:
                self.get_logger().warn("SO101 re-latch completed but a stop is pending; stopping")
                self._pose_calib_pos = None
                self._pose_calib_rot = None
                self._stop_so101_servo("re-latch completed while stop pending")
                return
            self._so101_started = True
            self._so101_reengage_required = False
            self.get_logger().info("SO101 baseline re-latched on trigger press")
        else:
            self.get_logger().error(response.message or "SO101 servo rejected re-latch")
            # Same as the exception path: baseline not re-latched, so drop the
            # clutch calib and re-arm on the next held frame.
            self._pose_calib_pos = None
            self._pose_calib_rot = None

    def _go_home_so101(self) -> bool:
        """Dispatch the shared transactional ArmReturnHome action."""
        if self._estop_active:
            return False
        if self._so101_home_cli is None or not self._so101_home_cli.server_is_ready():
            self.get_logger().warn("SO101 ArmReturnHome action not ready")
            return False
        if self._so101_stop_pending or self._so101_stalled:
            self.get_logger().warn("SO101 ArmReturnHome refused: stop pending / stream stalled")
            return False
        if self._so101_home_inflight:
            return False

        self._so101_home_inflight = True
        self._so101_home_cancel_pending = False
        self._so101_started = False
        self._so101_reengage_required = True
        goal = ArmReturnHome.Goal()
        goal.target_name = "home"
        try:
            future = self._so101_home_cli.send_goal_async(goal)
        except Exception as exc:  # noqa: BLE001
            self._so101_home_inflight = False
            self.get_logger().error(f"SO101 ArmReturnHome dispatch failed: {exc}")
            return False
        future.add_done_callback(self._on_so101_home_goal_response)
        return True

    def _on_so101_home_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._so101_home_inflight = False
            self._home_terminal = True
            self.get_logger().error(f"SO101 ArmReturnHome goal request failed: {exc}")
            self._release_home_gate_if_ready()
            return
        if not goal_handle.accepted:
            self._so101_home_inflight = False
            self._home_terminal = True
            self.get_logger().error("SO101 ArmReturnHome goal was rejected")
            self._release_home_gate_if_ready()
            return

        self._so101_home_goal_handle = goal_handle
        if self._so101_home_cancel_pending or self._so101_stop_pending or self._so101_stalled:
            goal_handle.cancel_goal_async()
        goal_handle.get_result_async().add_done_callback(self._on_so101_home_result)
        self.get_logger().info("SO101 ArmReturnHome accepted")

    def _on_so101_home_result(self, future) -> None:
        self._so101_home_inflight = False
        self._so101_home_goal_handle = None
        self._so101_home_cancel_pending = False
        self._home_terminal = True
        self._so101_started = False
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"SO101 ArmReturnHome result failed: {exc}")
            self._release_home_gate_if_ready()
            return

        log = self.get_logger().info if result.success else self.get_logger().error
        log(result.message or result.error_code or "SO101 ArmReturnHome completed")
        self._release_home_gate_if_ready()
        if self._so101_stop_pending and not self._so101_stop_inflight:
            self._stop_so101_servo("ArmReturnHome completed while stop pending")

    def _cancel_so101_home(self, reason: str) -> None:
        if not self._so101_home_inflight:
            return
        self._so101_home_cancel_pending = True
        if self._so101_home_goal_handle is not None:
            self._so101_home_goal_handle.cancel_goal_async()
        self.get_logger().warn(f"Canceling SO101 ArmReturnHome: {reason}")

    def _release_home_gate_if_ready(self) -> None:
        if self._homing and self._home_terminal and not self._home_deadman_held:
            self._homing = False

    def _control_callback(self) -> None:
        event = self._tcp_server.consume_connection_event()
        if event is not None:
            if event.startswith("connected"):
                addr = event.split(" ", 1)[1]
                self.get_logger().info("\n" + "=" * 60 + "\n" + f"  VR CLIENT CONNECTED from {addr}\n" + "=" * 60)
            elif event == "disconnected":
                self.get_logger().info(
                    "\n"
                    + "=" * 60
                    + "\n"
                    + "  VR CLIENT DISCONNECTED\n"
                    + "  Waiting for new connection...\n"
                    + "=" * 60
                )

        data = self._tcp_server.get_latest()
        if data is None:
            self._publish_all_zero()
            return

        # Stale-frame watchdog: the client is still TCP-connected but has stopped
        # sending. Do NOT republish the frozen frame (that would keep placo's
        # command fresh forever and let the arm chase a dead target). Treat a
        # stale frame as no data.
        if self._output_profile == "so101":
            age = self._tcp_server.get_latest_age_s()
            if age is not None and age > self._so101_command_stale_s:
                self._handle_so101_stale()
                return
            # NOTE: fresh frames do NOT clear _so101_stalled here. If the stream
            # recovers while the user is still holding the trigger, we must NOT
            # silently resume motion — the arm was stopped on purpose. The stall
            # is cleared only by a trigger RELEASE (see _control_so101_pose), and
            # the subsequent press re-baselines and restarts. This enforces
            # "recover ⇒ deliberate re-grip", not "recover ⇒ auto-resume".

        if self._output_profile == "so101":
            self._control_so101(data)
            return

        for side in ("left", "right"):
            ctrl = getattr(data, side)
            if ctrl is None:
                self._publish_zero(side)
                continue

            state = self._arm_state[side]
            linear, angular = self._compute_velocities(ctrl, state)
            self._publish_arm(side, linear, angular)
            self._publish_gripper(side, ctrl.grip_value)

    def _publish_all_zero(self) -> None:
        for side in ("left", "right"):
            self._arm_state[side].ema_linear = np.zeros(3)
            self._arm_state[side].ema_angular = np.zeros(3)
        if self._output_profile == "so101":
            # No fresh data (hard disconnect / client exit): request a servo stop
            # for both pose and velocity modes. Publishing zero velocity is not a
            # transactional stop and would not cancel an active ArmReturnHome action.
            if self._so101_input_mode == "pose":
                self._pose_calib_pos = None
                self._pose_calib_rot = None
            self._homing = False
            if not self._so101_stalled:
                self._so101_stalled = True
                self._stop_so101_servo("VR command stream disconnected")
            elif self._so101_stop_pending:
                self._stop_so101_servo("stop still pending while disconnected")
            return
        for side in ("left", "right"):
            self._publish_zero(side)

    def _handle_so101_stale(self) -> None:
        """Client is connected but its frames went stale. Drop the clutch so a
        resumed stream re-baselines from the new hand pose, and request a placo
        stop on the stall's rising edge so the arm stops tracking the last target
        instead of holding it. A failed/unavailable stop remains latched as
        pending and is retried on later stale ticks until placo confirms success.
        Stopping publication alone is not enough: placo latches the last pose
        reference and keeps driving to it.

        The gripper is intentionally not touched — with no fresh grip value the
        safe choice is to leave it where the last valid frame left it rather than
        force it open/closed on a comms glitch."""
        self._pose_calib_pos = None
        self._pose_calib_rot = None
        # Homing is also invalid once input is lost — a settled re-grip must
        # happen through the normal rising edge after the stream returns.
        self._homing = False
        if not self._so101_stalled:
            self._so101_stalled = True
            self._stop_so101_servo(f"VR command stream stalled (> {self._so101_command_stale_s:.2f}s)")
        elif self._so101_stop_pending:
            # Already stalled but the deadman stop never confirmed. Keep
            # retrying every stale tick until the servo acknowledges — a single
            # failed stop must not leave the arm tracking a dead target.
            self._stop_so101_servo("stop still pending during stall")

    def _control_so101(self, data: _DualArmVRData) -> None:
        ctrl = getattr(data, self._controller_side)
        if getattr(self, "_managed", False):
            self._control_managed(ctrl)
            return
        if self._estop_active:
            return
        if self._handle_so101_home_input(ctrl):
            return
        if self._so101_input_mode == "pose":
            self._control_so101_pose(ctrl)
            return
        if ctrl is None:
            # Controller absent (disconnect, not a deliberate release): publish
            # zero and, like the pose path, leave any stall LATCHED — recovery
            # requires a live controller reporting the trigger released.
            self._publish_so101(np.zeros(3), np.zeros(3), None)
            return

        # A live controller reporting the trigger released clears a stall — the
        # same "recover ⇒ deliberate re-grip" contract the pose path enforces.
        # The stale watchdog (_handle_so101_stale) sets _so101_stalled for ALL
        # so101 modes, but only the pose release path cleared it; without this,
        # velocity mode would be permanently locked out after any stale event
        # (_ensure_so101_started keeps refusing to (re)start while stalled).
        # Clearing on release is safe against a recovery-frame jump: on release
        # _compute_velocities resets prev_pos/EMA to zero, and the enable rising
        # edge re-latches calib_pos and emits zero velocity for that frame.
        if not ctrl.enabled and self._so101_stalled:
            self._so101_stalled = False

        # Still stalled (trigger held through a stale event, or stall not yet
        # cleared by a release): hold. The servo was stopped by the watchdog and
        # _ensure_so101_started refuses to restart it while stalled, so publish
        # zero rather than feed velocities the disabled servo would ignore — and
        # avoid a large recovery-frame velocity spike if it were re-enabled.
        if self._so101_stalled:
            self._publish_so101(np.zeros(3), np.zeros(3), ctrl.grip_value)
            return

        state = self._arm_state[self._controller_side]
        linear, angular = self._compute_velocities(ctrl, state)
        self._publish_so101(linear, angular, ctrl.grip_value)

    def _control_managed(self, ctrl):
        """Fresh trigger edges own the runtime session in both VR modes."""
        from std_msgs.msg import Empty

        if self._estop_active:
            return
        if ctrl is None or not ctrl.enabled:
            if self._so101_started or self._so101_recalib_inflight or self._so101_home_inflight:
                self._stop_so101_servo("VR deadman released")
            self._managed_release_required = ctrl is None
            self._pose_calib_pos = None
            self._pose_calib_rot = None
            self._arm_state[self._controller_side] = _ArmState()
            if ctrl is not None:
                self._so101_stalled = False
            return
        if self._so101_home_inflight:
            self._managed_lease_pub.publish(Empty())
            return
        if not self._so101_started:
            if self._managed_release_required or self._so101_stop_pending or self._so101_recalib_inflight:
                return
            self._managed_release_required = True
            self._pose_calib_pos = ctrl.position.copy()
            self._pose_calib_rot = ctrl.rotation
            self._recalibrate_so101_baseline()
            return
        self._managed_lease_pub.publish(Empty())
        secondary = bool(ctrl.secondary_button)
        if secondary and not self._secondary_prev:
            self._secondary_prev = True
            self._go_home_so101()
            return
        self._secondary_prev = secondary
        if self._so101_input_mode == "pose":
            self._control_so101_pose(ctrl)
        else:
            linear, angular = self._compute_velocities(ctrl, self._arm_state[self._controller_side])
            self._publish_so101(linear, angular, ctrl.grip_value)

    def _managed_runtime_status(self, status):
        invalid = status.stop_latched or status.active_mode != "stream" or status.lifecycle != "ACTIVE"
        if invalid and self._so101_started:
            self._so101_started = False
            self._managed_release_required = True
            self._pose_calib_pos = None
            self._pose_calib_rot = None
            self._so101_stalled = True

    def _on_estop(self, msg: Bool) -> None:
        if bool(msg.data):
            self._estop_active = True
            self._pose_calib_pos = None
            self._pose_calib_rot = None
            self._homing = False
            self._so101_stalled = True
            self._stop_so101_servo("emergency stop")
            return
        self._estop_active = False
        self.get_logger().info("VR emergency stop released; release and re-press trigger to resume")

    def _handle_so101_home_input(self, ctrl: _ControllerData | None) -> bool:
        """Own the Home button/gate for both pose and velocity input modes."""
        secondary = bool(getattr(ctrl, "secondary_button", False)) if ctrl is not None else False
        if secondary and not self._secondary_prev:
            self._secondary_prev = True
            if self._go_home_so101():
                self._pose_calib_pos = None
                self._pose_calib_rot = None
                state = self._arm_state[self._controller_side]
                state.prev_pos = None
                state.prev_rot = None
                state.enabled_prev = False
                state.ema_linear = np.zeros(3)
                state.ema_angular = np.zeros(3)
                self._homing = True
                self._home_terminal = False
                self._home_deadman_held = bool(ctrl is not None and ctrl.enabled)
                return True
        else:
            self._secondary_prev = secondary

        if not self._homing:
            if not self._so101_reengage_required:
                return False
            if ctrl is None or not ctrl.enabled:
                return False
            if self._so101_stalled or self._so101_stop_pending:
                return False
            if self._so101_recalib_inflight:
                return True
            if self._so101_input_mode == "pose":
                self._pose_calib_pos = ctrl.position.copy()
                self._pose_calib_rot = ctrl.rotation
            if not self._recalibrate_so101_baseline():
                self._pose_calib_pos = None
                self._pose_calib_rot = None
            return True
        if ctrl is not None:
            self._home_deadman_held = bool(ctrl.enabled)
        self._release_home_gate_if_ready()
        return self._homing

    def _control_so101_pose(self, ctrl: _ControllerData | None) -> None:
        """Pose passthrough: publish a RELATIVE (clutch) pose command.

        The PoseStamped is NOT an absolute EE pose. Relative to the clutch
        baseline latched at the trigger press:
          - position = (hand_pos - hand_calib) * position_scale — a base-frame
            displacement (already axis-swapped in _parse_controller);
          - orientation = the relative base-frame rotation delta
            ΔR_base = R_current * R_clutch^-1 (identity when position_only).
        placo adds this displacement/delta onto the EE pose it latched at its
        own enable (_ee0_p / _ee0_R), so a held hand holds the arm and a zero
        delta at press means no motion.
        """
        if ctrl is None or not ctrl.enabled:
            # Trigger released or controller absent: clear clutch baseline and
            # publish nothing for the ARM (placo holds last reference). Next
            # press re-grips. The GRIPPER is independent of enable — publish it
            # whenever controller data exists so it can be opened/closed without
            # holding the trigger.
            self._pose_calib_pos = None
            self._pose_calib_rot = None
            # A release is also what clears a stall. Fresh frames alone do NOT
            # resume motion (see _control_callback): the arm was stopped on
            # purpose, so recovery requires the user to let go and deliberately
            # re-grip. Only clear once we have a live controller reporting the
            # trigger released — an absent controller (ctrl is None) is a
            # disconnect, not a release, and must stay stalled. The stop-pending
            # latch is independent: it stays set until the servo confirms the
            # stop, blocking re-start even after the stall clears.
            if ctrl is not None:
                self._so101_stalled = False
                self._publish_so101_gripper(ctrl.grip_value)
            return

        # Enable rising edge (first frame with trigger held): latch clutch
        # baseline and emit zero displacement + identity rotation so the arm
        # does not jump. Also re-latch placo's EE baseline to this instant so
        # the zero point is consistent across both nodes (trigger press zeroes).
        if self._pose_calib_pos is None or self._pose_calib_rot is None:
            # Only engage the clutch if placo actually accepted the re-latch
            # (start service ready). If it is not ready yet (startup discovery),
            # roll back the calib so THIS press is a no-op and the next frame
            # retries — otherwise the following frame would emit a real
            # displacement with no inflight gate and jump the arm on first press.
            self._pose_calib_pos = ctrl.position.copy()
            self._pose_calib_rot = ctrl.rotation
            if not self._recalibrate_so101_baseline():
                self._pose_calib_pos = None
                self._pose_calib_rot = None
            self._publish_so101_pose(np.zeros(3), Rotation.identity(), ctrl.grip_value)
            return

        scale = float(self.get_parameter("position_scale").value)
        # Gate: until placo confirms it re-latched its EE baseline (recalib
        # callback clears the inflight flag), hold zero displacement. Otherwise
        # the first steady-state frame adds a real displacement onto placo's
        # STALE baseline (its pre-trigger reset pose) and the arm jumps hard on
        # the first press. Re-latch on trigger press is async (service round-trip
        # + 50Hz tick); publishing zero until it lands keeps both zero points
        # consistent, so the first non-zero command rides the fresh baseline.
        if self._so101_recalib_inflight:
            self._publish_so101_pose(np.zeros(3), Rotation.identity(), ctrl.grip_value)
            return
        rel_pos = (ctrl.position - self._pose_calib_pos) * scale
        # Orientation is RELATIVE to the clutch baseline, expressed as a
        # BASE-frame increment — the same frame as the position delta and the
        # pose_cmd_base topic. Contract:
        #   ΔR_base = R_current * R_clutch^-1     (this node)
        #   R_target = ΔR_base * R_clutch         (placo, left-multiply)
        # At the press instant R_current == R_clutch so ΔR_base is identity and
        # the EE attitude does not jump. Using R_clutch^-1 * R_current instead
        # would be a body/tool-frame increment, whose axis depends on the hand
        # attitude at trigger press — that is the coupling bug we are avoiding.
        # Both formulas live in vr_rotation (pure, ROS-free, directly tested).
        if self._so101_position_only:
            # Rotation passthrough disabled: hold the EE attitude fixed at the
            # clutch baseline (identity delta) and drive position only. Used
            # until the base-frame correction matrix is re-standardised, or when
            # a task wants pure translation. See vr_rotation.R_ROBOT_BASE_FROM_VR_BASE.
            rel_rot = Rotation.identity()
        else:
            rel_rot = compute_base_rotation_delta(ctrl.rotation, self._pose_calib_rot)
            # Re-express the base-frame delta from the VR base frame into the
            # robot base frame (frame conjugation); see vr_rotation.
            rel_rot = remap_base_rotation(rel_rot)
        self._publish_so101_pose(rel_pos, rel_rot, ctrl.grip_value)

    def _publish_so101_pose(
        self,
        rel_position: np.ndarray,
        rotation: Rotation,
        grip_value: float | None,
    ) -> None:
        """Publish a PoseStamped EE command + gripper for pose mode."""
        if getattr(self, "_managed", False) and not self._so101_started:
            return
        if self._so101_pose_pub is None:
            return
        base_frame = str(self.get_parameter("base_link_name").value)
        q = rotation.as_quat()  # (x, y, z, w)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.pose.position.x = float(rel_position[0])
        msg.pose.position.y = float(rel_position[1])
        msg.pose.position.z = float(rel_position[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._so101_pose_pub.publish(msg)
        self._publish_so101_gripper(grip_value)

    def _publish_so101_gripper(self, grip_value: float | None) -> None:
        if grip_value is None or self._so101_gripper_pub is None:
            return
        open_pos = float(self.get_parameter("so101_gripper_open").value)
        closed_pos = float(self.get_parameter("so101_gripper_closed").value)
        gripper_pos = open_pos + (closed_pos - open_pos) * float(np.clip(grip_value, 0.0, 1.0))
        if getattr(self, "_managed", False):
            from sensor_msgs.msg import JointState

            if not self._so101_started or not np.isfinite(gripper_pos):
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [self.get_parameter("gripper_joint_name").value]
            limit = self._joint_limits[msg.name[0]]
            msg.position = [min(limit["max"], max(limit["min"], gripper_pos))]
            self._so101_gripper_pub.publish(msg)
            return
        msg = Float64MultiArray()
        msg.data = [gripper_pos]
        self._so101_gripper_pub.publish(msg)

    def _compute_velocities(self, ctrl: _ControllerData, state: _ArmState) -> tuple:
        if state.calib_pos is None or state.calib_rot is None:
            state.calib_pos = ctrl.position.copy()
            state.calib_rot = ctrl.rotation
            state.prev_pos = None
            state.prev_rot = None
            state.enabled_prev = False
            self.get_logger().info(f"Arm calibrated on first data. Ref pos: {state.calib_pos}")
            return np.zeros(3), np.zeros(3)

        if not ctrl.enabled:
            state.prev_pos = None
            state.prev_rot = None
            state.enabled_prev = False
            state.ema_linear = np.zeros(3)
            state.ema_angular = np.zeros(3)
            return np.zeros(3), np.zeros(3)

        if not state.enabled_prev:
            state.calib_pos = ctrl.position.copy()
            state.calib_rot = ctrl.rotation
            state.prev_pos = None
            state.prev_rot = None
            state.enabled_prev = True
            return np.zeros(3), np.zeros(3)

        state.enabled_prev = True

        pos_cal = ctrl.position - state.calib_pos
        rot_cal = state.calib_rot.inv() * ctrl.rotation

        if state.prev_pos is None or state.prev_rot is None:
            state.prev_pos = pos_cal.copy()
            state.prev_rot = rot_cal
            return np.zeros(3), np.zeros(3)

        delta_pos = pos_cal - state.prev_pos
        delta_rot = state.prev_rot.inv() * rot_cal

        state.prev_pos = pos_cal.copy()
        state.prev_rot = rot_cal

        dt = self._control_dt
        linear_velocity = delta_pos.astype(float) / dt
        angular_velocity = delta_rot.as_rotvec() / dt

        max_lin = self.get_parameter("max_linear_speed").value
        max_ang = self.get_parameter("max_angular_speed").value
        lin_scale = self.get_parameter("linear_speed_scale").value
        ang_scale = self.get_parameter("angular_speed_scale").value

        linear_velocity *= lin_scale
        angular_velocity *= ang_scale

        lin_norm = float(np.linalg.norm(linear_velocity))
        if lin_norm > max_lin and lin_norm > 0:
            linear_velocity = linear_velocity * (max_lin / lin_norm)

        ang_norm = float(np.linalg.norm(angular_velocity))
        if ang_norm > max_ang and ang_norm > 0:
            angular_velocity = angular_velocity * (max_ang / ang_norm)

        lin_dz = self.get_parameter("linear_deadzone").value
        ang_dz = self.get_parameter("angular_deadzone").value
        if np.linalg.norm(linear_velocity) < lin_dz:
            linear_velocity = np.zeros(3)
        if np.linalg.norm(angular_velocity) < ang_dz:
            angular_velocity = np.zeros(3)

        alpha = self.get_parameter("velocity_ema_alpha").value
        state.ema_linear = alpha * linear_velocity + (1 - alpha) * state.ema_linear
        state.ema_angular = alpha * angular_velocity + (1 - alpha) * state.ema_angular

        return state.ema_linear.copy(), state.ema_angular.copy()

    def _publish_arm(self, side: str, linear: np.ndarray, angular: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        lin_frame = self.get_parameter("linear_frame_id").value
        ang_frame = self.get_parameter("angular_frame_id").value

        lin_pub = self._right_linear_pub if side == "right" else self._left_linear_pub
        ang_pub = self._right_angular_pub if side == "right" else self._left_angular_pub

        lin_msg = Vector3Stamped()
        lin_msg.header.stamp = stamp
        lin_msg.header.frame_id = lin_frame
        lin_msg.vector.x = float(linear[0])
        lin_msg.vector.y = float(linear[1])
        lin_msg.vector.z = float(linear[2])
        lin_pub.publish(lin_msg)

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = stamp
        ang_msg.header.frame_id = ang_frame
        ang_msg.vector.x = float(angular[0])
        ang_msg.vector.y = float(angular[1])
        ang_msg.vector.z = float(angular[2])
        ang_pub.publish(ang_msg)

    def _publish_so101(
        self,
        linear: np.ndarray,
        angular: np.ndarray,
        grip_value: float | None,
    ) -> None:
        if getattr(self, "_managed", False) and not self._so101_started:
            return
        if self._so101_linear_pub is None or self._so101_angular_pub is None:
            return

        # Convert tool-frame angular → base frame for placo (which treats its
        # angular input as base-frame and applies no transform). The adapter
        # passes linear through unchanged and self-zeroes angular until the
        # first TF lookup succeeds / when TF goes stale.
        linear_t = (float(linear[0]), float(linear[1]), float(linear[2]))
        angular_t = (float(angular[0]), float(angular[1]), float(angular[2]))
        if self._angular_adapter is not None:
            _, angular_base = self._angular_adapter.convert(linear_t, angular_t)
        else:
            angular_base = angular_t

        base_frame = str(self.get_parameter("base_link_name").value)
        stamp = self.get_clock().now().to_msg()
        lin_msg = Vector3Stamped()
        lin_msg.header.stamp = stamp
        lin_msg.header.frame_id = base_frame
        lin_msg.vector.x = linear_t[0]
        lin_msg.vector.y = linear_t[1]
        lin_msg.vector.z = linear_t[2]
        self._so101_linear_pub.publish(lin_msg)

        ang_msg = Vector3Stamped()
        ang_msg.header.stamp = stamp
        ang_msg.header.frame_id = base_frame
        ang_msg.vector.x = float(angular_base[0])
        ang_msg.vector.y = float(angular_base[1])
        ang_msg.vector.z = float(angular_base[2])
        self._so101_angular_pub.publish(ang_msg)

        self._publish_so101_gripper(grip_value)

    def _publish_zero(self, side: str) -> None:
        self._publish_arm(side, np.zeros(3), np.zeros(3))

    def _publish_gripper(self, side: str, grip_value: float) -> None:
        gripper_pos = 60.0 * (1.0 - grip_value)
        pub = self._right_gripper_pub if side == "right" else self._left_gripper_pub
        msg = Float64MultiArray()
        msg.data = [gripper_pos]
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VRTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
