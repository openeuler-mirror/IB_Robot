"""Runtime contract surface: canonical topic/service names and lifecycle vocabulary.

Every runtime (mock, ros2_control-backed facade, vendor adapter) serves the
same names so generic consumers never depend on the execution stack behind
them. Names are runtime-neutral: no controller, planner, or vendor terms.
"""

from __future__ import annotations

# --- status / mode / stop ---------------------------------------------------
STATUS_TOPIC = "/runtime_status"
SET_MODE_SERVICE = "/runtime/set_mode"
GET_STATUS_SERVICE = "/runtime/get_status"
STOP_SERVICE = "/runtime/stop"

# --- motion services (robot-motion-services spec) ---------------------------
COMPUTE_FK_SERVICE = "/motion/compute_fk"
COMPUTE_IK_SERVICE = "/motion/compute_ik"
MOVE_TO_POSE_SERVICE = "/motion/move_to_pose"
MOVE_TO_JOINT_SERVICE = "/motion/move_to_joint"

# --- base channels (robot-runtime-contract spec) -----------------------------
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
# Navigation gating keeps the names navigation stacks already use (the former
# motion-mode handshake); only the implementation behind them moved.
NAVIGATION_ENABLE_SERVICE = "/motion_mode/set_navigation_enabled"
NAVIGATION_ACK_TOPIC = "/motion_mode/base_navigation_enabled"

# --- lifecycle vocabulary ----------------------------------------------------
LIFECYCLE_CONNECTING = "CONNECTING"
LIFECYCLE_ACTIVE = "ACTIVE"
LIFECYCLE_DEGRADED = "DEGRADED"
LIFECYCLE_STOPPED = "STOPPED"
LIFECYCLE_FAULTED = "FAULTED"
LIFECYCLES = (LIFECYCLE_CONNECTING, LIFECYCLE_ACTIVE, LIFECYCLE_DEGRADED, LIFECYCLE_STOPPED, LIFECYCLE_FAULTED)

# --- move outcome vocabulary (MoveToPose / MoveToConfiguration.outcome) -------
OUTCOME_SUCCEEDED = "SUCCEEDED"
OUTCOME_REJECTED = "REJECTED"
OUTCOME_PLANNING_FAILED = "PLANNING_FAILED"
OUTCOME_EXECUTION_FAILED = "EXECUTION_FAILED"
OUTCOME_CANCELLED = "CANCELLED"

# --- stop policies -------------------------------------------------------------
STOP_HOLD = "HOLD"
STOP_TORQUE_OFF = "TORQUE_OFF"
STOP_POLICIES = (STOP_HOLD, STOP_TORQUE_OFF)

# The mode every runtime must declare: no command controller active, joints hold.
IDLE_MODE = "idle"
