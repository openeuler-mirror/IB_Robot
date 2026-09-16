#!/bin/bash
# Clean up ROS 2 processes, shared memory and residual daemon state

set -euo pipefail

# shellcheck disable=SC2059
log() { printf "[cleanup_ros] $1\n" "${@:2}"; }

# Never signal this script or any ancestor shell. This matters when cleanup is
# chained with diagnostics/launch commands whose parent command line contains a
# process pattern such as "ros2 launch" or "realsense2_camera".
_CLEANUP_PROTECTED_PIDS=("$$")
_cleanup_ancestor_pid="$PPID"
while [[ "$_cleanup_ancestor_pid" =~ ^[0-9]+$ ]] && (( _cleanup_ancestor_pid > 1 )); do
    _CLEANUP_PROTECTED_PIDS+=("$_cleanup_ancestor_pid")
    _cleanup_ancestor_pid="$(ps -o ppid= -p "$_cleanup_ancestor_pid" 2>/dev/null | tr -d ' ')"
done

safe_kill_matching() {
    local signal="$1"
    local pattern="$2"
    local pid protected
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        protected=false
        for protected_pid in "${_CLEANUP_PROTECTED_PIDS[@]}"; do
            if [[ "$pid" == "$protected_pid" ]]; then
                protected=true
                break
            fi
        done
        [[ "$protected" == true ]] && continue
        kill "-$signal" "$pid" 2>/dev/null || true
    done < <(pgrep -f -- "$pattern" 2>/dev/null || true)
}

log "Stopping ROS 2 launch parents..."
# 1. Graceful shutdown of launch/process group leaders first
safe_kill_matching SIGINT "ros2 launch"
safe_kill_matching SIGINT "move_group"
safe_kill_matching SIGINT "ign gazebo"
safe_kill_matching SIGINT "gz sim"
safe_kill_matching SIGINT "mujoco_ros2_control/ros2_control_node"
safe_kill_matching SIGINT "controller_manager/ros2_control_node"
sleep 2

log "Force killing remaining ROS 2 nodes..."
# 2. Force kill every known robot/ROS node class
safe_kill_matching KILL "ros2 launch"
safe_kill_matching KILL "move_group"
safe_kill_matching KILL "rviz2"
safe_kill_matching KILL "ign gazebo"
safe_kill_matching KILL "gz sim"
safe_kill_matching KILL "mujoco_ros2_control/ros2_control_node"
safe_kill_matching KILL "controller_manager/ros2_control_node"
safe_kill_matching KILL "robot_state_publisher"
safe_kill_matching KILL "realsense2_camera"

# IB-Robot application nodes
safe_kill_matching KILL "lerobot_policy_node"
safe_kill_matching KILL "action_dispatcher_node"
safe_kill_matching KILL "motion_server"
safe_kill_matching KILL "safety_guard_node"
safe_kill_matching KILL "skill_executor_node"
safe_kill_matching KILL "agent_plan_node"
safe_kill_matching KILL "task_entry_node"
safe_kill_matching KILL "task_executor_node"
safe_kill_matching KILL "perception_service_node"
safe_kill_matching KILL "geometric_grasp_node"
safe_kill_matching KILL "act_inference_node"
safe_kill_matching KILL "cmd_vel_bridge_node"

# IB-Robot grasp pipeline nodes (manipulation_execution / manipulation_service /
# embodied_agent / inference_service / perception_service generic model services)
safe_kill_matching KILL "pick_executor_node"
safe_kill_matching KILL "place_executor_node"
safe_kill_matching KILL "grasp_planner_node"
safe_kill_matching KILL "grasp_verifier_node"
safe_kill_matching KILL "task_planner_node"
safe_kill_matching KILL "pipeline_policy_node"
safe_kill_matching KILL "pure_inference_node"
safe_kill_matching KILL "model_service_node"

# Auxiliary TF/relay/camera helper nodes
safe_kill_matching KILL "static_tf_"
safe_kill_matching KILL "tf2_ros/static_transform_publisher"
safe_kill_matching KILL "interactive_marker"
safe_kill_matching KILL "so101_joint_current"
safe_kill_matching KILL "robot_config/topic_relay"
safe_kill_matching KILL "camera_info_relay"
safe_kill_matching KILL "depth_image_relay"
safe_kill_matching KILL "aligned_depth"
safe_kill_matching KILL "color_image_relay"
safe_kill_matching KILL "pointcloud_relay"
safe_kill_matching KILL "front_color_image_relay"
safe_kill_matching KILL "front_aligned"

# Parallel grasp IK/FK workers spawned by so101_ik_workers.launch.py
safe_kill_matching KILL "ik_worker"
safe_kill_matching KILL "so101_ik_workers"
sleep 1

# 3. Stop ROS 2 daemon so stale graph data is flushed
log "Stopping ROS 2 daemon..."
ros2 daemon stop 2>/dev/null || true

# 4. Clean shared memory (ROS 2 Humble uses /dev/shm)
log "Cleaning shared memory..."
rm -f /dev/shm/ros2_humble_* 2>/dev/null || true
rm -f /dev/shm/ros_humble_* 2>/dev/null || true
find /dev/shm -maxdepth 1 -user "$(id -un)" \
    \( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' \) \
    -delete 2>/dev/null || true

log "Done."
echo ""
echo "You can now run:"
echo "  source .shrc_local && export ROS_DOMAIN_ID=<your_id> && source install/setup.zsh && ros2 launch robot_config robot.launch.py ..."
