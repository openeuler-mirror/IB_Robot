#!/bin/bash
# Install LTTng and ros2_tracing dependencies for IB-Robot tracing.
#
# Usage: bash scripts/tracing/setup_tracing.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
WORKSPACE="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd -P)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
echo "=== IB-Robot Tracing Setup (ROS 2 ${ROS_DISTRO}) ==="
echo "Note: tracing dependencies are now installed by ./scripts/setup.sh."
echo "This script remains as a retrofit shortcut for already-initialized workspaces."

echo "--- Installing LTTng ---"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    lttng-tools \
    liblttng-ust-dev \
    python3-lttngust \
    python3-babeltrace \
    babeltrace2

echo "--- Installing ros2_tracing ---"
sudo apt-get install -y --no-install-recommends \
    "ros-${ROS_DISTRO}-ros2trace" \
    "ros-${ROS_DISTRO}-tracetools" \
    "ros-${ROS_DISTRO}-tracetools-analysis" \
    "ros-${ROS_DISTRO}-tracetools-launch" \
    "ros-${ROS_DISTRO}-tracetools-read" \
    "ros-${ROS_DISTRO}-tracetools-trace"

echo ""
echo "--- Verification ---"
echo -n "lttng: "; lttng --version 2>/dev/null || echo "NOT FOUND"
echo -n "ros2 trace: "; ros2 trace --help >/dev/null 2>&1 && echo "OK" || echo "NOT FOUND"
echo -n "babeltrace2: "; babeltrace2 --version 2>/dev/null | head -1 || echo "NOT FOUND"
echo -n "lttngust: "; python3 -c "import lttngust; print('OK')" 2>/dev/null || echo "NOT FOUND"

if ! groups | grep -q tracing; then
    sudo usermod -aG tracing "$(whoami)" 2>/dev/null || true
    echo "Note: added to 'tracing' group — may need re-login."
fi

echo ""
echo "=== Setup complete ==="
echo "Usage:"
echo "  ./scripts/setup.sh  # regular workspace setup now includes tracing deps"
echo "  ros2 launch robot_config robot.launch.py enable_tracing:=true ..."
echo "  # Or manually:"
echo "  bash scripts/tracing/start_trace.sh"
