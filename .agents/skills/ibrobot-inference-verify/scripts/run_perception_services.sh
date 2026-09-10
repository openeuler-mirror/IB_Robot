#!/bin/zsh
# Launch a perception_services mock stack, wait for all typed services to
# register, then call each one (call_perception_services.py) and tear down.
#
# MUST run under zsh: sourcing .shrc_local breaks under bash + `set -u`.
# Launch and verification MUST stay in the same shell invocation: background
# processes are killed when the invoking session/call ends.
#
# Usage: run_perception_services.sh <robot.yaml> <ros_domain_id> <logfile> [timeout_s]
set -e
YAML="$1"; DOMAIN="$2"; LOG="$3"; TIMEOUT="${4:-600}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
source .shrc_local >/dev/null 2>&1
export ROS_DOMAIN_ID="$DOMAIN"
export PYTHONUNBUFFERED=1
ros2 launch robot_config robot.launch.py \
  config_path:="$YAML" \
  control_mode:=model_inference use_sim:=true >"$LOG" 2>&1 &
LPID=$!
trap 'pkill -TERM -P $LPID 2>/dev/null; kill -TERM $LPID 2>/dev/null' EXIT

# expected endpoints come from the generated YAML itself
ENDPOINTS=$(python3 - "$YAML" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
for s in c["robot"]["perception_services"]["services"]:
    if s.get("enabled", True):
        print(s["endpoint"])
EOF
)
EXPECTED=$(echo "$ENDPOINTS" | wc -l | tr -d ' ')
PATTERN=$(echo "$ENDPOINTS" | paste -sd '|' -)

START=$(date +%s)
STATE="LAUNCH_TIMEOUT"
while true; do
  sleep 5
  if (( $(date +%s) - START > TIMEOUT )); then break; fi
  READY=$(ros2 service list 2>/dev/null | grep -cE "$PATTERN" || true)
  if (( READY >= EXPECTED )); then STATE="SERVICES_READY"; break; fi
done
echo "LAUNCH_STATE=$STATE ELAPSED=$(( $(date +%s) - START ))s"
if [[ "$STATE" == "SERVICES_READY" ]]; then
  sleep 5  # let sessions finish loading after registration
  python3 "$SCRIPT_DIR/call_perception_services.py"
fi
