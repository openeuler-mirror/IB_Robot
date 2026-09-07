#!/bin/zsh
# FullSubNet host verification via the speech_direction ROS pipeline.
#
# FullSubNet has NO typed model service: it is embedded in the streaming
# speech_direction node (VAD -> FullSubNet -> SRP-PHAT DOA). Verification =
# launch with wav replay + confirm direction messages (only published when
# VAD fired AND enhancement ran) + degraded=False.
#
# Prerequisites (see references/troubleshooting.md "speech_direction host
# layout gaps" and issue #125):
#   models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar        (download_speech_direction_models.sh)
#   models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.manifest.json
#   models/silero-vad/assets/silero_vad.onnx                               (download_speech_direction_models.sh)
#   a real-speech 6ch 16k wav (scripts/make_speech_wav.py)
#
# Usage: run_fullsubnet.sh <cpu|cuda> <ros_domain_id> [wav_path]
set -e
MODE="$1"; DOMAIN="$2"; WAV="${3:-/tmp/opencode/speech6ch.wav}"
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
source .shrc_local >/dev/null 2>&1
export ROS_DOMAIN_ID="$DOMAIN"
LOG=/tmp/opencode/fullsubnet_${MODE}.log
CFG=/tmp/opencode/speech_${MODE}.yaml

# generate the runtime config from the production one (timing on, disk writes off)
python3 - "$MODE" "$CFG" <<'EOF'
import sys, yaml
mode, out = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open("src/voice_asr_service/config/speech_direction.yaml"))
p = c["speech_direction_node"]["ros__parameters"]
p["silero_vad_backend"] = "onnx"
p["silero_vad_deployment"] = "torch_cpu"
p["fullsubnet_backend"] = f"stateful_torch_{mode}"
p["fullsubnet_deployment"] = f"torch_{mode}"
p["fullsubnet_timing_enabled"] = True
for k in ("diagnostics_save_raw6ch", "diagnostics_save_enh4ch", "diagnostics_save_frame_metrics",
          "diagnostics_save_gray_events", "diagnostics_high_throughput_enabled"):
    p[k] = False
yaml.dump(c, open(out, "w"), default_flow_style=False, allow_unicode=True, sort_keys=False)
EOF

PROFILE=custom   # config carries the backends; profile stays empty
ros2 launch voice_asr_service speech_direction.launch.py \
  profile:=$PROFILE config_file:=$CFG \
  speech_direction_input_source:=wav \
  speech_direction_wav_path:=$WAV \
  speech_direction_wav_replay_rate:=2.0 >"$LOG" 2>&1 &
LPID=$!
trap 'pkill -TERM -P $LPID 2>/dev/null; kill -TERM $LPID 2>/dev/null' EXIT
sleep 3
python3 - > /tmp/opencode/direction_${MODE}.txt <<'EOF'
import time
import rclpy
from rclpy.node import Node
from ibrobot_msgs.msg import SpeechDirection
rclpy.init()
node = Node("direction_listener")
count = [0]
def cb(m):
    count[0] += 1
    print(f"seq={m.seq_id} azimuth={m.azimuth_rad:.3f}rad ({m.azimuth_rad*57.2958:.1f}deg)", flush=True)
node.create_subscription(SpeechDirection, "/voice/speech_direction", cb, 10)
end = time.time() + 45
while rclpy.ok() and time.time() < end and count[0] < 8:
    rclpy.spin_once(node, timeout_sec=1.0)
node.destroy_node(); rclpy.shutdown()
EOF
echo "=== direction messages ==="
cat /tmp/opencode/direction_${MODE}.txt
echo "=== node state ==="
grep -aE "已启动|降级|degraded|Traceback" "$LOG" | sed 's/\x1b\[[0-9;]*m//g' | head -4
