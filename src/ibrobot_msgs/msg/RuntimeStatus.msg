# RuntimeStatus.msg - canonical robot runtime identity, lifecycle, and capabilities.
#
# Published by every robot runtime at a fixed rate (>= 1 Hz) and immediately on
# lifecycle, mode, capability, or fault change. See the robot-runtime-contract
# spec (change robot-runtime-abstraction).

# Runtime identity and version (e.g. "so101_robot", "0.1.0").
string runtime_name
string runtime_version

# Lifecycle: CONNECTING | ACTIVE | DEGRADED | STOPPED | FAULTED
string lifecycle

# Capabilities this runtime currently provides (canonical vocabulary names).
string[] capabilities

# JSON object of capability parameters, keyed by capability name
# (joint counts, rates, IK endpoint list, stop latency bounds, base geometry).
string capabilities_json

# Public interface description, validated by interface_description.schema.json.
# Includes schema version, identity, content digest, effective interface profiles
# and separately observed states. The same document can be exported as YAML.
string interface_description_json

# Currently active mode (a name from the runtime profile, e.g. "idle",
# "stream", "trajectory", "base_navigation").
string active_mode

# Declared modes and, for the active one, the controllers it activates.
string[] declared_modes
string[] active_controllers

# Streaming commands observed on a channel while that channel was not active.
# Parallel arrays: channel name -> count since runtime start.
string[] rejected_channels
uint32[] rejected_counts

# Stop latch. When stop_latched is true, only a mode request to "idle" is
# accepted; stop_policy records the policy that engaged it (HOLD | TORQUE_OFF).
bool stop_latched
string stop_policy

# Active faults (empty when healthy). Human-readable detail.
string[] faults

builtin_interfaces/Time stamp
