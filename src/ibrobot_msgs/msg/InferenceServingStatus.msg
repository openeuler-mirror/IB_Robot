# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# control_modes.<mode>.inference.scheduler.enable=true. Do not use on the legacy inference path.
# Product-session serving status published by each monolithic scheduled pipeline.
# Distinct from InferencePipelineStatus, which carries distributed transport handshake only.
uint8 STARTING=1
uint8 IDLE=2
uint8 RESETTING=3
uint8 ACTIVE=4
uint8 FAILED=5
uint8 CLOSING=6
std_msgs/Header header
string<=63 pipeline_id
string<=36 boot_id
uint64 sequence
uint8 state
string<=36 product_session_id
uint64 product_session_generation
builtin_interfaces/Time lease_expires_at
string<=64 deployment_fingerprint
string<=64 runtime_policy_fingerprint
string<=64 pipeline_compatibility_fingerprint
string<=128 configured_hardware_resource_id
string<=128 runtime_hardware_resource_id
uint32 hardware_priority_levels
# Policy/VLA-private execution capabilities; not a perception/voice protocol.
# schema_version=1 is the only accepted version. Publishers and consumers must
# rebuild together when this experimental interface changes.
uint32 scheduling_capability_schema_version
# Policy executor scheduling mode: SEQUENTIAL or INDEPENDENT.
string<=16 stage_policy
# Whether the loaded executor supports Global's priority-0 FIFO profile/EDF
# admission scheme. This is neither a hardware capability nor a finish guarantee;
# profile availability, resource contention and request deadlines are checked separately.
bool supports_priority_zero_deadline_admission
# Accepted public priorities are the continuous interval [0, this value].
int32 max_supported_public_priority
ibrobot_msgs/InferenceWorkCapacity[<=2] capacities
ibrobot_msgs/ScheduledInferenceError error
