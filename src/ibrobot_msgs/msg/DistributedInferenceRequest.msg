uint8 OP_INFER=1
uint8 OP_RESET=2
uint8 OP_CANCEL=3

uint32 protocol_version
uint8 operation
string pipeline_id
string request_id
string target_request_id
string session_id
uint64 session_generation
string deployment_fingerprint
# Absolute UTC/system-clock deadline. Zero means no explicit deadline.
builtin_interfaces/Time deadline
# Target observation time in the contract clock domain. Zero when no RTP stream is referenced.
builtin_interfaces/Time observation_timestamp
string prompt
# Parallel arrays identifying RTP-backed observations negotiated through VideoStreamDescriptor.
string[] stream_observation_keys
string[] stream_ids
# Non-streamed observations, including explicit DDS image mode, remain in tensors.
VariantsList tensors
# Timestamped history of non-streamed observations for post-hoc alignment
# with the cloud-selected video capture. Empty when no RTP stream is
# referenced (pure DDS mode stays on the request tick anchor).
uint64[] aligned_timestamps_ns
VariantsList[] aligned_tensors
