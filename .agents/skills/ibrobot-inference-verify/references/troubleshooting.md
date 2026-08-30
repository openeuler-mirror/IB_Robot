# Troubleshooting

## When to Read

- A verification tier failed with an unfamiliar error
- DDS or action_dispatcher behaves unexpectedly
- Model service plugin fails to initialize
- Board SSH or CANN environment issues

## Pass / Fail Criteria

| Tier | Pass criteria |
|------|----------------|
| 1 (pytest) | All tests pass, 0 collection errors |
| 2 (build) | All packages finished, 0 failures |
| 3 (policy mock) | `First inference received: chunk=100` in launch output |
| 4 (perception service) | Service registered + `success=True` or `runtime_state=ready` |
| 5 (board OM script) | `status=passed` with correct shapes + `finite=True` |
| 6 (board ROS mock) | Pipeline started + service `success=True` |
| 7 (board NPU) | `action_shape=(100, 6) finite=True` (slow, ~610 s) |

## Known Issues

### DDS stale action-server discovery

**Symptom**: `action_dispatcher` shows `in_progress=True gen=0` forever; pipeline
says "already processing another operation" even after the first launch exits.

**Cause**: Previous `ros2 launch` left DDS discovery entries; the new
`action_dispatcher` discovers a dead action server and dispatches a goal that
never completes, permanently blocking the pipeline.

**Fix**: Use a fresh `ROS_DOMAIN_ID` for each test run (e.g., 93, 94, 95, …).
Never reuse `ROS_DOMAIN_ID=42` for tests. If already stuck, kill all ROS
processes and wait 10 s before relaunching.

### adapter.json identity mismatch

**Symptom**: `model_service_node` crashes with "adapter identity mismatch:
expected {'interface': 'tensor_model', 'model_type': 'ram_plus', ...}, got
{'family': 'ram_plus', ...}".

**Cause**: `models/<bundle>/assets/adapter.json` still has the legacy `family`
field instead of v3 `interface` and `model_type`.

**Fix**: Add `interface: "tensor_model"` and `model_type: "<model_type>"` and
`operation: "<operation>"` to each `adapter.json`. Remove `family` if it
duplicates `model_type`.

Note: `models/` is `.gitignore`-d, so this is a local-only fix. Board bundles
may already have the correct fields.

### builder_options not unwrapped (SessionBuilderRegistry)

**Symptom**: `model_service_node` crashes with "missing required keyword-only
argument 'adapter'" when launching perception services.

**Cause**: `SessionBuilderRegistry.create()` filtered out `builder_options`
because the builder signature doesn't have a `builder_options` parameter.

**Fix**: Ensure `registry.py` has the `builder_options` unwrapping code:
```python
builder_options = kwargs.pop("builder_options", None)
if builder_options:
    kwargs.update(dict(builder_options))
```

### LeRobot `typing.Self` import error

**Symptom**: `ImportError: cannot import name 'Self' from 'typing'` on Python
3.10.

**Cause**: LeRobot v0.6.0 uses `typing.Self` (Python 3.11+); the v0.6.0 patch
stack backports it via `typing_compat` but the patch may not be applied.

**Fix**: Run `bash scripts/setup.sh --only-patch --yes` to apply the LeRobot
patch stack. On the board, use `IBR_LEROBOT_FORCE_REBUILD=1` if the board
is still on v0.5.1 patched branch.

### ACT policy "stack expects each tensor to be equal size"

**Symptom**: `RuntimeError: stack expects each tensor to be equal size, but
got [3, 1024] at entry 0 and [1024] at entry 1`.

**Cause**: LeRobot preprocessor outputs tensors without batch dimension;
`predict_action_chunk` treats the channel dimension as batch, causing
encoder layers to produce mismatched shapes.

**Fix**: Ensure `LeRobotTorchModelSession._execute` adds batch dimension
when the input tensor's dimension count equals the manifest-declared shape
dimension count (via `unsqueeze(0)`).

### Perception adapter identity mismatch on board

**Symptom**: a grasp or semantic perception plugin reports that its required
adapter identity does not match the bundle, for example
`tensor_model/sam2/prompt` versus `tensor_model/sam2/automatic`.

**Cause**: SAM2 box-prompt grasp segmentation and SAM2 automatic semantic
mapping are different service contracts. Grounding-DINO grasp detection is
also the raw `grounding_dino/detect` contract. Do not deploy these bundles under
one shared path or overwrite one with another.

**Fix**: Use `models/sam2.1_hiera_tiny_prompt_ascend` for
`SegmentDetectionsPlugin` and the official `sam2/automatic` bundle for
`SAM2GenerateMasksPlugin`. Verify the selected bundle's manifest and
`assets/adapter.json` before starting the model service.

### Board contract_mock crash with minimal YAML

**Symptom**: `contract_mock` crashes with `AttributeError: 'str' object has
no attribute 'get'` when using a hand-written minimal YAML.

**Cause**: The minimal YAML has `peripherals: {cameras: []}` (a dict with
empty list), but the loader expects `peripherals` to be a list of peripheral
dicts (copied from the production YAML).

**Fix**: Always start from a copy of the production `so101_single_arm.yaml`
and override fields, rather than writing a minimal YAML from scratch.

### Board RTPS_READER_HISTORY payload size error

**Symptom**: `RTPS_READER_HISTORY Error: Change payload size of '32' bytes is
larger than the history payload size of '19' bytes`.

**Cause**: DDS FastRTPS on openEuler Embedded has a small default payload
history; some message types exceed it.

**Impact**: Non-fatal warning; does not block inference or service calls.

## Host Verification Harness Issues (2026-08 batch verification)

### Harness script under bash + `set -u` cannot find ros2

**Symptom**: A shell script that runs `set -u` then `source .shrc_local`
prints `ros2: command not found`; under bash the source step can even hang.

**Cause**: ROS setup scripts are not nounset-safe; `set -u` aborts the source
mid-way (hidden by `2>/dev/null`), and `.shrc_local` is written for zsh.

**Fix**: Run harness scripts with `zsh`, and drop `set -u` (or set it only
after sourcing). All `scripts/run_*.sh` in this skill already comply.

### Background launch dies between verification steps

**Symptom**: `ros2 service list` showed the endpoints during one command, but
a follow-up call script gets `SERVICE_NOT_AVAILABLE`.

**Cause**: Background processes are killed when the invoking shell/session
ends; the launch from the previous invocation no longer exists.

**Fix**: Keep launch + readiness wait + service calls in ONE shell
invocation (`scripts/run_perception_services.sh` does exactly this).

### Deployment key naming differs per bundle

**Symptom**: Launch config validation fails: `Deployment 'torch_cpu' is not
present in .../inference_manifest.json; available deployments: [...]`.

**Cause**: Policy bundles use `cpu` / `torch-cpu` / `torch-cuda` while
perception bundles use `torch_cpu` / `torch_cuda`; guessing the wrong key is
rejected at config-validation time (this is by design, not a bug).

**Fix**: Resolve the key from the bundle manifest
(`scripts/generate_verify_yaml.py --device cpu|cuda` does this automatically).

### Ascend-only bundles cannot be verified on host

**Symptom**: Same validation error as above, and the manifest only lists
`ascend_310p` (or `ascend_310b`).

**Known bundles**: `models/grounding_dino_swint_seq8_1280x720`,
`models/grasp` (graspgen).

**Fix**: Skip on host and mark as "board-only" in the verification report;
verify via tiers 5–6 instead of forcing a host deployment.

### SigLIP2 returns "encoded 0 masks" or "timestamp does not match"

**Symptom 1**: `success=True` but `encoded 0 masks`, embeddings empty.

**Cause**: `EncodeEmbeddings` only encodes the masks in the request; image-only
requests short-circuit by contract.

**Symptom 2**: `success=False`, `mask 0 timestamp does not match the source
image`.

**Cause**: The mono8 mask must carry the exact same `header.stamp` as the
image; generating them with two `get_clock().now()` calls differs by nanoseconds.

**Fix**: Share one stamp across image and masks; always send ≥1 mask with
candidate labels. `scripts/call_perception_services.py` handles both.

### GenerateMasks response field mismatch in test scripts

**Symptom**: `AttributeError: 'GenerateMasks_Response' object has no attribute
'masks'`.

**Cause**: The masks are returned inside `detections.detections`
(`DetectionArray` → `Detection2D[]`, each with a `mask` image field), not a
top-level `masks` array.

**Fix**: Count/inspect via `len(response.detections.detections)`.

### SAM2 CPU calls hit client-side timeout

**Symptom**: `CALL_TIMEOUT(120s)` for `/perception/sam2/generate_masks` on CPU.

**Cause**: Automatic mask generation takes ~100 s per call on CPU
(~7.7 s on CUDA).

**Fix**: Use ≥240 s client timeout for sam2 on CPU; prefer the CUDA tier for
sam2 iteration.

## Speech Model Issues (2026-08-29 batch verification)

### ZipVoice MODEL_NOT_READY: "ModelSession.__init__() got an unexpected keyword argument 'domains'"

**Symptom**: `/voice_tts/synthesize` returns `MODEL_NOT_READY` with that
message in `model_service_node` log.

**Cause**: Refactor commit `31633ce1` (migrate model sessions to native
runtime) removed the `domains` parameter from `ModelSession.__init__`;
`voice_tts_service/zipvoice_onnx_adapter.py` was the one stale caller.
Masked in production by `exit_on_init_failure: false` graceful degradation.

**Fix**: Delete the `domains=kwargs.pop("domains", None),` line from the
`super().__init__` call (one-line change; keep it in mind until landed).

### speech_direction host layout gaps (issue #125)

**Symptom**: `require_configured_models` fails on host for missing files, or
Silero onnx runs from a `*.om` path.

**Cause**: `FullSubNetConfig`/`VadConfig` defaults point at board-side
artifact paths (`artifacts/ascend/...`) and are NOT ROS parameters; the torch
backends bypass bundle-manifest resolution entirely.

**Fix (local, no code change)**: copy the cumulative manifest json into
`models/voice_asr/artifacts/ascend/fullsubnet/` and the
`silero_vad_v5.onnx` content to
`artifacts/ascend/silero_vad/silero_vad_v6_310p_mixed16.om`
(onnxruntime loads by content). Long-term fix tracked in issue #125.

### Synthetic audio never triggers Silero VAD

**Symptom**: speech_direction runs, degraded=False, but zero
`/voice/speech_direction` messages.

**Cause**: Harmonic/formant synthesis scores <0.6 on Silero VAD (it is
trained to reject non-speech).

**Fix**: Use real speech — `scripts/make_speech_wav.py` pulls LibriSpeech
dummy samples (HF hub) and pre-scores them with the repo SileroVadEngine.

### onnxruntime-gpu CUDA EP silently falls back to CPU

**Symptom**: Synthesis succeeds with CUDA-first providers but latency ≈ CPU
baseline; log shows `Failed to load library libonnxruntime_providers_cuda.so
with error: libcublasLt.so.12 ...` / `libcurand.so.10 ...`.

**Cause**: ort-gpu needs CUDA 12 runtime libs that are not on the default
library path (only torch's bundled copies exist in the venv).

**Fix**: Mount the torch-cu126 nvidia lib dirs on `LD_LIBRARY_PATH`
(`run_zipvoice.sh cuda` does this: cublas/cudnn/curand/cuda_runtime/
cuda_nvrtc/cufft/cusolver/cusparse/nvjitlink). Always confirm with
`nvidia-smi --query-compute-apps` — absence of the node pid means CPU
fallback.

### FullSubNet timing logs invisible

**Symptom**: `fullsubnet_timing_enabled: true` but no `[FullSubNetTiming]`
lines in the launch log.

**Cause**: The enhancement modules use stdlib `logging` INFO, which ROS 2
launch output suppresses by default.

**Fix**: Use `scripts/bench_fullsubnet.py` for per-hop numbers; the ROS-level
proof is `degraded=False` + direction messages.
