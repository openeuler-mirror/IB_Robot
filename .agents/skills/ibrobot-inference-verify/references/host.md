# Host Verification Commands

## When to Read

- Running tiers 1–4 of the inference verification matrix
- Need the exact pytest, build, and ROS mock launch commands
- Need the perception typed-service call scripts

Reusable scripts live in this skill's `scripts/` directory. They encode the
pitfalls listed in `troubleshooting.md` (zsh requirement, stamp matching,
deployment-name resolution); prefer them over hand-rolling.

## Tier 1 — Unit & Contract Tests

```bash
source .shrc_local && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q src/inference_service/tests
```

Pass criteria: all tests pass (no collection errors).

## Tier 2 — ROS Overlay Build

Never invoke `colcon build` directly; always build through the project script
with a clean cache:

```bash
source .shrc_local && ./scripts/build.sh --clean
```

Pass criteria: all packages finished, 0 failures.

## Tier 3 — Policy ROS Mock (CPU/CUDA)

### Generate mock YAML and launch

The generator auto-resolves the per-bundle deployment key from
`inference_manifest.json` (`cpu` vs `torch-cpu` vs `torch_cpu` …), so only
`--device cpu|cuda` is needed:

```bash
python3 .agents/skills/ibrobot-inference-verify/scripts/generate_verify_yaml.py \
  policy --model models/pi05 --device cuda -o /tmp/opencode/pi05_cuda.yaml

zsh .agents/skills/ibrobot-inference-verify/scripts/run_policy_mock.sh \
  /tmp/opencode/pi05_cuda.yaml 94 /tmp/opencode/pi05_cuda.log 420
```

Pass criteria: script prints `RESULT=PASS` (`First inference received:
chunk=<n>, latency=...ms` in the log), plus
`Unified pipeline started: id=policy ... backend=torch`.

Run one launch per (model, deployment) combination; use a **fresh
`ROS_DOMAIN_ID`** each run (93–99, never 42).

### Manual fallback (equivalent steps)

```bash
source .shrc_local && export ROS_DOMAIN_ID=94 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    config_path:=/tmp/opencode/pi05_cuda.yaml \
    control_mode:=model_inference use_sim:=true
```

### Harness rules (do not skip)

- **Runner scripts must be zsh.** `source .shrc_local` under bash + `set -u`
  aborts (ROS setup scripts are not nounset-safe) → `ros2: command not found`.
- **Launch + verification must live in one shell invocation.** Background
  processes are killed when the invoking session/call ends; a launch started in
  one call cannot be probed from the next.
- Deployment keys differ per bundle (`cpu`/`torch-cpu` for policies,
  `torch_cpu`/`torch_cuda` for perception bundles). Config validation rejects
  any deployment missing from the manifest — resolve from the manifest, do not
  guess.
- Ascend-only bundles (`grounding_dino_swint_seq8_1280x720_ascend`,
  `models/grasp`) have no host deployment: skip them and note it in the report.

## Tier 4 — Perception ROS Typed Service

### Generate perception YAML, launch, and call

```bash
python3 .agents/skills/ibrobot-inference-verify/scripts/generate_verify_yaml.py \
  perception --device cpu -o /tmp/opencode/perception_cpu.yaml

zsh .agents/skills/ibrobot-inference-verify/scripts/run_perception_services.sh \
  /tmp/opencode/perception_cpu.yaml 91 /tmp/opencode/perception_cpu.log 600
```

The runner waits until every endpoint from the YAML appears in
`ros2 service list`, then calls siglip2 / ram_plus / sam2
(`--include-grounding` adds GroundingDetect when a host-runnable bundle
exists). Repeat with `--device cuda` for the CUDA tier.

Pass criteria per service: registered endpoint + `success=True` +
`runtime_state=ready`.

### Service contract notes (baked into the caller script)

- **SigLIP2** `EncodeEmbeddings` only encodes the masks supplied in the
  request; an image-only request returns `encoded 0 masks` without running the
  encoder. Send ≥1 mono8 mask **with the exact same header stamp as the image**
  (otherwise `validate_mask_batch` rejects: "mask 0 timestamp does not match
  the source image"). Verified output: 1 embedding, dim=1152.
- **RAM++** `RecognizeTags`: send `score_threshold`; ~136 tags on a synthetic
  red rectangle is normal.
- **SAM2** `GenerateMasks`: automatic mask generation on CPU takes ~100 s, so
  use a generous client timeout (default 240 s in the script). The response
  masks are under `detections.detections` (Detection2D[] with a `mask` image
  field), not a `masks` field.
- Mark perception services `required: false` in verification YAMLs so a broken
  bundle degrades to a failed service call instead of blocking the launch; the
  report should then list it as skipped/failed.

### Expected latency shape (RTX 3090 reference)

| Service | torch_cpu | torch_cuda |
|---------|-----------|------------|
| siglip2 encode (1 mask) | ~3.2 s | ~0.33 s |
| ram_plus tags | ~1.1 s | ~0.26 s |
| sam2 automatic masks | ~110 s | ~7.7 s |

A large CPU↔CUDA gap also confirms the CUDA deployment actually bound the GPU.

## Tier 4b — Speech Models (FullSubNet, ZipVoice)

Speech models do NOT follow the perception preset flow: fullsubnet is a
stateful streaming stage inside `speech_direction_node` (topic-based), while
zipvoice IS a standard `model_service_node` typed service. Both have dedicated
runner scripts.

### FullSubNet (speech_direction pipeline, stateful streaming)

```bash
# 1. prefetch the speech bundles (Ubuntu assets land in models/silero-vad
#    and models/fullsubnet; Ascend OM pairs come from NAS / the 310B export
#    flow and are optional on host):
python3 scripts/download_models.py --models silero-vad,fullsubnet
python3 scripts/verify_speech_direction_assets.py

# 2. real-speech test wav (synthetic audio FAILS Silero VAD):
python3 .agents/skills/ibrobot-inference-verify/scripts/make_speech_wav.py

# 3. run CPU then CUDA:
zsh .agents/skills/ibrobot-inference-verify/scripts/run_fullsubnet.sh cpu 81
zsh .agents/skills/ibrobot-inference-verify/scripts/run_fullsubnet.sh cuda 80

# 4. per-hop latency numbers (the node's [FullSubNetTiming] INFO logs are
#    suppressed by default python logging):
python3 .agents/skills/ibrobot-inference-verify/scripts/bench_fullsubnet.py
```

Pass criteria: `speech_direction_node 已启动 ... degraded=False` in the log +
≥1 `/voice/speech_direction` message (only published when VAD fired AND
FullSubNet enhancement ran — that IS the inference evidence).

Reference numbers (RTX 3090): cuda fb+sb ≈ 0.75 ms/hop, cpu ≈ 36 ms/hop
(cpu slightly exceeds the 32 ms realtime budget per hop).

### ZipVoice (typed model service, standard form)

```bash
zsh .agents/skills/ibrobot-inference-verify/scripts/run_zipvoice.sh cpu 79
zsh .agents/skills/ibrobot-inference-verify/scripts/run_zipvoice.sh cuda 78
```

- Bundle: `models/zipvoice` (deployments `ubuntu_onnx` + `ascend_310p`);
  service `/voice_tts/synthesize` (`ibrobot_msgs/srv/SynthesizeSpeech`).
- CPU works out of the box. CUDA requires onnxruntime-gpu — the runner flips
  `providers` to CUDA-first and restores it on exit, and mounts torch's
  bundled CUDA libs on `LD_LIBRARY_PATH`; it also prints `nvidia-smi`
  compute-apps to prove GPU binding (silent CPU fallback otherwise).
- CUDA only accelerates text_encoder + fm_decoder; the Vocos vocoder is
  hardcoded torch-CPU (partial CUDA coverage — expected).
- Reference numbers (RTX 3090, "你好，异构算力统一推理框架。"): cpu ≈ 3.2 s,
  cuda ≈ 0.42 s (7.5×), identical PCM output.
