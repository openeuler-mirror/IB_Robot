---
name: ibrobot-inference-verify
description: >-
  Unified inference runtime verification on host and board. Use when users mention
  "inference verification", "推理验证", "mock test", "ROS mock", "perception test",
  "policy test", "ACT test", "RAM++ test", "SAM2 test", "SigLIP2 test", "OM test",
  "Torch NPU test", "板端验证", "本机验证", "use_sim true", "platform mock",
  "model_service test", "typed service test", or need to verify that the unified
  inference runtime (ModelRuntimeHandle, ModelSession, RuntimeAssembly) works
  end-to-end through the ROS pipeline. Covers host CPU/CUDA Torch policy and
  perception models, board Ascend OM models, and board Torch NPU.
license: Apache-2.0
compatibility: >-
  Requires IB_Robot workspace with .shrc_local, ROS 2 Humble overlay, LeRobot
  patch stack applied, and model bundles under models/. Board tests require SSH
  access to an Ascend board: default ssh OPi_20T (Ascend 310B); when it is
  unreachable, ask the user for the access method of their 310B (Orange Pi) or
  310P (AI Station) board and the IB_Robot path on it. Perception ROS service
  tests require model bundles with v3 manifest and adapter.json containing
  interface/model_type fields.
---

# IB-Robot Inference Runtime Verification

End-to-end verification of the unified inference runtime through the ROS pipeline
on both host (CPU/CUDA Torch) and board (Ascend 310B OM, Torch NPU).

## When to Use

- After native runtime or inference_service code changes that affect
  `ModelSession`, `ModelRuntimeHandle`, `RuntimeAssembly`, `SessionBuilderRegistry`,
  or the policy/perception pipeline.
- Before merging a PR that touches `inference_service`, `perception_service`,
  `voice_tts_service`, `voice_asr_service`, or `manipulation_service`.
- When a user asks to "验证推理", "test inference", "跑一下 mock", or
  "board inference test".
- After LeRobot patch stack changes that affect policy loading or preprocessing.

## Hard Constraints

- Always `source .shrc_local` before any command (same as `ibrobot-env`).
- Use a **fresh `ROS_DOMAIN_ID`** per run (e.g., 93–99) to avoid stale DDS
  action-server discovery from previous launches.
- Never commit files under `models/` — that directory is `.gitignore`-d.
- Commit only `src/` changes; use `git add <specific-paths>`, never `git add .`.
- Board operations go through `ibrobot-launch`; SSH via `ssh OPi_20T`
  by default. If unreachable, resolve access via "Board Access Resolution".

## Board Access Resolution

Board tiers (5–7) run on an Ascend NPU development board. Resolve the board
target in this order:

1. **Default**: probe the preset target once with
   `ssh -o ConnectTimeout=5 OPi_20T true` (Ascend 310B, openEuler Embedded
   aarch64). Treat timeout, name resolution failure, or connection refusal as
   unreachable.
2. **Fallback**: when `OPi_20T` is unreachable, use the **ask-user tool**
   (interactive question prompt) to request two things from the user:
   - how to access their Ascend board — either a 310B board
     (Orange Pi / 香橙派) or a 310P board (AI Station) — e.g. an SSH alias,
     a host/IP with port and username, or a ready-to-use login command;
   - the absolute path of the IB_Robot workspace on that board.
3. Substitute the answers for `ssh OPi_20T` and `/IB_Robot` in every board
   command below. Never guess hosts, ports, credentials, or board paths;
   probe a user-provided alias once before running verification commands on it.

## Verification Matrix

| Tier | Scope | Method | Platform |
|------|-------|--------|----------|
| 1 | Unit & contract tests | `pytest` | Host |
| 2 | ROS overlay build | `./scripts/build.sh --clean` | Host |
| 3 | Policy ROS mock (ACT) | `ros2 launch ... use_sim:=true platform:=mock` | Host CPU |
| 4 | Perception ROS typed service | `ros2 launch` + `ros2 service call` | Host CPU |
| 5 | Board OM script | Python script via SSH | OPi_20T 310B |
| 6 | Board OM ROS mock | `ros2 launch` on board | OPi_20T 310B |
| 7 | Board Torch NPU (optional) | Python script via SSH | OPi_20T 310B NPU |

Not every PR needs all tiers. Use the table below to pick tiers.

## Tier Selection Guide

| Changed code | Required tiers |
|--------------|----------------|
| `model_sessions/`, `unified_runtime/`, `pipeline/` | 1, 2, 3 |
| `perception_service/`, `model_service_plugins.py` | 1, 2, 4 |
| `voice_tts_service/`, `voice_asr_service/` | 1, 2, 4b |
| `manipulation_service/` | 1, 2 |
| `backends/`, `runtime_composition.py` | 1, 2, 3 |
| LeRobot patch stack | 1, 3, 5 |
| Any `inference_service` change | 1, 2, 3, 5 |

## Core Workflow

1. **Tier 1 — Unit tests**: `pytest -q src/inference_service/tests`
2. **Tier 2 — Build**: `./scripts/build.sh --clean` (never raw `colcon build`)
3. **Tier 3 — Policy mock**: generate YAML with
   `scripts/generate_verify_yaml.py policy`, launch with
   `scripts/run_policy_mock.sh` (zsh), check `RESULT=PASS`
   (`action_dispatcher` reports `First inference received`)
4. **Tier 4 — Perception service**: generate YAML with
   `scripts/generate_verify_yaml.py perception`, launch + call each typed
   service via `scripts/run_perception_services.sh` (launch and calls must
   stay in one shell invocation)
4b. **Tier 4b — Speech models**: fullsubnet via
   `scripts/run_fullsubnet.sh` (streaming pipeline, topic-based evidence) +
   `scripts/bench_fullsubnet.py`; zipvoice via `scripts/run_zipvoice.sh`
   (typed service). See `references/host.md` Tier 4b for prerequisites.
5. **Tier 5 — Board OM**: SSH to the resolved board target (default `OPi_20T`,
   see "Board Access Resolution"), run OM script per model
6. **Tier 6 — Board ROS mock**: Launch ROS mock on board, call typed services

## Skill Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_verify_yaml.py` | Generate mock YAML for policy / perception tiers; auto-resolves the deployment key from each bundle manifest (`--device cpu\|cuda`) |
| `scripts/run_policy_mock.sh` | zsh: launch policy mock, poll for `First inference received`, tear down; prints `RESULT=PASS/TIMEOUT/FAIL` |
| `scripts/run_perception_services.sh` | zsh: launch perception mock, wait for all endpoints, run the caller, tear down |
| `scripts/call_perception_services.py` | Call siglip2 / ram_plus / sam2 typed services with a synthetic image (handles stamp matching and response layouts) |
| `scripts/make_speech_wav.py` | Build a real-speech 6ch 16k wav (LibriSpeech via HF hub + Silero pre-scoring; synthetic audio fails VAD) |
| `scripts/run_fullsubnet.sh` | zsh: speech_direction wav-replay run (cpu/cuda config generated inline); pass = degraded=False + direction messages |
| `scripts/bench_fullsubnet.py` | Per-hop fb/sb latency for the stateful torch executor (node timing logs are suppressed) |
| `scripts/run_zipvoice.sh` | zsh: typed `/voice_tts/synthesize` run; cuda mode flips providers (restored on exit) + mounts torch CUDA libs + nvidia-smi proof |
| `scripts/call_tts.py` | SynthesizeSpeech caller with segment details |
| `scripts/run_board_tier6.sh` | POSIX sh: board ROS mock orchestration (launch + typed service calls + teardown) |

Harness rules: run shell scripts with **zsh** (bash + `set -u` breaks
`.shrc_local` sourcing) and keep launch + verification in one shell
invocation. Detail: `references/host.md` and
`references/troubleshooting.md`.

## Internal References

| Purpose | Reference |
|---------|-----------|
| Host verification commands (tiers 1–4): pytest, build, mock YAML, service call scripts | `references/host.md` |
| Board verification commands (tiers 5–6): SSH, OM scripts, board ROS mock | `references/board.md` |
| Pass/fail criteria, known issues, troubleshooting (DDS stale, adapter.json, batch dim, harness pitfalls) | `references/troubleshooting.md` |
