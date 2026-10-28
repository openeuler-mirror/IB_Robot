# `scripts/setup/` — workspace bootstrap modules

This directory hosts the modular pieces sourced by `scripts/setup.sh`:

| File | Responsibility |
| --- | --- |
| `detect.sh` | OS / arch / Python detection; resolves `IBR_HOST_*` and `IBR_LEROBOT_PROFILES`. |
| `benchmark_profile.sh` | Ubuntu GPU preflight, validated PyTorch/TorchVision/TorchCodec CUDA profile, and provider ABI boundary for `--with-benchmark`; NVIDIA driver >=560.28.03 is required, local nvcc is not required and GraspGen is excluded. The generated benchmark constraints include `mujoco<3.9` (hf-libero 0.1.4's cap). |
| `platforms/<id>.sh` | Per-platform package managers, ROS source, and `platform_lerobot_profiles()` defaults. |
| `lerobot_patches.sh` | Drives `git am` of the curated lerobot patch stack into `libs/lerobot`. |
| `lerobot_resolve_active.py` | Resolves the active lerobot tag from `INDEX.yaml`; cross-checks the per-tag `manifest.yaml`. |
| `lerobot_filter_series.py` | Reads `manifest.yaml` + host facts, prints the patch series that actually applies. |
| `tests/test_lerobot_filter.sh` | Regression fixtures pinned to the canonical 3-platform matrix + tag-binding cases. |
| `tests/test_cann_torch_versions.sh` | Tests CANN detection and openEuler Torch ABI selection without a CANN installation. |
| `tests/test_openeuler_cv_bridge_abi.sh` | Tests installation of the OpenCV 4.13-compatible openEuler `cv_bridge` RPM. |
| `tests/test_setup_profile_training.sh` | Asserts `--profile training` never triggers ROS 2 / rosdep / colcon while `full` / `inference` keep their behaviour. |
| `tests/test_mujoco_sim_pin.sh` | Tests the three MuJoCo-sim-pin install orders, the benchmark `mujoco<3.9` constraint, the inference-profile skip, and the stdlib-only verification on an unbuilt workspace (issue #190). |

## Setup profiles (dependency-scope parameterization)

`setup.sh` controls the dependency installation scope via `--profile <name>`
(or the `IBR_SETUP_PROFILE` environment variable):

| Profile | Purpose |
| --- | --- |
| `full` (default) | Complete development workspace: hardware, teleop, perception, grasp, voice, sim, and dev tooling. |
| `inference` | Minimal policy-inference runtime for edge-cloud collaborative inference verification hosts. |
| `training` | **Ubuntu 22.04 only.** LeRobot training host: creates the venv, applies the LeRobot patch stack, and installs LeRobot plus the training dependencies only. No ROS 2 / rosdep / colcon. |

The single source of truth for the capability matrix is `_SETUP_PROFILE_TABLE` in [`common.sh`](./common.sh): one row per profile, cells separated by `|` (columns are the `rosdep` cell, the `submodules` scope, the `verify` strictness, and the comma-separated `features`), and any cell may be wrapped onto indented continuation lines — the physical lines are glued together before the cells are split and cell whitespace is normalised. The `rosdep` cell reads `<scope>:<paths>` (for example `full:src`, `narrow:src/a,src/b`, `skip:-`), so a profile cannot claim `skip` while still resolving a path set. `setup_profile_enables <feature>` answers "what is installed" and `setup_profile_get rosdep|rosdep_paths|submodules|verify` answers scope, paths, and verification strictness, so **adding a profile means adding one row**.

Differences of `inference` vs. `full`:

- **Submodules**: only `libs/lerobot` is initialized (LiDAR / bringup submodules and the ROS third-party patch stacks are skipped).
- **rosdep**: scope narrowed to `ibrobot_msgs`, `tensormsg`, `inference_manifest`, `torch_models`, `inference_service`, `model_utils`, `dataset_tools`; `robot_config` is still built (SSOT YAML) but excluded from rosdep resolution so nav2 / slam_toolbox / gz / camera / LiDAR bringup dependencies are not pulled in.
- **Python venv**: skips `hardware.txt`, WebPhone, `dev-tools.txt`, voice-tts, perception (SAM2 / Grounding-DINO / RAM++ / SigLIP2 and the audited wheels), FullSubNet, GraspGen, and the pre-commit / gitlint hooks; Ubuntu installs only `requirements/inference.txt` (ONNX toolchain), while openEuler keeps its complete OS dependencies and the CANN-matched `torch_npu` is installed by the accelerator axis; the lerobot editable install drops the teleop-only `kinematics` extra.
- **CANN Torch ABI**: LeRobot installs normally from its own declared dependencies (with the selected extras); `accel-cann-8.1.txt` (the hardware-validated 2.5.1 set) or `accel-cann-8.3.txt` (the officially paired 2.8.0 set) then overrides Torch per the detected CANN release, and the matching `constraints-cann-<release>.txt` protects it through `PIP_CONSTRAINT` for every later pip step. `openeuler-24.03.txt` supplies the OS packages only.
- **OpenCV / cv_bridge ABI**: openEuler aarch64 installs `opencv>=4.13.0` together with `ros-humble-cv-bridge>=3.2.1-2.oe2403`, the RPM rebuilt for OpenCV 4.13. Verification exercises BGR/RGB conversion, non-contiguous arrays, and mono8/mono16/32FC1 round trips in the workspace interpreter; it accepts later compatible RPM revisions and keeps the NumPy 1.x-compatible Python `cv2` wheel.
- **Verification**: tracing checks are skipped (lttng / tracetools are full-workspace diagnostics), and so is the Ubuntu tracing-tools post-install hook (`platform_post_install_rosdeps`).

```bash
# Edge-cloud collaborative inference verification host:
./scripts/setup.sh --yes --profile inference
./scripts/build.sh --packages-up-to inference_service
```

Differences of `training` vs. `full` (for Ubuntu 22.04 hosts that only run LeRobot training):

`training` supports only `ubuntu-22.04`. Other platforms are rejected after platform detection, before dependency previews, submodule synchronization or system installation. Neither `--only-patch` nor `--skip-verify` bypasses this restriction. The platform support of `full` / `inference` is unchanged. `training` cannot be combined with `--with-benchmark`; use `full` for benchmark setup.

- **ROS 2**: never installed. `scripts/install_ros.sh`, rosdep, colcon, and the ROS third-party patch stacks are all skipped, and no `install/` overlay is required.
- **Submodules**: only `libs/lerobot` is initialized.
- **System packages**: only the Python bootstrap (`python3-venv` / `python3-pip`) and `ffmpeg` (torchcodec video decoding) are required. All profiles share the Ubuntu dependency installation hook; apt handles already-installed packages idempotently. FFmpeg is declared explicitly instead of relying on ROS/OpenCV to install it transitively.
- **Python venv**: keeps `--system-site-packages` (same layout as `full` / `inference`) and installs only `base.txt`, `requirements/training.txt` (TensorBoard), and the LeRobot editable install (extras: `smolvla,pi,diffusion,dataset,deepdiff-dep`). It skips `hardware.txt`, WebPhone, `dev-tools.txt`, voice-tts, perception, FullSubNet, GraspGen, colcon, the `empy` / `rosdep` pins, the pre-commit / gitlint hooks, and the teleop-only `kinematics` extra.
- **NumPy / OpenCV**: the ROS 2 ABI pins (`numpy==1.26.4` / `opencv<4.12`) are deliberately **not** applied; LeRobot's own resolution is kept. CANN hosts now take that layer from the **accelerator axis** (see [accelerator dependency placement in the Chinese README](./README.md#加速器轴依赖落位)).
- **Verification**: ROS 2 / rosdep / colcon / cv_bridge / pygraphviz / tracing checks are skipped; the venv, `import lerobot`, the `lerobot.policies.act` API, and the `lerobot-train` executable are verified. When torchcodec is installed its decoder module is also loaded, surfacing missing FFmpeg libraries or ABI mismatches.
- **`.shrc_local`**: reads `venv/.ibrobot_setup_profile`; with `training` it labels the banner `(profile: training)`, reports the missing ROS 2 / `install/` / CANN entries as expected, and stops advertising the `cb` / `cbp` / `src` colcon aliases.

> Naming note: this `training` value is the **install-scope** profile. It is a different namespace from the LeRobot patch-filter `training` profile (`IBR_LEROBOT_PROFILES` / `--lerobot-profiles`); `--profile training` leaves the patch filter set untouched, so the training-relevant patches are still applied.

```bash
# Ubuntu 22.04 training host (no ROS 2):
./scripts/setup.sh --yes --profile training
source .shrc_local && lerobot-train --dataset.repo_id=<repo_id> --policy.type=act
```

## MuJoCo simulation version rule (issue #190)

The MuJoCo simulation platforms (SO-101, and the LeKiwi lidar stage one) run
on `mujoco_ros2_control` 0.1.2 / `mujoco_vendor` 0.1.1 (libmujoco 3.12.0)
plus `mujoco_3d_lidar` 0.1.2 (the MuJoCo lidar engine extension, which the
LeKiwi MID-360 simulation needs). The models are compiled by the workspace
Python `mujoco`, and its version must exactly equal the simulator's
libmujoco, therefore:

- The Python pin moved into a dedicated `requirements/mujoco-sim.txt`
  (`mujoco==3.12.0`), installed only on the Ubuntu `full` profile without
  benchmark dependencies and without `hf-libero`/`robosuite` already in the
  venv (`python_venv.sh::mujoco_sim_pin_mode`).
- The LIBERO benchmark and the MuJoCo ROS simulation are **mutually exclusive
  in one venv**: hf-libero 0.1.4 declares `mujoco<3.9` (robosuite 1.4 breaks
  on MuJoCo >= 3.10), so the pip constraints generated for `--with-benchmark`
  include `mujoco<3.9`; a venv that already holds a benchmark install skips
  the 3.12.0 pin with a warning — **recreate the venv** to use the MuJoCo
  simulation.
- `verify_env.sh::verify_mujoco_sim` re-checks the same versions with a
  stdlib-only snippet where the pin would be installed, and additionally
  checks the `mujoco_3d_lidar` ament `mujoco_plugins` resource, its package
  version (>= 0.1.2) and the extension library it points at (the ROS prefix
  is derived from `SETUP_ROS_SETUP_PATH`); every other profile and benchmark
  venv gets an info line at most. A missing package points at re-running
  `./scripts/setup.sh` with sudo (its rosdep step installs it); an installed
  package that is too old or broken names the apt command, because rosdep
  never upgrades an installed package.
- The runtime adds a strict check
  (`robot_runtime.mujoco.check_simulator_versions`): on an inconsistent or
  too-old simulator, `sim_platform:=mujoco` fails before model composition
  with an installation diagnostic.
- openEuler ships no MuJoCo rpm: beyond the existing
  `mujoco_ros2_control` / `mujoco_ros2_control_msgs` rosdep skip keys, this
  change adds `mujoco_3d_lidar` (the lidar engine extension has no openEuler
  rpm either). `imu_sensor_broadcaster` belongs to the ros2_controllers
  family (same source as `joint_state_broadcaster`) and is not skipped.

## CANN dependency layers

| File under `requirements/` | Usage |
| --- | --- |
| `accel-cann-8.1.txt` | Accelerator axis: CANN 8.1 Torch 2.5.1/Torch-NPU 2.5.1/TorchVision 0.20.1 (hardware-validated on Ascend310P1), installed on any OS where an Ascend NPU is detected. |
| `constraints-cann-8.1.txt` | Version constraints, not an install list. A function-local `PIP_CONSTRAINT` protects every pip invocation and subprocess; all protected versions are checked after the final install. |
| `accel-cann-8.3.txt` | Accelerator axis: CANN 8.3 Torch 2.8.0/Torch-NPU 2.8.0/TorchVision 0.23.0 (official pairing with aarch64 wheel availability verified; not hardware-validated yet), installed on any OS where an Ascend NPU is detected. |
| `constraints-cann-8.3.txt` | Same role as the 8.1 constraint file, for the 8.3 triple. |

Both profiles install LeRobot normally first (with the selected extras); `accel-cann-<release>.txt` then overrides to the release's paired set, and only afterwards do the constraints activate to protect every later pip install.
The Torch ABI, Transformers, tokenizers, safetensors, Pillow and OpenCV wheel constraints are independent of profile.
NumPy retains the repository's existing final ROS ABI restoration step and is verified as 1.26.4 after installation. Full's Rerun metadata requires NumPy 2, so NumPy is not included in the all-install constraint.
Transformers 5.3.0 preserves the frozen accuracy baseline. Its known security advisories, model upgrades, and other-model compatibility remain separate work.

## LeRobot patch dispatch

`libs/lerobot` ships one patch series per supported upstream tag under
`third_party/patches/lerobot/<tag>/`. The single source of truth that
selects which tag is active is `third_party/patches/lerobot/INDEX.yaml`
(see [Multi-tag layout](#multi-tag-layout) below). Within the active
tag, **not every patch applies to every host.** The applier filters the
raw `series.txt` through `lerobot_filter_series.py`, which gates each
patch on independent predicates declared in `manifest.yaml`:

- `python_min` / `python_max` — bounded interval (`>=3.10`, `<3.11`, ...).
- `profiles` — set intersection against the active profile set.

The active profile set is resolved with this precedence (highest first):

1. `--lerobot-profiles core,ascend,...` CLI flag on `setup.sh`.
2. `IBR_LEROBOT_PROFILES=core,ascend,...` environment variable.
3. `platform_lerobot_profiles` callback in the active platform script.
4. The fallback `core,ros,hardware,dev`.

### Common overrides for hardware bring-up

```bash
# Force the Ascend NPU patches on a desktop verifying inference parity:
./scripts/setup.sh --yes --lerobot-profiles core,ros,hardware,ascend

# Bring up an OpenHarmony device as if it were a vanilla core target:
./scripts/setup.sh --yes --lerobot-profiles core

# Explicitly bypass the filter (last-resort escape hatch — applies the raw
# series.txt verbatim, even patches that do not match host facts):
IBR_LEROBOT_FORCE_UNFILTERED=1 ./scripts/setup.sh --yes
```

### Diagnosing the filter

The applier prints a `[CTX] python=X.Y profiles=...` audit line and a
`KEEP` / `SKIP` decision for every patch. To preview without touching
`libs/lerobot`:

```bash
IBR_HOST_PYTHON_VERSION=3.11 \
IBR_LEROBOT_PROFILES=core,ros,hardware,openeuler \
  python3 scripts/setup/lerobot_filter_series.py \
    --manifest third_party/patches/lerobot/v0.6.0/manifest.yaml \
    --series   third_party/patches/lerobot/v0.6.0/series.txt
```

### When to add a new patch

1. Add the patch file under the active tag's directory
   (`third_party/patches/lerobot/<active_tag>/`; `<active_tag>` is whatever
   `INDEX.yaml.active_tag` currently points at).
2. Append it to that tag's `series.txt`.
3. Register a `patches[]` entry in the same directory's `manifest.yaml`
   declaring **explicit** `python_min` / `python_max` / `profiles`
   predicates. Default to the narrowest set that is known to be safe;
   broaden later once verified on additional platforms.
4. Add a fixture line to `scripts/setup/tests/test_lerobot_filter.sh` so the
   pre-commit hook catches accidental scope changes.

## Multi-tag layout

`third_party/patches/lerobot/INDEX.yaml` is the single source of truth
for which upstream lerobot tag the setup script targets. Each tag owns
its own directory of patches plus a `manifest.yaml` that re-declares the
binding so the resolver can fail closed on drift:

```text
third_party/patches/lerobot/
├── INDEX.yaml                  # active_tag + supported_tags + archived_tags
├── v0.6.0/                     # current active tag, one directory per supported tag
│   ├── manifest.yaml           # declares lerobot_tag + lerobot_commit_range
│   ├── series.txt
│   └── 0001-*.patch ...
└── v0.6.0/                     # (example: future tag added side-by-side)
    └── ...
```

The resolver (`lerobot_resolve_active.py`) enforces these invariants:

| Invariant | Failure mode |
| --- | --- |
| `INDEX.active_tag` matches a `supported_tags[]` entry | resolver exits 1 with "active_tag not found" |
| The active entry's `dir` exists with `manifest.yaml` + `series.txt` | resolver exits 1 with "directory does not exist" |
| `INDEX.supported_tags[i].upstream_commit` == `manifest.lerobot_commit_range.min` | resolver exits 1 with "INDEX vs manifest mismatch" |
| `INDEX.active_tag` is **not** present in `archived_tags[]` | resolver exits 1 with "archived tag" |
| `libs/lerobot` HEAD prior to patch application falls inside `manifest.lerobot_commit_range` | filter exits 1 with "HEAD not in commit_range" |

### Upgrading to a new lerobot tag

1. Bump the `libs/lerobot` submodule to the new upstream tip and record
   the resulting sha as `<NEW_COMMIT>`.
2. Create `third_party/patches/lerobot/<NEW_TAG>/` and rebase the
   patches you need on top of `<NEW_COMMIT>` (drop / port any that no
   longer apply). Author a fresh `manifest.yaml` with:
   ```yaml
   lerobot_tag: <NEW_TAG>
   lerobot_commit_range:
     min: <NEW_COMMIT>
     max: <NEW_COMMIT>     # widen later if you accept fast-forward upstreams
   ```
3. Append a `supported_tags[]` entry under `INDEX.yaml` pointing at the
   new directory, with matching `upstream_commit` and a unique
   `branch_name` (convention: `ibrobot/lerobot-<NEW_TAG>-patched`).
4. Flip `INDEX.yaml.active_tag` to `<NEW_TAG>`. Optionally move the
   previous tag's entry from `supported_tags[]` into `archived_tags[]`
   once it is no longer maintained — archived entries stay in tree for
   audit but the resolver refuses to use them.
5. Add fixture coverage for the new tag in
   `scripts/setup/tests/test_lerobot_filter.sh` and run the regression
   harness; both the resolver and the per-platform filter cases must
   pass before submitting.

### Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `lerobot_filter_series.py failed with exit 1`, "PyYAML required but missing" | venv lacks `python3-yaml` | `pip install pyyaml` in the workspace venv, or set `IBR_LEROBOT_FORCE_UNFILTERED=1` to skip filtering. |
| `failed to parse manifest`, exit 1 | `manifest.yaml` is malformed | Lint the YAML; the applier fails closed by design. |
| `lerobot_resolve_active.py failed`, "active_tag not found" | `INDEX.yaml.active_tag` does not match any `supported_tags[]` entry | Add the entry, or fix the `active_tag` value. |
| `lerobot_resolve_active.py failed`, "INDEX vs manifest mismatch" | `INDEX.supported_tags[i].upstream_commit` and `manifest.lerobot_commit_range.min` disagree | Update one to match the other; they MUST be kept in sync. |
| `libs/lerobot HEAD ... is not in the manifest commit_range` | submodule HEAD diverged from the active tag's declared range | Either checkout the expected base, fast-forward the manifest range, or add a new tag directory + flip `INDEX.yaml.active_tag`. |
| Worktree dirty, applier aborts | local edits in `libs/lerobot` | Stash / commit your edits, or set `IBR_LEROBOT_FORCE_REBUILD=1` to discard them and rebuild the patched branch from scratch. |
