# `scripts/setup/` — workspace bootstrap modules

This directory hosts the modular pieces sourced by `scripts/setup.sh`:

| File | Responsibility |
| --- | --- |
| `detect.sh` | OS / arch / Python detection; resolves `IBR_HOST_*` and `IBR_LEROBOT_PROFILES`. |
| `benchmark_profile.sh` | Ubuntu GPU preflight, validated PyTorch/TorchVision/TorchCodec CUDA profile, and provider ABI boundary for `--with-benchmark`; NVIDIA driver >=560.28.03 is required, local nvcc is not required and GraspGen is excluded. |
| `platforms/<id>.sh` | Per-platform package managers, ROS source, and `platform_lerobot_profiles()` defaults. |
| `lerobot_patches.sh` | Drives `git am` of the curated lerobot patch stack into `libs/lerobot`. |
| `lerobot_resolve_active.py` | Resolves the active lerobot tag from `INDEX.yaml`; cross-checks the per-tag `manifest.yaml`. |
| `lerobot_filter_series.py` | Reads `manifest.yaml` + host facts, prints the patch series that actually applies. |
| `tests/test_lerobot_filter.sh` | Regression fixtures pinned to the canonical 3-platform matrix + tag-binding cases. |
| `tests/test_cann_torch_versions.sh` | Tests CANN detection and openEuler Torch ABI selection without a CANN installation. |

## Setup profiles (dependency-scope parameterization)

`setup.sh` controls the dependency installation scope via `--profile <name>`
(or the `IBR_SETUP_PROFILE` environment variable):

| Profile | Purpose |
| --- | --- |
| `full` (default) | Complete development workspace: hardware, teleop, perception, grasp, voice, sim, and dev tooling. |
| `inference` | Minimal policy-inference runtime for edge-cloud collaborative inference verification hosts. |

Differences of `inference` vs. `full`:

- **Submodules**: only `libs/lerobot` is initialized (LiDAR / bringup submodules and the ROS third-party patch stacks are skipped).
- **rosdep**: scope narrowed to `ibrobot_msgs`, `tensormsg`, `inference_manifest`, `torch_models`, `inference_service`, `model_utils`, `dataset_tools`; `robot_config` is still built (SSOT YAML) but excluded from rosdep resolution so nav2 / slam_toolbox / gz / camera / LiDAR bringup dependencies are not pulled in.
- **Python venv**: skips `hardware.txt`, WebPhone, `dev-tools.txt`, voice-tts, perception (SAM2 / Grounding-DINO / RAM++ / SigLIP2 and the audited wheels), FullSubNet, GraspGen, and the pre-commit / gitlint hooks; Ubuntu installs only `requirements/inference.txt` (ONNX toolchain), while openEuler keeps its complete platform dependencies and a CANN-matched `torch_npu`; the lerobot editable install drops the teleop-only `kinematics` extra.
- **CANN 8.1 ABI**: installs LeRobot's non-Torch runtime dependencies from `requirements/lerobot-v0.6-cann-8.1.txt`; the full profile adds dataset/kinematics and Transformers 5.5 from `lerobot-v0.6-cann-8.1-full.txt`, while the local-only inference profile pins `transformers==5.3.0` through `lerobot-v0.6-cann-8.1-inference.txt` to preserve frozen PI0.5 accuracy (remote code and `save_pretrained` are not used); editable LeRobot is installed with `--no-deps`, and setup pins and verifies `torch==2.5.1`, `torch_npu==2.5.1`, and `torchvision==0.20.1`.
- **Verification**: tracing checks are skipped (lttng / tracetools are full-workspace diagnostics), and so is the Ubuntu tracing-tools post-install hook (`platform_post_install_rosdeps`).

```bash
# Edge-cloud collaborative inference verification host:
./scripts/setup.sh --yes --profile inference
./scripts/build.sh --packages-up-to inference_service
```

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
