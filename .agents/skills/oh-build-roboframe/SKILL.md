---
name: oh-build-roboframe
description: "Host-side OpenHarmony cross-build for RoboFrame release package. Use when user mentions `build_roboframe_oh.sh`, `OH_ROOT`, `OpenHarmony build`, `OH host build`, `编译板端 RoboFrame`, `交叉编译 inference_service`, `robot_config OpenHarmony`, `series.openharmony-5.1.0-musl.txt`, or needs the official host-side build pipeline. Triggers for 'BQ3588HM', 'RoboPi'."
---

# OpenHarmony Host-Side Build for RoboFrame

Use this skill for the official Ubuntu host-side cross-build of the RoboFrame release package (IB_Robot ROS 2 packages for OpenHarmony).

This skill is for:

- `scripts/openharmony/build_roboframe_oh.sh`
- `ibrobot_msgs`
- `tensormsg`
- `robot_config`
- `inference_service`
- BQ3588HM OpenHarmony host-side packaging

Do not use this skill for:

- Third-party ROS packages like `usb_cam` or `camera_ros`
- Board connectivity or file transfer only
- Kernel rebuild work
- RKNN model conversion itself

Use instead:

- `oh-cross-build-ros-pkg` for third-party ROS 2 packages
- `oh-access` for board transfer / shell
- `oh-rebuild-kernel` for kernel work
- `rknn-convert` for ONNX -> RKNN

## Source of Truth

Always treat these as authoritative:

- `scripts/openharmony/build_roboframe_oh.sh`
- `README.md` OpenHarmony section

The board runtime bundle must be produced by the official script, not by manually copying files into `install/`.

## Non-Negotiable Rules

1. Never hand-copy `libs/lerobot/src` into the OH install tree.
2. Always use `scripts/openharmony/build_roboframe_oh.sh` as the build entry.
3. The final runtime bundle must include the OpenHarmony lerobot patch stack from:
   `third_party/patches/lerobot/<active_tag>/series.openharmony-5.1.0-musl.txt`
4. Fail closed if the staged `install/lerobot/src` is missing upstream policy
   config or processor registration imports.
5. Prefer building from a `/tmp` working copy when the user wants to keep the main tree untouched, but rely on the script's own fallback clone logic for `libs/lerobot` runtime staging.

## Required Host Inputs

- `OH_ROOT`
- Docker image `voxelsky/ohos-ros-humble-builder:v0.1.5`
- Downloaded OH SDK / sysdeps / humble runtime tarballs under the documented host layout

Do not guess alternate host layouts. If `OH_ROOT` is missing, ask the user or use the documented layout only.

## Canonical Build Command

Run from a temporary working copy when requested, but keep `OH_ROOT` pointed at the shared host build root:

```bash
OH_ROOT="<oh_build_root>" \
./scripts/openharmony/build_roboframe_oh.sh \
  --oh-root "<oh_build_root>"
```

**Do NOT pass `--packages` to manually reduce the package list.** The default
`PACKAGES` array in `build_roboframe_oh.sh` is the authoritative complete list —
it includes all transitive dependencies (`embodied_common`, `voice_asr_service`)
and the SO-101 runtime suite (`so101_description`, `so101_motion`, `so101_robot`, ...).
Excluding packages causes runtime `ModuleNotFoundError` / `package not found`
errors on the board. If a package fails to build, fix the root cause (e.g.
stale build cache) rather than excluding it from the list.

If a clean build is needed (stale cache after prefix changes), remove the
build and install directories first:

```bash
docker run --rm -v "<oh_build_root>/custom_build_root/ibrobot_oh_ws:/ws" \
  alpine sh -c 'rm -rf /ws/build /ws/install'
```

## Expected Successful Runtime-Staging Signals

The build output must include all of the following:

- `Post-processing OpenHarmony runtime bundle...`
- `Preparing OpenHarmony-patched LeRobot runtime staging tree...`
- `Applying OpenHarmony lerobot runtime patch 0009-adaptive-weight-prerequisites.patch...`
- `Applying OpenHarmony lerobot runtime patch 0011-knowledge-distillation.patch...`

If these lines are missing, the build is not complete.

## Required Post-Build Verification

After a successful build, verify the staged runtime tree directly in host `install/lerobot/src`.

Check these files:

- `install/lerobot/src/lerobot/policies/__init__.py`
- `install/lerobot/src/lerobot/policies/factory.py`
- `install/lerobot/src/lerobot/policies/pi05/__init__.py`

They must preserve upstream registration behavior, for example:

- `PI0FastConfig` is imported by `policies/__init__.py`
- `SmolVLANewLineProcessor` is imported by `policies/__init__.py`
- `processor_pi05` is imported by `policies/pi05/__init__.py`

If these checks fail, stop and fix the build pipeline. Do not patch board files by hand.

## `/tmp` Working Copy Rule

When the user wants all edits/build invocation to happen outside the main tree:

1. create or refresh a `/tmp` working copy
2. run the official build script from that copy
3. let the script itself source `lerobot` from either:
   - a valid local git/submodule checkout, or
   - its fallback upstream clone path

Do not restore the old behavior of directly copying host `libs/lerobot/src`.

## Typical Recovery Cases

常见的 4 种恢复场景（lerobot 子模块损坏、training 依赖残留、stale build cache、missing package at runtime）见 `references/troubleshooting.md`。

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| 4 种典型恢复场景（lerobot 子模块、training 依赖、stale cache、missing package） | `references/troubleshooting.md` |

Do not expose these references as separate skills.

## Deployment Handoff

After this skill finishes a valid host build, the next normal step is:

1. package the generated `install/`
2. deploy it to `/data/roboframe/install` on the board

Board deployment itself is not the responsibility of this skill; use normal shell/HDC operations after the build succeeds.
