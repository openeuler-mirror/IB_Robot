---
name: ibrobot-build
description: "Handles the specialized build and environment setup process. Use when user needs to 'build', 'compile', 'colcon build', 'build specific package', '编译', '构建', '解决编译错误', 'fix build errors', 'cb', 'cbp', or needs to refresh the workspace state after code changes. Triggers for 'compile code', '怎么编译', '项目构建', or build infrastructure issues."
---

# IB-Robot Build & Environment Skill

This skill provides the mandatory procedure for building and running code in the IB-Robot workspace.

## ⚠️ CRITICAL: Always Execute from Project Root

**ALL commands in this skill MUST be executed from the IB-Robot project root directory!**

The project root is: `<project_root>`

**Why?** The `.shrc_local` script uses relative paths and expects to be sourced from the project root. If you're not in the root directory:
- `source .shrc_local` will fail with "No such file or directory"
- Build scripts won't be found
- Environment variables won't be set correctly

**Before executing ANY command in this skill:**
```bash
cd <project_root>
```

## Core Mandate: Environment Inheritance

Every execution that depends on project environment variables (ROS 2 Humble, venv, PYTHONPATH, **ROS_DOMAIN_ID**) **MUST** be preceded by sourcing `.shrc_local` and (when needed) exporting `ROS_DOMAIN_ID` within the same shell context.

**CRITICAL**: This workspace uses `ROS_DOMAIN_ID=42` to avoid conflicts with other ROS 2 systems. **ALWAYS** set this before running ROS 2 nodes or commands, or controllers and nodes will fail to communicate.

### 1. Building the Project

**Step 0: Change to project root (if not already there)**
```bash
cd <project_root>
```

Building does NOT require ROS_DOMAIN_ID, but needs PYTHONPATH and other environment setup:

```bash
source .shrc_local && ./scripts/build.sh
```

Or build specific package:
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```

**Why single call?** Each Bash tool call creates a new shell process. Environment variables set by `source` in one call are lost in the next call. Using `&&` keeps everything in the **same shell process**.

### 2. Running Nodes or Launch Files

**Step 0: Change to project root (if not already there)**
```bash
cd <project_root>
```

Any `ros2 run` or `ros2 launch` command **MUST** include both environment setup AND ROS_DOMAIN_ID:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py ...
```

### 3. ROS 2 Commands

**Step 0: Change to project root (if not already there)**
```bash
cd <project_root>
```

Any `ros2` command (topic list, node list, service call, etc.) also needs ROS_DOMAIN_ID:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic list
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 node list
```

### 4. Common Build Commands (Aliases)

The `.shrc_local` provides convenient aliases:
- `cb`: Full build (`colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release`)
- `cbp <pkg>`: Build specific package (`colcon build --symlink-install --packages-select <pkg>`)
- `src`: Refresh environment (`source install/setup.zsh`)

**Important**: These aliases only work in **interactive terminals**. When using the Bash tool (which uses non-interactive shells), use the full commands instead:

**Interactive Terminal (User's shell)**:
```bash
# Build (no ROS_DOMAIN_ID needed)
source .shrc_local && cb
source .shrc_local && cbp robot_config

# Run/launch (ROS_DOMAIN_ID required)
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py use_sim:=true
```

**Non-Interactive Shell (Bash tool)**:
```bash
# Build (no ROS_DOMAIN_ID needed)
source .shrc_local && colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config

# Run/launch (ROS_DOMAIN_ID required)
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && ros2 launch robot_config robot.launch.py use_sim:=true
```

**Note**: Use `--merge-install` flag because the workspace was created with merged layout.

## Build Script Details

The project uses `./scripts/build.sh` which handles:
1. Source ROS 2 Humble environment
2. Activate Python venv
3. Set PYTHONPATH for lerobot
4. Run colcon build with proper settings

### Manual Build (if needed)

```bash
source .shrc_local && colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Build Specific Package

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```

## OpenHarmony Host-Side Cross-Build Prerequisites

When the task is specifically about:

- `scripts/openharmony/build_ibrobot_oh_custom.sh`
- OpenHarmony host-side cross-builds
- `OH_ROOT`, `OH_DOWNLOAD_ROOT`, `OH_CUSTOM_ROOT`

follow these additional rules.

### These variables are host-side only

The following variables are for the **Ubuntu host build layout**, not for the board runtime:

- `OH_ROOT`
- `OH_DOWNLOAD_ROOT`
- `OH_CUSTOM_ROOT`

The authoritative repo doc is:

- `docs/BQ3588HM_OpenHarmony_ROS.md`
  - **第 2 节：统一放到一个外部目录：推荐的主机目录布局**
  - **第 4 节：`build_ibrobot_oh_custom.sh` 里的变量到底对应什么**

### If `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT` are missing

Do **not** guess undocumented host paths.

Instead, stop and ask the user to do one of:

1. prepare and provide `OH_ROOT` according to `docs/BQ3588HM_OpenHarmony_ROS.md`
2. provide explicit `--root`, `--sdk-tar`, `--sysdeps-tar`, and `--humble-tar` paths

If the user also lacks a working `hdc`, point them to:

- `docs/BQ3588HM_board_usage.md` → **第一阶段：HDC 调试工具准备**
- `docs/BQ3588HM_OpenHarmony_ROS.md` → **1.4 OpenHarmony ROS SDK**

## Environment Setup Details

### What .shrc_local Does

1. **ROS 2 Environment**: Sources `/opt/ros/humble/setup.zsh`
2. **Workspace**: Sources `install/setup.zsh`
3. **Python Venv**: Activates `venv/bin/activate`
4. **PYTHONPATH**: Adds `libs/lerobot/src` to Python path
5. **Aliases**: Defines `cb`, `cbp`, `src` shortcuts

### Critical Environment Variables

**ROS_DOMAIN_ID=42**: This MUST be set for all ROS 2 **runtime** operations. Without it:
- Controllers will fail to spawn
- Nodes cannot discover each other
- Topics and services will be invisible
- System will appear to hang or fail silently

**Not needed for**: Building, compilation, or any operation that doesn't communicate with ROS 2 nodes.

**PYTHONPATH**: Must include `libs/lerobot` for inference service.

### Critical for Inference Service

The inference_service package requires:
- `lerobot` module (from `libs/lerobot/src`)
- `torch` module (from venv)
- ROS 2 packages

**All of these are set up by `.shrc_local`**, which is why it must be sourced before any build or run command.

## Common Patterns

### Pattern 1: After Code Changes

```bash
# Build (no ROS_DOMAIN_ID)
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```

### Pattern 2: Running Tests

```bash
# Runtime (needs ROS_DOMAIN_ID)
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 test ...
```

### Pattern 3: Clean Build

```bash
# Build (no ROS_DOMAIN_ID)
source .shrc_local && rm -rf build install log && colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Pattern 4: Launch System

```bash
# Runtime (needs ROS_DOMAIN_ID)
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && ros2 launch robot_config robot.launch.py use_sim:=true
```

## Troubleshooting

### Issue: Controllers fail to spawn / Nodes cannot discover each other

**Root Cause**: ROS_DOMAIN_ID not set, causing DDS discovery to use default domain (0)

**Solution**: Always export ROS_DOMAIN_ID for runtime operations:
```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch ...
```

### Issue: ModuleNotFoundError: No module named 'lerobot'

**Root Cause**: PYTHONPATH not set, .shrc_local not sourced in current shell

**Solution**: Always source .shrc_local:
```bash
source .shrc_local && <your_command>
```

**Wrong** (won't work):
```python
# Bash call 1
source .shrc_local

# Bash call 2
ros2 launch robot_config robot.launch.py  # ← PYTHONPATH and ROS_DOMAIN_ID lost!
```

**Correct**:
```python
# Single Bash call
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py
```

### Issue: colcon build error - "install directory was created with the layout 'merged'"

**Root Cause**: Workspace uses merged install layout, but command doesn't include `--merge-install`

**Solution**: Add `--merge-install` flag:
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```

### Issue: Build Errors - venv not found

**Root Cause**: Virtual environment doesn't exist

**Solution**: Run setup script first:
```bash
./scripts/setup.sh
```

### Issue: Import errors after build

**Root Cause**: Environment not refreshed after build

**Solution**: Source setup in same command:
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config && source install/setup.zsh && python3 -c "import lerobot; print('OK')"
```

### Issue: `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT` not set for OpenHarmony builds

**Root Cause**: the host-side OpenHarmony cross-build layout has not been prepared yet

**Solution**: follow:

- `docs/BQ3588HM_OpenHarmony_ROS.md` → **第 2 节** and **第 4 节**

Then either:

```bash
export OH_ROOT="<your-unified-oh-root>"
./scripts/openharmony/build_ibrobot_oh_custom.sh --oh-root "$OH_ROOT" ...
```

or provide explicit tarball paths:

```bash
./scripts/openharmony/build_ibrobot_oh_custom.sh \
  --root <custom_build_root> \
  --sdk-tar <ohos-sdk-18-linux-aarch64-*.tar.gz> \
  --sysdeps-tar <ohos-*-sysdeps-*.tar.gz> \
  --humble-tar <ohos-humble-build-aarch64-*.tar.gz> ...
```

## Package-Specific Build Notes

### robot_config (Python package)
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```
- Fast build (~1 second)
- Generates contracts at launch time (not build time)

### inference_service (Python package)
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select inference_service
```
- Requires lerobot in PYTHONPATH
- Depends on rosetta_interfaces

### so101_hardware (C++ + Python)
```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select so101_hardware
```
- Has C++ ros2_control component
- Also has Python scripts

## When to Use This Skill

Invoke this skill when:
- ✅ Building any package
- ✅ Running ros2 commands
- ✅ Launching nodes
- ✅ Getting import errors
- ✅ Setting up the environment
- ✅ After git pull or code changes

Do NOT invoke for:
- ❌ Reading files (use Read tool directly)
- ❌ Editing YAML configs
- ❌ Code analysis tasks

## Quick Reference

**For Interactive Terminals (User's shell)**:
| Task | Command |
|------|---------|
| Full build | `source .shrc_local && cb` |
| Build package | `source .shrc_local && cbp <pkg>` |
| Refresh env | `source .shrc_local && src` |
| Launch robot | `source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py ...` |
| List topics | `source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic list` |
| Test import | `source .shrc_local && python3 -c "import lerobot"` |

**For Non-Interactive Shells (Bash tool)**:
| Task | Command |
|------|---------|
| Full build | `source .shrc_local && colcon build --symlink-install --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release` |
| Build package | `source .shrc_local && colcon build --symlink-install --merge-install --packages-select <pkg>` |
| Refresh env | `source .shrc_local && source install/setup.zsh` |
| Launch robot | `source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && ros2 launch robot_config robot.launch.py ...` |
| List topics | `source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic list` |
| Test import | `source .shrc_local && python3 -c "import lerobot"` |

## Architecture Context (Feb 2026 Refactoring)

After the refactoring:
- Contracts are auto-generated (no manual build step for contracts)
- Inference service needs PYTHONPATH injection (handled by .shrc_local)
- Unified launch system (single ros2 launch command)
- Dependencies: lerobot, torch must be in environment

**Key Point**: The build system is simpler now, but environment setup is more critical than ever due to lerobot integration and ROS_DOMAIN_ID requirements for runtime operations.
