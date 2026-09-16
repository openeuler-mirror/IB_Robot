---
name: ibrobot-build
description: "Build IB_Robot through ./scripts/build.sh (Ubuntu / openEuler Embedded). Use when user needs to 'build', 'compile', 'colcon build', 'build specific package', '编译', '构建', '解决编译错误', 'fix build errors', or needs to refresh the workspace state after code changes. NOT for OpenHarmony cross-builds — use oh-build-roboframe skill instead."
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

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| Build troubleshooting（ROS_DOMAIN_ID 缺失、lerobot 导入失败、merge-install 报错等） | `references/troubleshooting.md` |
| 特定 ROS 2 包的构建特性（robot_config / inference_service / so101_hardware） | `references/package-notes.md` |

Do not expose these references as separate skills.

## Core Mandate: Environment Inheritance

Every execution that depends on project environment variables (ROS 2 Humble, venv, PYTHONPATH, **ROS_DOMAIN_ID**) **MUST** be preceded by sourcing `.shrc_local` and (when needed) exporting `ROS_DOMAIN_ID` within the same shell context.

**CRITICAL**: This workspace uses `ROS_DOMAIN_ID=42` to avoid conflicts with other ROS 2 systems. **ALWAYS** set this before running ROS 2 nodes or commands, or controllers and nodes will fail to communicate.

### 1. Building the Project

```bash
cd <project_root>
source .shrc_local && ./scripts/build.sh
```

Or build specific package:
```bash
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config
```

Or build by package group (additive flags, resolved through the colcon
dependency closure so each robot/agent flag automatically brings its base
dependencies):
```bash
source .shrc_local && ./scripts/build.sh --so101              # SO-101 + base
source .shrc_local && ./scripts/build.sh --lekiwi             # LeKiwi + base
source .shrc_local && ./scripts/build.sh --agent              # Agent stack + base
source .shrc_local && ./scripts/build.sh --rosclaw            # rosclaw + base
source .shrc_local && ./scripts/build.sh --base --agent --lekiwi  # LeKiwi complete content
source .shrc_local && ./scripts/build.sh --list-groups --so101    # Inspect resolved packages
```

Group flags cannot be combined with `--this` or explicit `--packages-*`
selections after `--`. Group membership maps to `src/` path prefixes
defined in `scripts/build.sh` (`GROUP_PATHS`); adjust there when packages
move between groups.

**Never invoke `colcon build` directly.** All builds go through `./scripts/build.sh`; it applies the correct layout, symlink, and CMake settings (default dev mixin).

**Why single call?** Each Bash tool call creates a new shell process. Environment variables set by `source` in one call are lost in the next call. Using `&&` keeps everything in the **same shell process**.

### 2. Running Nodes or Launch Files

Any `ros2 run` or `ros2 launch` command **MUST** include both environment setup AND ROS_DOMAIN_ID:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py ...
```

### 3. ROS 2 Commands

Any `ros2` command (topic list, node list, service call, etc.) also needs ROS_DOMAIN_ID:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic list
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 node list
```

### 4. Common Build Commands

Builds work identically in interactive terminals and in the Bash tool (non-interactive shells): always through `./scripts/build.sh`.

```bash
# Build everything (default dev mixin)
source .shrc_local && ./scripts/build.sh

# Build specific package
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config

# Clean build (cache reset; use for layout errors, stale caches, or after infra changes)
source .shrc_local && ./scripts/build.sh --clean

# Run/launch (ROS_DOMAIN_ID required)
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && ros2 launch robot_config robot.launch.py use_sim:=true
```

**Note**: Do not use the `.shrc_local` aliases (`cb`, `cbp`) from scripts or agent tools — they only exist in interactive shells and expand to raw `colcon` invocations. The build script is the single entry point.

## Build Script Details

`./scripts/build.sh` handles:
1. Source ROS 2 Humble environment
2. Activate Python venv
3. Set PYTHONPATH for lerobot
4. Run colcon with proper settings (`dev` mixin by default: symlink-install, no tests)

### Clean Build (cache reset)

```bash
source .shrc_local && ./scripts/build.sh --clean
```

### Build Specific Package

```bash
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config
```

## Environment Setup Details

### What .shrc_local Does

1. **ROS 2 Environment**: Sources `/opt/ros/humble/setup.zsh`
2. **Workspace**: Sources `install/setup.zsh`
3. **Python Venv**: Activates `venv/bin/activate`
4. **PYTHONPATH**: Adds `libs/lerobot/src` to Python path
5. **Aliases**: Defines `cb`, `cbp`, `src` shortcuts for interactive shells only (not usable from agent Bash tools)

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
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config
```

### Pattern 2: Running Tests

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 test ...
```

### Pattern 3: Clean Build

```bash
source .shrc_local && ./scripts/build.sh --clean
```

### Pattern 4: Launch System

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && ros2 launch robot_config robot.launch.py use_sim:=true
```

遇到错误时参考 `references/troubleshooting.md`。特定包构建特性参考 `references/package-notes.md`。

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

**Builds (interactive and Bash tool alike)**:
| Task | Command |
|------|---------|
| Full build | `source .shrc_local && ./scripts/build.sh` |
| Build package | `source .shrc_local && ./scripts/build.sh -- --packages-select <pkg>` |
| Clean build | `source .shrc_local && ./scripts/build.sh --clean` |
| Refresh env | `source .shrc_local && source install/setup.zsh` |

**Runtime (ROS_DOMAIN_ID required)**:
| Task | Command |
|------|---------|
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
