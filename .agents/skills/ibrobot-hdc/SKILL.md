---
name: ibrobot-hdc
description: "Handles stable OpenHarmony HDC-over-TCP access to the Bearkey BQ3588HM board. Use when user wants to 'hdc shell', 'connect board', 'send file to OH', 'pull file from OH', 'OpenHarmony device', 'BQ3588HM', '网络调试', '推送文件', '拉取文件', or run commands on the flashed board."
---

# IB-Robot OpenHarmony HDC Skill

This skill standardizes all host-to-board interactions through the local SDK `hdc` binary and a
user-provided board target.

For board-local runtime facts such as ROS bootstrap, Python 3.12 availability, or the read-only-root workaround around `ros2ohos.env`, use `ibrobot-bq3588hm-oh`.

## Required Connection Inputs

Use these values as the default pattern:

```bash
HDC_BIN=hdc
HDC_TARGET=<board-ip>:8710
```

Prefer the `hdc` command from `$PATH` so the workflow is portable across different hosts.
Still call `"$HDC_BIN"` explicitly in commands after setting `HDC_BIN=hdc`.

Before running remote commands, the agent should confirm one of:

1. a TCP target such as `<board-ip>:8710`, or
2. that USB HDC is currently connected and will be used directly

Do **not** assume a fixed host-side SDK path or a fixed board IP. If the target is unknown, ask
the user to provide the TCP target IP or confirm USB availability.

### If `hdc` is missing on the host

If `command -v hdc` fails:

1. **stop and ask the user to install or provide HDC before continuing**
2. tell the user to add the extracted SDK `toolchains` directory to `PATH`
3. tell the user to persist that export in `~/.bashrc` or `~/.zshrc`

When asking the user, point them to these repo docs:

- `docs/BQ3588HM_board_usage.md` → **第一阶段：HDC 调试工具准备**
- `docs/BQ3588HM_OpenHarmony_ROS.md` → **1.4 OpenHarmony ROS SDK**

These two docs explain where the OpenHarmony SDK comes from, where to extract the `hdc`
binary from on the host, and how to export the `toolchains` directory into `PATH`.

## Core Rule: Prefer TCP, Not USB

For this board, USB HDC is a valid connection method and can be used directly instead of TCP.
However, USB sessions are more likely to become unstable after large transfers and can leave stale
server-side sessions. Prefer a user-provided TCP target such as `<board-ip>:8710` for automation,
shells, and file transfers whenever available.

## Standard Execution Patterns

### Check or establish the connection

```bash
"$HDC_BIN" list targets
"$HDC_BIN" tconn "$HDC_TARGET"
"$HDC_BIN" list targets
```

TCP targets require `tconn`; USB targets appear automatically.

### Run a one-off remote command

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell pwd
"$HDC_BIN" -t "$HDC_TARGET" shell 'ls -1 /data'
"$HDC_BIN" -t "$HDC_TARGET" shell 'cd /data && tar -zxpvf ohos-humble-build-aarch64-20260115100449.tar.gz'
```

### Open an interactive shell

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell
```

### Send files to the board

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send <local_path> <remote_path>
"$HDC_BIN" -t "$HDC_TARGET" file send ./artifact.tar.gz /data/artifact.tar.gz
```

### Receive files from the board

```bash
"$HDC_BIN" -t "$HDC_TARGET" file recv <remote_path> <local_path>
"$HDC_BIN" -t "$HDC_TARGET" file recv /data/result.log ./result.log
```

### Stream device logs

```bash
"$HDC_BIN" -t "$HDC_TARGET" hilog
```

### Reboot or change daemon mode

```bash
"$HDC_BIN" -t "$HDC_TARGET" target boot
"$HDC_BIN" -t "$HDC_TARGET" tmode port 8710
"$HDC_BIN" -t "$HDC_TARGET" tmode port close
```

Note: `tmode port 8710` is usually performed over USB first to move the daemon into TCP mode.
After that, prefer Ethernet or Wi-Fi plus `tconn`. If TCP has not been enabled yet, USB HDC is
still a valid transport and can be used directly.

## Recommended Automation Pattern

For reproducible agent actions, wrap every command with the fixed binary and target:

```bash
HDC_BIN=hdc
HDC_TARGET=<board-ip>:8710
"$HDC_BIN" -t "$HDC_TARGET" shell '<command>'
```

Examples:

```bash
HDC_BIN=hdc
HDC_TARGET=<board-ip>:8710
"$HDC_BIN" -t "$HDC_TARGET" shell 'cd /data && ls'
"$HDC_BIN" -t "$HDC_TARGET" file send ./local.tar.gz /data/local.tar.gz
```

## Stability Recovery

If TCP automation stops responding:

```bash
"$HDC_BIN" kill -r
"$HDC_BIN" tconn "$HDC_TARGET"
"$HDC_BIN" list targets -v
```

If the device is still reachable but the session is stale, remove and reconnect the TCP target:

```bash
"$HDC_BIN" tconn "$HDC_TARGET" -remove
"$HDC_BIN" tconn "$HDC_TARGET"
```

If USB is available and TCP is not ready yet, use it to enable TCP mode or operate directly:

```bash
"$HDC_BIN" shell ifconfig
"$HDC_BIN" tmode port 8710
"$HDC_BIN" tconn "$HDC_TARGET"
```

## Board-Specific Notes

- Current board: Bearkey BQ3588HM
- Current target: user-provided TCP target such as `<board-ip>:8710`, or a direct USB HDC session
- Current local SDK tool: `hdc` from `$PATH`
- Verified remote paths of interest: `/data`, `/system`, `/vendor`
- Current `/data` already contains:
  - `ohos-18-sysdeps-aarch64-20260115.tar.gz`
  - `ohos-humble-build-aarch64-20260115100449.tar.gz`

## When to Use This Skill

Invoke this skill when the user wants to:
- connect to the flashed OpenHarmony board
- run `hdc shell`
- push or pull files with `hdc`
- inspect `/data`, logs, or services on the board
- prepare the board for ROS 2 or IB-Robot runtime validation

Do NOT use this skill for:
- local workspace builds (`ibrobot-build`)
- local ROS 2 environment setup (`ibrobot-env`)
- launching the Ubuntu-side robot stack (`ibrobot-launch`)

## Quick Reference

| Task | Command |
|------|---------|
| List targets | `"$HDC_BIN" list targets` |
| Connect TCP target | `"$HDC_BIN" tconn "$HDC_TARGET"` |
| Remote shell command | `"$HDC_BIN" -t "$HDC_TARGET" shell '<cmd>'` |
| Interactive shell | `"$HDC_BIN" -t "$HDC_TARGET" shell` |
| Send file | `"$HDC_BIN" -t "$HDC_TARGET" file send <local> <remote>` |
| Receive file | `"$HDC_BIN" -t "$HDC_TARGET" file recv <remote> <local>` |
| Show logs | `"$HDC_BIN" -t "$HDC_TARGET" hilog` |
