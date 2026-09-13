#!/usr/bin/env bash

set -Eeo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
    echo "set ROS_DOMAIN_ID to the same value as the Pipeline first" >&2
    exit 2
fi
if [[ -z "${KIMICODE_API_KEY:-}" ]]; then
    echo "warning: KIMICODE_API_KEY is not set; vlm planner profiles (e.g. so101_agent_manual) will fail to plan" >&2
fi
cd "${ROOT_DIR}"
if [[ -n "${WORKSPACE:-}" && "${WORKSPACE}" != "${ROOT_DIR}" ]]; then
    echo "error: another IB_Robot workspace is already loaded; start a clean shell as described by ibrobot-worktree-env" >&2
    exit 2
fi
if [[ -z "${WORKSPACE:-}" ]]; then
    echo "warning: initializing Agent chat environment from ${ROOT_DIR}" >&2
    source "${ROOT_DIR}/.shrc_local"
fi
if [[ "${WORKSPACE:-}" != "${ROOT_DIR}" ]]; then
    echo "failed to initialize the IB_Robot worktree environment" >&2
    exit 2
fi
PYTHON_BIN="$(command -v python3)"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "IB-Robot Agent chat requires Python 3.10 or newer; activate the project venv first" >&2
    exit 2
fi
if [[ ! -t 0 || ! -t 1 ]]; then
    echo "Agent chat requires an interactive terminal; run it directly in a terminal, not through a pipe or background job" >&2
    exit 2
fi
echo "[Agent chat] connecting to /ibrobot_agent_node/ready ..." >&2
if command -v ibrobot_agent_chat >/dev/null 2>&1; then
    exec env PYTHONUNBUFFERED=1 ibrobot_agent_chat "$@"
fi
if command -v ros2 >/dev/null 2>&1; then
    exec env PYTHONUNBUFFERED=1 ros2 run ibrobot_agent ibrobot_agent_chat "$@"
fi
exec env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT_DIR}/src/ibrobot_agent/ibrobot_agent/chat_tui.py" "$@"
