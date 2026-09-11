#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
    echo "set ROS_DOMAIN_ID to the same value as the Pipeline first" >&2
    exit 2
fi
if [[ -z "${KIMICODE_API_KEY:-}" ]]; then
    echo "KIMICODE_API_KEY is not set; export it before starting Agent chat" >&2
    exit 2
fi
cd "${ROOT_DIR}"
if [[ -z "${WORKSPACE:-}" || "${WORKSPACE}" != "${ROOT_DIR}" ]]; then
    echo "warning: initializing Agent chat environment from ${ROOT_DIR}" >&2
    unset VIRTUAL_ENV PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH
    source "${ROOT_DIR}/.shrc_local"
fi
if [[ "${WORKSPACE:-}" != "${ROOT_DIR}" ]]; then
    echo "failed to initialize the IB_Robot worktree environment" >&2
    exit 2
fi
PYTHON_BIN="$(command -v python3)"
if [[ "$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "ROS Humble Agent chat requires Python 3.10; activate the project venv first" >&2
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
exec env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${ROOT_DIR}/src/ibrobot_agent/ibrobot_agent/chat_tui.py" "$@"
