#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UI_DIR="${ROOT_DIR}/web/ibrobot_tracing_ui"

cd "${ROOT_DIR}"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "Node.js 18+ and npm are required to build the tracing Web UI; install them explicitly first." >&2
    exit 1
fi
node_major=$(node -p 'Number(process.versions.node.split(".")[0])')
if [[ "${node_major}" -lt 18 ]]; then
    echo "Node.js 18 or newer is required to build the tracing Web UI (found $(node --version))" >&2
    exit 1
fi

if [[ ! -x "${UI_DIR}/node_modules/.bin/vite" || ! -x "${UI_DIR}/node_modules/.bin/vue-tsc" ]]; then
    echo "Tracing Web UI dependencies are missing. From the repository root, run explicitly:" >&2
    echo "  npm ci --prefix web/ibrobot_tracing_ui" >&2
    exit 1
fi

# ROS Humble setup scripts reference optional variables that are not nounset-safe.
# shellcheck disable=SC1091
set +u
source "${ROOT_DIR}/.shrc_local"
set -u
if ! python3 -c 'import fastapi; import uvicorn; from pydantic import TypeAdapter'; then
    echo "Tracing Web Python dependencies are missing or incompatible. From the repository root, run explicitly:" >&2
    echo "  source .shrc_local && python3 -m pip install -r requirements/tracing-web.txt" >&2
    exit 1
fi

npm run build --prefix "${UI_DIR}"
# EXTRA_ARGS override build.sh's default --base-paths src rather than append to it.
"${ROOT_DIR}/scripts/build.sh" -- --base-paths src tools/ibrobot_tracing_web \
    --packages-select ibrobot_tracing ibrobot_tracing_web

echo "Tracing Web UI and API built successfully."
