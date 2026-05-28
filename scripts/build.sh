#!/bin/bash
# build.sh - Modern ROS 2 Humble build script with mixin support
#
# Usage:
#   ./scripts/build.sh                    # Default dev build
#   ./scripts/build.sh --mixin release    # Release build
#   ./scripts/build.sh --mixin debug test # Debug with tests
#   ./scripts/build.sh --list-mixins      # Show available mixins
#   ./scripts/build.sh --packages-select tensormsg  # Build specific package
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(dirname "${SCRIPT_DIR}")}"
MIXIN_DIR="${WORKSPACE}/.colcon/mixin"

# Logging Utilities
RED='\033[0;31m'
NC='\033[0m' # No Color
log_info()    { echo -e "\033[0;32m[INFO] $*${NC}"; }
log_error()   { echo -e "${RED}[ERROR] $*${NC}"; }
log_warning() { echo -e "\033[1;33m[WARNING] $*${NC}"; }

# ============================================================================
# Help & Mixin Listing
# ============================================================================
show_help() {
    cat << 'EOF'
ROS 2 Humble Build Script

Usage: ./scripts/build.sh [OPTIONS] [-- COLCON_ARGS]

Options:
  --mixin NAME [NAME...]   Use specified mixin(s) (can combine multiple)
  --list-mixins            List available mixins and exit
  --clean                  Clean build (cmake-clean-cache)
  --this                   Build only packages in current directory
  -v, --verbose            Show detailed build output
  -h, --help               Show this help

Common mixins:
  dev          Development (debug, no tests, symlink-install) [DEFAULT]
  debug        Debug build with full symbols
  release      Optimized release build
  test         Enable testing
  ci           CI build (release + tests + linting)
  prod         Production (optimized, no debug/tests/tracing)
  asan         AddressSanitizer
  tsan         ThreadSanitizer

Examples:
  ./scripts/build.sh                           # Default dev build
  ./scripts/build.sh --mixin release           # Release build
  ./scripts/build.sh --mixin debug test        # Debug with tests
  ./scripts/build.sh --mixin asan              # With AddressSanitizer
  ./scripts/build.sh --clean --mixin release   # Clean release build
  ./scripts/build.sh -- --packages-select foo  # Pass args to colcon
EOF
}

list_mixins() {
    echo "Available mixins in ${MIXIN_DIR}:"
    echo ""
    if command -v yq &> /dev/null; then
        yq -r '.[] | "  \(.name)\t\(.description // "")"' "${MIXIN_DIR}/build.mixin.yaml" | column -t -s $'\t'
    else
        # Fallback: parse with grep/sed
        grep -E "^- name:|^  description:" "${MIXIN_DIR}/build.mixin.yaml" | \
        sed 'N;s/- name: \(.*\)\n  description: "\(.*\)"/  \1\t\2/' | \
        column -t -s $'\t'
    fi
    echo ""
    echo "Combine mixins: --mixin debug test asan"
}

# ============================================================================
# Argument Parsing
# ============================================================================
MIXINS=()
CLEAN_BUILD=false
BUILD_THIS=false
VERBOSE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mixin)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MIXINS+=("$1")
                shift
            done
            ;;
        --list-mixins)
            list_mixins
            exit 0
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --this)
            BUILD_THIS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default mixin if none specified
[[ ${#MIXINS[@]} -eq 0 ]] && MIXINS=("dev")

# ============================================================================
# Virtual Environment Setup
# ============================================================================
setup_venv() {
    local venv_paths=(
        "${WORKSPACE}/venv"
        "/home/ros/colcon_venv/venv"
        "${VIRTUAL_ENV:-}"
    )
    
    for venv in "${venv_paths[@]}"; do
        if [[ -n "${venv}" && -f "${venv}/bin/activate" ]]; then
            source "${venv}/bin/activate"
            export PATH="${venv}/bin:$PATH"
            return 0
        fi
    done
    return 1
}

ensure_python_deps() {
    [[ -z "${VIRTUAL_ENV:-}" ]] && return 0
    
    local deps=("typing_extensions:typing-extensions" "serial:pyserial" "feetech_servo_sdk:feetech-servo-sdk" "sherpa_onnx:sherpa-onnx" "soundfile:soundfile" "sounddevice:sounddevice")
    for dep in "${deps[@]}"; do
        local module="${dep%%:*}"
        local package="${dep##*:}"
        if ! python3 -c "import ${module}" 2>/dev/null; then
            echo "Installing ${package} in venv..."
            python3 -m pip install --quiet "${package}"
        fi
    done
}

check_lerobot_python_compat() {
    local pyproject="${WORKSPACE}/libs/lerobot/pyproject.toml"
    [[ ! -f "${pyproject}" ]] && return 0

    local min_python
    min_python="$(grep -E '^requires-python = ">=[0-9]+\.[0-9]+"' "${pyproject}" | sed -E 's/.*">=([0-9]+\.[0-9]+)".*/\1/' | head -n1)"
    [[ -z "${min_python}" ]] && return 0

    if ! python3 - "${min_python}" <<'PY'
import re
import sys

required = sys.argv[1]
match = re.fullmatch(r"(\d+)\.(\d+)", required)
if not match:
    raise SystemExit(0)

major, minor = map(int, match.groups())
raise SystemExit(0 if sys.version_info >= (major, minor) else 1)
PY
    then
            local current_py
            current_py="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
            log_error "libs/lerobot requires Python >= ${min_python}, but the active interpreter is ${current_py}."
            log_error "The current ROS 2 Humble workspace venv is not new enough for LeRobot v0.5.1."
            exit 1
    fi
}

require_setup_environment() {
    if ! setup_venv; then
        log_error "Virtual environment not found. Please run ./scripts/setup.sh first."
        exit 1
    fi

    # Verify colcon is importable from the venv. PYTHONNOUSERSITE=1 (set by
    # setup_venv) means a colcon installed under ~/.local cannot rescue us
    # here; it must live in the venv's site-packages. Surface a precise,
    # actionable error early instead of letting `python3 -m colcon` fail with
    # a bare "No module named colcon" deep inside the build pipeline.
    if ! python3 -m colcon --help >/dev/null 2>&1; then
        log_error "colcon is not importable from the workspace venv."
        log_error "This usually means setup.sh installed colcon into ~/.local"
        log_error "(via 'pip install --user') instead of into the venv, while"
        log_error "build.sh sets PYTHONNOUSERSITE=1 to ignore ~/.local."
        log_error ""
        log_error "Fix it by reinstalling colcon into the venv:"
        log_error "  source venv/bin/activate"
        log_error "  PYTHONNOUSERSITE=1 python3 -m pip install --upgrade colcon-common-extensions colcon-mixin"
        log_error ""
        log_error "Or re-run ./scripts/setup.sh which now installs colcon into venv automatically."
        exit 1
    fi

    check_lerobot_python_compat

    if ! python3 -c "import lerobot" 2>/dev/null; then
        log_warning "lerobot is not importable in the current venv."
        log_warning "Run ./scripts/setup.sh to install or repair the Python environment before building."
    fi

    if ! python3 -c "import numpy; assert numpy.__version__.startswith('1.26.')" 2>/dev/null; then
        log_warning "NumPy is not pinned to the expected ROS-compatible 1.26.x series."
        log_warning "Run ./scripts/setup.sh to repair the Python environment before building."
    fi
}

require_setup_environment


# ============================================================================
# ROS 2 Environment
# ============================================================================
source /opt/ros/humble/setup.sh

# Clean build: remove stale dirs BEFORE sourcing to prevent overlay chain leaks.
# Without this, install/setup.sh (which may chain to stale overlays like a
# dev_worktree) pollutes AMENT_PREFIX_PATH and colcon records it again.
if ${CLEAN_BUILD}; then
    log_info "Removing stale install/, build/, log/ to prevent overlay chain leaks..."
    rm -rf "${WORKSPACE}/install" "${WORKSPACE}/build" "${WORKSPACE}/log"
fi

if ! ${CLEAN_BUILD}; then
    [[ -f "${WORKSPACE}/install/setup.sh" ]] && source "${WORKSPACE}/install/setup.sh"
fi

# ============================================================================
# openEuler / RedHat FFmpeg header fix
# On these systems, FFmpeg headers live under /usr/include/ffmpeg/ instead of
# /usr/include/.  Export CPATH so that packages like usb_cam can find them.
# ============================================================================
if [[ -d /usr/include/ffmpeg ]]; then
    export CPATH="/usr/include/ffmpeg${CPATH:+:$CPATH}"
fi

# ============================================================================
# Build
# ============================================================================
cd "${WORKSPACE}"

# Build mixin arguments
MIXIN_ARGS=()
if [[ -f "${MIXIN_DIR}/build.mixin.yaml" ]]; then
    MIXIN_ARGS+=("--mixin-files" "${MIXIN_DIR}/build.mixin.yaml")
    MIXIN_ARGS+=("--mixin" "${MIXINS[@]}")
fi

# Clean build if requested
CLEAN_ARGS=()
${CLEAN_BUILD} && CLEAN_ARGS+=("--cmake-clean-cache")

# Build specific directory if --this
THIS_ARGS=()
${BUILD_THIS} && THIS_ARGS+=("--paths" "$(pwd)")

echo "════════════════════════════════════════════════════════════════════"
echo "Building with mixin(s): ${MIXINS[*]}"
echo "════════════════════════════════════════════════════════════════════"

# Select event handlers based on verbosity
EVENT_HANDLERS="status- summary-"
${VERBOSE} && EVENT_HANDLERS="console_cohesion+"

PYTHONNOUSERSITE=1 python3 -m colcon build \
    --continue-on-error \
    --parallel-workers "$(nproc)" \
    --merge-install \
    --symlink-install \
    --event-handlers ${EVENT_HANDLERS} \
    --cmake-args -Wno-dev \
    --base-paths src \
    "${MIXIN_ARGS[@]}" \
    "${CLEAN_ARGS[@]}" \
    "${THIS_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Build complete. Source with: source install/setup.sh"
