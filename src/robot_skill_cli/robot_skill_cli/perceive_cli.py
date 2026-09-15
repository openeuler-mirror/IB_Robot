"""ibrobot-perceive: read-only perception topic reader with hard allowlist and audit log.

This wrapper is the only sanctioned path for an LLM to read ROS perception topics.
It enforces a hard-coded source/field allowlist, logs every read for audit, runs
with a bounded timeout, and extracts the requested field from the ``ros2 topic echo``
YAML output. Direct ``ros2`` subcommands remain forbidden by POLICY.md.

Source allowlist (alias -> real ROS topic):
  voice_direction      -> /voice/speech_direction  (voice_asr_service, fixed contract)
  arm_joint_position   -> robot_config.moveit.joint_state_topic (SSOT-resolved)

  ``voice_direction`` is a fixed contract from ``voice_asr_service`` and does not
  depend on ``robot_config``. ``arm_joint_position`` resolves its topic from the
  same ``robot_config`` that MoveIt and the Capability Gateway consume, so multi-part
  robots publish the correct arm joint topic:
    so101_single_arm            -> /joint_states (wrapper fallback)
    lekiwi_handeye_realsense_grasp -> /arm_joint_state_broadcaster/joint_states
  Topic resolution reuses ``robot_config.config_path.resolve_robot_config_path`` (the
  SSOT path resolver); no second path priority is maintained. The resolved topic is
  only the *name* — it is not a security-sensitive surface. The security boundary is
  the hard-coded *field* set: widening it requires a source edit, never a config.yaml
  override.

Point-in-time read, not a persistent snapshot:
  The wrapper calls ``ros2 topic echo --once`` which returns the *next* message
  published on the topic within the timeout. For volatile event topics such as
  ``/voice/speech_direction`` (published only when voice activity is detected)
  there is no latched/persistent state to read; the returned value is a single
  point-in-time sample that may already be stale by the time it is consumed.
  Callers must treat the value as open-loop and may not assume it reflects the
  current world state at execution time. This is a deliberate contract, not a
  cache; do not widen it by wrapping the wrapper in a cache without declaring
  the freshness/liveness semantics here and in POLICY.md.

Output contract (literal for LLM injection into workflow_json):
  Scalars (int/float) print via ``str()`` (e.g. ``0.5236``); lists and dicts
  print via ``json.dumps()`` (e.g. ``[0.12, -0.31]``). Allowlist fields must be
  JSON-serializable scalars or scalar arrays; do not extend the allowlist to
  bool/None/nested objects without keeping ``_format_value`` in sync, because
  ``str(True)``/``str(None)``/``str({'k': 1})`` are not valid JSON and would
  break LLM injection into ``workflow_json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from robot_config.config_path import resolve_robot_config_path

# Hard-coded source allowlist. The *field* set is the security boundary: widening
# it requires a source edit, never a config.yaml override. The *topic* for
# config-backed sources is resolved from robot_config at runtime (SSOT), so
# multi-part robots publish the correct arm joint topic. ``voice_direction`` is a
# fixed contract from voice_asr_service and does not depend on robot_config.
PERCEPTION_ALLOWLIST: dict[str, dict[str, Any]] = {
    "arm_joint_position": {
        "fields": {"position"},
        "config_backed": True,
    },
    "voice_direction": {
        "fields": {"azimuth_rad", "seq_id"},
        "config_backed": False,
        "topic": "/voice/speech_direction",
    },
}

LOG_PATH = Path("/tmp/hermes-perceive.log")
TIMEOUT_SEC = 5
_VOICE_DIRECTION_TIMEOUT_SEC = 60
_DEFAULT_ARM_JOINT_TOPIC = "/joint_states"
_VOICE_DIRECTION_TYPE = "ibrobot_msgs/msg/SpeechDirection"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibrobot-perceive",
        description="Read one perception source field via the allowlisted wrapper.",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(PERCEPTION_ALLOWLIST),
        help="perception source alias (topic is resolved from robot_config for config-backed sources)",
    )
    parser.add_argument("--field", required=True, help="top-level YAML field to extract")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-name", help="robot_config name (same semantics as robot-skill)")
    config_group.add_argument("--config-path", help="explicit robot_config YAML path")
    return parser


def _log(source: str, field: str, status: str) -> None:
    ts = datetime.now().astimezone().isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(f"{ts} source={source} field={field} status={status}\n")


def _load_robot_section(config_name: str | None, config_path: str | Path | None) -> dict[str, Any]:
    """Read the raw ``robot`` section for topic resolution.

    Reuses ``resolve_robot_config_path`` (the SSOT path resolver) so no second path
    priority is maintained. Reads only the ``robot`` section without the full
    ``load_robot_config_dict`` validation, keeping ``ibrobot-perceive`` lightweight and
    decoupled from unrelated nav/grasp/perception config errors.
    """
    resolved_path = Path(resolve_robot_config_path(config_name=config_name, config_path=config_path))
    with resolved_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict) or "robot" not in data:
        raise ValueError(f"Invalid robot config: missing 'robot' section in {resolved_path}")
    robot_data = data["robot"]
    if not isinstance(robot_data, dict):
        raise ValueError(f"Invalid robot config: 'robot' section must be a mapping in {resolved_path}")
    return robot_data


def _resolve_topic(source: str, robot_config: dict[str, Any] | None) -> str:
    spec = PERCEPTION_ALLOWLIST[source]
    if not spec["config_backed"]:
        return str(spec["topic"])
    if source == "arm_joint_position":
        if robot_config is None:
            return _DEFAULT_ARM_JOINT_TOPIC
        return str(robot_config.get("moveit", {}).get("joint_state_topic", _DEFAULT_ARM_JOINT_TOPIC))
    raise ValueError(f"unknown config-backed source: {source}")


def _extract_field(stdout: str, field: str) -> object | None:
    # ros2 topic echo prints the message first and a trailing document separator;
    # parsing from the first separator would discard that message entirely.
    try:
        docs = list(yaml.safe_load_all(stdout))
    except yaml.YAMLError:
        marker = stdout.find("---\n")
        if marker < 0:
            raise
        docs = list(yaml.safe_load_all(stdout[marker:]))
    for doc in docs:
        if isinstance(doc, dict) and field in doc:
            return doc[field]
    return None


def _format_value(value: object) -> str:
    if isinstance(value, list | dict):
        return json.dumps(value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    spec = PERCEPTION_ALLOWLIST[args.source]
    if args.field not in spec["fields"]:
        allowed = ", ".join(sorted(spec["fields"]))
        _log(args.source, args.field, f"rejected_field (allowed={allowed})")
        print(f"ERROR: field {args.field} not allowed for {args.source}; allowed: {allowed}", file=sys.stderr)
        return 1

    robot_config: dict[str, Any] | None = None
    if spec["config_backed"]:
        try:
            robot_config = _load_robot_section(args.config_name, args.config_path)
        except (FileNotFoundError, ValueError) as e:
            _log(args.source, args.field, f"config_error: {e}")
            print(f"ERROR: failed to load robot_config for topic resolution: {e}", file=sys.stderr)
            return 1

    topic = _resolve_topic(args.source, robot_config)
    timeout_sec = _VOICE_DIRECTION_TIMEOUT_SEC if args.source == "voice_direction" else TIMEOUT_SEC
    command = ["ros2", "topic", "echo", "--once"]
    if args.source == "voice_direction":
        command.extend(
            [
                "--qos-reliability",
                "reliable",
                "--qos-history",
                "keep_last",
                "--qos-depth",
                "1",
                "--qos-durability",
                "volatile",
                topic,
                _VOICE_DIRECTION_TYPE,
            ]
        )
    else:
        command.append(topic)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _log(args.source, args.field, "timeout")
        print(f"ERROR: timed out reading {topic} ({timeout_sec}s)", file=sys.stderr)
        return 1

    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[:1]
        detail_str = detail[0] if detail else "unknown"
        _log(args.source, args.field, f"ros2_error: {detail_str[:200]}")
        print(f"ERROR: ros2 topic echo failed: {detail_str[:200]}", file=sys.stderr)
        return 1

    try:
        value = _extract_field(result.stdout, args.field)
    except yaml.YAMLError as e:
        _log(args.source, args.field, f"parse_error: {e}")
        print(f"ERROR: failed to parse topic output: {e}", file=sys.stderr)
        return 1

    if value is None:
        _log(args.source, args.field, "missing_field")
        print(f"ERROR: field {args.field} not found in topic data", file=sys.stderr)
        return 1

    _log(args.source, args.field, f"ok: {value}")
    print(_format_value(value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
