"""Runtime provider launch builder (robot-runtime-packaging: runtime selection by provider name).

When a robot configuration names ``runtime.provider``, robot bring-up
(hardware, controllers, motion services, runtime facade) is delegated to
that provider's ``launch/runtime.launch.py``. Generic launch never references
a robot-specific launch file, model path, or planning-framework launch; it
only resolves the provider through the ament index and gates upper layers on
``RuntimeStatus`` reconciliation (``wait_for_runtime``).

Configuration::

    robot:
      runtime:
        provider: so101_robot          # <provider>/launch/runtime.launch.py
        profile: so101_single_arm      # path, or a profile shipped by the provider
      capabilities:
        requires: [joint.state, joint.trajectory, motion.move_to_pose]

D12: the deployment peripheral inventory (``peripherals:`` and, for lidar
robots, ``navigation.fast_lio``) is serialized into a temp YAML and handed to
the runtime as the ``peripherals`` launch argument -- data passthrough only,
the runtime owns the driver composition.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from robot_config.interface_binding import (
    bind_robot_interfaces,
    required_interface_ids,
    required_interface_requirements,
)
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.runtime")

RUNTIME_LAUNCH_FILE = "runtime.launch.py"


def runtime_provider(robot_config: dict[str, Any]) -> str:
    """The configured provider name, or '' when the configuration uses in-config bring-up."""
    return str((robot_config.get("runtime") or {}).get("provider", "") or "").strip()


def runtime_profile(robot_config: dict[str, Any]) -> str:
    return str((robot_config.get("runtime") or {}).get("profile", "") or "").strip()


def required_capabilities(robot_config: dict[str, Any]) -> list[str]:
    return [str(name) for name in ((robot_config.get("capabilities") or {}).get("requires") or [])]


def resolve_runtime_launch(provider: str) -> str:
    """Path of ``<provider>/launch/runtime.launch.py``; fails fast naming provider and search path."""
    if provider == "mock_runtime":
        # The mock runtime is the contract-layer reference implementation
        # shipped inside the robot_runtime package (core-only independence
        # gate provider); its entry is mock_runtime.launch.py.
        try:
            share = get_package_share_directory("robot_runtime")
        except PackageNotFoundError as exc:  # pragma: no cover - robot_config depends on robot_runtime
            raise RuntimeError("robot_runtime is not installed; the mock runtime provider is unavailable") from exc
        launch_path = os.path.join(share, "launch", "mock_runtime.launch.py")
        if os.path.isfile(launch_path):
            return launch_path
    try:
        share = get_package_share_directory(provider)
    except PackageNotFoundError as exc:
        prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        raise RuntimeError(
            f"runtime.provider {provider!r} is not installed (searched AMENT_PREFIX_PATH: {prefixes}). "
            "Install the robot runtime package or fix robot.runtime.provider."
        ) from exc
    launch_path = os.path.join(share, "launch", RUNTIME_LAUNCH_FILE)
    if not os.path.isfile(launch_path):
        raise RuntimeError(
            f"runtime.provider {provider!r} ships no {RUNTIME_LAUNCH_FILE} (looked in {os.path.dirname(launch_path)})"
        )
    return launch_path


def _write_peripherals_fragment(robot_config: dict[str, Any]) -> str:
    """Serialize the deployment peripherals/fast_lio inventory for the runtime.

    Returns "" when the configuration declares nothing beyond profile defaults.
    """
    fragment: dict[str, Any] = {}
    peripherals = robot_config.get("peripherals") or []
    fast_lio = (robot_config.get("navigation") or {}).get("fast_lio") or {}
    if peripherals:
        fragment["peripherals"] = peripherals
    if fast_lio.get("enabled"):
        fragment["fast_lio"] = fast_lio
    if not fragment:
        return ""
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="ibrobot_peripherals_", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(fragment, handle, default_flow_style=False, sort_keys=False)
    logger.info(f"peripherals fragment for the runtime: {handle.name}")
    return handle.name


def generate_runtime_provider_actions(
    robot_config: dict[str, Any],
    *,
    use_sim: bool,
    display: bool = False,
    readiness_timeout_s: float = 60.0,
    description_output: str | None = None,
) -> tuple[list, Node]:
    """Return (provider launch include, wait_for_runtime readiness/reconciliation node)."""
    provider = runtime_provider(robot_config)
    if not provider:
        raise RuntimeError("robot.runtime.provider is required for runtime-provider bring-up")
    profile = runtime_profile(robot_config)
    if not profile:
        raise RuntimeError(
            f"robot.runtime.profile is required when runtime.provider={provider!r} "
            "(a path, or a profile name shipped by the provider)"
        )
    launch_path = resolve_runtime_launch(provider)
    peripherals_file = _write_peripherals_fragment(robot_config)
    # Stream-family modes always start idle: policy and teleop must admit
    # explicitly from the observed idle state; only trajectory execution may
    # begin active because it cannot move before a request arrives.
    mode_name = str(robot_config.get("default_control_mode", "") or "")
    mode_config = (robot_config.get("control_modes") or {}).get(mode_name) or {}
    runtime_mode = str(mode_config.get("runtime_mode", "") or "").strip()
    initial_mode = "trajectory" if runtime_mode == "trajectory" else ""
    logger.info(
        f"runtime.provider={provider}: including {launch_path} "
        f"(profile={profile}, simulated={use_sim}, initial_mode={initial_mode or 'profile default'}, "
        f"peripherals={peripherals_file or 'profile defaults'})"
    )

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments={
            "profile": profile,
            "simulated": "true" if use_sim else "false",
            "display": "true" if display else "false",
            "peripherals": peripherals_file,
            **({"initial_mode": initial_mode} if initial_mode else {}),
            **(
                {"instance_id": str(robot_config["runtime"]["instance_id"])}
                if (robot_config.get("runtime") or {}).get("instance_id")
                else {}
            ),
        }.items(),
    )
    required = required_capabilities(robot_config)
    interface_ids = required_interface_ids(robot_config) if description_output else []
    instance_id = (robot_config.get("runtime") or {}).get("instance_id")
    waiter = Node(
        package="robot_runtime",
        executable="wait_for_runtime",
        name="wait_for_runtime",
        output="screen",
        arguments=[
            "--runtime-name",
            provider,
            "--timeout",
            str(float(readiness_timeout_s)),
            *(["--required", *required] if required else []),
            *(["--description-output", description_output] if description_output else []),
            *(["--require-interfaces", *interface_ids] if interface_ids else []),
            *(
                ["--interface-requirements", json.dumps(required_interface_requirements(robot_config))]
                if interface_ids
                else []
            ),
            *(["--instance-id", instance_id] if instance_id else []),
        ],
    )
    return [include], waiter


def materialize_interface_config(robot_config: dict[str, Any], descriptor: dict, destination: Path) -> dict:
    """Validate then atomically publish one complete, bound consumer configuration."""
    from robot_config.loader import validate_robot_config_dict

    effective = bind_robot_interfaces(robot_config, descriptor, require_ready=True)
    validate_robot_config_dict(effective)
    snapshot = {key: value for key, value in effective.items() if key != "_config_path"}
    encoded = yaml.safe_dump({"robot": snapshot}, sort_keys=False)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, encoding="utf-8", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    effective["_config_path"] = str(destination)
    return effective


def generate_bound_runtime_actions(
    robot_config: dict[str, Any],
    construct_consumers,
    *,
    use_sim: bool,
    display: bool = False,
    readiness_timeout_s: float = 60.0,
) -> list:
    """Start the provider once; only construct consumers after the live description is bound."""
    directory = Path(tempfile.mkdtemp(prefix="ibrobot_interfaces_"))
    description_path = directory / "description.yaml"
    effective_path = directory / "robot.yaml"
    try:
        providers, waiter = generate_runtime_provider_actions(
            robot_config,
            use_sim=use_sim,
            display=display,
            readiness_timeout_s=readiness_timeout_s,
            description_output=str(description_path),
        )
    except Exception:
        shutil.rmtree(directory)
        raise

    def on_ready(event, _context):
        if _context.is_shutdown:
            return []
        if event.returncode != 0:
            reason = f"Runtime interface readiness failed (returncode={event.returncode}); consumers were not started"
            logger.error(reason)
            return [EmitEvent(event=Shutdown(reason=reason))]
        try:
            from robot_runtime.interface_description import load_description

            effective = materialize_interface_config(robot_config, load_description(description_path), effective_path)
            logger.info(f"Runtime interfaces bound; consumer snapshot: {effective_path}")
            return construct_consumers(effective)
        except Exception as exc:
            reason = f"Runtime interface binding failed; consumers were not started: {exc}"
            logger.error(reason)
            return [EmitEvent(event=Shutdown(reason=reason))]

    def on_shutdown(_event, _context):
        # Recorders persist this path in dataset/bag metadata for offline conversion.
        if not effective_path.is_file():
            shutil.rmtree(directory, ignore_errors=True)

    return [
        RegisterEventHandler(OnShutdown(on_shutdown=on_shutdown)),
        RegisterEventHandler(OnProcessExit(target_action=waiter, on_exit=on_ready)),
        *providers,
        waiter,
    ]
