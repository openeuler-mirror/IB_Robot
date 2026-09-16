#!/usr/bin/env python3
"""
Configuration Validation Script for IB Robot

This script validates the consistency of joint configurations across multiple
configuration files, enforcing the DRY (Don't Repeat Yourself) principle.

Usage:
    python validate_config.py [--robot-config PATH] [--verbose]

Exit Codes:
    0: All validations passed
    1: Configuration errors found
    2: File not found or parse error

Used in CI/CD to catch configuration drift before deployment.
"""

import argparse
import sys
from pathlib import Path

import yaml


class ConfigValidator:
    """Validates robot configuration consistency."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {"INFO": "ℹ", "WARNING": "⚠", "ERROR": "✗", "SUCCESS": "✓"}.get(level, "•")
            print(f"{prefix} {message}")

    def load_yaml(self, path: Path) -> dict:
        """Load a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def resolve_ros_path(self, path_str: str, base_dir: Path) -> Path:
        """Resolve ROS-style path substitutions.

        Supports:
        - $(find package) → searches in common locations
        - $(env VAR) → reads environment variable
        """
        import os
        import re

        # Handle $(find package)
        find_pattern = r"\$\(find\s+(\w+)\)"
        match = re.search(find_pattern, path_str)
        if match:
            package_name = match.group(1)
            # Try common package locations
            search_paths = [
                base_dir / ".." / package_name,
                base_dir / ".." / ".." / package_name,
                Path(f"/opt/ros/humble/share/{package_name}"),
            ]

            # Also check install directory
            workspace_root = base_dir
            while workspace_root.parent != workspace_root:
                if (workspace_root / "install").exists():
                    search_paths.insert(0, workspace_root / "install" / package_name / "share" / package_name)
                    break
                workspace_root = workspace_root.parent

            for search_path in search_paths:
                if search_path.exists():
                    resolved = path_str.replace(match.group(0), str(search_path.resolve()))
                    return self.resolve_ros_path(resolved, base_dir)

            raise FileNotFoundError(f"Could not find package '{package_name}'")

        # Handle $(env VAR)
        env_pattern = r"\$\(env\s+(\w+)\)"
        match = re.search(env_pattern, path_str)
        if match:
            env_var = match.group(1)
            env_value = os.environ.get(env_var)
            if not env_value:
                raise ValueError(f"Environment variable '{env_var}' not set")
            resolved = path_str.replace(match.group(0), env_value)
            return self.resolve_ros_path(resolved, base_dir)

        return Path(path_str)

    def validate_joints_config(self, robot_config_path: Path) -> tuple[dict, set, set, set]:
        """Validate and extract joint configuration from robot_config."""
        self.log(f"Loading robot config: {robot_config_path}")

        config = self.load_yaml(robot_config_path)
        robot_cfg = config.get("robot", config)

        joints_cfg = robot_cfg.get("joints", {})
        if not joints_cfg:
            self.errors.append("No 'joints' configuration found in robot_config")
            return {}, set(), set(), set()

        arm_joints = set(joints_cfg.get("arm", []))
        gripper_joints = set(joints_cfg.get("gripper", []))
        all_joints = set(joints_cfg.get("all", []))

        # Validate consistency
        expected_all = arm_joints | gripper_joints
        if all_joints != expected_all:
            self.warnings.append(
                f"'all' joints list ({sorted(all_joints)}) does not match arm + gripper union ({sorted(expected_all)})"
            )

        self.log(f"Arm joints: {sorted(arm_joints)}")
        self.log(f"Gripper joints: {sorted(gripper_joints)}")
        self.log(f"All joints: {sorted(all_joints)}", "SUCCESS")

        return joints_cfg, arm_joints, gripper_joints, all_joints

    def validate_controller_config(
        self, controllers_config_path: Path, expected_arm: set[str], expected_gripper: set[str], expected_all: set[str]
    ) -> bool:
        """Validate controller configuration consistency."""
        self.log(f"\nValidating controller config: {controllers_config_path}")

        try:
            config = self.load_yaml(controllers_config_path)
        except Exception as e:
            self.errors.append(f"Failed to load controller config: {e}")
            return False

        controllers_to_check = [
            ("arm_position_controller", expected_arm),
            ("arm_trajectory_controller", expected_arm),
            ("gripper_position_controller", expected_gripper),
            ("gripper_trajectory_controller", expected_gripper),
            ("joint_state_broadcaster", expected_all),
        ]

        all_valid = True
        for controller_name, expected_joints in controllers_to_check:
            controller_cfg = config.get(controller_name, {}).get("ros__parameters", {})
            if not controller_cfg:
                self.log(f"{controller_name}: not found (skipping)", "WARNING")
                continue

            actual_joints = set(controller_cfg.get("joints", []))
            if actual_joints != expected_joints:
                self.errors.append(
                    f"{controller_name} joint mismatch:\n"
                    f"  Expected: {sorted(expected_joints)}\n"
                    f"  Found:    {sorted(actual_joints)}"
                )
                all_valid = False
            else:
                self.log(f"{controller_name}: ✓", "SUCCESS")

        return all_valid

    def validate_moveit_config(
        self, moveit_config_path: Path, expected_arm: set[str], expected_gripper: set[str]
    ) -> bool:
        """Validate MoveIt controller configuration consistency."""
        self.log(f"\nValidating MoveIt config: {moveit_config_path}")

        try:
            config = self.load_yaml(moveit_config_path)
        except Exception as e:
            self.warnings.append(f"Failed to load MoveIt config: {e}")
            self.warnings.append("Skipping MoveIt validation (may not be configured)")
            return True

        manager_cfg = config.get("moveit_simple_controller_manager", {})
        controllers_to_check = [
            ("arm_trajectory_controller", expected_arm),
            ("gripper_trajectory_controller", expected_gripper),
        ]

        all_valid = True
        for controller_name, expected_joints in controllers_to_check:
            controller_cfg = manager_cfg.get(controller_name, {})
            if not controller_cfg:
                self.log(f"{controller_name}: not found (skipping)", "WARNING")
                continue

            actual_joints = set(controller_cfg.get("joints", []))
            if actual_joints != expected_joints:
                self.errors.append(
                    f"MoveIt {controller_name} joint mismatch:\n"
                    f"  Expected: {sorted(expected_joints)}\n"
                    f"  Found:    {sorted(actual_joints)}"
                )
                all_valid = False
            else:
                self.log(f"MoveIt {controller_name}: ✓", "SUCCESS")

        return all_valid

    def run_validation(
        self, robot_config_path: Path, controllers_config_path: Path = None, moveit_config_path: Path = None
    ) -> bool:
        """Run complete configuration validation."""
        self.log("=" * 60)
        self.log("IB Robot Configuration Validation")
        self.log("=" * 60)

        # Step 1: Load and validate robot_config
        try:
            joints_cfg, arm_joints, gripper_joints, all_joints = self.validate_joints_config(robot_config_path)
        except Exception as e:
            self.errors.append(f"Failed to validate robot_config: {e}")
            return False

        if not joints_cfg:
            return False

        # Step 2: Resolve and validate controller config path
        if controllers_config_path is None:
            robot_config = self.load_yaml(robot_config_path)
            robot_cfg = robot_config.get("robot", robot_config)
            ros2_control = robot_cfg.get("ros2_control", {})
            ctrl_path_str = ros2_control.get("controllers_config", "")

            if ctrl_path_str:
                try:
                    controllers_config_path = self.resolve_ros_path(ctrl_path_str, robot_config_path.parent)
                except Exception as e:
                    self.warnings.append(f"Could not resolve controllers_config path: {e}")

        # Step 3: Validate controller config
        if controllers_config_path and controllers_config_path.exists():
            self.validate_controller_config(controllers_config_path, arm_joints, gripper_joints, all_joints)
        else:
            self.warnings.append("Controllers config not found, skipping validation")

        # Step 4: Validate MoveIt config (if provided)
        if moveit_config_path and moveit_config_path.exists():
            self.validate_moveit_config(moveit_config_path, arm_joints, gripper_joints)

        # Step 5: Print summary
        self.log("\n" + "=" * 60)
        self.log("Validation Summary")
        self.log("=" * 60)

        if self.warnings:
            self.log(f"\nWarnings ({len(self.warnings)}):", "WARNING")
            for warning in self.warnings:
                print(f"  • {warning}")

        if self.errors:
            self.log(f"\nErrors ({len(self.errors)}):", "ERROR")
            for error in self.errors:
                print(f"  • {error}")
            self.log("\n✗ Configuration validation FAILED", "ERROR")
            return False

        self.log("\n✓ All configuration validations passed", "SUCCESS")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate IB Robot configuration consistency")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("src/robot_config/config/robots/so101_single_arm.yaml"),
        help="Path to robot configuration YAML",
    )
    parser.add_argument(
        "--controllers-config", type=Path, help="Path to controllers configuration YAML (auto-resolved if not provided)"
    )
    parser.add_argument("--moveit-config", type=Path, help="Path to MoveIt controllers configuration YAML")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Try to auto-detect MoveIt config if not provided
    moveit_config = args.moveit_config
    if not moveit_config:
        candidate = Path("src/robots/so101/so101_motion/config/so101/moveit_controllers.yaml")
        if candidate.exists():
            moveit_config = candidate

    validator = ConfigValidator(verbose=args.verbose)

    try:
        success = validator.run_validation(args.robot_config, args.controllers_config, moveit_config)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Validation failed with exception: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
