"""Contract validation for robot configuration."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ContractSynthesisError(Exception):
    """Raised when contract synthesis fails due to architectural errors."""
    pass

def validate_control_mode_config(robot_config: Dict, control_mode: str) -> None:
    """Validate control mode configuration BEFORE synthesis.

    This performs architectural validation to catch configuration errors
    at system startup time, not during inference.

    Args:
        robot_config: Full robot configuration dict
        control_mode: Control mode name to validate

    Raises:
        ContractSynthesisError: If configuration is architecturally invalid
    """
    errors = []
    warnings = []

    # Check 1: Control mode exists
    control_modes = robot_config.get('control_modes', {})
    if control_mode not in control_modes:
        errors.append(f"Control mode '{control_mode}' not defined in robot_config")
        raise ContractSynthesisError("\n".join(errors))

    mode_config = control_modes[control_mode]

    # Check 2: Inference configuration
    inference_config = mode_config.get('inference', {})
    if inference_config.get('enabled', False):

        # Check 2.1: Model reference exists
        model_name = inference_config.get('model')
        models = robot_config.get('models', {})
        if model_name not in models:
            errors.append(
                f"Control mode '{control_mode}' requires model '{model_name}' "
                f"but it's not defined in robot_config.models"
            )

        # Check 2.2: Contract observations exist (Single Source of Truth)
        contract_config = robot_config.get('contract', {})
        observations = contract_config.get('observations', [])
        
        if not observations:
            errors.append(
                f"No observations defined in contract section. "
                f"Please add 'contract.observations' to robot_config.yaml"
            )
        else:
            # Validate observation peripheral references
            for obs_spec in observations:
                peripheral_name = obs_spec.get('peripheral')
                if peripheral_name:
                    peripheral = next(
                        (p for p in robot_config.get('peripherals', []) if p['name'] == peripheral_name),
                        None
                    )
                    if not peripheral:
                        errors.append(
                            f"Observation '{obs_spec.get('key')}' references peripheral '{peripheral_name}' "
                            f"but it's not defined in robot_config.peripherals"
                        )

    # Check 3: Executor configuration
    executor_config = mode_config.get('executor', {})
    executor_type = executor_config.get('type')
    if executor_type and executor_type not in ['topic', 'action']:
        errors.append(
            f"Invalid executor type '{executor_type}' in mode '{control_mode}'. "
            f"Must be 'topic' or 'action'"
        )

    # Check 4: Controllers exist
    controllers = mode_config.get('controllers', [])
    ros2_control_config = robot_config.get('ros2_control', {})
    defined_controllers = ros2_control_config.get('controllers', [])

    for ctrl in controllers:
        if ctrl not in defined_controllers:
            warnings.append(
                f"Controller '{ctrl}' used in mode '{control_mode}' "
                f"but not listed in ros2_control.controllers"
            )

    # Report results
    if warnings:
        logger.warning(f"Configuration warnings for mode '{control_mode}':")
        for warning in warnings:
            logger.warning(f"  - {warning}")

    if errors:
        error_msg = (
            f"Architectural errors in control mode '{control_mode}':\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
        logger.error(error_msg)
        raise ContractSynthesisError(error_msg)
