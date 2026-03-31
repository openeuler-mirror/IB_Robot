"""MoveIt 2 launch builders.

This module handles:
- MoveIt 2 core node generation (move_group)
- RViz visualization for MoveIt
- Inclusion of external MoveIt launch files
"""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.moveit")


def generate_moveit_nodes(robot_config, control_mode, use_sim=False, display=True):
    """Generate MoveIt 2 nodes.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag
        display: Whether to launch RViz visualization

    Returns:
        List of launch actions for MoveIt 2
    """
    actions = []
    
    # Check if MoveIt is needed for this control mode
    # Usually enabled for 'moveit_planning' or any mode with 'moveit' in name
    with_moveit = 'moveit' in control_mode.lower()
    
    if not with_moveit:
        return actions

    logger.info(f"Generating MoveIt Nodes (mode: {control_mode}, display: {display})")

    # Find MoveIt launch file
    try:
        moveit_package_dir = get_package_share_directory('robot_moveit')
        moveit_launch_file = Path(moveit_package_dir) / 'launch' / 'so101_moveit.launch.py'

        if moveit_launch_file.exists():
            # Get joint_names from robot_config to pass to MoveIt launch
            joint_names = robot_config['joints']['arm']
            # Convert list to space-separated string for launch argument
            joint_names_str = ' '.join(joint_names)

            # Get MoveIt gateway parameters from robot_config
            arm_group_name = robot_config['moveit']['arm_group_name']
            base_link = robot_config['moveit']['base_link']
            ee_link = robot_config['moveit']['ee_link']
            shoulder_link = robot_config['moveit']['shoulder_link']

            # Include MoveIt launch file
            moveit_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(moveit_launch_file)),
                launch_arguments={
                    'is_sim': 'True' if use_sim else 'False',
                    'display': 'True' if display else 'False',
                    'joint_names': joint_names_str,
                    'arm_group_name': arm_group_name,
                    'base_link': base_link,
                    'ee_link': ee_link,
                    'shoulder_link': shoulder_link,
                }.items()
            )
            actions.append(moveit_launch)
            logger.info(f"Added MoveIt launch (is_sim={use_sim}, display={display})")
        else:
            logger.warning(f"MoveIt launch file not found at {moveit_launch_file}")
    except Exception as e:
        logger.warning(f"Could not find robot_moveit package: {e}")
        logger.info("Continuing without MoveIt...")

    return actions
