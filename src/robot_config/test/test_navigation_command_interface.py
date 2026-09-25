from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_navigation_command_api_and_cli_are_installed():
    action = ROOT / "src/ibrobot_msgs/action/ExecuteNavigation.action"
    setup = ROOT / "src/robot_navigation/setup.py"
    cli = ROOT / "src/robot_navigation/robot_navigation/nav_cmd.py"
    retired_cli = ROOT / "src/robot_navigation/robot_navigation/nav_request.py"
    retired_script = ROOT / "scripts/nav_request.sh"

    content = action.read_text(encoding="utf-8")
    assert "geometry_msgs/PoseStamped target_pose" in content
    assert "translation commands use meters" in content
    assert "TURN_LEFT and TURN_RIGHT use radians" in content
    assert "ABSOLUTE_POSE ignores value" in content
    assert "relative commands ignore target_pose" in content
    assert "string state" in content
    assert "geometry_msgs/PoseStamped resolved_target_pose" in content
    setup_content = setup.read_text(encoding="utf-8")
    assert "navigation_command_server = robot_navigation.navigation_command_server:main" in setup_content
    assert "nav_cmd = robot_navigation.nav_cmd:main" in setup_content
    assert "nav_request =" not in setup_content
    cli_content = cli.read_text(encoding="utf-8")
    assert "/navigation/execute" not in cli_content
    assert "/navigation/cancel_current" in cli_content
    assert '"leftward": ExecuteNavigation.Goal.STRAFE_LEFT' in cli_content
    assert '"rightward": ExecuteNavigation.Goal.STRAFE_RIGHT' in cli_content
    assert "strafe-left" not in cli_content
    assert "strafe-right" not in cli_content
    assert not retired_cli.exists()
    assert not retired_script.exists()


def test_lidar_navigation_stage_enables_command_adapter_on_dynamic_chain():
    config_path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    navigation = config["nav_stages"]["navigation"]["navigation"]
    command = navigation["command_server"]

    assert navigation["nav2_bringup"]["dyn_avoid_enabled"] is True
    assert command == {
        "enabled": True,
        "action_name": "/navigation/execute",
        "cancel_service_name": "/navigation/cancel_current",
        "nav2_action_name": "/navigate_to_pose",
        "nav2_result_timeout": 300.0,
        "stop_velocity_topic": "/cmd_vel_safe",
        "cancel_response_timeout": 2.0,
    }
    assert "command_server" not in config["nav_stages"]["mapping"]["navigation"]


def test_lifecycle_startup_is_ros_native_and_keeps_warmup_boundary():
    launch_path = ROOT / "src/robot_navigation/launch/nav2_bringup.launch.py"
    launch_content = launch_path.read_text(encoding="utf-8")

    assert '"autostart": "false"' in launch_content
    assert "navigation_lifecycle_coordinator" in launch_content
    assert "ExecuteProcess(" not in launch_content
    assert "TimerAction(" not in launch_content
    assert "ros2 service call" not in launch_content


def test_real_navigation_allows_slow_board_action_acknowledgements():
    params_path = ROOT / "src/robot_navigation/config/nav2_params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))

    bt_params = params["bt_navigator"]["ros__parameters"]
    assert bt_params["default_server_timeout"] == 1000


def test_real_navigation_loads_only_default_tree_plugins():
    params_path = ROOT / "src/robot_navigation/config/nav2_params.yaml"
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))

    assert params["bt_navigator"]["ros__parameters"]["plugin_lib_names"] == [
        "nav2_compute_path_to_pose_action_bt_node",
        "nav2_compute_path_through_poses_action_bt_node",
        "nav2_remove_passed_goals_action_bt_node",
        "nav2_follow_path_action_bt_node",
        "nav2_clear_costmap_service_bt_node",
        "nav2_goal_updated_condition_bt_node",
        "nav2_spin_action_bt_node",
        "nav2_wait_action_bt_node",
        "nav2_back_up_action_bt_node",
        "nav2_rate_controller_bt_node",
        "nav2_recovery_node_bt_node",
        "nav2_pipeline_sequence_bt_node",
        "nav2_round_robin_node_bt_node",
    ]


def test_navigation_include_does_not_leak_disabled_autostart_to_sibling_actions():
    launch_path = ROOT / "src/robot_navigation/launch/nav2_bringup.launch.py"
    launch_content = launch_path.read_text(encoding="utf-8")

    assert "GroupAction(" in launch_content
    assert "scoped=True" in launch_content
