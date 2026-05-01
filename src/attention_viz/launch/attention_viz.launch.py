#!/usr/bin/env python3
"""Attention visualization standalone launch file."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="file",
            description="Visualization mode: 'interactive', 'realtime' or 'file'",
        ),
        DeclareLaunchArgument(
            "save_dir",
            default_value="attention_visualizations",
            description="Directory to save heatmap images (file mode)",
        ),
        DeclareLaunchArgument(
            "attention_topic",
            default_value="/attention/weights",
            description="Topic for attention weight messages",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Disable GUI and save/publish heatmaps in file mode",
        ),
        DeclareLaunchArgument(
            "update_frequency",
            default_value="10.0",
            description="Maximum heatmap update frequency in Hz",
        ),
        DeclareLaunchArgument(
            "heatmap_topic_prefix",
            default_value="/visualization/heatmap",
            description="Topic prefix for generated heatmap images",
        ),
        Node(
            package="attention_viz",
            executable="attention_visualization_node",
            name="attention_visualization",
            parameters=[
                {
                    "visualization_mode": LaunchConfiguration("mode"),
                    "attention_topic": LaunchConfiguration("attention_topic"),
                    "save_dir": LaunchConfiguration("save_dir"),
                    "headless": ParameterValue(
                        LaunchConfiguration("headless"),
                        value_type=bool,
                    ),
                    "update_frequency": ParameterValue(
                        LaunchConfiguration("update_frequency"),
                        value_type=float,
                    ),
                    "heatmap_topic_prefix": LaunchConfiguration("heatmap_topic_prefix"),
                    "queries_to_visualize": [0, 20, 40, 60, 80],
                    "layer_idx": -1,
                    "batch_idx": 0,
                    "average_heads": True,
                    "blend_alpha": 0.4,
                },
            ],
            output="screen",
        ),
    ])
