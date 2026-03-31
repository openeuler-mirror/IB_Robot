"""Voice ASR launch builder for robot_config."""

from typing import Any, Dict, List

from launch_ros.actions import Node
from robot_config.logger_utils import get_colored_logger

from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.voice_asr")


def generate_voice_asr_nodes(robot_config: Dict[str, Any]) -> List[Node]:
    """Generate voice ASR nodes from robot_config YAML."""
    voice_asr_config = robot_config.get("voice_asr", {})
    if not voice_asr_config.get("enabled", False):
        logger.info("Voice ASR disabled, skipping")
        return []

    model_path = voice_asr_config.get("model_path", "")
    tokens_path = voice_asr_config.get("tokens_path", "")

    if not model_path:
        raise ValueError(
            "voice_asr.model_path is required when voice_asr.enabled is true"
        )

    node_params = {
        "active_mode": voice_asr_config.get("active_mode", "manual"),
        "language": voice_asr_config.get("language", "zh"),
        "model_path": resolve_ros_path(model_path) if model_path else "",
        "tokens_path": resolve_ros_path(tokens_path) if tokens_path else "",
        "provider": voice_asr_config.get("provider", "cpu"),
        "model_type": voice_asr_config.get("model_type", "auto"),
        "max_recording_duration": voice_asr_config.get("max_recording_duration", 10.0),
        "vad_sensitivity": voice_asr_config.get("vad_sensitivity", 0.5),
        "publish_partial": voice_asr_config.get("publish_partial", True),
        "output_topic": voice_asr_config.get("output_topic", "/voice_command"),
        "sample_rate": voice_asr_config.get("sample_rate", 16000),
        "chunk_size": voice_asr_config.get("chunk_size", 512),
        "buffer_seconds": voice_asr_config.get("buffer_seconds", 5.0),
        "device_index": voice_asr_config.get("device_index", -1),
    }

    node_name = voice_asr_config.get("node_name", "voice_asr_node")
    logger.info(f"Voice ASR enabled, launching node '{node_name}'")
    logger.info(f"  output_topic: {node_params['output_topic']}")

    return [
        Node(
            package="voice_asr_service",
            executable="voice_asr_node",
            name=node_name,
            output="screen",
            parameters=[node_params],
        )
    ]
