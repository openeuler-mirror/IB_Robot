"""Resolve dispatcher feedback from public contracts without provider imports."""

from robot_config.contract_utils import qos_profile_from_dict
from robot_config.interface_binding import InterfaceBindingError


def resolve_joint_feedback(contract, default_topic, default_qos, runtime=None):
    """Return the feedback topic, real ROS QoS and local freshness budget."""
    runtime = runtime if isinstance(runtime, dict) else {}
    topic, qos = default_topic, default_qos
    if runtime.get("provider"):
        from robot_runtime.interface_description import validate_description

        description = runtime.get("interface_description")
        if not isinstance(description, dict):
            raise InterfaceBindingError(
                "description_required", "runtime.interface_description", "bound snapshot required"
            )
        try:
            validate_description(description)
        except ValueError as exc:
            raise InterfaceBindingError("invalid_descriptor", "runtime.interface_description", str(exc)) from exc
        interface = description["interfaces"].get("joint.state", {})
        if (interface.get("kind"), interface.get("direction"), interface.get("message_type")) != (
            "topic",
            "publish",
            "sensor_msgs/msg/JointState",
        ):
            raise InterfaceBindingError("invalid_interface", "joint.state", "JointState publication required")
        topic = interface["endpoint"]
        qos = qos_profile_from_dict(interface["qos"])

    bound = [
        observation
        for observation in contract.observations
        if (observation._interface_source or {}).get("id") == "joint.state"
    ]
    if bound:
        topics = {observation.topic for observation in bound}
        if len(topics) != 1 or (runtime.get("provider") and topic not in topics):
            raise InterfaceBindingError("topic_mismatch", "joint.state", "ambiguous feedback endpoints")
        topic = bound[0].topic
    observations = [
        observation
        for observation in contract.observations
        if observation.topic == topic and observation.type == "sensor_msgs/msg/JointState"
    ]
    if observations:
        observation = observations[0]
        qos = qos_profile_from_dict(observation.qos) or qos
        max_age_ms = int(observation.align.max_age_ms) if observation.align else 0
    else:
        max_age_ms = 0
    return topic, qos, max_age_ms * 1_000_000
