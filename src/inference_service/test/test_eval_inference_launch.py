import importlib.util
from pathlib import Path

from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def test_eval_launch_propagates_pipeline_scoped_video_control_topics():
    launch_path = Path(__file__).parents[1] / "launch" / "eval_inference.launch.py"
    spec = importlib.util.spec_from_file_location("eval_inference_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    description = module.generate_launch_description()
    arguments = {entity.name: entity for entity in description.entities if isinstance(entity, DeclareLaunchArgument)}
    node = next(entity for entity in description.entities if isinstance(entity, Node))
    parameters = {key[0].text: value[0] for key, value in node._Node__parameters[0].items()}

    assert "video_descriptor_topic" in arguments
    assert "video_status_topic" in arguments
    assert parameters["video_descriptor_topic"].variable_name[0].text == "video_descriptor_topic"
    assert parameters["video_status_topic"].variable_name[0].text == "video_status_topic"


def test_distributed_launches_preserve_runtime_options_as_json_text():
    launch_dir = Path(__file__).parents[1] / "launch"
    for filename in ("cloud_inference.launch.py", "local_distributed_inference.launch.py"):
        spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), launch_dir / filename)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        description = module.generate_launch_description()
        nodes = [entity for entity in description.entities if isinstance(entity, Node)]
        assert nodes
        for node in nodes:
            runtime_options = next(
                value for key, value in node._Node__parameters[0].items() if key[0].text == "runtime_options_json"
            )
            assert isinstance(runtime_options, ParameterValue)
            assert runtime_options.value_type is str
