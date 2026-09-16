"""Real ROS discovery -> YAML snapshot -> IB-Robot contract binding, without motors."""

import json
import subprocess
import sys
import time

import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import GetRuntimeStatus
from robot_runtime.interface_description import load_description, validate_description
from robot_runtime.mock_runtime_node import MockRuntime, default_profile


def test_status_snapshot_and_consumer_binding(tmp_path):
    profile = default_profile(base=False)
    profile["runtime"].update(name="schema_robot", instance_id="unit-1")
    profile["peripherals"] = [
        {
            "type": "camera",
            "name": "front",
            "driver": "opencv",
            "width": 16,
            "height": 12,
            "fps": 20,
            "pixel_format": "rgb8",
            "frame_id": "front_optical",
        }
    ]
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    rclpy.init(args=["--ros-args", "-p", f"profile:={path}"])
    runtime = MockRuntime()
    caller = Node("schema_caller")
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(runtime)
    executor.add_node(caller)
    output = tmp_path / "description.yaml"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "robot_runtime.wait_for_runtime",
            "--runtime-name",
            "schema_robot",
            "--instance-id",
            "unit-1",
            "--timeout",
            "12",
            "--require-interfaces",
            "camera.front.color",
            "joint.state",
            "--interface-requirements",
            '{"camera.front.color": {"width": 16, "height": 12, "min_fps": 10}}',
            "--description-output",
            str(output),
            "--ros-args",
            "-r",
            "__node:=schema_waiter",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while process.poll() is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert process.poll() == 0, process.communicate(timeout=2)
        descriptor = load_description(output)
        image = descriptor["states"]["camera.front.color"]
        assert image["state"] == "ready"
        assert image["observed_profile"]["width"] == 16
        assert image["observed_profile"]["encoding"] == "rgb8"
        assert image["observed_profile"]["fps"] > 10
        client = caller.create_client(GetRuntimeStatus, "/runtime/get_status")
        assert client.wait_for_service(timeout_sec=3)
        future = client.call_async(GetRuntimeStatus.Request())
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        current = json.loads(future.result().status.interface_description_json)
        validate_description(current)
        assert current["digest"] == descriptor["digest"]

        # The runtime-only gate intentionally has no robot_config installed.
        try:
            from robot_config.loader import build_contract_from_robot_config_dict, load_robot_config_dict
        except ImportError:
            return
        config = {
            "robot": {
                "name": "interface_client",
                "default_control_mode": "teleop",
                "runtime": {
                    "provider": "schema_robot",
                    "profile": str(path),
                    "instance_id": "unit-1",
                    "interface_description": str(output),
                },
                "contract": {
                    "observations": [
                        {
                            "key": "observation.images.front",
                            "interface": "camera.front.color",
                            "requires": {"width": 16, "height": 12, "min_fps": 10},
                            "image": {"resize": [8, 8], "encoding": "rgb8"},
                        }
                    ]
                },
            }
        }
        consumer = tmp_path / "client.yaml"
        consumer.write_text(yaml.safe_dump(config))
        loaded = load_robot_config_dict(consumer)
        contract = build_contract_from_robot_config_dict(loaded)
        assert contract.observations[0].topic == "/camera/front/image_raw"
        assert contract.observations[0].image["resize"] == [8, 8]
        assert not loaded.get("peripherals")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        executor.shutdown(timeout_sec=2)
        caller.destroy_node()
        runtime.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
