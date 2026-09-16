"""Exercise the live binding launch boundary without executing any ROS processes."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from launch import LaunchContext
from launch.actions import EmitEvent, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.event_handlers import OnProcessExit
from launch.events.process import ProcessExited
from launch_ros.actions import Node

from robot_config.launch_builders import control, execution, perception, recording, runtime, sim_backend
from robot_config.loader import load_robot_config_dict
from robot_runtime.interface_description import description_digest, validate_description


@pytest.fixture
def startup(monkeypatch, tmp_path, public_description):
    source = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location(
        "embodied_interface_launch_test", source / "embodied_bringup/launch/embodied_pipeline.launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: str(source / package))
    profile = {"width": 640, "height": 480, "fps": 30.0, "encoding": "bgr8"}
    descriptor = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "test", "runtime_name": "test_robot", "runtime_version": "1.0.0"},
        "execution": "physical",
        "interfaces": {
            "camera.front.color": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/image",
                "message_type": "sensor_msgs/msg/Image",
                "qos": {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5},
                "frame_id": "front_optical",
                "configured_profile": profile,
                "supported_profiles": None,
                "camera_info_topic": "/unit/front/info",
            },
            "camera.front.color_info": {
                "capability": "perception.camera",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/front/info",
                "message_type": "sensor_msgs/msg/CameraInfo",
                "qos": {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5},
                "frame_id": "front_optical",
            },
        },
        "states": {
            "camera.front.color": {
                "state": "ready",
                "observed_profile": copy.deepcopy(profile),
                "observed_frame_id": "front_optical",
                "last_seen": 1_800_000_000.0,
                "detail": "",
            },
        },
    }
    descriptor["model"] = public_description["model"]
    for key, interface in public_description["interfaces"].items():
        descriptor["interfaces"].setdefault(key, interface)
    descriptor["digest"] = description_digest(descriptor)
    validate_description(descriptor)
    config = {
        "name": "consumer",
        "type": "test",
        "runtime": {"provider": "test_robot", "profile": "test", "instance_id": "unit-1"},
        "default_control_mode": "teleop",
        "skill_required_control_mode": "moveit_planning",
        "controller_startup_timeout": {"hardware": 17.0, "sim": 23.0},
        "control_modes": {"moveit_planning": {"controllers": ["arm_trajectory_controller"]}},
        "contract": {
            "observations": [
                {
                    "key": "observation.images.front",
                    "interface": "camera.front.color",
                    "image": {"resize": [224, 224], "encoding": "rgb8"},
                }
            ]
        },
        "embodied": {
            "enabled": False,
            "entry_mode": "hermes",
            "skill_catalog_profile": "test",
            "safety": {"motion_authorized": True},
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": {"enabled": True, "handler": "sorting_hat_v1", "summary": "Sort"}},
        },
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"robot": config}), encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_path": str(path),
            "control_mode": "moveit_planning",
            "with_embodied": "true",
            "with_perception": "true",
            "entry_mode": "hermes",
            "with_moveit": "false",
            "with_inference": "true",
            "record": "true",
            "record_mode": "episodic",
            "voice_asr_realtime_pre_roll_seconds": "1.25",
        }
    )
    providers = []
    seen = []
    waiter = ExecuteProcess(cmd=["waiter-is-never-executed"])
    owner = LogInfo(msg="mock provider")

    def provider(robot_config, **options):
        providers.append((copy.deepcopy(robot_config), options))
        assert len(providers) == 1, "runtime provider was started twice"
        return [owner], waiter

    def observe(kind, effective, options):
        snapshot = Path(effective["_config_path"])
        assert snapshot != path and snapshot.is_file(), "consumer constructed before snapshot"
        assert yaml.safe_load(snapshot.read_text())["robot"] == {
            key: value for key, value in effective.items() if key != "_config_path"
        }
        seen.append((kind, effective, options))

    def base_execution(effective, *_args, **options):
        observe("base", effective, options)
        return []

    real_recording = recording.generate_recording_nodes
    real_embodied = module.generate_embodied_nodes

    def base_recording(effective, *args, **options):
        observe("recording", effective, options)
        return real_recording(effective, *args, **options)

    def embodied(effective, *args, **options):
        observe("embodied", effective, options)
        return real_embodied(effective, *args, **options)

    def forbidden(*_args, **_kwargs):
        pytest.fail("unexpected hardware, simulation, or controller waiter construction")

    monkeypatch.setattr(runtime, "generate_runtime_provider_actions", provider)
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(execution, "generate_execution_nodes", base_execution)
    monkeypatch.setattr(recording, "generate_recording_nodes", base_recording)
    monkeypatch.setattr(module, "generate_embodied_nodes", embodied)
    monkeypatch.setattr(control, "generate_ros2_control_nodes", forbidden)
    monkeypatch.setattr(sim_backend, "get_sim_backend", forbidden)
    for name in ("generate_camera_nodes", "generate_lidar_nodes", "generate_tf_nodes"):
        monkeypatch.setattr(perception, name, forbidden)

    def prepare():
        groups = module.launch_setup(context)
        assert not seen and not providers
        assert len(groups) == 1 and isinstance(groups[0], GroupAction)
        actions = []
        for action in groups[0].execute(context):
            if isinstance(action, OpaqueFunction):
                monkeypatch.setitem(action._OpaqueFunction__function.__globals__, "load_robot_config", forbidden)
            actions.extend(action.execute(context) or [])
        assert not seen and len(providers) == 1
        assert actions[-2:] == [owner, waiter]
        assert not any(isinstance(action, IncludeLaunchDescription) for action in actions)
        return actions

    def finish(actions, returncode=0):
        event = ProcessExited(
            action=actions[-1], name="waiter", cmd=[], cwd=None, env=None, pid=1, returncode=returncode
        )
        handler = next(
            action.event_handler
            for action in actions
            if isinstance(getattr(action, "event_handler", None), OnProcessExit)
        )
        assert handler.matches(event)
        return list(handler.handle(event, context))

    return SimpleNamespace(
        module=module,
        config=config,
        descriptor=descriptor,
        path=path,
        context=context,
        providers=providers,
        seen=seen,
        prepare=prepare,
        finish=finish,
    )


@pytest.mark.parametrize("use_sim", [False, True])
@pytest.mark.parametrize("authorize_motion", [None, "false", "true"])
def test_consumers_share_one_live_snapshot(startup, use_sim, authorize_motion):
    startup.context.launch_configurations["use_sim"] = str(use_sim).lower()
    startup.context.launch_configurations["moveit_display"] = "true"
    if authorize_motion is not None:
        startup.context.launch_configurations["authorize_motion"] = authorize_motion
    startup.descriptor["execution"] = "simulated" if use_sim else "physical"
    startup.descriptor["digest"] = description_digest(startup.descriptor)
    actions = startup.prepare()
    unbound, options = startup.providers[0]
    assert "topic" not in unbound["contract"]["observations"][0]
    assert options["use_sim"] is use_sim and options["display"] is True
    assert options["readiness_timeout_s"] == (23.0 if use_sim else 17.0)
    description_path = Path(options["description_output"])
    description_path.write_text(yaml.safe_dump(startup.descriptor), encoding="utf-8")
    returned = startup.finish(actions)

    assert [kind for kind, *_rest in startup.seen] == ["base", "recording", "embodied", "embodied"]
    effective = startup.seen[0][1]
    assert all(config is effective for _kind, config, _options in startup.seen)
    assert effective["default_control_mode"] == "moveit_planning"
    assert effective["embodied"]["enabled"] and effective["embodied"]["perception"]["enabled"]
    assert effective["voice_asr"]["realtime_pre_roll_seconds"] == 1.25
    assert effective["contract"]["observations"][0]["topic"] == "/unit/front/image"
    assert effective["contract"]["observations"][0]["image"] == {"resize": [224, 224], "encoding": "rgb8"}
    assert startup.seen[0][2]["use_sim_time"] is False
    for _kind, _config, consumer_options in startup.seen[2:]:
        assert consumer_options["motion_authorized"] is (authorize_motion == "true")
    assert startup.seen[2][2]["include_motion"] is False
    assert startup.seen[2][2]["include_perception"] is True
    assert startup.seen[3][2]["include_visual_games"] is False
    assert startup.seen[3][2]["include_perception"] is False
    assert len(startup.providers) == 1
    assert all(isinstance(action, Node) for action in returned)
    executables = [action.node_executable for action in returned]
    assert "wait_for_controllers" not in executables and "wait_for_runtime" not in executables
    assert executables.count("perception_service_node") == 1
    assert executables.count("visual_game_gateway_node") == 1
    assert executables.count("skill_executor_node") == 1
    recorder = next(action for action in returned if action.node_package == "dataset_tools")
    params = {"".join(part.text for part in name): value for name, value in recorder._Node__parameters[0].items()}
    assert yaml.safe_load("".join(part.text for part in params["robot_config_path"])) == effective["_config_path"]
    assert load_robot_config_dict(effective["_config_path"])["contract"] == effective["contract"]


@pytest.mark.parametrize("failure", ["waiter", "missing_descriptor", "stale", "wrong_runtime", "bad_descriptor"])
def test_binding_failure_aborts_all_consumers(startup, failure):
    actions = startup.prepare()
    description_path = Path(startup.providers[0][1]["description_output"])
    if failure == "stale":
        startup.descriptor["states"]["camera.front.color"]["state"] = "stale"
    elif failure == "wrong_runtime":
        startup.descriptor["robot"]["runtime_name"] = "other_robot"
    elif failure == "bad_descriptor":
        startup.descriptor["schema_version"] = 99
    startup.descriptor["digest"] = description_digest(startup.descriptor)
    if failure != "missing_descriptor":
        description_path.write_text(yaml.safe_dump(startup.descriptor), encoding="utf-8")
    returned = startup.finish(actions, returncode=1 if failure == "waiter" else 0)
    assert not startup.seen and len(startup.providers) == 1
    assert len(returned) == 1 and isinstance(returned[0], EmitEvent)
    assert not (description_path.parent / "robot.yaml").exists()


@pytest.mark.parametrize("snapshot", ["embedded", "missing_file"])
def test_offline_descriptor_cannot_skip_live_readiness(startup, snapshot):
    startup.config["runtime"]["interface_description"] = (
        startup.descriptor if snapshot == "embedded" else "missing-offline-description.yaml"
    )
    startup.config["_interfaces_deferred"] = False
    startup.path.write_text(yaml.safe_dump({"robot": startup.config}), encoding="utf-8")
    actions = startup.prepare()
    returned = startup.finish(actions, returncode=1)
    assert not startup.seen and len(startup.providers) == 1
    assert len(returned) == 1 and isinstance(returned[0], EmitEvent)


@pytest.mark.parametrize("consumer", ["base", "embodied"])
def test_consumer_construction_failure_aborts_launch(startup, monkeypatch, consumer):
    def fail(*_args, **_kwargs):
        raise RuntimeError("test consumer construction failed")

    if consumer == "base":
        monkeypatch.setattr(execution, "generate_execution_nodes", fail)
    else:
        monkeypatch.setattr(startup.module, "generate_embodied_nodes", fail)
    actions = startup.prepare()
    Path(startup.providers[0][1]["description_output"]).write_text(yaml.safe_dump(startup.descriptor), encoding="utf-8")
    returned = startup.finish(actions)
    assert len(startup.providers) == 1
    assert len(returned) == 1 and isinstance(returned[0], EmitEvent)
    assert not any(kind == "embodied" for kind, *_rest in startup.seen)


def test_snapshot_preserves_development_catalog_location(startup, tmp_path):
    catalog_root = tmp_path / "src/skill_catalog"
    catalog_root.mkdir(parents=True)
    source_path = tmp_path / "src/robot_config/config/robots/input.yaml"
    source_path.parent.mkdir(parents=True)
    startup.config["embodied"].update(
        skill_catalog_source_mode="development", skill_catalog_source_root="src/skill_catalog"
    )
    source_path.write_text(yaml.safe_dump({"robot": startup.config}), encoding="utf-8")
    startup.context.launch_configurations["config_path"] = str(source_path)
    actions = startup.prepare()
    assert startup.providers[0][0]["embodied"]["skill_catalog_source_root"] == str(catalog_root)
    Path(startup.providers[0][1]["description_output"]).write_text(yaml.safe_dump(startup.descriptor), encoding="utf-8")
    returned = startup.finish(actions)
    assert not any(isinstance(action, EmitEvent) for action in returned)
    snapshot = startup.seen[0][1]["_config_path"]
    embodied = load_robot_config_dict(snapshot)["embodied"]
    assert embodied["skill_catalog_source_mode"] == "development"
    assert embodied["skill_catalog_source_root"] == str(catalog_root)


def test_embodied_disabled_still_waits_for_base_binding(startup):
    startup.context.launch_configurations["with_embodied"] = "false"
    actions = startup.prepare()
    Path(startup.providers[0][1]["description_output"]).write_text(yaml.safe_dump(startup.descriptor), encoding="utf-8")
    returned = startup.finish(actions)
    assert [kind for kind, *_rest in startup.seen] == ["base", "recording"]
    assert len(startup.providers) == 1
    assert len(returned) == 1 and returned[0].node_package == "dataset_tools"


def test_legacy_config_keeps_visual_startup_and_controller_barrier(startup, monkeypatch):
    startup.config.pop("runtime")
    observation = startup.config["contract"]["observations"][0]
    observation.pop("interface")
    observation.update(topic="/legacy/image", type="sensor_msgs/msg/Image")
    startup.path.write_text(yaml.safe_dump({"robot": startup.config}), encoding="utf-8")
    seen = []
    visual, motion = LogInfo(msg="visual consumer"), LogInfo(msg="motion consumer")

    def embodied(config, _mode, **options):
        seen.append(config["_config_path"])
        assert options["motion_authorized"] is False
        return [motion] if options.get("include_visual_games") is False else [visual]

    monkeypatch.setattr(startup.module, "generate_embodied_nodes", embodied)
    actions = startup.module.launch_setup(startup.context)
    assert not startup.providers and seen == [str(startup.path), str(startup.path)]
    assert isinstance(actions[0], IncludeLaunchDescription)
    arguments = dict(actions[0].launch_arguments)
    assert arguments["config_path"] == str(startup.path)
    assert arguments["with_embodied"] == "false" and arguments["with_perception"] == "false"
    assert visual in actions and motion not in actions
    assert actions[-1].node_executable == "wait_for_controllers"
    assert startup.finish(actions) == [motion]
