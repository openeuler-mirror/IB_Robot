"""Check the launch command without starting the Web API or importing its runtime."""

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.substitutions import ExecutableInPackage

_LAUNCH_PATH = Path(__file__).resolve().parents[1] / "launch" / "tracing_web.launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("tracing_web_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None
assert _LAUNCH_SPEC.loader is not None
web_launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(web_launch)


@pytest.fixture
def launch_setup():
    context = LaunchContext()
    description = web_launch.generate_launch_description()
    for action in description.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    setup = next(action for action in description.entities if isinstance(action, OpaqueFunction))
    return context, setup


@pytest.mark.parametrize(
    ("overrides", "expected_arguments"),
    [
        (
            {},
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--trace-root",
                "~/.ros/tracing",
                "--allowed-host",
                "localhost",
                "--allowed-host",
                "127.0.0.1",
            ],
        ),
        (
            {
                "host": "0.0.0.0",
                "port": "8443",
                "trace_root": "/tmp/trace captures",
                "allowed_hosts": " robot.local, ,192.0.2.1, ",
                "allow_unauthenticated_lan": "True",
                "certfile": "/tmp/tls files/cert.pem",
                "keyfile": "/tmp/tls files/key.pem",
            },
            [
                "--host",
                "0.0.0.0",
                "--port",
                "8443",
                "--trace-root",
                "/tmp/trace captures",
                "--allowed-host",
                "robot.local",
                "--allowed-host",
                "192.0.2.1",
                "--allow-unauthenticated-lan",
                "--certfile",
                "/tmp/tls files/cert.pem",
                "--keyfile",
                "/tmp/tls files/key.pem",
            ],
        ),
    ],
)
def test_web_launch_passes_only_cli_arguments(launch_setup, overrides, expected_arguments):
    context, setup = launch_setup
    context.launch_configurations.update(overrides)

    actions = setup.execute(context)

    assert len(actions) == 1
    process = actions[0]
    arguments = [perform_substitutions(context, argument) for argument in process.cmd[1:]]
    assert "--ros-args" not in arguments
    assert type(process) is ExecuteProcess
    assert arguments == expected_arguments
    executable = process.cmd[0][0]
    assert isinstance(executable, ExecutableInPackage)
    assert perform_substitutions(context, executable.package) == "ibrobot_tracing_web"
    assert perform_substitutions(context, executable.executable) == "ibrobot-tracing-web"


@pytest.mark.parametrize("tls_argument", ["certfile", "keyfile"])
def test_web_launch_rejects_partial_tls_configuration(launch_setup, tls_argument):
    context, setup = launch_setup
    context.launch_configurations[tls_argument] = "/tmp/tls.pem"

    with pytest.raises(RuntimeError, match="certfile and keyfile must be provided together"):
        setup.execute(context)
