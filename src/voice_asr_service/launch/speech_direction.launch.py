import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# CANN 安装路径：默认标准安装位置，可通过 ASCEND_TOOLKIT_HOME 环境变量覆盖，
# 兼容自定义 install path、容器挂载、非 root 安装等部署环境。
_CANN_ROOT = Path(os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"))


def _prepend_env(current: str, entries: tuple[Path, ...]) -> str:
    """为节点显式补齐CANN运行环境，避免依赖调用者是否source。"""
    values = [str(path) for path in entries if path.exists()]
    if current:
        values.append(current)
    return os.pathsep.join(values)


_MODEL_PATH_KEYS = ("speech_direction_inference_bundle", "silero_vad_inference_bundle")


def _workspace_root() -> Path:
    """从 colcon 包安装前缀定位标准工作区根目录。"""
    package_prefix = Path(get_package_prefix("voice_asr_service")).resolve()
    # 标准独立安装为 <workspace>/install/<package>，merge-install 为 <workspace>/install。
    if package_prefix.parent.name == "install":
        return package_prefix.parent.parent
    if package_prefix.name == "install":
        return package_prefix.parent
    # 自定义 install base 无稳定的工作区反推规则；保留可用默认值，并允许 models_root 显式覆盖。
    return package_prefix.parent


def _load_profile_overrides(profiles_path: str | Path, profile: str) -> dict[str, object]:
    """读取只包含平台差异的小型 profile，禁止复制公共算法参数。"""
    path = Path(profiles_path)
    with path.open(encoding="utf-8") as profile_file:
        document = yaml.safe_load(profile_file)
    profiles = document.get("profiles") if isinstance(document, Mapping) else None
    if not isinstance(profiles, Mapping) or profile not in profiles:
        raise ValueError(f"平台 profile 不存在: {profile}")
    overrides = profiles[profile]
    if not isinstance(overrides, Mapping):
        raise ValueError(f"平台 profile {profile} 必须是 mapping")
    allowed = {
        "silero_vad_backend",
        "silero_vad_deployment",
        "fullsubnet_backend",
        "fullsubnet_deployment",
    }
    unexpected = set(overrides) - allowed
    if unexpected:
        raise ValueError(f"平台 profile {profile} 包含公共算法字段: {sorted(unexpected)}")
    return dict(overrides)


def _validate_profile_combination(profile: str, params: Mapping[str, object]) -> None:
    """在 launch 边界拒绝半切换配置，避免后端与模型路径错配。"""
    if profile == "ascend_310p":
        expected = {"silero_vad_backend": "ascend", "fullsubnet_backend": "ascend"}
    elif profile == "ubuntu_cuda":
        expected = {
            "silero_vad_backend": "onnx",
            "fullsubnet_backend": "stateful_torch_cuda",
        }
    else:
        return
    mismatched = {key: (params.get(key), value) for key, value in expected.items() if params.get(key) != value}
    if mismatched:
        raise ValueError(f"平台 profile {profile} 后端组合错误: {mismatched}")


def _expected_deployment_backend(backend: str) -> str:
    """Map a node backend selector to its manifest backend family."""
    if backend == "ascend":
        return "ascend"
    if backend == "onnx":
        return "onnx"
    if backend in {"stateful_torch_cuda", "stateful_torch_cpu", "torch"}:
        return "torch"
    raise ValueError(f"未知的 speech_direction 后端: {backend}")


def _resolve_bundle_path(value: str, models_root: Path, default_name: str) -> Path:
    path = Path(value.strip()).expanduser() if value.strip() else models_root / default_name
    if not path.is_absolute():
        path = models_root / path
    return path


def _validate_deployment_platform(params: Mapping[str, object], models_root: Path) -> None:
    """校验 deployment 的 manifest backend 与声明的平台后端一致。"""
    from inference_manifest import ManifestValidationError, load_inference_manifest

    checks = (
        (
            str(params.get("silero_vad_backend", "")),
            str(params.get("silero_vad_deployment", "")),
            _resolve_bundle_path(str(params.get("silero_vad_inference_bundle", "")), models_root, "silero-vad"),
            "silero_vad",
        ),
        (
            str(params.get("fullsubnet_backend", "")),
            str(params.get("fullsubnet_deployment", "")),
            _resolve_bundle_path(str(params.get("speech_direction_inference_bundle", "")), models_root, "fullsubnet"),
            "fullsubnet",
        ),
    )
    for backend, deployment, bundle_path, label in checks:
        if not backend or not deployment:
            continue
        expected = _expected_deployment_backend(backend)
        try:
            validated = load_inference_manifest(bundle_path, deployment)
        except ManifestValidationError as exc:
            raise ValueError(f"{label} deployment '{deployment}' 无法从 {bundle_path} 加载: {exc}") from exc
        actual = validated.deployment.backend
        if actual != expected:
            raise ValueError(
                f"{label}_backend={backend} 需要 {expected} deployment，但 {deployment} 的 manifest backend 是 {actual}"
            )


def _load_speech_direction_parameters(
    config_path: str | Path,
    models_root: Path,
    *,
    profile: str = "ascend_310p",
    profiles_path: str | Path | None = None,
    parameter_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """加载基础 YAML、应用平台小覆盖，并相对 models 根解析模型路径。"""
    config_file_path = Path(config_path)
    with config_file_path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)

    # 在 launch 边界拒绝畸形配置，避免节点收到难以定位的参数错误。
    if not isinstance(document, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: YAML document 必须是 mapping")
    node_config = document.get("speech_direction_node")
    if not isinstance(node_config, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: speech_direction_node 必须存在且为 mapping")
    parameter_config = node_config.get("ros__parameters")
    if not isinstance(parameter_config, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: speech_direction_node.ros__parameters 必须存在且为 mapping")

    params = dict(parameter_config)
    if profiles_path is not None:
        params.update(_load_profile_overrides(profiles_path, profile))
    if parameter_overrides:
        params.update(parameter_overrides)
    _validate_profile_combination(profile, params)
    for key in _MODEL_PATH_KEYS:
        value = params.get(key)
        # 兼容后端未启用时允许其模型路径为空；有值时仍统一锚定 models_root。
        if not isinstance(value, str):
            raise ValueError(f"配置文件 {config_file_path}: speech_direction_node.ros__parameters.{key} 必须是字符串")
        if not value.strip():
            continue
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = models_root / path
        params[key] = str(path.resolve())
    return params


def _launch_setup(context):
    config_path = LaunchConfiguration("config_file").perform(context)
    profile = LaunchConfiguration("profile").perform(context)
    profiles_path = LaunchConfiguration("profiles_file").perform(context)
    models_root = Path(LaunchConfiguration("models_root").perform(context)).expanduser()
    channel_indices = LaunchConfiguration("microphone_channel_indices").perform(context).strip()
    parameter_overrides: dict[str, object] = {}
    for launch_name, parameter_name in (
        ("microphone_channels", "audio_channels"),
        ("microphone_sample_rate", "sample_rate"),
        ("speech_direction_audio_topic", "audio_topic"),
    ):
        value = LaunchConfiguration(launch_name).perform(context).strip()
        if value:
            parameter_overrides[parameter_name] = (
                int(value) if parameter_name in {"sample_rate", "audio_channels"} else value
            )
    if channel_indices:
        parsed_channels = yaml.safe_load(channel_indices)
        if not isinstance(parsed_channels, list):
            raise ValueError("microphone_channel_indices must be a YAML list")
        parameter_overrides["channel_indices"] = parsed_channels
    mount_yaw_deg = LaunchConfiguration("speech_direction_mount_yaw_deg").perform(context).strip()
    if mount_yaw_deg:
        parameter_overrides["mount_yaw_deg"] = float(mount_yaw_deg)
    params = _load_speech_direction_parameters(
        config_path,
        models_root,
        profile=profile,
        profiles_path=profiles_path,
        parameter_overrides=parameter_overrides,
    )
    _validate_deployment_platform(params, models_root)
    # 只有 Ascend ACL profile 需要 CANN；Ubuntu CUDA 不注入无关环境。
    environment = {}
    if params.get("silero_vad_backend") == "ascend" or params.get("fullsubnet_backend") == "ascend":
        environment = {
            "ASCEND_HOME_PATH": str(_CANN_ROOT),
            "PYTHONPATH": _prepend_env(os.environ.get("PYTHONPATH", ""), (_CANN_ROOT / "python" / "site-packages",)),
            "LD_LIBRARY_PATH": _prepend_env(
                os.environ.get("LD_LIBRARY_PATH", ""),
                (_CANN_ROOT / "lib64", _CANN_ROOT / "runtime" / "lib64"),
            ),
            "PATH": _prepend_env(os.environ.get("PATH", ""), (_CANN_ROOT / "bin",)),
        }
    return [
        Node(
            package="voice_asr_service",
            executable="speech_direction_node",
            name="speech_direction_node",
            parameters=[params],
            additional_env=environment,
            output="screen",
        )
    ]


def generate_launch_description():
    package_config = Path(get_package_share_directory("voice_asr_service")) / "config"
    default_config = str(package_config / "speech_direction.yaml")
    default_profiles = str(package_config / "speech_direction_profiles.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="ascend_310p",
                description="平台配置: ascend_310p(默认) / ubuntu_cuda / custom",
            ),
            DeclareLaunchArgument(
                "profiles_file",
                default_value=default_profiles,
                description="仅包含平台差异的 profile YAML",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="speech_direction ROS 参数 YAML",
            ),
            DeclareLaunchArgument(
                "models_root",
                default_value=str(_workspace_root() / "models"),
                description="模型根目录；YAML 中相对模型路径以此为基准",
            ),
            DeclareLaunchArgument("microphone_sample_rate", default_value=""),
            DeclareLaunchArgument("microphone_channels", default_value=""),
            DeclareLaunchArgument("speech_direction_audio_topic", default_value=""),
            DeclareLaunchArgument("microphone_channel_indices", default_value=""),
            DeclareLaunchArgument("speech_direction_mount_yaw_deg", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
