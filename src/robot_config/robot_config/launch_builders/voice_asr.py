"""Voice ASR launch builder for robot_config."""

from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.voice_asr")
_VOICE_ASR_REPO_ROOT = Path(__file__).resolve().parents[4]
_VOICE_ASR_REALTIME_MODES = {"continuous", "wake_word"}
_VOICE_ASR_MISSING_ERROR = "voice_asr.enabled=true requires the voice_asr_service package to be installed"


def _load_voice_asr_service():
    try:
        from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("voice_asr_service"):
            raise ModuleNotFoundError(_VOICE_ASR_MISSING_ERROR) from exc
        raise

    return VOICE_ASR_DEFAULTS


def resolve_voice_asr_path(path: str) -> str:
    """Resolve Voice ASR paths relative to the workspace root when needed."""
    resolved = resolve_ros_path(path)
    if not resolved:
        return resolved

    resolved_path = Path(resolved).expanduser()
    if resolved_path.is_absolute():
        return str(resolved_path)
    return str((_VOICE_ASR_REPO_ROOT / resolved_path).resolve())


def voice_asr_mode_requires_streaming(active_mode: str) -> bool:
    return active_mode in _VOICE_ASR_REALTIME_MODES


def _validate_audio_contract(validated, *, sample_rate: int, chunk_size: int, require_frame: bool) -> list[str]:
    """Check a deployment's declared audio contract against the node's audio input."""
    contract = getattr(validated.deployment, "audio_contract", None)
    if contract is None:
        return []
    errors: list[str] = []
    if contract.sample_rate_hz is not None and contract.sample_rate_hz != sample_rate:
        errors.append(
            f"audio contract sample_rate_hz={contract.sample_rate_hz} does not match configured sample_rate={sample_rate}"
        )
    if contract.channels is not None and contract.channels != 1:
        errors.append(f"audio contract requires channels={contract.channels}, expected mono input")
    if contract.sample_dtype is not None and contract.sample_dtype != "float32":
        errors.append(f"audio contract requires sample_dtype={contract.sample_dtype}, expected float32")
    if require_frame:
        if contract.frame_size is not None and contract.frame_size != chunk_size:
            errors.append(
                f"audio contract frame_size={contract.frame_size} does not match configured chunk_size={chunk_size}"
            )
        if contract.execution_mode is not None and contract.execution_mode not in {"streaming", "both"}:
            errors.append(f"audio contract execution_mode={contract.execution_mode} is not streaming-capable")
    return errors


def validate_voice_asr_model_config(
    bundle_path: str,
    deployment: str,
    require_streaming: bool = False,
    sample_rate: int = 16000,
    chunk_size: int = 512,
    vad_bundle_path: str = "",
    vad_deployment: str = "",
) -> list[str]:
    if not bundle_path:
        return ["voice_asr.bundle_path is required when voice_asr.enabled is true"]
    if not deployment:
        return ["voice_asr.deployment is required when voice_asr.enabled is true"]
    if not Path(bundle_path).is_dir():
        return [f"Voice ASR bundle path not found: {bundle_path}"]
    try:
        from inference_manifest import load_inference_manifest

        validated = load_inference_manifest(bundle_path, deployment)
    except Exception as exc:
        return [f"Voice ASR bundle/deployment is invalid: {exc}"]
    if require_streaming and not {"encoder", "decoder"}.issubset(validated.resolved_artifacts):
        return ["Voice ASR realtime streaming requires encoder and decoder deployment artifacts"]
    errors = _validate_audio_contract(validated, sample_rate=sample_rate, chunk_size=chunk_size, require_frame=False)
    if vad_bundle_path and vad_deployment:
        try:
            vad_validated = load_inference_manifest(vad_bundle_path, vad_deployment)
        except Exception as exc:
            return errors + [f"Voice ASR VAD bundle/deployment is invalid: {exc}"]
        errors += _validate_audio_contract(
            vad_validated, sample_rate=sample_rate, chunk_size=chunk_size, require_frame=True
        )
    return errors


def generate_voice_asr_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Generate voice ASR nodes from robot_config YAML."""
    voice_asr_config = robot_config.get("voice_asr", {})
    if not voice_asr_config.get("enabled", False):
        logger.info("Voice ASR disabled, skipping")
        return []

    stale_fields = {"model_path", "tokens_path", "provider", "model_type", "auto_download_model"}
    configured_stale_fields = sorted(stale_fields.intersection(voice_asr_config))
    if configured_stale_fields:
        raise ValueError(
            "voice_asr uses deprecated raw model fields; configure bundle_path/deployment instead: "
            + ", ".join(configured_stale_fields)
        )

    voice_asr_defaults = _load_voice_asr_service()

    active_mode = voice_asr_config.get("active_mode", "manual")
    bundle_path = resolve_voice_asr_path(voice_asr_config.get("bundle_path", voice_asr_defaults["bundle_path"]))
    deployment = str(voice_asr_config.get("deployment", voice_asr_defaults["deployment"]))
    vad_bundle_path = resolve_voice_asr_path(
        voice_asr_config.get("vad_bundle_path", voice_asr_defaults["vad_bundle_path"])
    )
    vad_deployment = str(voice_asr_config.get("vad_deployment", voice_asr_defaults["vad_deployment"]))
    validation_errors = validate_voice_asr_model_config(
        bundle_path=bundle_path,
        deployment=deployment,
        require_streaming=voice_asr_mode_requires_streaming(active_mode),
        sample_rate=int(voice_asr_config.get("sample_rate", voice_asr_defaults["sample_rate"])),
        chunk_size=int(voice_asr_config.get("chunk_size", voice_asr_defaults["chunk_size"])),
        vad_bundle_path=vad_bundle_path,
        vad_deployment=vad_deployment,
    )
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    node_params = {
        "active_mode": active_mode,
        "language": voice_asr_config.get("language", voice_asr_defaults["language"]),
        "bundle_path": bundle_path,
        "deployment": deployment,
        "max_recording_duration": voice_asr_config.get(
            "max_recording_duration", voice_asr_defaults["max_recording_duration"]
        ),
        "vad_sensitivity": voice_asr_config.get("vad_sensitivity", voice_asr_defaults["vad_sensitivity"]),
        "vad_bundle_path": resolve_voice_asr_path(
            voice_asr_config.get("vad_bundle_path", voice_asr_defaults["vad_bundle_path"])
        ),
        "vad_deployment": voice_asr_config.get("vad_deployment", voice_asr_defaults["vad_deployment"]),
        "realtime_pre_roll_seconds": voice_asr_config.get(
            "realtime_pre_roll_seconds", voice_asr_defaults["realtime_pre_roll_seconds"]
        ),
        "publish_partial": voice_asr_config.get("publish_partial", voice_asr_defaults["publish_partial"]),
        "output_topic": voice_asr_config.get("output_topic", voice_asr_defaults["output_topic"]),
        "sample_rate": voice_asr_config.get("sample_rate", voice_asr_defaults["sample_rate"]),
        "chunk_size": voice_asr_config.get("chunk_size", voice_asr_defaults["chunk_size"]),
        "buffer_seconds": voice_asr_config.get("buffer_seconds", voice_asr_defaults["buffer_seconds"]),
        "exit_on_init_failure": voice_asr_config.get(
            "exit_on_init_failure", voice_asr_defaults["exit_on_init_failure"]
        ),
    }
    audio_io = robot_config.get("audio_io", {})
    if not is_audio_io_enabled(audio_io):
        raise ValueError("voice_asr.enabled=true requires audio_io.enabled=true")
    microphone_name = str(audio_io.get("microphone", ""))
    microphone_params = find_microphone_params(robot_config.get("peripherals", []), microphone_name)
    node_params["audio_topic"] = str(audio_io.get("capture_stamped_topic", "/audio/capture_stamped"))
    node_params["audio_channels"] = int(microphone_params.get("channels", 1))
    node_params["audio_input_channel"] = int(
        voice_asr_config.get("audio_input_channel", voice_asr_defaults.get("audio_input_channel", 1))
    )

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
