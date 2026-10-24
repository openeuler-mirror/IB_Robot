"""Package PI0.5 and SmolVLA TCIM outputs into unified HMM deployments."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

from inference_manifest import DeviceLink, PublishedArtifactIndex
from model_utils.inference_manifest_export import (
    RuntimeABI,
    RuntimeTensor,
    artifact_bindings,
    compiled_deployment,
    load_published_artifact_index,
    package_deployment_artifact,
    read_tcim_abi,
    upsert_deployment,
)

_PI05_ROLES = ("prefill", "action_in_proj", "time_mlp", "decode", "action_out_proj")
_SMOLVLA_ROLES = ("prefill", "action")
_CACHE_PATTERN = re.compile(r"past_(?:key|value)_(\d+)")


def write_hmm_deployment(
    bundle_root: str | Path,
    config: Mapping[str, object],
    *,
    vision_hmm: str | Path,
    vision_abi_path: str | Path,
    embedding_path: str | Path,
    role_artifacts: Mapping[str, tuple[str | Path, str | Path]],
    state_projection_path: str | Path | None = None,
    deployment_name: str = "hmm",
    target_soc: str = "lq50",
    target_runtime: str = "tcim",
    vision_layout: str = "NCHW",
) -> Path:
    """Dispatch HMM packaging by policy family and reject unsupported ACT packages."""

    policy_type = str(config.get("type", "")).lower()
    common = {
        "bundle_root": bundle_root,
        "config": config,
        "vision_hmm": vision_hmm,
        "vision_abi_path": vision_abi_path,
        "embedding_path": embedding_path,
        "role_artifacts": role_artifacts,
        "deployment_name": deployment_name,
        "target_soc": target_soc,
        "target_runtime": target_runtime,
        "vision_layout": vision_layout,
    }
    if policy_type == "pi05":
        if state_projection_path is not None:
            raise ValueError("PI0.5 HMM packaging does not use a state projection artifact")
        return write_pi05_hmm_deployment(**common)
    if policy_type == "smolvla":
        if state_projection_path is None:
            raise ValueError("SmolVLA HMM packaging requires state_projection_path")
        return write_smolvla_hmm_deployment(
            **common,
            state_projection_path=state_projection_path,
        )
    raise ValueError(f"HMM deployment packaging supports only PI0.5 and SmolVLA, not policy type {policy_type!r}")


def write_pi05_hmm_deployment(
    bundle_root: str | Path,
    config: Mapping[str, object],
    *,
    vision_hmm: str | Path,
    vision_abi_path: str | Path,
    embedding_path: str | Path,
    role_artifacts: Mapping[str, tuple[str | Path, str | Path]],
    deployment_name: str = "hmm",
    target_soc: str = "lq50",
    target_runtime: str = "tcim",
    vision_layout: str = "NCHW",
) -> Path:
    """Package the PI0.5 vision/prefill/decode projection graph."""

    if str(config.get("type", "")).lower() != "pi05":
        raise ValueError("PI0.5 HMM packaging requires policy type 'pi05'")
    _require_role_set(role_artifacts, _PI05_ROLES, "PI0.5")
    cameras = _visual_features(config)
    chunk_size = _positive_int(config, "chunk_size", "PI0.5")
    max_action_dim = _positive_int(config, "max_action_dim", "PI0.5")
    tokenizer_length = _positive_int(config, "tokenizer_max_length", "PI0.5")

    vision_abi = read_tcim_abi(vision_abi_path)
    role_abis = {role: read_tcim_abi(role_artifacts[role][1]) for role in _PI05_ROLES}
    prefill = role_abis["prefill"]
    action_in = role_abis["action_in_proj"]
    time_mlp = role_abis["time_mlp"]
    decode = role_abis["decode"]
    action_out = role_abis["action_out_proj"]
    _validate_vision_abi(vision_abi, vision_layout, "PI0.5")
    _require_tensor_names(prefill.inputs, ("prefix_embs", "attention_mask", "position_ids"), "PI0.5 prefill inputs")
    _require_tensor_names(
        decode.inputs[:4],
        ("action_embs", "attention_mask", "position_ids", "condition"),
        "PI0.5 decode inputs",
    )
    _require_single_io(action_in, "action_in", "action_in_proj_out", "PI0.5 action_in_proj")
    _require_single_io(time_mlp, "time_emb", "time_mlp_out", "PI0.5 time_mlp")
    _require_single_io(action_out, "action_out", "action_out_proj_out", "PI0.5 action_out_proj")

    expected_action = (1, chunk_size, max_action_dim)
    if action_in.inputs[0].shape != expected_action or action_out.outputs[0].shape != expected_action:
        raise ValueError(f"PI0.5 HMM noise and action ABI must use shape {expected_action}")
    _require_compatible(action_in.outputs[0], decode.inputs[0], "PI0.5 action projection to decode")
    _require_compatible(time_mlp.outputs[0], decode.inputs[3], "PI0.5 time projection to decode")
    _require_compatible(decode.outputs[0], action_out.inputs[0], "PI0.5 decode to action projection")

    prefill_caches = prefill.outputs
    decode_caches = decode.inputs[4:]
    _validate_cache_sequence(tuple(tensor.name for tensor in prefill_caches), "PI0.5 prefill")
    _validate_cache_sequence(tuple(tensor.name for tensor in decode_caches), "PI0.5 decode")
    prefill_outputs = {tensor.name: tensor for tensor in prefill_caches}
    if tuple(prefill_outputs) != tuple(tensor.name for tensor in decode_caches):
        raise ValueError("PI0.5 prefill cache outputs and decode cache inputs must match exactly")
    for target in decode_caches:
        _require_compatible(prefill_outputs[target.name], target, f"PI0.5 cache {target.name}")

    image_tokens = _image_token_count(vision_abi.outputs[0], "PI0.5")
    actual_prefix = len(cameras) * image_tokens + tokenizer_length
    prefix_capacity = prefill.inputs[0].shape[1]
    if prefix_capacity < actual_prefix:
        raise ValueError(
            f"PI0.5 prefill capacity {prefix_capacity} is smaller than configured prefix length {actual_prefix}"
        )
    if prefill.inputs[0].shape[-1] != vision_abi.outputs[0].shape[-1]:
        raise ValueError("PI0.5 vision and prefill hidden sizes do not match")

    root = Path(bundle_root).expanduser().resolve(strict=True)
    published = load_published_artifact_index(root)
    execution, artifacts, bindings, image_semantics = _package_vision_roles(
        root,
        cameras,
        vision_hmm,
        vision_abi,
        backend="hmm",
        deployment_name=deployment_name,
        vision_layout=vision_layout,
        published=published,
    )
    artifacts["embedding"] = (
        package_deployment_artifact(
            root,
            embedding_path,
            backend="hmm",
            deployment_name=deployment_name,
            role="embedding",
            published=published,
        ),
        "pt",
    )
    embedding_abi = RuntimeABI(
        inputs=tuple(
            RuntimeTensor(f"image_{index}", index, vision_abi.outputs[0].dtype, vision_abi.outputs[0].shape)
            for index in range(len(cameras))
        )
        + (
            RuntimeTensor("tokens", len(cameras), "int64", (1, tokenizer_length)),
            RuntimeTensor("language_mask", len(cameras) + 1, "bool", (1, tokenizer_length)),
        ),
        outputs=(
            _synthetic_tensor("prefix_embeddings", 0, prefill.inputs[0]),
            _synthetic_tensor("prefix_attention", 1, prefill.inputs[1]),
            _synthetic_tensor("prefix_positions", 2, prefill.inputs[2]),
            _synthetic_tensor("decode_attention", 3, decode.inputs[1]),
            _synthetic_tensor("decode_positions", 4, decode.inputs[2]),
        ),
    )
    embedding_inputs = {f"image_{index}": semantic for index, semantic in enumerate(image_semantics)}
    embedding_inputs.update(
        {
            "tokens": "observation.language.tokens",
            "language_mask": "observation.language.attention_mask",
        }
    )
    bindings["embedding"] = artifact_bindings(
        embedding_abi,
        input_semantics=embedding_inputs,
        output_semantics={
            "prefix_embeddings": "internal.prefix_embeddings",
            "prefix_attention": "internal.prefix_attention",
            "decode_attention": "internal.decode_attention",
            "prefix_positions": "internal.prefix_positions",
            "decode_positions": "internal.decode_positions",
        },
    )

    cache_semantics = {tensor.name: _cache_semantic(tensor.name) for tensor in prefill_caches}
    bindings["prefill"] = artifact_bindings(
        prefill,
        input_semantics={
            "prefix_embs": "internal.prefix_embeddings",
            "attention_mask": "internal.prefix_attention",
            "position_ids": "internal.prefix_positions",
        },
        output_semantics={tensor.name: cache_semantics[tensor.name] for tensor in prefill.outputs},
    )
    bindings["action_in_proj"] = artifact_bindings(
        action_in,
        input_semantics={"action_in": "noise"},
        output_semantics={"action_in_proj_out": "internal.action_embedding"},
    )
    bindings["time_mlp"] = artifact_bindings(
        time_mlp,
        input_semantics={"time_emb": "time"},
        output_semantics={"time_mlp_out": "internal.time_condition"},
    )
    bindings["decode"] = artifact_bindings(
        decode,
        input_semantics={
            "action_embs": "internal.action_embedding",
            "attention_mask": "internal.decode_attention",
            "position_ids": "internal.decode_positions",
            "condition": "internal.time_condition",
            **cache_semantics,
        },
        output_semantics={decode.outputs[0].name: "internal.suffix_hidden"},
    )
    bindings["action_out_proj"] = artifact_bindings(
        action_out,
        input_semantics={"action_out": "internal.suffix_hidden"},
        output_semantics={"action_out_proj_out": "action"},
    )
    for role in _PI05_ROLES:
        artifacts[role] = _package_hmm_role(root, role, role_artifacts[role][0], deployment_name, published)
    execution.extend(("embedding", *_PI05_ROLES))
    links = tuple(
        DeviceLink(
            semantic=semantic,
            producer="prefill",
            consumer="decode",
            transport="device_pointer",
            owner="producer",
            lifetime="inference",
        )
        for semantic in cache_semantics.values()
    )
    deployment = compiled_deployment(
        root,
        backend="hmm",
        target_soc=target_soc,
        target_runtime=target_runtime,
        artifacts=artifacts,
        published=published,
        execution=execution,
        bindings=bindings,
        device_links=links,
    )
    return upsert_deployment(root, deployment_name, deployment).manifest_path


def write_smolvla_hmm_deployment(
    bundle_root: str | Path,
    config: Mapping[str, object],
    *,
    vision_hmm: str | Path,
    vision_abi_path: str | Path,
    embedding_path: str | Path,
    state_projection_path: str | Path,
    role_artifacts: Mapping[str, tuple[str | Path, str | Path]],
    deployment_name: str = "hmm",
    target_soc: str = "lq50",
    target_runtime: str = "tcim",
    vision_layout: str = "NCHW",
) -> Path:
    """Package the SmolVLA vision/prefill/action graph."""

    if str(config.get("type", "")).lower() != "smolvla":
        raise ValueError("SmolVLA HMM packaging requires policy type 'smolvla'")
    if config.get("add_image_special_tokens", False) is not False:
        raise ValueError("SmolVLA HMM packaging does not support add_image_special_tokens=true")
    _require_role_set(role_artifacts, _SMOLVLA_ROLES, "SmolVLA")
    cameras = _visual_features(config)
    tokenizer_length = _positive_int(config, "tokenizer_max_length", "SmolVLA")
    max_state_dim = _positive_int(config, "max_state_dim", "SmolVLA")
    chunk_size = _positive_int(config, "chunk_size", "SmolVLA")
    max_action_dim = _positive_int(config, "max_action_dim", "SmolVLA")

    vision_abi = read_tcim_abi(vision_abi_path)
    prefill = read_tcim_abi(role_artifacts["prefill"][1])
    action = read_tcim_abi(role_artifacts["action"][1])
    _validate_vision_abi(vision_abi, vision_layout, "SmolVLA")
    _require_tensor_names(prefill.inputs, ("prefix_embs", "attention_mask", "position_ids"), "SmolVLA prefill inputs")
    if len(action.inputs) < 5:
        raise ValueError("SmolVLA action ABI requires noise, time, prefix mask, and cache inputs")
    _require_tensor_names(action.inputs[:3], ("x_t", "timestep", "prefix_pad_masks"), "SmolVLA action inputs")
    if len(action.outputs) != 1 or action.outputs[0].name != "v_t":
        raise ValueError("SmolVLA action ABI must expose exactly one output named 'v_t'")

    expected_action = (1, chunk_size, max_action_dim)
    if action.inputs[0].shape != expected_action or action.outputs[0].shape != expected_action:
        raise ValueError(f"SmolVLA HMM noise and action ABI must use shape {expected_action}")
    image_tokens = _image_token_count(vision_abi.outputs[0], "SmolVLA")
    actual_prefix = len(cameras) * image_tokens + tokenizer_length + 1
    prefix_capacity = prefill.inputs[0].shape[1]
    if prefix_capacity < actual_prefix:
        raise ValueError(
            f"SmolVLA prefill capacity {prefix_capacity} is smaller than configured prefix length {actual_prefix}"
        )
    if prefill.inputs[0].shape[-1] != vision_abi.outputs[0].shape[-1]:
        raise ValueError("SmolVLA vision and prefill hidden sizes do not match")
    if prefill.inputs[1].shape != (1, prefix_capacity, prefix_capacity):
        raise ValueError("SmolVLA prefill attention ABI must be square at the prefix capacity")
    if prefill.inputs[2].shape != (1, prefix_capacity) or action.inputs[2].shape != (1, prefix_capacity):
        raise ValueError("SmolVLA position and prefix mask ABIs must match the prefill capacity")

    prefill_cache_names = tuple(tensor.name for tensor in prefill.outputs)
    action_cache_names = tuple(tensor.name for tensor in action.inputs[3:])
    _validate_cache_sequence(prefill_cache_names, "SmolVLA prefill")
    _validate_cache_sequence(action_cache_names, "SmolVLA action")
    prefill_outputs = {tensor.name: tensor for tensor in prefill.outputs}
    missing = [name for name in action_cache_names if name not in prefill_outputs]
    if missing:
        raise ValueError(f"SmolVLA action cache inputs are missing from prefill outputs: {missing}")
    for consumer in action.inputs[3:]:
        _require_compatible(prefill_outputs[consumer.name], consumer, f"SmolVLA cache {consumer.name}")

    root = Path(bundle_root).expanduser().resolve(strict=True)
    published = load_published_artifact_index(root)
    execution, artifacts, bindings, image_semantics = _package_vision_roles(
        root,
        cameras,
        vision_hmm,
        vision_abi,
        backend="hmm",
        deployment_name=deployment_name,
        vision_layout=vision_layout,
        published=published,
    )
    artifacts.update(
        {
            "embedding": (
                package_deployment_artifact(
                    root,
                    embedding_path,
                    backend="hmm",
                    deployment_name=deployment_name,
                    role="embedding",
                    published=published,
                ),
                "pt",
            ),
            "state_projection": (
                package_deployment_artifact(
                    root,
                    state_projection_path,
                    backend="hmm",
                    deployment_name=deployment_name,
                    role="state_projection",
                    published=published,
                ),
                "pt",
            ),
            "prefill": _package_hmm_role(root, "prefill", role_artifacts["prefill"][0], deployment_name, published),
            "action": _package_hmm_role(root, "action", role_artifacts["action"][0], deployment_name, published),
        }
    )
    embedding_abi = RuntimeABI(
        inputs=tuple(
            RuntimeTensor(f"image_{index}", index, vision_abi.outputs[0].dtype, vision_abi.outputs[0].shape)
            for index in range(len(cameras))
        )
        + (
            RuntimeTensor("tokens", len(cameras), "int64", (1, tokenizer_length)),
            RuntimeTensor("language_mask", len(cameras) + 1, "bool", (1, tokenizer_length)),
            RuntimeTensor("state", len(cameras) + 2, "float32", (1, max_state_dim)),
        ),
        outputs=(
            _synthetic_tensor("prefix_embeddings", 0, prefill.inputs[0]),
            _synthetic_tensor("prefix_pad_masks", 1, action.inputs[2]),
            _synthetic_tensor("attention_mask", 2, prefill.inputs[1]),
            _synthetic_tensor("position_ids", 3, prefill.inputs[2]),
        ),
    )
    embedding_inputs = {f"image_{index}": semantic for index, semantic in enumerate(image_semantics)}
    embedding_inputs.update(
        {
            "tokens": "observation.language.tokens",
            "language_mask": "observation.language.attention_mask",
            "state": "observation.state",
        }
    )
    bindings["embedding"] = artifact_bindings(
        embedding_abi,
        input_semantics=embedding_inputs,
        output_semantics={
            "prefix_embeddings": "internal.prefix_embeddings",
            "prefix_pad_masks": "internal.prefix_pad_masks",
            "attention_mask": "internal.attention_mask",
            "position_ids": "internal.position_ids",
        },
    )
    consumed = {name: _cache_semantic(name) for name in action_cache_names}
    bindings["prefill"] = artifact_bindings(
        prefill,
        input_semantics={
            "prefix_embs": "internal.prefix_embeddings",
            "attention_mask": "internal.attention_mask",
            "position_ids": "internal.position_ids",
        },
        output_semantics={
            tensor.name: consumed.get(tensor.name, f"diagnostic.prefill.cache.{tensor.index}")
            for tensor in prefill.outputs
        },
    )
    bindings["action"] = artifact_bindings(
        action,
        input_semantics={
            "x_t": "noise",
            "timestep": "time",
            "prefix_pad_masks": "internal.prefix_pad_masks",
            **consumed,
        },
        output_semantics={"v_t": "action"},
    )
    execution.extend(("embedding", "prefill", "action"))
    links = tuple(
        DeviceLink(
            semantic=semantic,
            producer="prefill",
            consumer="action",
            transport="device_pointer",
            owner="producer",
            lifetime="inference",
        )
        for semantic in consumed.values()
    )
    deployment = compiled_deployment(
        root,
        backend="hmm",
        target_soc=target_soc,
        target_runtime=target_runtime,
        artifacts=artifacts,
        published=published,
        execution=execution,
        bindings=bindings,
        device_links=links,
    )
    return upsert_deployment(root, deployment_name, deployment).manifest_path


def package_hmm_deployment(
    *,
    bundle_root: str | Path,
    deployment_name: str,
    target_soc: str,
    target_runtime: str,
    spec_path: str | Path,
) -> Path:
    """Package a policy-specific HMM graph from a small path-only JSON spec."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    spec_file = Path(spec_path).expanduser().resolve(strict=True)
    with spec_file.open(encoding="utf-8") as stream:
        spec = json.load(stream)
    if not isinstance(spec, dict):
        raise ValueError(f"HMM packaging spec must be a JSON object: {spec_file}")
    with (root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"LeRobot config must be a JSON object: {root / 'config.json'}")

    vision = _path_pair(spec.get("vision"), "vision", spec_file)
    embedding = _resolve_spec_path(spec_file, _required_string(spec.get("embedding"), "embedding"))
    roles_value = spec.get("roles")
    if not isinstance(roles_value, dict):
        raise ValueError("HMM packaging spec requires a roles object")
    roles = {role: _path_pair(value, f"roles.{role}", spec_file) for role, value in roles_value.items()}
    state_projection_value = spec.get("state_projection")
    state_projection = (
        _resolve_spec_path(spec_file, _required_string(state_projection_value, "state_projection"))
        if state_projection_value is not None
        else None
    )
    return write_hmm_deployment(
        root,
        config,
        vision_hmm=vision[0],
        vision_abi_path=vision[1],
        embedding_path=embedding,
        role_artifacts=roles,
        state_projection_path=state_projection,
        deployment_name=deployment_name,
        target_soc=target_soc,
        target_runtime=target_runtime,
        vision_layout=_required_string(spec.get("vision_layout", "NCHW"), "vision_layout").upper(),
    )


def _package_vision_roles(
    root: Path,
    cameras: tuple[str, ...],
    vision_hmm: str | Path,
    vision_abi: RuntimeABI,
    *,
    backend: str,
    deployment_name: str,
    vision_layout: str,
    published: PublishedArtifactIndex,
) -> tuple[list[str], dict[str, tuple[Path, str]], dict, list[str]]:
    execution: list[str] = []
    artifacts: dict[str, tuple[Path, str]] = {}
    bindings = {}
    image_semantics: list[str] = []
    for camera in cameras:
        role = _vision_role(camera)
        image_semantic = f"internal.image_embedding.{role.removeprefix('vision_')}"
        execution.append(role)
        image_semantics.append(image_semantic)
        artifacts[role] = (
            package_deployment_artifact(
                root,
                vision_hmm,
                backend=backend,
                deployment_name=deployment_name,
                role=role,
                published=published,
                force_copy=True,
            ),
            "hmm",
        )
        bindings[role] = artifact_bindings(
            vision_abi,
            input_semantics={vision_abi.inputs[0].name: camera},
            output_semantics={vision_abi.outputs[0].name: image_semantic},
            image_layouts={camera: vision_layout},
        )
    return execution, artifacts, bindings, image_semantics


def _package_hmm_role(
    root: Path, role: str, source: str | Path, deployment_name: str, published: PublishedArtifactIndex
) -> tuple[Path, str]:
    return (
        package_deployment_artifact(
            root,
            source,
            backend="hmm",
            deployment_name=deployment_name,
            role=role,
            published=published,
        ),
        "hmm",
    )


def _visual_features(config: Mapping[str, object]) -> tuple[str, ...]:
    features = config.get("input_features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot config requires an input_features object")
    cameras = tuple(
        semantic
        for semantic, feature in features.items()
        if isinstance(semantic, str) and isinstance(feature, dict) and str(feature.get("type", "")).upper() == "VISUAL"
    )
    if not cameras:
        raise ValueError("HMM packaging requires at least one VISUAL input feature")
    return cameras


def _positive_int(config: Mapping[str, object], key: str, policy: str) -> int:
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{policy} config requires positive integer {key!r}")
    return value


def _validate_vision_abi(abi: RuntimeABI, layout: str, policy: str) -> None:
    if layout not in {"NCHW", "NHWC"}:
        raise ValueError("vision_layout must be NCHW or NHWC")
    if len(abi.inputs) != 1 or len(abi.outputs) != 1:
        raise ValueError(f"{policy} vision HMM ABI requires exactly one input and one output")
    image = abi.inputs[0]
    channel_axis = 1 if layout == "NCHW" else 3
    if len(image.shape) != 4 or image.shape[channel_axis] != 3:
        raise ValueError(f"{policy} vision HMM ABI is incompatible with {layout} RGB input")


def _image_token_count(tensor: RuntimeTensor, policy: str) -> int:
    if len(tensor.shape) != 3 or tensor.shape[0] != 1:
        raise ValueError(f"{policy} vision output must have shape (1, tokens, hidden)")
    return tensor.shape[1]


def _require_role_set(
    role_artifacts: Mapping[str, tuple[str | Path, str | Path]],
    expected: tuple[str, ...],
    policy: str,
) -> None:
    actual = set(role_artifacts)
    wanted = set(expected)
    if actual != wanted:
        raise ValueError(
            f"{policy} HMM roles must be exactly {list(expected)} "
            f"(missing={sorted(wanted - actual)}, unexpected={sorted(actual - wanted)})"
        )


def _require_tensor_names(tensors: tuple[RuntimeTensor, ...], expected: tuple[str, ...], description: str) -> None:
    names = tuple(tensor.name for tensor in tensors)
    if names != expected:
        raise ValueError(f"{description} must be {list(expected)}, got {list(names)}")


def _require_single_io(abi: RuntimeABI, input_name: str, output_name: str, description: str) -> None:
    if tuple(tensor.name for tensor in abi.inputs) != (input_name,) or tuple(tensor.name for tensor in abi.outputs) != (
        output_name,
    ):
        raise ValueError(f"{description} ABI must be {input_name} -> {output_name}")


def _require_compatible(source: RuntimeTensor, target: RuntimeTensor, description: str) -> None:
    if source.dtype != target.dtype or source.shape != target.shape:
        raise ValueError(f"{description} ABI mismatch: {source.dtype}{source.shape} != {target.dtype}{target.shape}")


def _validate_cache_sequence(names: tuple[str, ...], description: str) -> None:
    if not names or len(names) % 2:
        raise ValueError(f"{description} cache ABI must contain key/value pairs")
    expected = tuple(name for index in range(len(names) // 2) for name in (f"past_key_{index}", f"past_value_{index}"))
    if names != expected:
        raise ValueError(f"{description} cache ABI must be interleaved contiguous key/value pairs")


def _cache_semantic(name: str) -> str:
    match = _CACHE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid SmolVLA cache tensor name {name!r}")
    kind = "key" if name.startswith("past_key_") else "value"
    return f"internal.past_{kind}.{match.group(1)}"


def _synthetic_tensor(name: str, index: int, source: RuntimeTensor) -> RuntimeTensor:
    return RuntimeTensor(name, index, source.dtype, source.shape)


def _vision_role(camera_semantic: str) -> str:
    suffix = camera_semantic.removeprefix("observation.images.")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix).strip("_.-")
    if not normalized:
        raise ValueError(f"Cannot derive a vision role from camera semantic {camera_semantic!r}")
    return f"vision_{normalized}"


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path_pair(value: object, name: str, spec_path: Path) -> tuple[Path, Path]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object with artifact and abi paths")
    artifact = _resolve_spec_path(spec_path, _required_string(value.get("artifact"), f"{name}.artifact"))
    abi = _resolve_spec_path(spec_path, _required_string(value.get("abi"), f"{name}.abi"))
    return artifact, abi


def _resolve_spec_path(spec_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = spec_path.parent / path
    return path.resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--deployment", default="hmm")
    parser.add_argument("--target-soc", default="lq50")
    parser.add_argument("--target-runtime", default="tcim")
    parser.add_argument("--spec", required=True, help="Path-only HMM packaging JSON")
    args = parser.parse_args()
    manifest = package_hmm_deployment(
        bundle_root=args.bundle_root,
        deployment_name=args.deployment,
        target_soc=args.target_soc,
        target_runtime=args.target_runtime,
        spec_path=args.spec,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
