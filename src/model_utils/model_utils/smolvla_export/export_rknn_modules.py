#!/usr/bin/env python3
"""Export SmolVLA as 3 segmented ONNX modules for RKNN conversion.

Mirrors the HMM module split (vision → prefill → action) but outputs standard
ONNX instead of HMONNX, so each module can be converted via rknn-toolkit2.

Modules:
  vision:  pixel_values [1,3,512,512] → image_embeddings [1,N,960]
  prefill: prefix_embs [1,177,960] + attention_mask + position_ids → 32 KV cache tensors
  action:  x_t [1,50,32] + timestep + prefix_pad_masks + 32 KV tensors → v_t [1,50,32]

Also saves token_embedding.pt (CPU-side text embedding) and meta_info.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnxsim import simplify

from model_utils.inference_manifest_export import (
    RuntimeABI,
    RuntimeTensor,
    artifact_bindings,
    compiled_deployment,
    copy_policy_metadata_bundle,
    load_published_artifact_index,
    package_deployment_artifact,
    read_runtime_abi,
    upsert_deployment,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEROBOT_SRC = REPO_ROOT / "libs" / "lerobot" / "src"


def ensure_lerobot_importable(lerobot_src: str | Path | None = None) -> Path:
    lerobot_src_path = Path(lerobot_src) if lerobot_src is not None else DEFAULT_LEROBOT_SRC
    lerobot_src_path = lerobot_src_path.resolve()
    if not lerobot_src_path.exists():
        raise FileNotFoundError(f"LeRobot src path not found: {lerobot_src_path}")
    if str(lerobot_src_path) not in sys.path:
        sys.path.insert(0, str(lerobot_src_path))
    return lerobot_src_path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_hf_cached_model_path(model_id_or_path: str) -> str:
    model_path = Path(model_id_or_path)
    if model_path.exists():
        return str(model_path.resolve())
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.exists():
            return str(snapshot_dir.resolve())
    if snapshots_dir.exists():
        candidates = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
        if candidates:
            return str(candidates[-1].resolve())
    return model_id_or_path


def configure_hf_offline_env():
    os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def patch_vision_embeddings_for_static_export(image_height: int, image_width: int) -> None:
    """Remove boolean indexing from the fixed-size vision export graph."""
    from transformers.models.smolvlm import modeling_smolvlm

    def forward(self, pixel_values, patch_attention_mask=None):
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        nb_patches_h = image_height // self.patch_size
        nb_patches_w = image_width // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side,
            1.0,
            1 / self.num_patches_per_side,
            device=pixel_values.device,
        )
        h_indices = torch.arange(nb_patches_h, device=pixel_values.device, dtype=torch.float32)
        w_indices = torch.arange(nb_patches_w, device=pixel_values.device, dtype=torch.float32)
        fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
        fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        position_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten().unsqueeze(0)
        return embeddings + self.position_embedding(position_ids.to(dtype=torch.long))

    modeling_smolvlm.SmolVLMVisionEmbeddings.forward = forward


@torch.no_grad()
def load_smolvla_policy(model_path: str, device: torch.device, lerobot_src: str | Path | None = None):
    ensure_lerobot_importable(lerobot_src)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    configure_hf_offline_env()
    config = PreTrainedConfig.from_pretrained(model_path)
    if hasattr(config, "vlm_model_name") and isinstance(config.vlm_model_name, str):
        local_vlm_path = Path(model_path) / config.vlm_model_name
        config.vlm_model_name = (
            str(local_vlm_path.resolve())
            if local_vlm_path.exists()
            else resolve_hf_cached_model_path(config.vlm_model_name)
        )
    if hasattr(config, "device"):
        config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(model_path, config=config, strict=False)
    policy.eval()
    policy.to(device)
    return policy


# ── Vision module ──────────────────────────────────────────────────────────


class SmolVLAVisionPart(nn.Module):
    """Vision tower + connector with static patch_attention_mask (NonZero fix)."""

    def __init__(self, vlm_with_expert: nn.Module, image_height: int = 512, image_width: int = 512):
        super().__init__()
        vlm_model = vlm_with_expert.get_vlm_model()
        self.vision_model = vlm_model.vision_model
        self.connector = vlm_model.connector
        patch_size = self.vision_model.config.patch_size
        nb_h = image_height // patch_size
        nb_w = image_width // patch_size
        self.register_buffer(
            "patch_attention_mask",
            torch.ones(1, nb_h, nb_w, dtype=torch.bool),
            persistent=False,
        )
        self.vision_model.eval()
        self.connector.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.contiguous()
        pixel_values = pixel_values * 2.0 - 1.0
        vision_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=self.patch_attention_mask,
        ).last_hidden_state
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states


# ── Prefill module ─────────────────────────────────────────────────────────


def flatten_cache_dict(past_key_values: dict, num_layers: int) -> tuple[torch.Tensor, ...]:
    flat: list[torch.Tensor] = []
    for i in range(num_layers):
        c = past_key_values[i]
        flat.append(c["key_states"])
        flat.append(c["value_states"])
    return tuple(flat)


def build_prefill_input_names() -> list[str]:
    return ["prefix_embs", "attention_mask", "position_ids"]


def build_cache_output_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


class SmolVLAPrefillPart(nn.Module):
    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        self.vlm_with_expert = vlm_with_expert
        self.num_layers = vlm_with_expert.num_vlm_layers

    def forward(self, prefix_embs, attention_mask, position_ids) -> tuple[torch.Tensor, ...]:
        prefix_dtype = self.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=attention_mask.to(device=prefix_embs.device, dtype=torch.bool),
            position_ids=position_ids.to(device=prefix_embs.device, dtype=torch.long),
            past_key_values=None,
            inputs_embeds=[prefix_embs.to(dtype=prefix_dtype), None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return flatten_cache_dict(past_key_values, self.num_layers)


# ── Action module ──────────────────────────────────────────────────────────


def build_cache_input_names(num_layers: int) -> list[str]:
    names: list[str] = []
    for i in range(num_layers):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


class SmolVLAActionPart(nn.Module):
    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.flow_model = flow_model
        self.num_layers = flow_model.vlm_with_expert.num_vlm_layers

    def forward(self, x_t, timestep, prefix_pad_masks, *flat_cache) -> torch.Tensor:
        past_key_values: dict[int, dict[str, torch.Tensor]] = {}
        for i in range(self.num_layers):
            past_key_values[i] = {
                "key_states": flat_cache[i * 2],
                "value_states": flat_cache[i * 2 + 1],
            }
        x_t = x_t.to(dtype=self.flow_model.action_in_proj.weight.dtype)
        timestep = timestep.to(device=x_t.device, dtype=torch.float32)
        prefix_pad_masks = prefix_pad_masks.to(device=x_t.device, dtype=torch.bool)
        return self.flow_model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


# ── ONNX export helpers ────────────────────────────────────────────────────


def export_onnx(model, inputs, onnx_file, input_names, output_names, opset=17):
    print(f"  Exporting ONNX → {onnx_file.name} ...")
    torch.onnx.export(
        model,
        inputs,
        str(onnx_file),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        verbose=False,
        do_constant_folding=True,
    )


def simplify_onnx(onnx_file, simplified_file, input_shapes):
    print(f"  Simplifying → {simplified_file.name} ...")
    onnx_model = onnx.load(str(onnx_file))
    try:
        import onnx_graphsurgeon as gs

        graph = gs.import_onnx(onnx_model)
        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)
        print("    graphsurgeon toposort applied")
    except Exception as exc:
        print(f"    Warning: graphsurgeon toposort failed: {exc}")
    try:
        model_simplified, check = simplify(onnx_model, test_input_shapes=input_shapes)
        if not check:
            print("    Warning: onnxsim check failed, saving anyway.")
    except Exception as exc:
        print(f"    Warning: onnxsim failed ({exc}); keeping toposorted ONNX.")
        model_simplified = onnx_model
    onnx.save(model_simplified, str(simplified_file))


def reject_onnx_ops(onnx_file: Path, unsupported_ops: set[str]) -> None:
    model = onnx.load(str(onnx_file), load_external_data=False)
    found = sorted({node.op_type for node in model.graph.node} & unsupported_ops)
    if found:
        raise RuntimeError(f"Unsupported ONNX operators in {onnx_file}: {', '.join(found)}")


def write_smolvla_rknn_deployment(
    bundle_root: str | Path,
    config: dict,
    *,
    vision_rknn: str | Path,
    vision_abi_path: str | Path,
    prefill_rknn: str | Path,
    prefill_abi_path: str | Path,
    action_rknn: str | Path,
    action_abi_path: str | Path,
    embedding_path: str | Path,
    state_projection_path: str | Path,
    deployment_name: str = "rknn",
    target_soc: str = "rk3588",
    target_runtime: str = "rknn-lite2",
) -> Path:
    """Package compiler-introspected SmolVLA RKNN modules into the unified manifest."""

    if str(config.get("type", "")).lower() != "smolvla":
        raise ValueError("SmolVLA RKNN packaging requires policy type 'smolvla'")
    cameras = [
        semantic
        for semantic, feature in config.get("input_features", {}).items()
        if isinstance(feature, dict) and str(feature.get("type", "")).upper() == "VISUAL"
    ]
    if not cameras:
        raise ValueError("SmolVLA RKNN packaging requires at least one VISUAL input feature")

    vision_abi = read_runtime_abi(vision_abi_path)
    prefill_abi = read_runtime_abi(prefill_abi_path)
    action_abi = read_runtime_abi(action_abi_path)
    _validate_smolvla_rknn_abis(config, cameras, vision_abi, prefill_abi, action_abi)

    root = Path(bundle_root).expanduser().resolve(strict=True)
    published = load_published_artifact_index(root)
    execution: list[str] = []
    artifacts: dict[str, tuple[Path, str]] = {}
    bindings = {}
    image_semantics: list[str] = []
    packaged_vision = package_deployment_artifact(
        root,
        vision_rknn,
        backend="rknn",
        deployment_name=deployment_name,
        role="vision",
        published=published,
        force_copy=True,
    )
    vision_roles: list[str] = []
    for camera in cameras:
        role = _vision_role(camera)
        vision_roles.append(role)
        image_semantic = f"internal.image_embedding.{role.removeprefix('vision_')}"
        execution.append(role)
        image_semantics.append(image_semantic)
        artifacts[role] = (packaged_vision, "rknn")
        bindings[role] = artifact_bindings(
            vision_abi,
            input_semantics={vision_abi.inputs[0].name: camera},
            output_semantics={vision_abi.outputs[0].name: image_semantic},
        )

    embedding_artifact = package_deployment_artifact(
        root,
        embedding_path,
        backend="rknn",
        deployment_name=deployment_name,
        role="embedding",
        published=published,
    )
    state_projection_artifact = package_deployment_artifact(
        root,
        state_projection_path,
        backend="rknn",
        deployment_name=deployment_name,
        role="state_projection",
        published=published,
    )
    prefill_artifact = package_deployment_artifact(
        root,
        prefill_rknn,
        backend="rknn",
        deployment_name=deployment_name,
        role="prefill",
        published=published,
    )
    action_artifact = package_deployment_artifact(
        root,
        action_rknn,
        backend="rknn",
        deployment_name=deployment_name,
        role="action",
        published=published,
    )
    artifacts.update(
        {
            "embedding": (embedding_artifact, "pt"),
            "prefill": (prefill_artifact, "rknn"),
            "action": (action_artifact, "rknn"),
            "state_projection": (state_projection_artifact, "pt"),
        }
    )

    prefix_embeddings, attention_mask, position_ids = prefill_abi.inputs
    noise, time_value, prefix_pad_masks, *action_cache_inputs = action_abi.inputs
    embedding_abi = RuntimeABI(
        inputs=tuple(
            RuntimeTensor(f"image_{index}", index, vision_abi.outputs[0].dtype, vision_abi.outputs[0].shape)
            for index in range(len(cameras))
        )
        + (
            RuntimeTensor("tokens", len(cameras), "int64", (1, int(config["tokenizer_max_length"]))),
            RuntimeTensor(
                "language_mask",
                len(cameras) + 1,
                "bool",
                (1, int(config["tokenizer_max_length"])),
            ),
            RuntimeTensor("state", len(cameras) + 2, "float32", (1, int(config["max_state_dim"]))),
        ),
        outputs=(
            RuntimeTensor("prefix_embeddings", 0, prefix_embeddings.dtype, prefix_embeddings.shape),
            RuntimeTensor("prefix_pad_masks", 1, prefix_pad_masks.dtype, prefix_pad_masks.shape),
            RuntimeTensor("attention_mask", 2, attention_mask.dtype, attention_mask.shape),
            RuntimeTensor("position_ids", 3, position_ids.dtype, position_ids.shape),
        ),
    )
    embedding_input_semantics = {f"image_{index}": semantic for index, semantic in enumerate(image_semantics)}
    embedding_input_semantics.update(
        {
            "tokens": "observation.language.tokens",
            "language_mask": "observation.language.attention_mask",
            "state": "observation.state",
        }
    )
    bindings["embedding"] = artifact_bindings(
        embedding_abi,
        input_semantics=embedding_input_semantics,
        output_semantics={
            "prefix_embeddings": "internal.prefix_embeddings",
            "prefix_pad_masks": "internal.prefix_pad_masks",
            "attention_mask": "internal.attention_mask",
            "position_ids": "internal.position_ids",
        },
    )
    cache_semantics = {_cache_name(tensor.name): _cache_semantic(tensor.name) for tensor in prefill_abi.outputs}
    bindings["prefill"] = artifact_bindings(
        prefill_abi,
        input_semantics={
            prefix_embeddings.name: "internal.prefix_embeddings",
            attention_mask.name: "internal.attention_mask",
            position_ids.name: "internal.position_ids",
        },
        output_semantics=cache_semantics,
    )
    action_input_semantics = {
        noise.name: "noise",
        time_value.name: "time",
        prefix_pad_masks.name: "internal.prefix_pad_masks",
    }
    action_input_semantics.update(
        {_cache_name(tensor.name): _cache_semantic(tensor.name) for tensor in action_cache_inputs}
    )
    bindings["action"] = artifact_bindings(
        action_abi,
        input_semantics=action_input_semantics,
        output_semantics={action_abi.outputs[0].name: "action"},
    )
    execution.extend(("embedding", "prefill", "action"))
    deployment = compiled_deployment(
        root,
        backend="rknn",
        target_soc=target_soc,
        target_runtime=target_runtime,
        artifacts=artifacts,
        published=published,
        execution=execution,
        bindings=bindings,
        artifact_share_groups={role: "vision" for role in vision_roles},
    )
    return upsert_deployment(root, deployment_name, deployment).manifest_path


def _validate_smolvla_rknn_abis(
    config: dict,
    cameras: list[str],
    vision_abi: RuntimeABI,
    prefill_abi: RuntimeABI,
    action_abi: RuntimeABI,
) -> None:
    if len(vision_abi.inputs) != 1 or len(vision_abi.outputs) != 1:
        raise ValueError("SmolVLA vision RKNN ABI requires exactly one input and one output")
    if vision_abi.inputs[0].layout not in {"NCHW", "NHWC"}:
        raise ValueError("SmolVLA vision RKNN ABI must declare NCHW or NHWC input layout")
    if tuple(tensor.name for tensor in prefill_abi.inputs) != ("prefix_embs", "attention_mask", "position_ids"):
        raise ValueError("SmolVLA prefill RKNN ABI inputs must be prefix_embs, attention_mask, position_ids")
    for key in ("tokenizer_max_length", "max_state_dim", "chunk_size", "max_action_dim"):
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"SmolVLA config requires positive integer {key!r}")
    cache_names = tuple(tensor.name for tensor in prefill_abi.outputs)
    if not cache_names or cache_names != tuple(build_cache_output_names(len(cache_names) // 2)):
        raise ValueError("SmolVLA prefill RKNN ABI must expose interleaved past_key_N/past_value_N outputs")
    expected_action_inputs = ("x_t", "timestep", "prefix_pad_masks", *cache_names)
    action_inputs = tuple(tensor.name for tensor in action_abi.inputs)
    if action_inputs != expected_action_inputs:
        if action_inputs[:1] == ("past_kv_tensor",):
            raise ValueError(
                "Legacy flattened-KV SmolVLA action RKNN is unsupported; rebuild the segmented action model"
            )
        raise ValueError(f"SmolVLA action RKNN ABI inputs must be {list(expected_action_inputs)}")
    if len(action_abi.outputs) != 1 or action_abi.outputs[0].name != "v_t":
        raise ValueError("SmolVLA action RKNN ABI must expose exactly one output named 'v_t'")

    image_tokens = vision_abi.outputs[0].shape[1]
    hidden_size = vision_abi.outputs[0].shape[-1]
    tokenizer_length = int(config["tokenizer_max_length"])
    expected_prefix = len(cameras) * image_tokens + tokenizer_length + 1
    if prefill_abi.inputs[0].shape != (1, expected_prefix, hidden_size):
        raise ValueError(
            f"SmolVLA prefill prefix ABI must be (1, {expected_prefix}, {hidden_size}), "
            f"got {prefill_abi.inputs[0].shape}"
        )
    if action_abi.inputs[2].shape != (1, expected_prefix):
        raise ValueError(f"SmolVLA action prefix_pad_masks ABI must be (1, {expected_prefix})")
    expected_action = (1, int(config["chunk_size"]), int(config["max_action_dim"]))
    if action_abi.inputs[0].shape != expected_action or action_abi.outputs[0].shape != expected_action:
        raise ValueError(f"SmolVLA noise and action RKNN ABI must use shape {expected_action}")


def _vision_role(camera_semantic: str) -> str:
    suffix = camera_semantic.removeprefix("observation.images.")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix).strip("_.-")
    if not normalized:
        raise ValueError(f"Cannot derive a vision role from camera semantic {camera_semantic!r}")
    return f"vision_{normalized}"


def _cache_name(name: str) -> str:
    if not re.fullmatch(r"past_(?:key|value)_\d+", name):
        raise ValueError(f"Invalid SmolVLA cache tensor name {name!r}")
    return name


def _cache_semantic(name: str) -> str:
    prefix, kind, index = name.split("_", 2)
    return f"internal.{prefix}_{kind}.{index}"


# ── Main export ────────────────────────────────────────────────────────────


def _bundle_root(args, out_dir: Path) -> Path:
    """Return the bundle root that receives metadata and the manifest.

    Defaults to ``out_dir`` so standalone runs keep working; pass
    ``--bundle_root`` to keep conversion intermediates out of the bundle.
    """

    explicit = getattr(args, "bundle_root", None)
    return Path(explicit).expanduser().resolve() if explicit else out_dir


def export_all(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    export_dtype = torch.float16 if device.type == "cuda" else torch.float32

    patch_vision_embeddings_for_static_export(args.image_height, args.image_width)

    policy = load_smolvla_policy(args.model_path, device, args.lerobot_src)
    flow_model = policy.model
    vlm_with_expert = flow_model.vlm_with_expert
    config = flow_model.config

    image_h, image_w = args.image_height, args.image_width
    chunk_size = config.chunk_size
    max_action_dim = config.max_action_dim
    num_layers = vlm_with_expert.num_vlm_layers
    text_config = vlm_with_expert.config.text_config
    prefix_hidden_size = text_config.hidden_size
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    print(f"Device: {device}, dtype: {export_dtype}")
    print(f"num_layers: {num_layers}, kv_heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"chunk_size: {chunk_size}, max_action_dim: {max_action_dim}")

    out_dir = Path(args.output_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir_value = getattr(args, "calibration_dir", None)
    calibration_dir = Path(calibration_dir_value) if calibration_dir_value else None
    if calibration_dir is not None:
        calibration_dir.mkdir(parents=True, exist_ok=True)

    # ── Vision ──
    print("\n=== Vision module ===")
    vision_model = SmolVLAVisionPart(vlm_with_expert, image_h, image_w)
    vision_model.eval()
    vision_model.to(device=device, dtype=export_dtype)
    dummy_pixel = torch.rand(1, 3, image_h, image_w, dtype=export_dtype, device=device)
    if calibration_dir is not None:
        torch.save((dummy_pixel.detach().cpu(),), calibration_dir / "vision.pt")

    with torch.no_grad():
        image_embeddings = vision_model(dummy_pixel)
    print(f"  image_embeddings shape: {image_embeddings.shape}")
    n_image_tokens = image_embeddings.shape[1]
    camera_count = len(config.image_features)
    derived_prefix_length = camera_count * n_image_tokens + config.tokenizer_max_length + 1
    prefix_length = args.prefix_length if args.prefix_length is not None else derived_prefix_length
    if prefix_length != derived_prefix_length:
        raise ValueError(
            f"--prefix_length={prefix_length} does not match camera/token ABI-derived length {derived_prefix_length}"
        )
    print(f"Image: {image_h}x{image_w}, prefix_length: {prefix_length}")

    vision_onnx = onnx_dir / "smolvla_vision.onnx"
    vision_simp = onnx_dir / "smolvla_vision_simp.onnx"
    export_onnx(
        vision_model,
        dummy_pixel,
        vision_onnx,
        ["pixel_values"],
        ["image_embeddings"],
        opset=args.opset,
    )
    simplify_onnx(vision_onnx, vision_simp, {"pixel_values": [1, 3, image_h, image_w]})
    reject_onnx_ops(vision_simp, {"NonZero"})

    # ── Token embedding (CPU) ──
    print("\n=== Token embedding (CPU) ===")
    token_embedding = vlm_with_expert.get_vlm_model().text_model.get_input_embeddings()
    emb_file = out_dir / "token_embedding.pt"
    torch.save(
        {k: v.detach().cpu() for k, v in token_embedding.state_dict().items()},
        emb_file,
    )
    print(f"  Saved: {emb_file}")

    state_projection_file = out_dir / "state_projection.pt"
    torch.save(
        {k: v.detach().cpu() for k, v in flow_model.state_proj.state_dict().items()},
        state_projection_file,
    )
    print(f"  Saved: {state_projection_file}")

    # ── Prefill ──
    print("\n=== Prefill module ===")
    prefill_model = SmolVLAPrefillPart(vlm_with_expert)
    prefill_model.eval()
    prefill_model.to(device=device, dtype=export_dtype)

    prefix_embs = torch.randn(1, prefix_length, prefix_hidden_size, device=device, dtype=export_dtype)
    attn_mask = torch.ones(1, prefix_length, prefix_length, device=device, dtype=torch.long)
    pos_ids = torch.arange(prefix_length, device=device, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        flat_cache = prefill_model(prefix_embs, attn_mask, pos_ids)
    if calibration_dir is not None:
        torch.save(
            tuple(tensor.detach().cpu() for tensor in (prefix_embs, attn_mask, pos_ids)),
            calibration_dir / "prefill.pt",
        )
    cache_shape = list(flat_cache[0].shape)
    print(f"  KV cache: {num_layers} layers × 2 = {len(flat_cache)} tensors, shape={cache_shape}")

    prefill_names_in = build_prefill_input_names()
    prefill_names_out = build_cache_output_names(num_layers)
    prefill_onnx = onnx_dir / "smolvla_prefill.onnx"
    prefill_simp = onnx_dir / "smolvla_prefill_simp.onnx"
    export_onnx(
        prefill_model,
        (prefix_embs, attn_mask, pos_ids),
        prefill_onnx,
        prefill_names_in,
        prefill_names_out,
        opset=args.opset,
    )
    simplify_onnx(
        prefill_onnx,
        prefill_simp,
        {
            "prefix_embs": list(prefix_embs.shape),
            "attention_mask": list(attn_mask.shape),
            "position_ids": list(pos_ids.shape),
        },
    )

    # ── Action ──
    print("\n=== Action module ===")
    action_model = SmolVLAActionPart(flow_model)
    action_model.eval()
    action_dtype = torch.float32
    action_model.to(device=device, dtype=action_dtype)

    x_t = torch.randn(1, chunk_size, max_action_dim, device=device, dtype=action_dtype)
    timestep = torch.ones(1, device=device, dtype=torch.float32)
    prefix_pad_masks = torch.ones(1, prefix_length, device=device, dtype=torch.bool)
    action_cache = tuple(tensor.to(dtype=action_dtype) for tensor in flat_cache)
    action_inputs = (x_t, timestep, prefix_pad_masks, *action_cache)
    if calibration_dir is not None:
        torch.save(
            tuple(tensor.detach().cpu() for tensor in action_inputs),
            calibration_dir / "action.pt",
        )

    action_names_in = ["x_t", "timestep", "prefix_pad_masks", *build_cache_input_names(num_layers)]
    action_onnx = onnx_dir / "smolvla_action.onnx"
    action_simp = onnx_dir / "smolvla_action_simp.onnx"
    export_onnx(
        action_model,
        action_inputs,
        action_onnx,
        action_names_in,
        ["v_t"],
        opset=args.opset,
    )
    action_shapes = {
        "x_t": list(x_t.shape),
        "timestep": list(timestep.shape),
        "prefix_pad_masks": list(prefix_pad_masks.shape),
    }
    for name, tensor in zip(action_names_in[3:], flat_cache, strict=True):
        action_shapes[name] = list(tensor.shape)
    simplify_onnx(action_onnx, action_simp, action_shapes)

    # ── Export metadata (not a runtime manifest) ──
    meta = {
        "schema_version": 1,
        "policy_type": "smolvla",
        "kind": "smolvla-rknn-export-metadata",
        "model_path": args.model_path,
        "image_height": image_h,
        "image_width": image_w,
        "prefix_length": prefix_length,
        "n_image_tokens": n_image_tokens,
        "prefix_hidden_size": prefix_hidden_size,
        "num_layers": num_layers,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "chunk_size": chunk_size,
        "max_action_dim": max_action_dim,
        "cache_shape_per_tensor": cache_shape,
        "modules": {
            "vision": {
                "onnx": "onnx/smolvla_vision_simp.onnx",
                "input_names": ["pixel_values"],
                "output_names": ["image_embeddings"],
                "input_shapes": {"pixel_values": [1, 3, image_h, image_w]},
            },
            "prefill": {
                "onnx": "onnx/smolvla_prefill_simp.onnx",
                "input_names": prefill_names_in,
                "output_names": prefill_names_out,
                "input_shapes": {
                    "prefix_embs": list(prefix_embs.shape),
                    "attention_mask": list(attn_mask.shape),
                    "position_ids": list(pos_ids.shape),
                },
            },
            "action": {
                "onnx": "onnx/smolvla_action_simp.onnx",
                "input_names": action_names_in,
                "output_names": ["v_t"],
                "input_shapes": action_shapes,
            },
        },
        "embedding": "token_embedding.pt",
        "state_projection": "state_projection.pt",
        "notes": [
            "3-module RKNN split mirroring HMM architecture.",
            "vision: pixel_values → image_embeddings (NPU)",
            "prefill: prefix_embs → KV cache (NPU)",
            "action: KV cache + x_t → v_t (NPU)",
            "token_embedding runs on CPU (text token lookup).",
            "Denoise loop (num_steps=10) runs on host CPU, calling action NPU each step.",
        ],
    }
    meta_file = out_dir / "smolvla_rknn_export.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n=== Meta info saved: {meta_file} ===")

    copied = copy_policy_metadata_bundle(args.model_path, _bundle_root(args, out_dir))
    print(f"  Copied {len(copied)} required LeRobot semantic files")

    print(f"\nDone! Output: {out_dir}")
    print(f"  ONNX files: {onnx_dir}")
    print(f"  Token embedding: {emb_file}")
    print(f"  State projection: {state_projection_file}")
    print(f"  Config: {meta_file}")
    print("\nNext: convert each ONNX to RKNN:")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_vision_simp.onnx --output {onnx_dir}/smolvla_vision.rknn --mode float16")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_prefill_simp.onnx --output {onnx_dir}/smolvla_prefill.rknn --mode float16")
    print("  .venv-rknn/bin/python .agents/skills/rknn-convert/convert_to_rknn.py \\")
    print(f"    --onnx {onnx_dir}/smolvla_action_simp.onnx --output {onnx_dir}/smolvla_action.rknn --mode float16")
    print("Then package the compiler-emitted *.abi.json files with package-compiled-deployment.")


def _package_existing(args) -> None:
    out_dir = Path(args.output_dir).expanduser().resolve()
    bundle_root = _bundle_root(args, out_dir)
    copy_policy_metadata_bundle(args.model_path, bundle_root)
    with (bundle_root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    onnx_dir = out_dir / "onnx"
    embedding = out_dir / "token_embedding.pt"
    state_projection = out_dir / "state_projection.pt"
    if not embedding.is_file():
        embedding = bundle_root / "token_embedding.pt"
    if not state_projection.is_file():
        state_projection = bundle_root / "state_projection.pt"
    manifest_path = write_smolvla_rknn_deployment(
        bundle_root,
        config,
        vision_rknn=args.vision_rknn or onnx_dir / "smolvla_vision.rknn",
        vision_abi_path=args.vision_abi or onnx_dir / "smolvla_vision.rknn.abi.json",
        prefill_rknn=args.prefill_rknn or onnx_dir / "smolvla_prefill.rknn",
        prefill_abi_path=args.prefill_abi or onnx_dir / "smolvla_prefill.rknn.abi.json",
        action_rknn=args.action_rknn or onnx_dir / "smolvla_action.rknn",
        action_abi_path=args.action_abi or onnx_dir / "smolvla_action.rknn.abi.json",
        embedding_path=embedding,
        state_projection_path=state_projection,
        deployment_name=args.deployment,
        target_soc=args.target_soc,
        target_runtime=args.target_runtime,
    )
    print(f"Unified manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SmolVLA as 3 segmented ONNX modules for RKNN")
    parser.add_argument("--model_path", type=str, required=True, help="SmolVLA policy path")
    parser.add_argument("--lerobot_src", type=str, default=str(DEFAULT_LEROBOT_SRC))
    parser.add_argument("--output_dir", type=str, default="models/_work/smolvla/rknn", help="Work output directory")
    parser.add_argument(
        "--bundle_root",
        type=str,
        default=None,
        help="Bundle root receiving LeRobot metadata and the unified manifest (default: same as --output_dir)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--prefix_length", type=int, default=None, help="Prefix sequence length (default: derive)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--package_only", action="store_true", help="Package existing RKNN compiler outputs only")
    parser.add_argument("--deployment", default="rknn", help="Unified manifest deployment name")
    parser.add_argument("--target_soc", default="rk3588")
    parser.add_argument("--target_runtime", default="rknn-lite2")
    parser.add_argument("--vision_rknn", default=None)
    parser.add_argument("--vision_abi", default=None)
    parser.add_argument("--prefill_rknn", default=None)
    parser.add_argument("--prefill_abi", default=None)
    parser.add_argument("--action_rknn", default=None)
    parser.add_argument("--action_abi", default=None)
    parsed = parser.parse_args()
    if parsed.package_only:
        _package_existing(parsed)
    else:
        export_all(parsed)
