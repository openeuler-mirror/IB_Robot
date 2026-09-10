"""Verified ZipVoice-Distill adapter for the Ascend 310P1 bucket OMs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from inference_manifest import CompiledDeployment
from inference_service.backends import RuntimeContext
from inference_service.backends.errors import BackendInferenceError as SessionInferenceError
from inference_service.backends.errors import BackendLoadError as SessionLoadError
from inference_service.model_sessions import AscendOmModelSession
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest
from voice_tts_service.errors import BackendInferenceError, BackendLoadError
from voice_tts_service.zipvoice_shared import ChineseTokenizer, PromptProfile, cross_fade_concat, timesteps

_PromptProfile = PromptProfile
_ChineseTokenizer = ChineseTokenizer


class ZipVoiceAscendSession(AscendOmModelSession):
    """Run the complete ZipVoice host-orchestrated pipeline in one model session."""

    allowed_runtime_options = AscendOmModelSession.allowed_runtime_options | {"prompt_profile"}
    execution_contract = "request-iterative"
    orchestration_visibility = "session"

    def __init__(self, device_id: int = 0, *, prompt_profile: str = "default", **kwargs) -> None:
        super().__init__(device_id, **kwargs)
        self._root: Path | None = None
        self._prompt_profile = prompt_profile
        self._config: dict[str, Any] = {}
        self._text_role = ""
        self._flow_role = ""
        self._tokenizer = None
        self._prompt = None
        self._vocos = None
        self._torch = None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendLoadError(f"failed to read ZipVoice 310P config {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise BackendLoadError("ZipVoice 310P config must be a JSON object")
        return value

    def _asset(self, field: str) -> Path:
        value = self._config.get(field)
        if not isinstance(value, str) or not value:
            raise BackendLoadError(f"ZipVoice 310P config field {field!r} must be a relative path")
        path = (self._root / value).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise BackendLoadError(f"ZipVoice asset path escapes bundle root: {value}") from exc
        if not path.is_file() and not path.is_dir():
            raise BackendLoadError(f"ZipVoice asset is unavailable: {path}")
        return path

    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None:
        deployment = context.deployment
        model = context.validated_manifest.manifest.model
        if (model.interface, model.model_type, model.operation) != ("tensor_model", "zipvoice", "synthesize"):
            raise SessionLoadError("ZipVoice session requires tensor_model/zipvoice/synthesize")
        if not isinstance(deployment, CompiledDeployment) or context.backend != "ascend":
            raise SessionLoadError("ZipVoice Ascend session requires a compiled Ascend deployment")
        if context.target_runtime != "acl":
            raise SessionLoadError("ZipVoice Ascend session requires target.runtime='acl'")
        if deployment.target.soc not in {"Ascend310P1", "Ascend310B1"}:
            raise SessionLoadError(
                f"verified ZipVoice OM requires Ascend310P1 or Ascend310B1, got {deployment.target.soc!r}"
            )
        self._root = context.validated_manifest.bundle_root
        self._config = self._load_json(self._root / "assets" / "zipvoice_310p.json")
        text_role = str(self._config.get("text_role", "text_encoder"))
        flow_role = str(self._config.get("flow_role", "flow_decoder_1537"))
        unknown_roles = sorted({text_role, flow_role} - set(deployment.execution))
        if unknown_roles:
            raise SessionLoadError(f"ZipVoice config references undeclared Ascend roles: {unknown_roles}")
        self._text_role = text_role
        self._flow_role = flow_role
        super()._load(context, rollback)
        rollback.defer(self._release_assets)
        self._load_assets()

    def _load_assets(self) -> None:
        profile_name = self._prompt_profile
        profiles = self._config.get("prompt_profiles", {})
        if not isinstance(profiles, dict) or profile_name not in profiles:
            raise BackendLoadError(f"unknown ZipVoice prompt profile {profile_name!r}")
        profile_relative = profiles[profile_name]
        if not isinstance(profile_relative, str) or not profile_relative:
            raise BackendLoadError(f"ZipVoice prompt profile {profile_name!r} must be a relative path")
        profile_path = (self._root / profile_relative).resolve()
        try:
            profile_path.relative_to(self._root)
        except ValueError as exc:
            raise BackendLoadError(f"ZipVoice prompt profile escapes bundle root: {profile_relative}") from exc
        try:
            with np.load(profile_path, allow_pickle=False) as profile:
                tokens = np.asarray(profile["prompt_tokens"], dtype=np.int64)
                features = np.asarray(profile["prompt_features"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise BackendLoadError(f"failed to load ZipVoice prompt profile {profile_path}: {exc}") from exc
        if tokens.ndim != 2 or tokens.shape[0] != 1 or features.ndim != 3 or features.shape[0] != 1:
            raise BackendLoadError("ZipVoice prompt profile tensor shapes are invalid")
        if features.shape[2] != 100 or not np.isfinite(features).all():
            raise BackendLoadError("ZipVoice prompt profile features are invalid")
        self._prompt = PromptProfile(tokens=tokens, features=features)
        self._tokenizer = ChineseTokenizer(self._asset("tokens_path"))
        self._load_vocos()

    def _load_vocos(self) -> None:
        try:
            import torch

            from voice_tts_service.vocos_backend import ZipVoiceVocos
        except (ImportError, OSError) as exc:
            raise BackendLoadError(f"ZipVoice Vocos dependency is unavailable: {exc}") from exc
        vocos = ZipVoiceVocos()
        checkpoint = torch.load(self._asset("vocos_checkpoint_path"), map_location="cpu", weights_only=True)
        incompatible = vocos.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint is missing keys: {incompatible.missing_keys}")
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("feature_extractor.")]
        if unexpected:
            raise BackendLoadError(f"ZipVoice Vocos checkpoint has unexpected keys: {unexpected}")
        self._torch = torch
        self._vocos = vocos.eval()

    @staticmethod
    def _timesteps(num_steps: int, t_shift: float) -> np.ndarray:
        return timesteps(num_steps, t_shift)

    @staticmethod
    def _max_target_tokens(
        flow_frames: int,
        prompt_frames: int,
        prompt_tokens: int,
        speed: float,
        text_capacity: int,
    ) -> int:
        for target in range(text_capacity, 0, -1):
            frames = int(np.ceil(prompt_frames / prompt_tokens * (prompt_tokens + target) / speed))
            if frames <= flow_frames:
                return target
        raise BackendInferenceError("flow bucket cannot fit one target token")

    @staticmethod
    def _cross_fade_concat(waves: list[np.ndarray], sample_rate: int, seconds: float) -> np.ndarray:
        return cross_fade_concat(waves, sample_rate, seconds)

    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        context.check("backend")
        if any(value is None for value in (self._tokenizer, self._prompt, self._torch, self._vocos)):
            raise SessionInferenceError("ZipVoice 310P session assets are not loaded")
        try:
            text = np.asarray(request.inputs["tts.text"], dtype=np.uint8).tobytes().decode("utf-8")
        except (KeyError, UnicodeDecodeError, ValueError) as exc:
            raise SessionInferenceError("ZipVoice request text is not valid UTF-8", code="invalid_text") from exc
        if (
            np.asarray(request.inputs.get("tts.prompt_audio", ())).size
            or np.asarray(request.inputs.get("tts.prompt_text", ()), dtype=np.uint8).size
        ):
            raise SessionInferenceError(
                "the verified Ascend310P1 deployment uses a fixed prompt profile and does not support request prompts",
                code="unsupported_prompt",
            )
        config = self._config
        text_capacity = int(config.get("text_capacity", 256))
        flow_frames = int(config.get("flow_frames", 1537))
        speed = float(config.get("speed", 1.0))
        num_steps = int(config.get("num_steps", 4))
        t_shift = float(config.get("t_shift", 0.5))
        sample_rate = int(config.get("sample_rate", 24000))
        prompt = self._prompt
        token_strings = self._tokenizer.text_to_tokens(text)
        chunk_capacity = self._max_target_tokens(
            flow_frames,
            prompt.frame_count,
            int(prompt.tokens.shape[1]),
            speed,
            text_capacity,
        )
        chunks = self._tokenizer.chunk_tokens(token_strings, chunk_capacity)
        timesteps = self._timesteps(num_steps, t_shift)
        rng = np.random.default_rng(int(config.get("seed", 42)))
        waves = [self._synthesize_chunk(chunk, rng, timesteps) for chunk in chunks]
        if not waves:
            raise SessionInferenceError("ZipVoice frontend produced no token chunks")
        wave = self._cross_fade_concat(waves, sample_rate, float(config.get("cross_fade_sec", 0.1)))
        if wave.size == 0 or not np.isfinite(wave).all():
            raise SessionInferenceError("ZipVoice produced invalid audio")
        return {"tts.audio": np.ascontiguousarray(np.clip(wave, -1.0, 1.0), dtype=np.float32)}

    def _synthesize_chunk(self, tokens: list[str], rng, timesteps: np.ndarray) -> np.ndarray:
        config = self._config
        prompt = self._prompt
        text_capacity = int(config.get("text_capacity", 256))
        flow_frames = int(config.get("flow_frames", 1537))
        ids = self._tokenizer.tokens_to_ids(tokens)
        padded = np.full((1, text_capacity), self._tokenizer.pad_id, dtype=np.int64)
        padded[0, : len(ids)] = ids
        text_inputs = {
            "host.zipvoice.tokens": padded,
            "host.zipvoice.tokens_len": np.asarray(len(ids), dtype=np.int64),
            "host.zipvoice.prompt_tokens": prompt.tokens,
            "host.zipvoice.prompt_features_len": np.asarray(prompt.frame_count, dtype=np.int64),
            "host.zipvoice.speed": np.asarray(float(config.get("speed", 1.0)), dtype=np.float32),
        }
        values: dict[str, object] = dict(text_inputs)
        text_outputs = self._run_role(0, self._text_role, values)
        text_full = np.asarray(text_outputs["host.zipvoice.text_condition"], dtype=np.float32).reshape(1, -1, 100)
        features_len = int(np.asarray(text_outputs["host.zipvoice.features_len"]).reshape(()))
        mask_full = np.asarray(text_outputs["host.zipvoice.padding_mask"], dtype=np.bool_).reshape(1, -1)
        if not prompt.frame_count < features_len <= flow_frames:
            raise SessionInferenceError(
                f"ZipVoice chunk requires {features_len} frames outside supported range "
                f"({prompt.frame_count + 1}..{flow_frames})"
            )
        text_condition = np.zeros((1, flow_frames, 100), dtype=np.float32)
        padding_mask = np.ones((1, flow_frames), dtype=np.bool_)
        copy_frames = min(flow_frames, text_full.shape[1])
        text_condition[:, :copy_frames] = text_full[:, :copy_frames]
        padding_mask[:, :copy_frames] = mask_full[:, :copy_frames]
        text_condition[:, features_len:] = 0.0
        padding_mask[:, features_len:] = True
        speech_condition = np.zeros((1, flow_frames, 100), dtype=np.float32)
        speech_condition[:, : prompt.frame_count] = prompt.features
        x = rng.standard_normal((1, flow_frames, 100), dtype=np.float32)
        for step in range(len(timesteps) - 1):
            values.update(
                {
                    "host.zipvoice.t": np.asarray(timesteps[step], dtype=np.float32),
                    "host.zipvoice.flow_x": x,
                    "host.zipvoice.flow_text_condition": text_condition,
                    "host.zipvoice.speech_condition": speech_condition,
                    "host.zipvoice.flow_padding_mask": padding_mask,
                    "host.zipvoice.guidance_scale": np.asarray(
                        float(config.get("guidance_scale", 3.0)), dtype=np.float32
                    ),
                }
            )
            outputs = self._run_role(1, self._flow_role, values)
            velocity = np.asarray(outputs["host.zipvoice.velocity"], dtype=np.float32).reshape(x.shape)
            if not np.isfinite(velocity).all():
                raise SessionInferenceError("ZipVoice flow decoder produced NaN or Inf")
            x += velocity * np.float32(timesteps[step + 1] - timesteps[step])
        generated = x[:, prompt.frame_count : features_len, :]
        features = np.transpose(generated, (0, 2, 1)) / np.float32(config.get("feature_scale", 0.1))
        with self._torch.inference_mode():
            return self._vocos(self._torch.from_numpy(features)).cpu().numpy()[0]

    def _release_assets(self) -> None:
        self._vocos = None
        self._torch = None
        self._prompt = None
        self._tokenizer = None

    def _close(self) -> None:
        self._release_assets()
        self._root = None
        self._config = {}
        self._text_role = ""
        self._flow_role = ""
        super()._close()
