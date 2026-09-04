"""Provider-independent benchmark I/O descriptors and canonical observations.

This module is deliberately pure Python.  It owns the semantic boundary shared
by benchmark adapters, ``robot.contract`` and model ``config.json`` files; it
never imports ROS, a benchmark provider, or an inference implementation.

Image features use one canonical payload convention: ``uint8`` ``HWC``.  A
LeRobot model manifest normally declares visual shapes as ``CHW``; the manifest
parser normalizes that declaration to the canonical payload convention before
compatibility comparison.  Vector state and action features use ``float32``
with layout ``C``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np


class IODescriptorError(ValueError):
    """Raised when an I/O descriptor or payload is malformed."""


class IOCompatibilityError(IODescriptorError):
    """Raised when adapter, robot, and model descriptors disagree."""


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IODescriptorError(f"{label} must be a non-empty string")
    return value.strip()


def _shape(value: Any, label: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise IODescriptorError(f"{label} must be a sequence of positive integers")
    result = tuple(value)
    if not result or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in result):
        raise IODescriptorError(f"{label} must be a non-empty sequence of positive integers")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise IODescriptorError(f"{label} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class FeatureDescriptor:
    """Semantic description of one canonical observation or action feature."""

    key: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    optional: bool = False

    SUPPORTED_KINDS: ClassVar[frozenset[str]] = frozenset({"image", "state", "action"})
    SUPPORTED_LAYOUTS: ClassVar[Mapping[str, frozenset[str]]] = MappingProxyType(
        {
            "image": frozenset({"CHW", "HWC"}),
            "state": frozenset({"C"}),
            "action": frozenset({"C"}),
        }
    )

    def __post_init__(self) -> None:
        key = _non_empty_string(self.key, "feature key")
        kind = _non_empty_string(self.kind, f"feature {key!r} kind").lower()
        if kind not in self.SUPPORTED_KINDS:
            raise IODescriptorError(
                f"feature {key!r} has unsupported kind {kind!r}; expected one of {sorted(self.SUPPORTED_KINDS)}"
            )
        try:
            dtype = np.dtype(_non_empty_string(self.dtype, f"feature {key!r} dtype")).name
        except TypeError as exc:
            raise IODescriptorError(f"feature {key!r} has unsupported dtype {self.dtype!r}") from exc
        shape = _shape(self.shape, f"feature {key!r} shape")
        layout = _non_empty_string(self.layout, f"feature {key!r} layout").upper()
        if layout not in self.SUPPORTED_LAYOUTS[kind]:
            raise IODescriptorError(
                f"feature {key!r} kind {kind!r} has unsupported layout {layout!r}; "
                f"expected one of {sorted(self.SUPPORTED_LAYOUTS[kind])}"
            )
        if kind == "image" and len(shape) != 3:
            raise IODescriptorError(f"image feature {key!r} must have rank 3, got shape {shape}")
        if kind in {"state", "action"} and len(shape) != 1:
            raise IODescriptorError(f"{kind} feature {key!r} must have rank 1, got shape {shape}")
        optional = _boolean(self.optional, f"feature {key!r} optional")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "optional", optional)


@dataclass(frozen=True, slots=True)
class BenchmarkIODescriptor:
    """Complete canonical observation/action surface for one benchmark."""

    observations: tuple[FeatureDescriptor, ...]
    actions: tuple[FeatureDescriptor, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        actions = tuple(self.actions)
        if not observations:
            raise IODescriptorError("descriptor must declare at least one observation")
        if not actions:
            raise IODescriptorError("descriptor must declare at least one action")
        self._validate_group(observations, "observation", expected_kinds={"image", "state"})
        self._validate_group(actions, "action", expected_kinds={"action"})
        overlap = sorted({item.key for item in observations} & {item.key for item in actions})
        if overlap:
            raise IODescriptorError(f"observation/action keys must be disjoint; duplicates: {overlap}")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "actions", actions)

    @staticmethod
    def _validate_group(
        features: tuple[FeatureDescriptor, ...],
        label: str,
        *,
        expected_kinds: set[str],
    ) -> None:
        seen: set[str] = set()
        for feature in features:
            if not isinstance(feature, FeatureDescriptor):
                raise IODescriptorError(f"{label} descriptors must contain FeatureDescriptor values")
            if feature.kind not in expected_kinds:
                raise IODescriptorError(f"{label} feature {feature.key!r} has invalid kind {feature.kind!r}")
            if feature.key in seen:
                raise IODescriptorError(f"duplicate {label} feature key {feature.key!r}")
            seen.add(feature.key)

    @property
    def observation_map(self) -> Mapping[str, FeatureDescriptor]:
        return MappingProxyType({item.key: item for item in self.observations})

    @property
    def action_map(self) -> Mapping[str, FeatureDescriptor]:
        return MappingProxyType({item.key: item for item in self.actions})


@dataclass(frozen=True, slots=True)
class ObservationSample:
    """One captured canonical observation payload."""

    key: str
    value: np.ndarray
    capture_timestamp_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        key = _non_empty_string(self.key, "observation sample key")
        if not isinstance(self.value, np.ndarray):
            raise IODescriptorError(f"observation sample {key!r} value must be a numpy.ndarray")
        if not isinstance(self.capture_timestamp_ns, int) or isinstance(self.capture_timestamp_ns, bool):
            raise IODescriptorError(f"observation sample {key!r} capture_timestamp_ns must be an int")
        if self.capture_timestamp_ns <= 0:
            raise IODescriptorError(f"observation sample {key!r} capture_timestamp_ns must be positive")
        domain = _non_empty_string(self.clock_domain, f"observation sample {key!r} clock_domain")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "clock_domain", domain)


@dataclass(frozen=True, slots=True)
class ObservationBatch(Mapping[str, np.ndarray]):
    """Atomic observation set belonging to one provider Episode transaction.

    ``episode_transaction_id`` is adapter-local and deliberately not the ROS
    public ``episode_id``.  A reset uses sequence 0 and subsequent steps use
    monotonically increasing sequences while retaining the same transaction.
    """

    samples: tuple[ObservationSample, ...]
    episode_transaction_id: str
    sequence_id: int
    legacy_compat: bool = False

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise IODescriptorError("observation batch must contain at least one sample")
        transaction = _non_empty_string(self.episode_transaction_id, "episode_transaction_id")
        if not isinstance(self.sequence_id, int) or isinstance(self.sequence_id, bool) or self.sequence_id < 0:
            raise IODescriptorError("observation batch sequence_id must be a non-negative int")
        _boolean(self.legacy_compat, "observation batch legacy_compat")
        seen: set[str] = set()
        domains: set[str] = set()
        for sample in samples:
            if not isinstance(sample, ObservationSample):
                raise IODescriptorError("observation batch samples must contain ObservationSample values")
            if sample.key in seen:
                raise IODescriptorError(f"observation batch contains duplicate key {sample.key!r}")
            seen.add(sample.key)
            domains.add(sample.clock_domain)
        if len(domains) != 1:
            raise IODescriptorError(f"observation batch samples must share one clock domain; got {sorted(domains)}")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "episode_transaction_id", transaction)

    def __getitem__(self, key: str) -> np.ndarray:
        for sample in self.samples:
            if sample.key == key:
                return sample.value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (sample.key for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def clock_domain(self) -> str:
        return self.samples[0].clock_domain

    @property
    def capture_timestamp_ns(self) -> int:
        return max(sample.capture_timestamp_ns for sample in self.samples)

    @classmethod
    def from_mapping(
        cls,
        observations: Mapping[str, np.ndarray],
        *,
        episode_transaction_id: str,
        sequence_id: int,
        capture_timestamp_ns: int | None = None,
        clock_domain: str = "system",
        legacy_compat: bool = False,
    ) -> ObservationBatch:
        if not isinstance(observations, Mapping) or not observations:
            raise IODescriptorError("observations must be a non-empty mapping")
        timestamp = time.time_ns() if capture_timestamp_ns is None else capture_timestamp_ns
        return cls(
            samples=tuple(
                ObservationSample(
                    key=key,
                    value=value,
                    capture_timestamp_ns=timestamp,
                    clock_domain=clock_domain,
                )
                for key, value in observations.items()
            ),
            episode_transaction_id=episode_transaction_id,
            sequence_id=sequence_id,
            legacy_compat=legacy_compat,
        )


def coerce_observation_batch(
    value: ObservationBatch | Mapping[str, np.ndarray],
    *,
    episode_transaction_id: str | None,
    sequence_id: int,
) -> ObservationBatch:
    """Bounded compatibility adapter for legacy mapping-returning adapters."""
    if isinstance(value, ObservationBatch):
        if value.sequence_id != sequence_id:
            raise IODescriptorError(
                f"observation batch sequence mismatch: expected {sequence_id}, got {value.sequence_id}"
            )
        if episode_transaction_id is not None and value.episode_transaction_id != episode_transaction_id:
            raise IODescriptorError(
                "observation batch Episode transaction mismatch: "
                f"expected {episode_transaction_id!r}, got {value.episode_transaction_id!r}"
            )
        return value
    if not isinstance(value, Mapping):
        raise IODescriptorError(f"adapter observations must be ObservationBatch or Mapping, got {type(value).__name__}")
    transaction = episode_transaction_id or f"legacy-{time.time_ns()}"
    return ObservationBatch.from_mapping(
        value,
        episode_transaction_id=transaction,
        sequence_id=sequence_id,
        legacy_compat=True,
    )


def validate_observation_batch(batch: ObservationBatch, descriptor: BenchmarkIODescriptor) -> None:
    """Validate one canonical batch against the adapter's declared descriptor."""
    expected = descriptor.observation_map
    actual_keys = set(batch)
    unknown = sorted(actual_keys - set(expected))
    missing = sorted(key for key, feature in expected.items() if not feature.optional and key not in actual_keys)
    if unknown:
        raise IODescriptorError(f"observation batch contains undeclared keys: {unknown}")
    if missing:
        raise IODescriptorError(f"observation batch is missing required keys: {missing}")
    for sample in batch.samples:
        feature = expected[sample.key]
        payload = sample.value
        actual_dtype = np.dtype(payload.dtype).name
        if actual_dtype != feature.dtype:
            raise IODescriptorError(
                f"observation {sample.key!r} dtype mismatch: expected {feature.dtype}, got {actual_dtype}"
            )
        if tuple(payload.shape) != feature.shape:
            raise IODescriptorError(
                f"observation {sample.key!r} shape mismatch: expected {feature.shape}, got {tuple(payload.shape)}"
            )
        if (
            feature.kind in {"state", "action"}
            and np.issubdtype(payload.dtype, np.floating)
            and not bool(np.all(np.isfinite(payload)))
        ):
            raise IODescriptorError(f"observation {sample.key!r} contains NaN or Inf")


def validate_action_payloads(actions: Mapping[str, np.ndarray], descriptor: BenchmarkIODescriptor) -> None:
    """Validate decoded actions before a provider's native step is entered."""
    expected = descriptor.action_map
    actual_keys = set(actions)
    unknown = sorted(actual_keys - set(expected))
    missing = sorted(key for key, feature in expected.items() if not feature.optional and key not in actual_keys)
    if unknown:
        raise IODescriptorError(f"action payload contains undeclared keys: {unknown}")
    if missing:
        raise IODescriptorError(f"action payload is missing required keys: {missing}")
    for key, payload in actions.items():
        if not isinstance(payload, np.ndarray):
            raise IODescriptorError(f"action {key!r} must be a numpy.ndarray")
        feature = expected[key]
        actual_dtype = np.dtype(payload.dtype).name
        if actual_dtype != feature.dtype:
            raise IODescriptorError(f"action {key!r} dtype mismatch: expected {feature.dtype}, got {actual_dtype}")
        if tuple(payload.shape) != feature.shape:
            raise IODescriptorError(
                f"action {key!r} shape mismatch: expected {feature.shape}, got {tuple(payload.shape)}"
            )
        if np.issubdtype(payload.dtype, np.floating) and not bool(np.all(np.isfinite(payload))):
            raise IODescriptorError(f"action {key!r} contains NaN or Inf")


def validate_io_compatibility(
    *,
    adapter: BenchmarkIODescriptor,
    robot: BenchmarkIODescriptor,
    model: BenchmarkIODescriptor,
) -> None:
    """Require exact semantic agreement across adapter, robot, and model."""
    _compare_descriptor_group("observations", adapter.observations, robot.observations, "adapter", "robot")
    _compare_descriptor_group("observations", robot.observations, model.observations, "robot", "model")
    _compare_descriptor_group("actions", adapter.actions, robot.actions, "adapter", "robot")
    _compare_descriptor_group("actions", robot.actions, model.actions, "robot", "model")


def _compare_descriptor_group(
    group: str,
    left: tuple[FeatureDescriptor, ...],
    right: tuple[FeatureDescriptor, ...],
    left_label: str,
    right_label: str,
) -> None:
    left_map = {item.key: item for item in left}
    right_map = {item.key: item for item in right}
    if left_map != right_map:
        raise IOCompatibilityError(
            f"{left_label}/{right_label} {group} descriptor mismatch: expected {left_map}, got {right_map}"
        )


def descriptor_from_robot_contract(robot: Mapping[str, Any]) -> BenchmarkIODescriptor:
    """Parse the canonical ``robot.contract`` mapping."""
    contract = robot.get("contract")
    if not isinstance(contract, Mapping):
        raise IODescriptorError("robot.contract must be a mapping")
    observations_raw = contract.get("observations")
    actions_raw = contract.get("actions")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise IODescriptorError("robot.contract.observations must be a non-empty list")
    if not isinstance(actions_raw, list) or not actions_raw:
        raise IODescriptorError("robot.contract.actions must be a non-empty list")
    observations = tuple(_robot_observation(item, index) for index, item in enumerate(observations_raw))
    actions = tuple(_robot_action(item, index) for index, item in enumerate(actions_raw))
    return BenchmarkIODescriptor(observations=observations, actions=actions)


def _robot_observation(item: Any, index: int) -> FeatureDescriptor:
    if not isinstance(item, Mapping):
        raise IODescriptorError(f"robot.contract.observations[{index}] must be a mapping")
    key = _non_empty_string(item.get("key"), f"robot.contract.observations[{index}].key")
    ros_type = _non_empty_string(item.get("type"), f"robot observation {key!r} type")
    optional = item.get("optional", False)
    if ros_type == "sensor_msgs/msg/Image":
        image = item.get("image")
        if not isinstance(image, Mapping):
            raise IODescriptorError(f"image observation {key!r} must declare image metadata")
        encoding = str(image.get("encoding", "")).lower()
        if encoding != "rgb8":
            raise IODescriptorError(f"image observation {key!r} requires rgb8 encoding, got {encoding!r}")
        resize = _shape(image.get("resize"), f"image observation {key!r} resize")
        if len(resize) != 2:
            raise IODescriptorError(f"image observation {key!r} resize must be [H, W]")
        return FeatureDescriptor(key, "image", "uint8", (resize[0], resize[1], 3), "HWC", optional)
    if ros_type == "ibrobot_msgs/msg/StampedFloat32MultiArray":
        selector = item.get("selector")
        names = selector.get("names") if isinstance(selector, Mapping) else None
        if not isinstance(names, list) or not names:
            raise IODescriptorError(f"state observation {key!r} must declare selector.names")
        return FeatureDescriptor(key, "state", "float32", (len(names),), "C", optional)
    raise IODescriptorError(f"robot observation {key!r} has unsupported modality/type {ros_type!r}")


def _robot_action(item: Any, index: int) -> FeatureDescriptor:
    if not isinstance(item, Mapping):
        raise IODescriptorError(f"robot.contract.actions[{index}] must be a mapping")
    key = _non_empty_string(item.get("key"), f"robot.contract.actions[{index}].key")
    selector = item.get("selector")
    names = selector.get("names") if isinstance(selector, Mapping) else None
    if not isinstance(names, list) or not names:
        raise IODescriptorError(f"action {key!r} must declare selector.names")
    publish = item.get("publish")
    if not isinstance(publish, Mapping):
        raise IODescriptorError(f"action {key!r} must declare publish metadata")
    ros_type = publish.get("type")
    if ros_type != "std_msgs/msg/Float32MultiArray":
        raise IODescriptorError(f"action {key!r} has unsupported type {ros_type!r}")
    return FeatureDescriptor(key, "action", "float32", (len(names),), "C", item.get("optional", False))


def descriptor_from_model_config(model: Mapping[str, Any]) -> BenchmarkIODescriptor:
    """Parse and normalize a LeRobot-style model ``config.json`` mapping."""
    inputs = model.get("input_features")
    outputs = model.get("output_features")
    if not isinstance(inputs, Mapping) or not inputs:
        raise IODescriptorError("model input_features must be a non-empty mapping")
    if not isinstance(outputs, Mapping) or not outputs:
        raise IODescriptorError("model output_features must be a non-empty mapping")
    observations = tuple(_model_feature(key, spec, action=False) for key, spec in inputs.items())
    actions = tuple(_model_feature(key, spec, action=True) for key, spec in outputs.items())
    return BenchmarkIODescriptor(observations=observations, actions=actions)


def _model_feature(key: Any, spec: Any, *, action: bool) -> FeatureDescriptor:
    key_str = _non_empty_string(key, "model feature key")
    if not isinstance(spec, Mapping):
        raise IODescriptorError(f"model feature {key_str!r} must be a mapping")
    type_name = _non_empty_string(spec.get("type"), f"model feature {key_str!r} type").upper()
    optional = spec.get("optional", False)
    raw_shape = _shape(spec.get("shape"), f"model feature {key_str!r} shape")
    explicit_layout = spec.get("layout")
    if action:
        if type_name != "ACTION":
            raise IODescriptorError(f"model output {key_str!r} has unsupported type {type_name!r}")
        return FeatureDescriptor(key_str, "action", spec.get("dtype", "float32"), raw_shape, "C", optional)
    if type_name == "VISUAL":
        source_layout = str(explicit_layout or "CHW").upper()
        if source_layout not in {"CHW", "HWC"}:
            raise IODescriptorError(f"model visual feature {key_str!r} has unsupported layout {source_layout!r}")
        if explicit_layout is None:
            # LeRobot manifests historically omit layout while declaring CHW
            # model shapes. Normalize that legacy representation to the HWC
            # benchmark payload boundary. An explicit layout is authoritative
            # and therefore participates in mismatch detection.
            if len(raw_shape) != 3:
                raise IODescriptorError(f"model visual feature {key_str!r} CHW shape must have rank 3")
            return FeatureDescriptor(
                key_str,
                "image",
                spec.get("dtype", "uint8"),
                (raw_shape[1], raw_shape[2], raw_shape[0]),
                "HWC",
                optional,
            )
        return FeatureDescriptor(key_str, "image", spec.get("dtype", "uint8"), raw_shape, source_layout, optional)
    if type_name == "STATE":
        return FeatureDescriptor(key_str, "state", spec.get("dtype", "float32"), raw_shape, "C", optional)
    raise IODescriptorError(f"model input {key_str!r} has unsupported modality/type {type_name!r}")


def resolve_model_config_path(robot: Mapping[str, Any]) -> Path:
    """Resolve the selected benchmark pipeline's model ``config.json``."""
    control_mode = str(robot.get("default_control_mode", "model_inference"))
    modes = robot.get("control_modes")
    mode = modes.get(control_mode) if isinstance(modes, Mapping) else None
    inference = mode.get("inference") if isinstance(mode, Mapping) else None
    pipelines = inference.get("pipelines") if isinstance(inference, Mapping) else None
    if not isinstance(pipelines, Mapping) or not pipelines:
        raise IODescriptorError("benchmark evaluation requires at least one inference pipeline")
    executor = mode.get("executor", {}) if isinstance(mode, Mapping) else {}
    selected = executor.get("inference_pipeline") if isinstance(executor, Mapping) else None
    if selected is None:
        if len(pipelines) != 1:
            raise IODescriptorError("benchmark executor must select one inference pipeline")
        selected = next(iter(pipelines))
    if not isinstance(selected, str) or not selected:
        raise IODescriptorError("benchmark executor.inference_pipeline must select one pipeline")
    pipeline = pipelines.get(selected)
    if not isinstance(pipeline, Mapping):
        raise IODescriptorError(f"benchmark executor selects unknown pipeline {selected!r}")
    model_path = pipeline.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise IODescriptorError("inference pipeline model_path is required")
    candidate = Path(model_path)
    if not candidate.is_absolute():
        config_path = Path(str(robot.get("_config_path", ""))).resolve()
        workspace_root = next((parent for parent in config_path.parents if (parent / "models").is_dir()), None)
        if workspace_root is None:
            raise IODescriptorError("cannot resolve workspace root for relative model_path")
        candidate = workspace_root / candidate
    config_json = candidate / "config.json"
    if not config_json.is_file():
        raise IODescriptorError(f"model config.json does not exist: {config_json}")
    return config_json


def descriptor_from_robot_model(robot: Mapping[str, Any]) -> BenchmarkIODescriptor:
    """Load and parse the selected model descriptor from a robot mapping."""
    path = resolve_model_config_path(robot)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IODescriptorError(f"failed to read model config.json {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise IODescriptorError(f"model config.json must contain an object: {path}")
    return descriptor_from_model_config(raw)
