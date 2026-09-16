"""Ordered multi-module execution plans derived only from manifest bindings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np

from inference_manifest import ArtifactBindings, DeviceLink, TensorBinding


class ExecutionPlanError(ValueError):
    """Raised when declared multi-module tensor ownership is inconsistent."""


@dataclass(frozen=True)
class ExecutionRolePlan:
    name: str
    position: int
    bindings: ArtifactBindings


@dataclass(frozen=True)
class HostInternalLink:
    semantic: str
    producer: str
    consumers: tuple[str, ...]
    producer_position: int
    last_consumer_position: int
    owner: Literal["execution_frame"] = "execution_frame"
    lifetime: Literal["through_last_consumer"] = "through_last_consumer"


@dataclass(frozen=True)
class DeviceLinkMetadata:
    """Descriptive device topology; it never contains a pointer value."""

    semantic: str
    producer: str
    consumer: str
    producer_binding: Literal["output", "input"]
    transport: Literal["device_pointer"]
    owner: Literal["producer", "consumer"]
    lifetime: Literal["inference"]


@dataclass(frozen=True)
class ExecutionPlan:
    roles: tuple[ExecutionRolePlan, ...]
    host_links: tuple[HostInternalLink, ...]
    device_links: tuple[DeviceLinkMetadata, ...]

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    def role(self, name: str) -> ExecutionRolePlan:
        for role in self.roles:
            if role.name == name:
                return role
        raise ExecutionPlanError(f"unknown execution role {name!r}")

    def device_links_for_consumer(self, role: str) -> tuple[DeviceLinkMetadata, ...]:
        return tuple(link for link in self.device_links if link.consumer == role)


@dataclass(frozen=True)
class ExecutionFrameSnapshot:
    """Immutable host-owned continuation captured between functional stages."""

    next_position: int
    host_tensors: Mapping[str, np.ndarray]


def _compatible_shapes(producer: TensorBinding, consumer: TensorBinding) -> bool:
    if len(producer.shape) != len(consumer.shape):
        return False
    return all(
        produced == consumed or produced == -1 or consumed == -1
        for produced, consumed in zip(producer.shape, consumer.shape, strict=True)
    )


def _validate_link_bindings(
    semantic: str,
    producer_binding: TensorBinding,
    consumer_binding: TensorBinding,
) -> None:
    if producer_binding.dtype != consumer_binding.dtype:
        raise ExecutionPlanError(
            f"internal tensor {semantic!r} dtype differs between producer "
            f"({producer_binding.dtype}) and consumer ({consumer_binding.dtype})"
        )
    if not _compatible_shapes(producer_binding, consumer_binding):
        raise ExecutionPlanError(
            f"internal tensor {semantic!r} shape differs between producer "
            f"{producer_binding.shape} and consumer {consumer_binding.shape}"
        )


def build_execution_plan(
    execution: Sequence[str],
    bindings: Mapping[str, ArtifactBindings],
    device_links: Sequence[DeviceLink] = (),
) -> ExecutionPlan:
    """Build and validate an ordered host/device internal tensor topology."""

    role_names = tuple(execution)
    if not role_names:
        raise ExecutionPlanError("execution plan requires at least one role")
    if len(role_names) != len(set(role_names)):
        raise ExecutionPlanError("execution plan contains duplicate roles")
    if set(bindings) != set(role_names):
        raise ExecutionPlanError(
            "execution binding roles must exactly match ordered roles "
            f"(missing={sorted(set(role_names) - set(bindings))}, "
            f"unexpected={sorted(set(bindings) - set(role_names))})"
        )

    positions = {role: position for position, role in enumerate(role_names)}
    role_plans = tuple(
        ExecutionRolePlan(name=role, position=position, bindings=bindings[role])
        for position, role in enumerate(role_names)
    )

    producers: dict[str, tuple[str, TensorBinding]] = {}
    consumers: dict[str, list[tuple[str, TensorBinding]]] = {}
    for role in role_names:
        for binding in bindings[role].outputs:
            if not binding.semantic.startswith("internal."):
                continue
            if binding.semantic in producers:
                previous_role = producers[binding.semantic][0]
                raise ExecutionPlanError(
                    f"internal tensor {binding.semantic!r} has multiple producers: {previous_role!r} and {role!r}"
                )
            producers[binding.semantic] = (role, binding)
        for binding in bindings[role].inputs:
            if binding.semantic.startswith("internal."):
                consumers.setdefault(binding.semantic, []).append((role, binding))

    device_source_inputs = {(link.producer, link.semantic) for link in device_links if link.producer_binding == "input"}
    device_target_inputs = {(link.consumer, link.semantic) for link in device_links}
    input_sourced_semantics = {semantic for _role, semantic in device_source_inputs}
    ambiguous_semantics = sorted(input_sourced_semantics & set(producers))
    if ambiguous_semantics:
        raise ExecutionPlanError(
            f"input-sourced device links cannot share semantics with internal outputs: {ambiguous_semantics}"
        )

    for semantic, semantic_consumers in consumers.items():
        if semantic not in producers:
            undeclared = [
                role
                for role, _binding in semantic_consumers
                if (role, semantic) not in device_source_inputs and (role, semantic) not in device_target_inputs
            ]
            if undeclared:
                raise ExecutionPlanError(
                    f"internal tensor {semantic!r} has consumers but no producer for roles {undeclared}"
                )
            continue
        producer_role, producer_binding = producers[semantic]
        for consumer_role, consumer_binding in semantic_consumers:
            if positions[producer_role] >= positions[consumer_role]:
                raise ExecutionPlanError(
                    f"internal tensor {semantic!r} producer {producer_role!r} must execute before {consumer_role!r}"
                )
            _validate_link_bindings(semantic, producer_binding, consumer_binding)

    unused_outputs = sorted(set(producers) - set(consumers))
    if unused_outputs:
        raise ExecutionPlanError(f"internal outputs have no consumers: {unused_outputs}")

    device_pairs: set[tuple[str, str, str]] = set()
    device_metadata: list[DeviceLinkMetadata] = []
    for link in device_links:
        if link.transport != "device_pointer":
            raise ExecutionPlanError(f"internal link {link.semantic!r} has unsupported transport {link.transport!r}")
        if link.owner not in {"producer", "consumer"} or link.lifetime != "inference":
            raise ExecutionPlanError(f"internal link {link.semantic!r} has invalid ownership or lifetime metadata")
        if link.producer not in positions or link.consumer not in positions:
            raise ExecutionPlanError(f"internal link {link.semantic!r} references an unknown role")
        pair = (link.semantic, link.producer, link.consumer)
        if pair in device_pairs:
            raise ExecutionPlanError(f"duplicate device link declaration for {link.semantic!r}")
        device_pairs.add(pair)

        source_bindings = (
            bindings[link.producer].inputs if link.producer_binding == "input" else bindings[link.producer].outputs
        )
        source_matches = [binding for binding in source_bindings if binding.semantic == link.semantic]
        if len(source_matches) != 1:
            raise ExecutionPlanError(
                f"device link {link.semantic!r} has no matching producer {link.producer_binding} on {link.producer!r}"
            )
        producer_binding = source_matches[0]
        consumer_bindings = {
            role: binding for role, binding in consumers.get(link.semantic, ()) if role == link.consumer
        }
        try:
            consumer_binding = consumer_bindings[link.consumer]
        except KeyError as exc:
            raise ExecutionPlanError(
                f"device link {link.semantic!r} has no matching input on consumer {link.consumer!r}"
            ) from exc
        if positions[link.producer] >= positions[link.consumer]:
            raise ExecutionPlanError(f"device link {link.semantic!r} producer must execute before its consumer")
        _validate_link_bindings(link.semantic, producer_binding, consumer_binding)
        device_metadata.append(
            DeviceLinkMetadata(
                semantic=link.semantic,
                producer=link.producer,
                consumer=link.consumer,
                producer_binding=link.producer_binding,
                transport=link.transport,
                owner=link.owner,
                lifetime=link.lifetime,
            )
        )

    host_links: list[HostInternalLink] = []
    for semantic, (producer_role, _producer_binding) in producers.items():
        host_consumers = tuple(
            role
            for role, _binding in sorted(consumers[semantic], key=lambda item: positions[item[0]])
            if (semantic, producer_role, role) not in device_pairs
        )
        if not host_consumers:
            continue
        host_links.append(
            HostInternalLink(
                semantic=semantic,
                producer=producer_role,
                consumers=host_consumers,
                producer_position=positions[producer_role],
                last_consumer_position=max(positions[role] for role in host_consumers),
            )
        )

    return ExecutionPlan(
        roles=role_plans,
        host_links=tuple(sorted(host_links, key=lambda link: (link.producer_position, link.semantic))),
        device_links=tuple(
            sorted(
                device_metadata, key=lambda link: (positions[link.producer], positions[link.consumer], link.semantic)
            )
        ),
    )


class ExecutionFrame:
    """Owns host-visible intermediates for exactly one ordered inference."""

    def __init__(self, plan: ExecutionPlan) -> None:
        self._plan = plan
        self._next_position = 0
        self._active_role: str | None = None
        self._host_tensors: dict[str, np.ndarray] = {}
        self._loop_start: int | None = None
        self._loop_end: int | None = None
        self._loop_iterations = 0
        self._completed_loop_iterations = 0

    @property
    def live_host_semantics(self) -> tuple[str, ...]:
        return tuple(sorted(self._host_tensors))

    def configure_loop(self, roles: Sequence[str], iteration_count: int) -> None:
        """Allow one contiguous role region to repeat a bounded number of times."""

        if self._active_role is not None or self._loop_start is not None:
            raise ExecutionPlanError("execution frame loop region is already active or configured")
        if type(iteration_count) is not int or iteration_count < 1:
            raise ExecutionPlanError("execution frame loop iteration_count must be a positive integer")
        loop_roles = tuple(roles)
        if not loop_roles:
            raise ExecutionPlanError("execution frame loop region requires at least one role")
        if self._next_position >= len(self._plan.roles):
            raise ExecutionPlanError("execution frame has no remaining roles for a loop region")
        end = self._next_position + len(loop_roles)
        declared = self._plan.role_names[self._next_position : end]
        if declared != loop_roles:
            raise ExecutionPlanError(
                f"execution frame loop roles must be the next contiguous plan region; expected {declared!r}"
            )
        self._loop_start = self._next_position
        self._loop_end = end - 1
        self._loop_iterations = iteration_count

    def begin_role(self, role: str) -> Mapping[str, np.ndarray]:
        if self._active_role is not None:
            raise ExecutionPlanError(f"execution role {self._active_role!r} has not finished")
        if self._next_position >= len(self._plan.roles):
            raise ExecutionPlanError("execution frame has no remaining roles")
        expected = self._plan.roles[self._next_position].name
        if role != expected:
            raise ExecutionPlanError(f"execution role {role!r} is out of order; expected {expected!r}")

        required = {
            link.semantic: self._host_tensors[link.semantic] for link in self._plan.host_links if role in link.consumers
        }
        self._active_role = role
        return MappingProxyType(required)

    def snapshot(self) -> ExecutionFrameSnapshot:
        if self._active_role is not None or self._loop_start is not None:
            raise ExecutionPlanError("execution frame can only snapshot at a functional stage boundary")
        tensors = MappingProxyType(
            {semantic: np.array(value, copy=True, order="C") for semantic, value in self._host_tensors.items()}
        )
        return ExecutionFrameSnapshot(self._next_position, tensors)

    def restore(self, snapshot: ExecutionFrameSnapshot) -> None:
        if self._active_role is not None or self._next_position != 0 or self._host_tensors:
            raise ExecutionPlanError("execution frame restore requires a fresh frame")
        if snapshot.next_position < 0 or snapshot.next_position > len(self._plan.roles):
            raise ExecutionPlanError("execution frame snapshot position is invalid")
        self._next_position = snapshot.next_position
        self._host_tensors = {
            semantic: np.array(value, copy=True, order="C") for semantic, value in snapshot.host_tensors.items()
        }

    def finish_role(self, role: str, semantic_outputs: Mapping[str, np.ndarray] | None = None) -> None:
        if self._active_role != role:
            raise ExecutionPlanError(f"cannot finish inactive execution role {role!r}")
        outputs = semantic_outputs or {}
        position = self._next_position

        produced_values: dict[str, np.ndarray] = {}
        for link in self._plan.host_links:
            if link.producer != role:
                continue
            try:
                value = outputs[link.semantic]
            except KeyError as exc:
                raise ExecutionPlanError(
                    f"execution role {role!r} did not provide host-visible output {link.semantic!r}"
                ) from exc
            if not isinstance(value, np.ndarray):
                raise ExecutionPlanError(f"host-visible output {link.semantic!r} must be a NumPy array")
            produced_values[link.semantic] = np.array(value, copy=True, order="C")

        self._host_tensors.update(produced_values)

        for link in self._plan.host_links:
            if link.last_consumer_position == position and not self._retain_for_later_loop_iteration(link):
                self._host_tensors.pop(link.semantic, None)

        self._active_role = None
        if self._loop_end == position:
            self._completed_loop_iterations += 1
            if self._completed_loop_iterations < self._loop_iterations:
                self._next_position = self._loop_start
            else:
                self._next_position += 1
        else:
            self._next_position += 1

    def _retain_for_later_loop_iteration(self, link: HostInternalLink) -> bool:
        if self._loop_start is None or self._loop_end is None:
            return False
        if self._completed_loop_iterations >= self._loop_iterations - 1:
            return False
        return link.producer_position < self._loop_start <= link.last_consumer_position <= self._loop_end

    def close(self) -> None:
        self._host_tensors.clear()
        self._active_role = None
