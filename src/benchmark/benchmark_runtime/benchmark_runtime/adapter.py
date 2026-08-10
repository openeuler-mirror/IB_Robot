"""Pure-Python benchmark adapter protocol.

benchmark plugin contract scope: defines only the :class:`BenchmarkAdapter` ABC. It does NOT import
``rclpy`` and does NOT import any concrete benchmark (LIBERO, robosuite,
MuJoCo, SAPIEN, IsaacLab). No default environment, deterministic adapter or
LIBERO adapter is implemented here.

Each environment node process creates exactly one adapter instance and is the
only caller of ``configure``/``reset``/``step``/``render``/``close``.
Evaluator, dispatcher and reporter must not bypass the environment node to
operate the adapter directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from benchmark_runtime.io_descriptor import BenchmarkIODescriptor
from benchmark_runtime.models import (
    BenchmarkCapabilities,
    BenchmarkEnvironmentConfig,
    BenchmarkTask,
    NativeArtifact,
    ResetRequest,
    ResetResult,
    StepResult,
)
from benchmark_runtime.plan import BenchmarkPlan


class PreNativeValidationError(Exception):
    """Raised by an adapter when action validation fails BEFORE the native
    environment step is entered.

    Pre-native validation: the generic environment node catches this exception and calls
    ``abort_step()`` (release the reserved identity slot without poisoning),
    allowing the caller to retry the same step. Any OTHER exception from
    ``adapter.step()`` is treated as a native step entry failure: the episode
    is poisoned, the step is not retried, and a new reset is required.

    Concrete adapters MUST raise this exception (not a generic ``RuntimeError``)
    when pre-native validation (dimension, range, dtype, gripper sign) fails.
    They MUST NOT raise it after the native ``env.step()`` has been called.
    """


class BenchmarkAdapter(ABC):
    """Stable, ROS-agnostic contract implemented by every concrete benchmark.

    Implementations must not import ``rclpy`` and must not load a policy
    checkpoint. Policy inference stays in ``inference_service``; observation
    and action feature keys/shapes/dtype come from the SSOT YAML Contract,
    not from adapter-local constant tables.
    """

    @property
    @abstractmethod
    def capabilities(self) -> BenchmarkCapabilities:
        """Capabilities advertised by this adapter."""

    @abstractmethod
    def configure(self, config: BenchmarkEnvironmentConfig) -> None:
        """Apply the resolved environment config. Heavy imports and context
        creation (MuJoCo, GPU) happen here, not during plugin discovery."""

    @abstractmethod
    def list_tasks(self) -> list[BenchmarkTask]:
        """Return the tasks available in the current suite."""

    def get_io_descriptor(self) -> BenchmarkIODescriptor:
        """Return the adapter-owned canonical observation/action descriptor.

        Production runtimes require this descriptor before the first reset.
        The default remains explicit so older adapters fail at the startup
        boundary rather than silently bypassing contract validation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not declare a benchmark I/O descriptor")

    def get_plan(self) -> BenchmarkPlan:
        """Return the immutable provider-resolved plan.

        Adapters that do not provide resolved plans fail explicitly without changing the base protocol.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support resolved benchmark plans")

    def on_task_finalized(self, suite: str, task_id: int) -> None:
        """Release task-scoped native resources after a committed task.

        The default is intentionally unsupported. The environment only calls
        this hook after the ledger has validated task finalization.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support task-finalized lifecycle notification")

    def get_native_reset_payload(self) -> dict[str, Any] | None:
        """Return the last committed reset's raw native payload, if supported.

        The generic runtime treats this as an optional adapter-owned seam. A
        concrete adapter may return raw observations and a native environment
        handle; it must not expose these through ROS messages.
        """
        return None

    def get_native_step_payload(self) -> dict[str, Any] | None:
        """Return the last committed step's raw native payload, if supported."""
        return None

    @abstractmethod
    def reset(self, request: ResetRequest) -> ResetResult:
        """Reset the environment for a new episode.

        Implementations must NOT assign ``episode_id`` or ``step_id`` in the
        returned :class:`ResetResult`; those are owned by the generic
        environment runtime (and the ROS ``ResetBenchmark.srv`` response) to
        guarantee exactly-once semantics across all adapters.
        """

    @abstractmethod
    def step(self, action: dict[str, np.ndarray]) -> StepResult:
        """Execute one action and return the next observation/result."""

    @abstractmethod
    def render(self) -> np.ndarray | None:
        """Return a rendered frame, or ``None`` when rendering is unsupported."""

    @abstractmethod
    def collect_native_artifacts(self) -> list[NativeArtifact]:
        """Collect benchmark-native artifacts produced so far."""

    @abstractmethod
    def close(self) -> None:
        """Release all environment resources."""
