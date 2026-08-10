"""Native report exporter protocol.

benchmark plugin contract scope: defines only the :class:`NativeReportExporter` ABC. It only
consumes the generic runtime models from :mod:`benchmark_runtime.models`.
It must NOT control the environment or policy, must NOT write the canonical
report, and must NOT import ``rclpy``.

benchmark plugin contract does NOT provide a no-op exporter or an actual exporter; those arrive in
later Work Packages. A benchmark that has no native report format will use a
no-op exporter added by the evaluator/report Work Package, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark_runtime.models import EpisodeResult, NativeArtifact, RunManifest, RunSummary, StepEvent


class NativeReportExporter(ABC):
    """Consumes generic runtime events to produce benchmark-native artifacts.

    The exporter only reads the standard runtime events. It writes to
    ``outputs/.../native/<benchmark_type>/`` and registers binary formats
    in the artifact index. Exporter failure must not erase the canonical
    report by default (release profile may configure otherwise).
    """

    @abstractmethod
    def on_run_started(self, manifest: RunManifest) -> None:
        """Called once at the start of a run, before any step."""

    @abstractmethod
    def on_step(self, event: StepEvent) -> None:
        """Called for each completed step event."""

    @abstractmethod
    def on_episode_finished(self, result: EpisodeResult) -> None:
        """Called when an episode reaches a terminal state."""

    def on_task_started(self, *args, **kwargs) -> None:
        """Optional native task lifecycle hook."""
        del args, kwargs

    def on_episode_reset(self, *args, **kwargs) -> None:
        """Optional hook for a committed raw native reset payload."""
        del args, kwargs

    def on_native_step(self, *args, **kwargs) -> None:
        """Optional hook for a committed raw native step payload."""
        del args, kwargs

    def finalize_episode(self, result: EpisodeResult) -> list[NativeArtifact]:
        """Optional idempotent episode finalization hook."""
        del result
        return []

    def finalize_task(self) -> list[NativeArtifact]:
        """Optional idempotent task finalization hook."""
        return []

    @abstractmethod
    def finalize(self, summary: RunSummary) -> list[NativeArtifact]:
        """Called once at the end of a run; returns native artifacts."""
