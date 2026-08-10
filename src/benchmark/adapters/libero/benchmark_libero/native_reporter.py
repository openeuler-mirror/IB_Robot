"""LIBERO-native artifact reporter used by the generic benchmark runtime.

The reporter deliberately keeps the native boundary in this adapter package:
LIBERO's ``VideoWriter`` and ``get_sim_state`` are only imported/called from
here.  The zero-argument plugin factory remains cheap; run/task configuration
arrives through :meth:`on_run_started` and the explicit native lifecycle
methods are available to the environment integration without changing the
generic runtime protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_runtime.models import EpisodeResult, NativeArtifact, RunManifest, RunSummary, StepEvent
from benchmark_runtime.native_report import NativeReportExporter

_MISSING = object()
_DEFAULT_FPS = 30
_DEFAULT_CAMERA_NAME = "agentview_image"


@dataclass
class _EpisodeState:
    episode_index: int
    init_state_id: int | None = None
    states: list[Any] = field(default_factory=list)
    result: EpisodeResult | None = None
    video_failed: bool = False
    state_failed: bool = False
    artifact_failures: list[dict[str, str]] = field(default_factory=list)
    terminal_frame_appended: bool = False
    last_observation: Mapping[str, Any] | None = None


@dataclass
class _TaskState:
    suite: str
    task_id: int
    name: str
    prompt: str
    planned_episodes: int | None
    video_enabled: bool
    fps: int
    camera_name: str
    single_video: bool
    save_sim_states: bool
    task_dir: Path
    video_dir: Path
    writer: Any = None
    writer_saved: bool = False
    episodes: dict[int, _EpisodeState] = field(default_factory=dict)
    artifacts: list[NativeArtifact] = field(default_factory=list)
    artifact_failures: list[dict[str, str]] = field(default_factory=list)
    finalized: bool = False


class LiberoNativeReporter(NativeReportExporter):
    """Own LIBERO's task-scoped videos, simulator states and native reports.

    ``NativeReportExporter`` supplies the generic event compatibility methods.
    A production environment should use the richer methods in this class so
    that raw observations and the native simulator object remain inside the
    adapter process::

        on_run_started(manifest)
        on_task_started(...)
        on_episode_reset(raw_obs, native_env=env, ...)
        on_native_step(raw_obs, native_env=env, ...)
        finalize_episode(result)
        finalize_task()
        finalize(summary)

    All artifact errors are retained in task state and do not mutate the
    supplied native ``EpisodeResult``.
    """

    def __init__(
        self,
        *,
        video_writer_factory: Callable[..., Any] | None = None,
        torch_module: Any = _MISSING,
    ) -> None:
        self._video_writer_factory = video_writer_factory
        self._torch_module = torch_module
        self._run_manifest: RunManifest | None = None
        self._run_root: Path | None = None
        self._native_root: Path | None = None
        self._config: dict[str, Any] = {}
        self._active_task: _TaskState | None = None
        self._tasks: dict[tuple[str, int], _TaskState] = {}
        self._artifacts: list[NativeArtifact] = []
        self._run_failures: list[dict[str, str]] = []

    # ------------------------------------------------------------------ #
    # Generic exporter interface
    # ------------------------------------------------------------------ #

    def on_run_started(self, manifest: RunManifest) -> None:
        if manifest is None:
            return
        if not isinstance(manifest, RunManifest):
            raise TypeError("manifest must be a RunManifest")
        if self._run_manifest is not None:
            raise RuntimeError("LIBERO native reporter run is already started")
        self._run_manifest = manifest
        self._run_root = Path(manifest.output_ref)
        self._native_root = self._run_root / "native" / "libero"
        self._config = self._resolve_config(manifest)
        self._native_root.mkdir(parents=True, exist_ok=True)

    def on_step(self, event: StepEvent) -> None:
        """Compatibility path for generic events.

        The environment integration should call :meth:`on_native_step` with
        raw OffScreenRenderEnv observations.  This fallback uses the generic
        observation mapping only when it contains the configured camera key;
        it intentionally cannot synthesize simulator states.
        """
        if event is None or self._active_task is None:
            return
        if (event.suite, event.task_id) != (self._active_task.suite, self._active_task.task_id):
            return
        observations = event.result.observations
        if self._active_task.camera_name not in observations:
            return
        self.on_native_step(
            observations,
            result=event.result,
            episode_index=event.episode_id,
            sim_state=_MISSING,
        )

    def on_episode_finished(self, result: EpisodeResult) -> None:
        if result is None:
            return
        self.finalize_episode(result)

    def finalize(self, summary: RunSummary) -> list[NativeArtifact]:
        del summary
        if self._active_task is not None and not self._active_task.finalized:
            self.finalize_task()
        self._artifacts = [artifact for task in self._tasks.values() for artifact in task.artifacts]
        return list(self._artifacts)

    # ------------------------------------------------------------------ #
    # Native lifecycle
    # ------------------------------------------------------------------ #

    def on_task_started(
        self,
        task: Any = None,
        *,
        suite: str | None = None,
        task_id: int | None = None,
        task_name: str | None = None,
        prompt: str | None = None,
        planned_episodes: int | None = None,
        video_enabled: bool | None = None,
        fps: int | None = None,
        camera_name: str | None = None,
        single_video: bool | None = None,
        save_sim_states: bool | None = None,
    ) -> None:
        """Open exactly one native writer scope for a task."""
        if self._native_root is None:
            raise RuntimeError("on_run_started must be called before opening a task")
        if self._active_task is not None:
            raise RuntimeError("previous LIBERO task must be finalized before switching tasks")

        values = _task_values(task, suite=suite, task_id=task_id, task_name=task_name, prompt=prompt)
        task_suite = str(values["suite"])
        task_id_value = int(values["task_id"])
        task_dir = self._native_root / "tasks" / f"task_{task_id_value:03d}"
        video_dir = task_dir / "videos"
        task_config = self._config
        enabled = bool(task_config.get("video_enabled", False) if video_enabled is None else video_enabled)
        task = _TaskState(
            suite=task_suite,
            task_id=task_id_value,
            name=str(values["task_name"]),
            prompt=str(values["prompt"]),
            planned_episodes=planned_episodes,
            video_enabled=enabled,
            fps=int(task_config.get("fps", _DEFAULT_FPS) if fps is None else fps),
            camera_name=str(
                task_config.get("camera_name", _DEFAULT_CAMERA_NAME) if camera_name is None else camera_name
            ),
            single_video=bool(task_config.get("single_video", False) if single_video is None else single_video),
            save_sim_states=bool(
                task_config.get("save_sim_states", False) if save_sim_states is None else save_sim_states
            ),
            task_dir=task_dir,
            video_dir=video_dir,
        )
        task_dir.mkdir(parents=True, exist_ok=True)
        if task.video_enabled:
            try:
                task.writer = self._make_video_writer(task)
            except Exception as exc:  # artifact failure; execution may continue
                self._record_failure(task, "writer_create", exc)
                task.video_enabled = False
        self._active_task = task
        self._tasks[(task.suite, task.task_id)] = task

    # Alias kept explicit for callers that use the plan vocabulary.
    open_task = on_task_started

    def on_episode_reset(
        self,
        observations: Mapping[str, Any],
        *,
        episode_index: int,
        init_state_id: int | None = None,
        sim_state: Any = _MISSING,
        native_env: Any = None,
    ) -> None:
        """Record the post-reset observation and initial native simulator state."""
        task = self._require_task()
        if episode_index in task.episodes:
            raise RuntimeError(f"episode {episode_index} was already reset")
        episode = _EpisodeState(episode_index=episode_index, init_state_id=init_state_id)
        task.episodes[episode_index] = episode
        if task.episodes and len(task.episodes) > 1 and task.writer is not None and not task.single_video:
            self._safe_writer_reset(task, episode)
        episode.last_observation = observations
        self._append_observation(task, episode, observations, done=False)
        self._capture_state(task, episode, sim_state=sim_state, native_env=native_env)

    def reset_episode(self, *args: Any, **kwargs: Any) -> None:
        """Readable alias for :meth:`on_episode_reset`."""
        self.on_episode_reset(*args, **kwargs)

    def on_native_step(
        self,
        observations: Mapping[str, Any],
        *,
        episode_index: int,
        result: Any = None,
        done: bool | None = None,
        sim_state: Any = _MISSING,
        native_env: Any = None,
    ) -> None:
        """Record one committed native step, including its post-step state."""
        task = self._require_task()
        episode = task.episodes.get(episode_index)
        if episode is None:
            raise RuntimeError(f"episode {episode_index} must be reset before stepping")
        if episode.result is not None:
            raise RuntimeError(f"episode {episode_index} is already finalized")
        episode.last_observation = observations
        terminal = bool(done) if done is not None else _result_done(result)
        if terminal:
            if not episode.terminal_frame_appended:
                self._append_observation(task, episode, observations, done=True)
                episode.terminal_frame_appended = True
        else:
            self._append_observation(task, episode, observations, done=False)
        self._capture_state(task, episode, sim_state=sim_state, native_env=native_env)

    record_step = on_native_step

    def finalize_episode(self, result: EpisodeResult) -> list[NativeArtifact]:
        """Persist the simulator-state prefix and retain the immutable outcome."""
        task = self._require_task()
        if (result.suite, result.task_id) != (task.suite, task.task_id):
            raise ValueError("episode result does not match active task")
        episode = task.episodes.get(result.episode_index)
        if episode is None:
            raise RuntimeError(f"episode {result.episode_index} was not reset")
        if episode.result is not None:
            return []
        if task.writer is not None and not episode.terminal_frame_appended and episode.last_observation is not None:
            self._append_observation(task, episode, episode.last_observation, done=True)
            episode.terminal_frame_appended = True
        episode.result = result
        if task.save_sim_states:
            self._save_episode_states(task, episode, partial=bool(result.error_category))
        return [artifact for artifact in task.artifacts if artifact.episode_id == result.episode_index]

    finish_episode = finalize_episode

    def finalize_task(self) -> list[NativeArtifact]:
        """Save the task writer once, normalize filenames, and write native reports."""
        task = self._require_task()
        if task.finalized:
            return list(task.artifacts)
        if task.writer is not None and not task.writer_saved:
            try:
                task.writer.save()
                task.writer_saved = True
                self._normalize_video_artifacts(task)
            except Exception as exc:
                self._record_failure(task, "video_finalize", exc)
        self._write_task_stats(task)
        self._write_task_manifest(task)
        task.finalized = True
        self._active_task = None
        return list(task.artifacts)

    finish_task = finalize_task

    @staticmethod
    def _video_frame_count(task: _TaskState, episode_index: int | None = None) -> int | None:
        buffer = getattr(task.writer, "image_buffer", None)
        if not isinstance(buffer, Mapping):
            return None
        if episode_index is None:
            return sum(len(frames) for frames in buffer.values())
        frames = buffer.get(episode_index)
        return len(frames) if isinstance(frames, list | tuple) else 0

    def artifact_status(
        self, *, scope: str, task_id: int | None = None, episode_index: int | None = None
    ) -> dict[str, Any]:
        """Return machine-readable artifact and failure status for one scope."""
        tasks = [task for task in self._tasks.values() if task_id is None or task.task_id == task_id]
        artifacts = []
        failures: list[dict[str, Any]] = []
        for task in tasks:
            for artifact in task.artifacts:
                if scope == "episode" and artifact.episode_id != episode_index:
                    continue
                if scope == "task" and artifact.scope not in {"episode", "task"}:
                    continue
                artifacts.append(
                    {
                        "ref": artifact.path,
                        "kind": artifact.kind,
                        "scope": artifact.scope,
                        "required": True,
                        "status": "available",
                        "metadata": dict(artifact.metadata),
                    }
                )
            if scope == "episode" and episode_index in task.episodes:
                expected_paths: list[tuple[Path, str, str, dict[str, Any]]] = []
                if task.video_enabled:
                    expected_paths.append(
                        (
                            task.video_dir / ("video.mp4" if task.single_video else f"episode_{episode_index:03d}.mp4"),
                            "video",
                            "episode" if not task.single_video else "task",
                            {
                                "fps": task.fps,
                                "camera_name": task.camera_name,
                                "single_video": task.single_video,
                                "frame_count": self._video_frame_count(task, episode_index),
                            },
                        )
                    )
                expected_paths.extend(
                    [
                        (task.task_dir / "evaluation.stats", "native_report", "task", {}),
                        (task.task_dir / "task_manifest.json", "native_report", "task", {}),
                    ]
                )
                for expected, kind, artifact_scope, metadata in expected_paths:
                    expected_ref = self._artifact_path(expected)
                    if all(item["ref"] != expected_ref for item in artifacts):
                        artifacts.append(
                            {
                                "ref": expected_ref,
                                "kind": kind,
                                "scope": artifact_scope,
                                "required": True,
                                "status": "pending",
                                "metadata": metadata,
                            }
                        )
            for failure in task.artifact_failures:
                failure_episode = failure.get("episode_index")
                if scope == "episode" and str(episode_index) != str(failure_episode):
                    continue
                failures.append(dict(failure))
        return {"artifacts": artifacts, "failures": failures}

    def capture_sim_state(self, native_env: Any) -> Any:
        """Call the native environment state API without importing MuJoCo here."""
        getter = getattr(native_env, "get_sim_state", None)
        if not callable(getter):
            raise AttributeError("native_env must provide get_sim_state()")
        return getter()

    # ------------------------------------------------------------------ #
    # Configuration and artifact helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_config(manifest: RunManifest) -> dict[str, Any]:
        metadata = dict(manifest.metadata)
        native = metadata.get("native_reporter", metadata.get("artifacts", metadata))
        if not isinstance(native, Mapping):
            native = {}
        video = native.get("video", native)
        if not isinstance(video, Mapping):
            video = {}
        result = dict(native)
        result.update(
            {
                "video_enabled": video.get("enabled", native.get("video_enabled", False)),
                "fps": video.get("fps", native.get("fps", _DEFAULT_FPS)),
                "camera_name": video.get("camera_name", native.get("camera_name", _DEFAULT_CAMERA_NAME)),
                "single_video": video.get("single_video", native.get("single_video", False)),
                "save_sim_states": native.get("save_sim_states", metadata.get("save_sim_states", False)),
            }
        )
        return result

    def _make_video_writer(self, task: _TaskState) -> Any:
        factory = self._video_writer_factory
        if factory is None:
            from libero.libero.utils.video_utils import VideoWriter  # noqa: PLC0415

            factory = VideoWriter
        return factory(
            str(task.video_dir),
            save_video=task.video_enabled,
            fps=task.fps,
            single_video=task.single_video,
        )

    @staticmethod
    def _safe_writer_reset(task: _TaskState, episode: _EpisodeState) -> None:
        try:
            task.writer.reset()
        except Exception as exc:
            task.artifact_failures.append(_failure("video_reset", exc))
            episode.video_failed = True
            episode.artifact_failures.append(_failure("video_reset", exc))

    def _append_observation(
        self, task: _TaskState, episode: _EpisodeState, observations: Mapping[str, Any], *, done: bool
    ) -> None:
        if task.writer is None:
            return
        try:
            task.writer.append_obs(observations, done=done, idx=episode.episode_index, camera_name=task.camera_name)
        except Exception as exc:
            episode.video_failed = True
            failure = _failure("video_append", exc)
            episode.artifact_failures.append(failure)
            task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})

    def _capture_state(self, task: _TaskState, episode: _EpisodeState, *, sim_state: Any, native_env: Any) -> None:
        if not task.save_sim_states:
            return
        if sim_state is _MISSING:
            if native_env is None:
                failure = {"stage": "state_capture", "error": "sim_state or native_env is required"}
                episode.state_failed = True
                episode.artifact_failures.append(failure)
                task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})
                return
            try:
                sim_state = self.capture_sim_state(native_env)
            except Exception as exc:
                failure = _failure("state_capture", exc)
                episode.state_failed = True
                episode.artifact_failures.append(failure)
                task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})
                return
        episode.states.append(sim_state)

    def _save_episode_states(self, task: _TaskState, episode: _EpisodeState, *, partial: bool) -> None:
        state_dir = task.task_dir / "sim_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        base = state_dir / f"episode_{episode.episode_index:03d}"
        metadata: dict[str, Any] = {
            "task_id": task.task_id,
            "suite": task.suite,
            "episode_index": episode.episode_index,
            "serializer": None,
            "partial": partial,
            "state_count": len(episode.states),
            "executed_steps": episode.result.steps if episode.result is not None else max(len(episode.states) - 1, 0),
            "initial_state_present": bool(episode.states),
            "states": [],
        }
        path: Path | None = None
        try:
            torch = self._load_torch()
            path = base.with_suffix(".pt")
            torch.save(episode.states, path)
            metadata["serializer"] = "torch"
        except (ImportError, TypeError, ValueError, AttributeError, pickle.PicklingError):
            if path is not None:
                path.unlink(missing_ok=True)
            try:
                path = base.with_suffix(".npz")
                arrays = {f"state_{index:06d}": np.asarray(state) for index, state in enumerate(episode.states)}
                np.savez(path, **arrays)
                metadata["serializer"] = "numpy"
            except Exception as exc:
                episode.state_failed = True
                failure = _failure("state_save", exc)
                episode.artifact_failures.append(failure)
                task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})
                return
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            episode.state_failed = True
            failure = _failure("state_save", exc)
            episode.artifact_failures.append(failure)
            task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})
            return
        metadata["states"] = [
            {
                "name": f"state_{index:06d}",
                "shape": list(np.asarray(state).shape),
                "dtype": str(np.asarray(state).dtype),
            }
            for index, state in enumerate(episode.states)
        ]
        sidecar = base.with_suffix(".metadata.json")
        try:
            metadata.update(_file_metadata(path))
            _atomic_json(sidecar, metadata)
        except Exception as exc:
            episode.state_failed = True
            failure = _failure("state_metadata_save", exc)
            episode.artifact_failures.append(failure)
            task.artifact_failures.append(failure | {"episode_index": str(episode.episode_index)})
            return
        task.artifacts.append(
            NativeArtifact(
                name=path.name,
                kind="trajectory",
                path=self._artifact_path(path),
                mime_type="application/octet-stream",
                scope="episode",
                suite=task.suite,
                task_id=task.task_id,
                episode_id=episode.episode_index,
                metadata={
                    "serializer": metadata["serializer"],
                    "state_count": len(episode.states),
                    "partial": partial,
                },
            )
        )
        task.artifacts.append(
            NativeArtifact(
                name=sidecar.name,
                kind="native_report",
                path=self._artifact_path(sidecar),
                mime_type="application/json",
                scope="episode",
                suite=task.suite,
                task_id=task.task_id,
                episode_id=episode.episode_index,
            )
        )

    def _write_task_stats(self, task: _TaskState) -> None:
        completed = [
            episode
            for episode in task.episodes.values()
            if episode.result is not None and episode.result.success is not None
        ]
        successful = sum(bool(episode.result.success) for episode in completed)
        payload = {"loss": None, "success_rate": (successful / len(completed) if completed else None)}
        path = task.task_dir / "evaluation.stats"
        temporary = task.task_dir / ".evaluation.stats.tmp"
        try:
            torch = self._load_torch()
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            task.artifacts.append(
                NativeArtifact(
                    name=path.name,
                    kind="native_report",
                    path=self._artifact_path(path),
                    mime_type="application/octet-stream",
                    scope="task",
                    suite=task.suite,
                    task_id=task.task_id,
                    metadata={"completed_native": len(completed), "successful_completed": successful},
                )
            )
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            self._record_failure(task, "native_stats_save", exc)

    def _write_task_manifest(self, task: _TaskState) -> None:
        path = task.task_dir / "task_manifest.json"
        completed_episodes = sum(episode.result is not None for episode in task.episodes.values())
        incomplete = task.planned_episodes is not None and completed_episodes < task.planned_episodes
        payload = {
            "suite": task.suite,
            "task_id": task.task_id,
            "name": task.name,
            "prompt": task.prompt,
            "planned_episodes": task.planned_episodes,
            "completed_episodes": completed_episodes,
            "partial": bool(task.artifact_failures or incomplete),
            "video": {
                "enabled": task.video_enabled,
                "fps": task.fps,
                "camera_name": task.camera_name,
                "single_video": task.single_video,
            },
            "artifacts": [
                *[artifact.path for artifact in task.artifacts],
                self._artifact_path(path),
            ],
            "artifact_failures": task.artifact_failures,
        }
        try:
            _atomic_json(path, payload)
            task.artifacts.append(
                NativeArtifact(
                    name=path.name,
                    kind="native_report",
                    path=self._artifact_path(path),
                    mime_type="application/json",
                    scope="task",
                    suite=task.suite,
                    task_id=task.task_id,
                )
            )
        except Exception as exc:
            self._record_failure(task, "task_manifest_save", exc)

    def _normalize_video_artifacts(self, task: _TaskState) -> None:
        if not task.video_enabled:
            return
        if task.single_video:
            path = task.video_dir / "video.mp4"
            if path.is_file():
                task.artifacts.append(_video_artifact(self, task, path, None))
            else:
                self._record_failure(task, "video_missing", FileNotFoundError(str(path)))
            return
        for episode in sorted(task.episodes.values(), key=lambda item: item.episode_index):
            source = task.video_dir / f"{episode.episode_index}.mp4"
            target = task.video_dir / f"episode_{episode.episode_index:03d}.mp4"
            if source.is_file():
                try:
                    source.replace(target)
                except Exception as exc:
                    self._record_failure(task, "video_rename", exc, episode.episode_index)
                    continue
                task.artifacts.append(_video_artifact(self, task, target, episode.episode_index))
            elif episode.result is not None and not episode.video_failed:
                self._record_failure(task, "video_missing", FileNotFoundError(str(source)), episode.episode_index)

    def _record_failure(self, task: _TaskState, stage: str, exc: Exception, episode_index: int | None = None) -> None:
        failure = _failure(stage, exc)
        if episode_index is not None:
            failure["episode_index"] = str(episode_index)
        task.artifact_failures.append(failure)
        self._run_failures.append({"task_id": str(task.task_id), **failure})
        if episode_index is not None and episode_index in task.episodes:
            task.episodes[episode_index].artifact_failures.append(failure)

    def _load_torch(self) -> Any:
        if self._torch_module is not _MISSING:
            if self._torch_module is None:
                raise ImportError("torch is disabled")
            return self._torch_module
        import torch  # noqa: PLC0415

        return torch

    def _require_task(self) -> _TaskState:
        if self._active_task is None:
            raise RuntimeError("no LIBERO task is open")
        return self._active_task

    def _artifact_path(self, path: Path) -> str:
        if self._run_root is None:
            return str(path)
        try:
            return str(path.relative_to(self._run_root))
        except ValueError:
            return str(path)


def _task_values(
    task: Any,
    *,
    suite: str | None,
    task_id: int | None,
    task_name: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    def value(name: str, explicit: Any, default: Any = None) -> Any:
        if explicit is not None:
            return explicit
        if isinstance(task, Mapping):
            return task.get(name, default)
        return getattr(task, name, default) if task is not None else default

    resolved = {
        "suite": value("suite", suite),
        "task_id": value("task_id", task_id),
        "task_name": value("name", task_name, f"task_{value('task_id', task_id, 0)}"),
        "prompt": value("prompt", prompt, ""),
    }
    if resolved["suite"] is None or resolved["task_id"] is None:
        raise ValueError("task suite and task_id are required")
    return resolved


def _result_done(result: Any) -> bool:
    return bool(getattr(result, "terminated", False) or getattr(result, "truncated", False))


def _failure(stage: str, exc: Exception) -> dict[str, str]:
    return {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}


def _file_metadata(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _video_artifact(
    reporter: LiberoNativeReporter, task: _TaskState, path: Path, episode_id: int | None
) -> NativeArtifact:
    return NativeArtifact(
        name=path.name,
        kind="video",
        path=reporter._artifact_path(path),
        mime_type="video/mp4",
        scope="episode" if episode_id is not None else "task",
        suite=task.suite,
        task_id=task.task_id,
        episode_id=episode_id,
        metadata={
            "fps": task.fps,
            "camera_name": task.camera_name,
            "single_video": task.single_video,
            "frame_count": reporter._video_frame_count(task, episode_id),
        },
    )


def create_native_reporter() -> NativeReportExporter:
    """Zero-argument plugin factory; native dependencies remain lazy."""
    return LiberoNativeReporter()
