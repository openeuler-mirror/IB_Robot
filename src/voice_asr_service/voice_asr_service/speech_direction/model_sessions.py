"""Manifest-backed session adapters for speech direction models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, suppress

import numpy as np

from inference_service.backends import RuntimeContext
from inference_service.model_sessions import ModelSession
from inference_service.unified_runtime import ExecutionContext, ModelRequest


class SpeechDirectionRoleRunner:
    """Protocol adapter retained for the Host FullSubNet and VAD layers."""

    def __init__(
        self,
        session: ModelSession,
        context: RuntimeContext,
        *,
        owns_session: bool = False,
    ) -> None:
        self.session = session
        self.context = context
        self.backend = getattr(context, "backend", "ascend") if context is not None else "ascend"
        self._owns_session = bool(owns_session)
        self._request_counter = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def last_timing_ms(self) -> dict[str, float]:
        # Forward the underlying session's per-request timing (e.g. the Torch
        # FullSubNet session's fb/sb hop latencies) to host consumers; sessions
        # without timing support yield an empty mapping.
        timing = getattr(self.session, "last_timing_ms", None)
        return dict(timing) if isinstance(timing, dict) else {}

    def _invoke(self, role: str, values: Mapping[str, object]) -> Mapping[str, object]:
        if self._execution_context is not None:
            request = ModelRequest(values, {"role": role})
            return self.session.execute_role(role, values, request, self._execution_context)
        self._request_counter += 1
        context = ExecutionContext(f"speech-direction-{self._request_counter}")
        request = ModelRequest(values, {"role": role})
        return self.session.execute_role(role, values, request, context)

    @contextmanager
    def execution_scope(self):
        """Keep FB, Host feature assembly, and SB in one admitted request."""
        if self._execution_context is not None:
            raise RuntimeError("speech direction execution scope is already active")
        self._request_counter += 1
        self._execution_context = ExecutionContext(f"speech-direction-{self._request_counter}")
        try:
            yield
        finally:
            self._execution_context = None

    def infer_named(self, values: Mapping[str, object]) -> Mapping[str, object]:
        return self._invoke("silero_vad", values)

    def run_fb(self, frame: np.ndarray) -> np.ndarray:
        output = self._invoke("fullsubnet_fb", {"host.fullsubnet.fb_spectrum": np.ascontiguousarray(frame)})
        return np.asarray(output["host.fullsubnet.fb_features"], dtype=np.float32)

    def run_sb(self, frame: np.ndarray) -> np.ndarray:
        output = self._invoke("fullsubnet_sb", {"host.fullsubnet.sb_features": np.ascontiguousarray(frame)})
        return np.asarray(output["host.fullsubnet.sb_mask"], dtype=np.float32)

    def infer(self, audio: np.ndarray) -> float:
        values: dict[str, object] = {"host.silero.audio": np.ascontiguousarray(audio)}
        # Only supply inputs the deployment actually declares: ONNX deployments
        # bind host.silero.sample_rate (sourced from the audio contract), while
        # Ascend OM deployments fold the sample rate into the graph and reject
        # unexpected semantics at validation time.
        deployment = getattr(self.context, "deployment", None) if self.context is not None else None
        bindings = getattr(deployment, "bindings", {}).get("silero_vad") if deployment is not None else None
        declared = {binding.semantic for binding in bindings.inputs} if bindings is not None else set()
        if "host.silero.sample_rate" in declared:
            contract = getattr(deployment, "audio_contract", None)
            if contract is not None and contract.sample_rate_hz is not None:
                values["host.silero.sample_rate"] = np.asarray(contract.sample_rate_hz, dtype=np.int64)
        output = self._invoke("silero_vad", values)
        return float(np.asarray(output["host.silero.prob"]).reshape(-1)[0])

    inference = infer

    def reset_state(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.session.reset(ExecutionContext(f"speech-direction-reset-{self._request_counter}"))

    def close(self) -> None:
        # Production Sessions are owned by ModelRuntimeHandle.  Keeping this
        # adapter non-owning prevents the host pipeline and the handle from
        # closing the same device resource twice.  Standalone callers can opt
        # into the old convenience behavior explicitly.
        if self._owns_session:
            self.session.close()


class SpeechDirectionSessionResources:
    """Lifecycle owner for the role Sessions used by one Speech stream."""

    def __init__(self, sessions: Mapping[str, tuple[ModelSession, RuntimeContext]]) -> None:
        if not sessions:
            raise ValueError("Speech Direction requires at least one model Session")
        self._entries = tuple((str(role), session, context) for role, (session, context) in sessions.items())
        if any(not role or session is None or context is None for role, session, context in self._entries):
            raise ValueError("Speech Direction Session entries must contain role, session, and context")
        self._loaded = False
        self._closed = False

    @property
    def sessions(self) -> Mapping[str, ModelSession]:
        return {role: session for role, session, _context in self._entries}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def add(self, role: str, session: ModelSession, context: RuntimeContext) -> None:
        """Register a role before handle ownership is transferred."""

        if self._loaded or self._closed:
            raise RuntimeError("cannot add a Session after resources are loaded or closed")
        if not role or session is None or context is None:
            raise ValueError("Speech Direction Session entries must contain role, session, and context")
        if any(existing_role == role for existing_role, _session, _context in self._entries):
            raise ValueError(f"duplicate Speech Direction Session role: {role}")
        self._entries = (*self._entries, (role, session, context))

    def load(self, _context: object = None) -> None:
        if self._closed:
            raise RuntimeError("Speech Direction Session resources are closed")
        if self._loaded:
            return
        loaded: list[ModelSession] = []
        try:
            for _role, session, context in self._entries:
                session.load(context)
                loaded.append(session)
        except Exception:
            for session in reversed(loaded):
                with suppress(Exception):
                    session.close()
            raise
        self._loaded = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for _role, session, _context in reversed(self._entries):
            try:
                session.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Speech Direction Session cleanup failed: {errors[0]}") from errors[0]


__all__ = ["SpeechDirectionRoleRunner", "SpeechDirectionSessionResources"]
