"""Optional action-expert attention collection for native Torch policies."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import numpy as np

from robot_config.inference_runtime_options import AUTO_HORIZON_RUNTIME_OPTION_DEFAULTS as _OPTION_DEFAULTS

from .auto_horizon import estimate_execution_horizon


class ActionSelfAttentionCollector:
    """Collect eager action-expert attention for one policy request."""

    def __init__(
        self,
        *,
        hold_threshold: float,
        entropy_quantile: float,
        run_length: int,
        prediction_horizon: int,
        sampling_step: int = 3,
    ) -> None:
        if prediction_horizon < 1 or sampling_step < 1 or run_length < 1:
            raise ValueError("prediction_horizon, sampling_step, and run_length must be positive")
        self._hold_threshold = hold_threshold
        self._entropy_quantile = entropy_quantile
        self._run_length = run_length
        self._prediction_horizon = prediction_horizon
        self._sampling_step = sampling_step
        self._handles: list[Any] = []
        self._layer_count = 0
        self._call_count = 0
        self._sum: np.ndarray | None = None
        self._count = 0
        self._lock = threading.Lock()

    def install(self, policy: object) -> bool:
        """Install hooks on modules belonging to the action expert."""
        model = getattr(policy, "model", policy)
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            return False
        for name, module in named_modules():
            if not (name.endswith("self_attn") and ("gemma_expert" in name or "action_expert" in name)):
                continue
            register = getattr(module, "register_forward_hook", None)
            if callable(register):
                self._handles.append(register(self._capture))
        self._layer_count = len(self._handles)
        return bool(self._handles)

    def begin(self) -> None:
        with self._lock:
            self._sum = None
            self._count = 0
            self._call_count = 0

    def estimate(self) -> Mapping[str, object]:
        with self._lock:
            if self._sum is None or self._count == 0:
                return {"auto_horizon_available": False}
            attention = self._sum / self._count
        try:
            estimate = estimate_execution_horizon(
                attention,
                hold_threshold=self._hold_threshold,
                entropy_quantile=self._entropy_quantile,
                run_length=self._run_length,
            )
        except (TypeError, ValueError) as exc:
            return {"auto_horizon_available": False, "auto_horizon_error": str(exc)}
        return {
            "auto_horizon_available": True,
            "execution_horizon": estimate.horizon,
            "auto_horizon": {
                "forward_horizon": estimate.forward_horizon,
                "backward_horizon": estimate.backward_horizon,
                "join_row": estimate.join_row,
                "entropy_threshold": estimate.entropy_threshold,
            },
        }

    def close(self) -> None:
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()
        self.begin()

    def _capture(self, _module: object, _inputs: tuple[object, ...], output: object) -> None:
        with self._lock:
            call_index = self._call_count
            self._call_count += 1
        if self._layer_count < 1 or call_index // self._layer_count + 1 != self._sampling_step:
            return
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return
        detach = getattr(output[1], "detach", None)
        if not callable(detach):
            return
        values = detach().float().cpu().numpy()
        if values.ndim == 4:
            values = values.mean(axis=(0, 1))
        elif values.ndim == 3:
            values = values.mean(axis=0)
        if values.ndim != 2:
            return
        horizon = self._prediction_horizon
        if values.shape[0] < horizon or values.shape[1] < horizon:
            return
        values = values[-horizon:, -horizon:]
        if not np.isfinite(values).all() or (values < 0).any():
            return
        with self._lock:
            if self._sum is None:
                self._sum = np.zeros_like(values, dtype=np.float64)
            if self._sum.shape == values.shape:
                self._sum += values
                self._count += 1


def build_action_attention_collector(policy: object, options: Mapping[str, object]):
    """Build and install a collector when the runtime option is enabled."""
    if options.get("auto_horizon_enabled", False) is not True:
        return None
    config = getattr(policy, "config", None)
    prediction_horizon = next(
        (
            int(getattr(config, name))
            for name in ("chunk_size", "action_horizon", "n_action_steps")
            if getattr(config, name, 0)
        ),
        0,
    )
    if prediction_horizon < 1:
        raise ValueError("AutoHorizon requires a positive policy prediction horizon")
    num_steps = int(getattr(config, "num_inference_steps", 0) or 0)
    sampling_step = int(options.get("auto_horizon_sampling_step", _OPTION_DEFAULTS["auto_horizon_sampling_step"]))
    if num_steps and sampling_step > num_steps:
        raise ValueError(
            f"auto_horizon_sampling_step={sampling_step} exceeds the policy's num_inference_steps={num_steps}"
        )
    collector = ActionSelfAttentionCollector(
        hold_threshold=float(
            options.get("auto_horizon_hold_threshold", _OPTION_DEFAULTS["auto_horizon_hold_threshold"])
        ),
        entropy_quantile=float(
            options.get("auto_horizon_entropy_quantile", _OPTION_DEFAULTS["auto_horizon_entropy_quantile"])
        ),
        run_length=int(options.get("auto_horizon_run_length", _OPTION_DEFAULTS["auto_horizon_run_length"])),
        prediction_horizon=prediction_horizon,
        sampling_step=int(options.get("auto_horizon_sampling_step", _OPTION_DEFAULTS["auto_horizon_sampling_step"])),
    )
    model = getattr(policy, "model", policy)
    for name, module in getattr(model, "named_modules", lambda: ())():
        if "gemma_expert" in name or "action_expert" in name:
            config = getattr(module, "config", None)
            if config is not None:
                config._attn_implementation = "eager"  # noqa: SLF001
    if not collector.install(policy):
        raise ValueError(
            "AutoHorizon is enabled but no action-expert self_attn modules were found "
            "(expected '*gemma_expert*self_attn' or '*action_expert*self_attn')"
        )
    return collector


__all__ = ["ActionSelfAttentionCollector", "build_action_attention_collector"]
