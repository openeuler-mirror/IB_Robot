"""LIBERO action codec (LIBERO runtime, Pre-native validation; LIBERO action boundary action-boundary reconciliation).

Owns the LIBERO-specific action validation and canonicalization:

- the generic runtime decodes the ROS ``VariantsList`` into a dict of numpy
  arrays via ``decode_step_action_generic``;
- this module validates that the decoded dict contains exactly one key
  ``action`` with a finite ``float32[7]``;
- the codec then canonicalizes the action to reproduce the pinned native
  robosuite effective command before ``env.step()``.

LIBERO action boundary action-boundary reconciliation:

The pinned native LIBERO environment (robosuite) applies two different
actuator semantics to the 7D action:

- **arm dimensions 0:6** (delta_xyz, delta_axis_angle): the OSC_POSE
  controller calls ``BaseController.scale_action`` which applies
  ``np.clip(action, input_min, input_max)`` with ``input_min=-1,
  input_max=1``. Values outside ``[-1, 1]`` are silently clipped to the
  boundary.

- **gripper dimension 6**: ``PandaGripper.format_action`` calls
  ``np.sign(action)`` to binarize the gripper command. Any negative
  value maps to -1 (open); any positive value maps to +1 (close);
  zero maps to 0.

The ACT checkpoint's MEAN_STD postprocessor can produce finite float32
values slightly outside ``[-1, 1]`` (e.g. gripper -1.045, arm-z 1.047).
These overshoots are caused by the unnormalizer, not by a model bug.
Rejecting them before native entry would be incompatible with the pinned
native execution semantics.

Therefore this codec explicitly reproduces the native canonicalization:

.. code-block:: python

    canonical_arm = np.clip(raw_action[0:6], -1.0, 1.0)
    canonical_gripper = -1.0 if raw_action[6] < 0 else
                        +1.0 if raw_action[6] > 0 else
                         0.0

The canonicalized ``float32[7]`` is what is sent to ``env.step()``,
recorded in evidence, and treated as the committed action.

This is **not** inference normalization. It is environment/action-codec
ownership: the codec converts a finite policy-space environment action
into the exact effective command that pinned robosuite would execute.

The raw postprocessed action is preserved in the ROS StepBenchmark request
and in diagnostics. Canonicalization happens only at the concrete
LIBERO environment boundary, not in inference_service, not in the
bundled preprocessor/postprocessor, not in the generic benchmark_runtime,
and not in action_dispatch.

The codec still rejects before native entry:
- missing ``action`` key;
- extra keys;
- non-ndarray payload;
- wrong dtype (int32, float64);
- wrong rank/shape;
- NaN, +Inf, -Inf.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark_runtime.action_codec import ActionDecodeError, decode_step_action_generic
from benchmark_runtime.adapter import PreNativeValidationError

_ALLOWED_ACTION_DIM = 7
_ACTION_KEY = "action"
_ARM_DIMS = 6  # dimensions 0:6 are arm (delta_xyz + delta_axis_angle)
_GRIPPER_INDEX = 6  # dimension 6 is gripper


def decode_libero_action(action_msg: Any) -> np.ndarray:
    """Decode a StepBenchmark ``VariantsList`` into a canonical ``float32[7]``.

    Performs generic decode (``decode_step_action_generic``) and then
    delegates ALL LIBERO-specific validation and canonicalization to
    :func:`validate_libero_action_dict`. There is exactly one concrete
    validation/canonicalization entry point.

    Raises :class:`PreNativeValidationError` on any validation failure so
    the generic environment node can ``abort_step()`` without poisoning.
    Returns the canonicalized contiguous ``float32[7]`` effective command.
    """
    try:
        decoded = decode_step_action_generic(action_msg)
    except ActionDecodeError as exc:
        raise PreNativeValidationError(f"action decode rejected: {exc}") from exc

    return validate_libero_action_dict(decoded)


def validate_libero_action_dict(action: dict[str, np.ndarray]) -> np.ndarray:
    """Validate and canonicalize a generic-decoded action dict.

    Validation (before canonicalization):
    - exactly one key ``action``;
    - ``isinstance(arr, np.ndarray)``;
    - ``np.dtype(arr.dtype) == np.dtype(np.float32)`` (no silent cast);
    - ``np.all(np.isfinite(arr))`` (NaN/Inf rejected);
    - optional leading singleton batch dim: ``(1, 7)`` -> ``(7,)``;
    - shape ``(7,)``.

    Canonicalization (LIBERO action boundary):
    - arm dimensions 0:6: ``np.clip(raw, -1.0, 1.0)`` — reproduces
      robosuite ``BaseController.scale_action`` clipping;
    - gripper dimension 6: sign-equivalent canonicalization to
      ``{-1.0, 0.0, +1.0}`` — reproduces ``PandaGripper.format_action``
      ``np.sign(action)`` binarization;
    - returns contiguous ``float32[7]``;
    - does not modify the input array;
    - does not perform checkpoint normalization/unnormalization;
    - does not invert gripper sign.

    Raises :class:`PreNativeValidationError` on any validation failure.
    Returns the canonicalized contiguous ``float32[7]`` effective command.
    """
    extra_keys = sorted(set(action.keys()) - {_ACTION_KEY})
    if extra_keys:
        raise PreNativeValidationError(
            f"action payload must contain only the 'action' key; unexpected keys: {extra_keys}"
        )
    if _ACTION_KEY not in action:
        raise PreNativeValidationError("action payload is missing the 'action' key")

    arr = action[_ACTION_KEY]
    if not isinstance(arr, np.ndarray):
        raise PreNativeValidationError(
            f"action payload must be np.ndarray under 'action' key; got {type(arr).__name__}"
        )

    if np.dtype(arr.dtype) != np.dtype(np.float32):
        raise PreNativeValidationError(
            f"action payload dtype must be float32; got {arr.dtype}. "
            "The wire contract requires float32; int/float64 are rejected."
        )

    if not bool(np.all(np.isfinite(arr))):
        raise PreNativeValidationError("action payload contains NaN or Inf; finite float32[7] required")

    # Optional leading singleton batch dimension: (1, 7) -> (7,)
    if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == _ALLOWED_ACTION_DIM:
        arr = arr.reshape(_ALLOWED_ACTION_DIM)
    elif arr.ndim != 1 or arr.shape[0] != _ALLOWED_ACTION_DIM:
        raise PreNativeValidationError(
            f"action payload shape must be ({_ALLOWED_ACTION_DIM},) or "
            f"(1, {_ALLOWED_ACTION_DIM}); got shape {tuple(arr.shape)}"
        )

    return _canonicalize_libero_action(arr)


def _canonicalize_libero_action(raw: np.ndarray) -> np.ndarray:
    """Convert a finite float32[7] into the pinned native effective command.

    - arm 0:6: ``np.clip(raw, -1.0, 1.0)`` — reproduces robosuite
      ``BaseController.scale_action``.
    - gripper 6: sign-equivalent to ``{-1.0, 0.0, +1.0}`` — reproduces
      ``PandaGripper.format_action`` via ``np.sign(action)``.

    Does not modify the input array. Returns a new contiguous ``float32[7]``.
    """
    canonical = raw.astype(np.float32, copy=True)
    # arm dimensions: clip to [-1, 1] (robosuite BaseController.scale_action)
    canonical[:_ARM_DIMS] = np.clip(canonical[:_ARM_DIMS], np.float32(-1.0), np.float32(1.0))
    # gripper dimension: sign-equivalent canonicalization (PandaGripper.format_action)
    gripper = float(canonical[_GRIPPER_INDEX])
    if gripper > 0.0:
        canonical[_GRIPPER_INDEX] = np.float32(1.0)
    elif gripper < 0.0:
        canonical[_GRIPPER_INDEX] = np.float32(-1.0)
    else:
        canonical[_GRIPPER_INDEX] = np.float32(0.0)
    return np.ascontiguousarray(canonical, dtype=np.float32)
