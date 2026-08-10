"""LIBERO observation codec (LIBERO runtime).

Converts a raw LIBERO ``OffScreenRenderEnv`` observation dictionary into the
checkpoint-compatible policy observation contract expected by the
``Deepkar/libero-test-act`` ACT checkpoint and the LeRobot LIBERO processor.

Boundary ownership for the 180-degree image orientation correction:

- The checkpoint-compatible LIBERO path requires exactly one 180-degree image
  correction (flip both H and W axes) per camera frame, matching the pinned
  LeRobot ``LiberoProcessorStep`` reference
  (``libs/lerobot/src/lerobot/processor/env_processor.py``).
- This codec owns the correction exactly once. The published ROS images are
  "checkpoint-oriented" so production preprocessing must NOT apply a
  second flip. A double-flip or no flip is a rejection defect.
- Parity is established against the LeRobot reference using ``torch.flip``
  on dims ``[2, 3]`` (H, W) after the LeRobot convention. The codec performs
  the same flip on the HWC NumPy image (``[::-1, ::-1, :]``) and produces a
  contiguous RGB uint8 HWC ``[256, 256, 3]`` tensor.

State mapping (8D float32):

```text
raw robot0_eef_pos[3]            -> observation.state[0:3]
raw robot0_eef_quat[4] xyzw      -> axis-angle[3] -> observation.state[3:6]
raw robot0_gripper_qpos[2]       -> observation.state[6:8]
```

Quaternion convention matches the pinned LeRobot processor:

```text
input order: x, y, z, w
output: axis * angle
small denominator -> zero vector
float32 output
```

Validation rejects missing keys, wrong shape, wrong channel count,
unsupported dtype and NaN/Inf.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class ObservationCodecError(ValueError):
    """Raised when a raw LIBERO observation cannot be converted cleanly."""


# Required raw LIBERO keys. Each must be present and shape-checked.
_REQUIRED_RAW_KEYS: tuple[str, ...] = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)

# Output policy keys, in canonical order matching the checkpoint contract:
POLICY_IMAGE_KEY = "observation.images.image"
POLICY_IMAGE2_KEY = "observation.images.image2"
POLICY_STATE_KEY = "observation.state"

# Output image dimensions (the ACT checkpoint was trained at 256x256 and
# HuggingFaceVLA/libero uses 256x256). The codec refuses to silently resize
# wrong shapes; the LIBERO env is constructed at 256x256 explicitly.
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_STATE_DIM = 8

# Small denominator guard matching LeRobot's `LiberoProcessorStep._quat2axisangle`.
_QUAT_SMALL_DENOMINATOR = 1e-10


def _require_raw_key(raw_obs: Mapping[str, Any], key: str) -> Any:
    if key not in raw_obs:
        raise ObservationCodecError(f"raw observation is missing required key '{key}'")
    return raw_obs[key]


def _validate_image(payload: Any, key: str) -> np.ndarray:
    if not isinstance(payload, np.ndarray):
        raise ObservationCodecError(f"raw image '{key}' must be a numpy.ndarray, got {type(payload).__name__}")
    if payload.dtype != np.uint8:
        raise ObservationCodecError(f"raw image '{key}' dtype must be uint8, got {payload.dtype}")
    if payload.ndim != 3 or payload.shape[2] != 3:
        raise ObservationCodecError(f"raw image '{key}' must be HWC [H,W,3], got shape {payload.shape}")
    h, w, _ = payload.shape
    if h != _IMAGE_HEIGHT or w != _IMAGE_WIDTH:
        raise ObservationCodecError(
            f"raw image '{key}' has shape {payload.shape}; "
            f"LIBERO env must be constructed at [{_IMAGE_HEIGHT},{_IMAGE_WIDTH},3]"
        )
    return payload


def _flip_image_180(image: np.ndarray) -> np.ndarray:
    """Flip both H and W axes (a 180-degree rotation in image space).

    Matches ``torch.flip(img, dims=[2, 3])`` from
    ``LiberoProcessorStep._process_observation``. The input is HWC uint8; we
    flip the first two axes (H and W) and return a contiguous array.
    """
    return np.ascontiguousarray(image[::-1, ::-1, :])


def quat_to_axis_angle_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert a single xyzw quaternion to a 3D axis-angle vector.

    Matches ``LiberoProcessorStep._quat2axisangle`` (pinned LeRobot
    ``libs/lerobot/src/lerobot/processor/env_processor.py``):

    - input order: x, y, z, w
    - output: axis * angle
    - small denominator -> zero vector
    - float32 output

    Pure NumPy; does not require torch.
    """
    if not isinstance(quat, np.ndarray):
        raise ObservationCodecError(f"eef_quat must be a numpy.ndarray, got {type(quat).__name__}")
    if quat.shape != (4,):
        raise ObservationCodecError(f"eef_quat must have shape (4,), got {quat.shape}")
    q = quat.astype(np.float32, copy=True)
    if not np.all(np.isfinite(q)):
        raise ObservationCodecError("eef_quat contains NaN or Inf")

    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den <= _QUAT_SMALL_DENOMINATOR:
        # Near-singular: zero rotation or numerical edge case. Match LeRobot
        # by returning a zero axis-angle vector.
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * float(np.arccos(w))
    axis = q[:3] / den
    axis_angle = axis.astype(np.float32) * np.float32(angle)
    return np.ascontiguousarray(axis_angle.astype(np.float32))


def _validate_state_components(
    eef_pos: np.ndarray,
    eef_quat: np.ndarray,
    gripper_qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(eef_pos, np.ndarray):
        raise ObservationCodecError(f"eef_pos must be a numpy.ndarray, got {type(eef_pos).__name__}")
    if eef_pos.shape != (3,):
        raise ObservationCodecError(f"eef_pos must have shape (3,), got {eef_pos.shape}")

    if not isinstance(eef_quat, np.ndarray):
        raise ObservationCodecError(f"eef_quat must be a numpy.ndarray, got {type(eef_quat).__name__}")
    if eef_quat.shape != (4,):
        raise ObservationCodecError(f"eef_quat must have shape (4,), got {eef_quat.shape}")

    if not isinstance(gripper_qpos, np.ndarray):
        raise ObservationCodecError(f"gripper_qpos must be a numpy.ndarray, got {type(gripper_qpos).__name__}")
    if gripper_qpos.shape != (2,):
        raise ObservationCodecError(f"gripper_qpos must have shape (2,), got {gripper_qpos.shape}")

    for name, arr in (("eef_pos", eef_pos), ("eef_quat", eef_quat), ("gripper_qpos", gripper_qpos)):
        if not np.all(np.isfinite(arr)):
            raise ObservationCodecError(f"{name} contains NaN or Inf")

    eef_pos_f32 = np.ascontiguousarray(eef_pos, dtype=np.float32)
    gripper_qpos_f32 = np.ascontiguousarray(gripper_qpos, dtype=np.float32)
    return eef_pos_f32, eef_quat, gripper_qpos_f32


def convert_observation(raw_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert a raw LIBERO observation mapping into the policy contract.

    Returns a mapping with the three canonical keys:

    - ``observation.images.image``  : contiguous RGB uint8 HWC [256,256,3]
    - ``observation.images.image2`` : contiguous RGB uint8 HWC [256,256,3]
    - ``observation.state``         : contiguous finite float32 [8]

    The 180-degree image orientation correction is applied exactly once here.
    The state vector is built in the order ``[eef_pos(3), axis_angle(3),
    gripper_qpos(2)]`` matching the LeRobot LIBERO processor. The output is
    suitable for direct ROS publication via the generic observation publisher.

    Raises :class:`ObservationCodecError` for any missing key, wrong shape,
    wrong dtype, NaN/Inf or non-finite state. No silent resize / normalize
    / unit conversion.
    """
    if not isinstance(raw_obs, Mapping):
        raise ObservationCodecError(f"raw observation must be a Mapping, got {type(raw_obs).__name__}")

    for key in _REQUIRED_RAW_KEYS:
        _require_raw_key(raw_obs, key)

    raw_image = _validate_image(raw_obs["agentview_image"], "agentview_image")
    raw_image2 = _validate_image(raw_obs["robot0_eye_in_hand_image"], "robot0_eye_in_hand_image")

    eef_pos, eef_quat, gripper_qpos = _validate_state_components(
        raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_quat"], raw_obs["robot0_gripper_qpos"]
    )

    axis_angle = quat_to_axis_angle_xyzw(eef_quat)

    state = np.concatenate([eef_pos, axis_angle, gripper_qpos], axis=0)
    if state.shape != (_STATE_DIM,):
        raise ObservationCodecError(f"assembled state has shape {state.shape}; expected ({_STATE_DIM},)")
    state = np.ascontiguousarray(state, dtype=np.float32)

    image_out = _flip_image_180(raw_image)
    image2_out = _flip_image_180(raw_image2)

    return {
        POLICY_IMAGE_KEY: image_out,
        POLICY_IMAGE2_KEY: image2_out,
        POLICY_STATE_KEY: state,
    }
