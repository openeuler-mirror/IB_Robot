"""Host-side GraspGen math the session runs between the compiled OM roles.

These functions reproduce the upstream CUDA semantics of NVlabs GraspGen on a
CPU host so that the eight compiled Ascend OM sub-graphs can be fed deterministic
inputs. The implementation mirrors the reference runtime shipped with the model
export pipeline (``ModelZoo-PyTorch/ACL_PyTorch/.../GraspGen/runtime.py``) and
intentionally avoids any CUDA-specific extension such as ``pointnet2_ops``.

The module is deliberately dependency-free (NumPy only) so the packager, the
adapter and the tests can import it without the Ascend ACL runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from model_utils.graspgen_contract import GRASPGEN_NPOINTS, GRASPGEN_NSAMPLES, GRASPGEN_RADII


def furthest_point_sample(xyz: np.ndarray, npoint: int) -> np.ndarray:
    points = np.ascontiguousarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {points.shape}")
    if len(points) == 0 or npoint <= 0:
        raise ValueError("FPS requires at least one input point and one output point")

    indices = np.zeros(npoint, dtype=np.int64)
    distances = np.full(len(points), np.float32(1e10), dtype=np.float32)
    valid = np.sum(points * points, axis=1, dtype=np.float32) > np.float32(1e-3)
    distances[~valid] = np.float32(-1.0)
    delta = np.empty_like(points)
    distance = np.empty(len(points), dtype=np.float32)
    has_valid = bool(np.any(valid))
    old = 0
    for output_index in range(1, npoint):
        np.subtract(points, points[old], out=delta)
        np.multiply(delta, delta, out=delta)
        np.sum(delta, axis=1, dtype=np.float32, out=distance)
        np.minimum(distances, distance, out=distances, where=valid)
        old = int(np.argmax(distances)) if has_valid else 0
        indices[output_index] = old
    return indices


def ball_query(xyz: np.ndarray, centers: np.ndarray, radius: float, nsample: int) -> np.ndarray:
    points = np.ascontiguousarray(xyz, dtype=np.float32)
    query = np.ascontiguousarray(centers, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("xyz and centers must both have shape [N, 3]")
    if nsample <= 0 or radius <= 0:
        raise ValueError("ball query radius and nsample must be positive")

    distance2 = np.empty((len(query), len(points)), dtype=np.float32)
    work = np.empty_like(distance2)
    np.subtract(query[:, None, 0], points[None, :, 0], out=distance2)
    np.multiply(distance2, distance2, out=distance2)
    for coordinate in (1, 2):
        np.subtract(query[:, None, coordinate], points[None, :, coordinate], out=work)
        np.multiply(work, work, out=work)
        np.add(distance2, work, out=distance2)
    radius2 = np.float32(radius) * np.float32(radius)
    indices = np.zeros((len(query), nsample), dtype=np.int64)
    for query_index, row in enumerate(distance2):
        matches = np.flatnonzero(row < radius2)[:nsample]
        if len(matches):
            indices[query_index].fill(int(matches[0]))
            indices[query_index, : len(matches)] = matches
    return indices


def remove_point_cloud_outliers(
    points: np.ndarray,
    threshold: float = 0.014,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the points whose mean L1 distance to their k nearest others is below threshold.

    The sampler's own rule, without its all-pairs distance matrix: exact L1
    neighbours come from a KD tree and only their distances are computed, in
    float32 like the sampler. An X2 bottle mask is ~27k points, which the
    all-pairs form took ~40 s per request to filter on the model host.
    The point's own row is excluded, duplicates of it are not.
    """
    values = np.ascontiguousarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {values.shape}")
    if len(values) <= k:
        return values.copy(), np.empty((0, 3), dtype=np.float32)
    if k <= 0 or threshold <= 0:
        raise ValueError("outlier threshold and k must be positive")

    from scipy.spatial import cKDTree

    # k + 1 so the k nearest others are there whether or not the row itself is.
    _, indices = cKDTree(values).query(values, k=k + 1, p=1, workers=-1)
    distance = np.sum(np.abs(values[:, None, :] - values[indices]), axis=-1, dtype=np.float32)
    distance[indices == np.arange(len(values))[:, None]] = np.float32(np.inf)
    distance.sort(axis=1)
    keep = np.mean(distance[:, :k], axis=1, dtype=np.float32) < np.float32(threshold)
    return values[keep], values[~keep]


def group_xyz(xyz: np.ndarray, centers: np.ndarray, indices: np.ndarray) -> np.ndarray:
    grouped = np.take(xyz.T, indices, axis=1)
    grouped -= centers.T[:, :, None]
    return np.ascontiguousarray(grouped[None], dtype=np.float32)


def group_features(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError(f"features must have shape [1, C, N], got {features.shape}")
    return np.ascontiguousarray(np.take(features[0], indices, axis=1)[None], dtype=np.float32)


@dataclass(frozen=True)
class PointNetGeometry:
    stage1_input: np.ndarray
    stage2_relative_xyz: np.ndarray
    stage2_indices: np.ndarray
    stage2_xyz: np.ndarray


def build_pointnet_geometry(
    xyz: np.ndarray,
    npoints: tuple[int, int] = GRASPGEN_NPOINTS,
    radii: tuple[float, float] = GRASPGEN_RADII,
    nsamples: tuple[int, int] = GRASPGEN_NSAMPLES,
) -> PointNetGeometry:
    points = np.ascontiguousarray(xyz, dtype=np.float32)
    stage1_fps = furthest_point_sample(points, npoints[0])
    stage1_xyz = points[stage1_fps]
    stage1_indices = ball_query(points, stage1_xyz, radii[0], nsamples[0])
    stage1_input = group_xyz(points, stage1_xyz, stage1_indices)

    stage2_fps = furthest_point_sample(stage1_xyz, npoints[1])
    stage2_xyz = stage1_xyz[stage2_fps]
    stage2_indices = ball_query(stage1_xyz, stage2_xyz, radii[1], nsamples[1])
    stage2_relative_xyz = group_xyz(stage1_xyz, stage2_xyz, stage2_indices)
    return PointNetGeometry(stage1_input, stage2_relative_xyz, stage2_indices, stage2_xyz)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(axis_angle, dtype=np.float32)
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True).astype(np.float32)
    factor = np.float32(0.5) * np.sinc(angles * np.float32(0.5 / math.pi)).astype(np.float32)
    quaternions = np.concatenate(
        [np.cos(angles * np.float32(0.5)).astype(np.float32), vectors * factor],
        axis=-1,
    )
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = np.float32(2.0) / np.sum(quaternions * quaternions, axis=-1, dtype=np.float32)
    matrix = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return np.ascontiguousarray(matrix.reshape(vectors.shape[:-1] + (3, 3)), dtype=np.float32)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    rotations = np.ascontiguousarray(matrix, dtype=np.float32)
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in [3, 3], got {rotations.shape}")
    flat = rotations.reshape(rotations.shape[:-2] + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.moveaxis(flat, -1, 0)
    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            np.float32(0.0),
        )
    ).astype(np.float32)
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    denominator = np.float32(2.0) * np.maximum(q_abs[..., None], np.float32(0.1))
    candidates = quat_by_rijk / denominator
    choice = np.argmax(q_abs, axis=-1)
    quaternions = np.take_along_axis(candidates, choice[..., None, None], axis=-2)[..., 0, :]
    quaternions = np.where(quaternions[..., :1] < 0, -quaternions, quaternions)
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True).astype(np.float32)
    half_angles = np.arctan2(norms, quaternions[..., :1]).astype(np.float32)
    factor = np.float32(0.5) * np.sinc(half_angles * np.float32(1.0 / math.pi)).astype(np.float32)
    return np.ascontiguousarray(quaternions[..., 1:] / factor, dtype=np.float32)


def rt_to_matrix(sample: np.ndarray, kappa: float) -> np.ndarray:
    values = np.ascontiguousarray(sample, dtype=np.float32)
    matrices = np.zeros((len(values), 4, 4), dtype=np.float32)
    matrices[:, :3, :3] = axis_angle_to_matrix(values[:, 3:] * np.float32(math.pi))
    matrices[:, :3, 3] = values[:, :3] / np.float32(kappa)
    matrices[:, 3, 3] = 1.0
    return matrices


def matrix_to_rt(matrices: np.ndarray, kappa: float) -> np.ndarray:
    poses = np.ascontiguousarray(matrices, dtype=np.float32)
    translation = poses[:, :3, 3] * np.float32(kappa)
    rotation = matrix_to_axis_angle(poses[:, :3, :3]) / np.float32(math.pi)
    return np.ascontiguousarray(np.concatenate([translation, rotation], axis=-1), dtype=np.float32)


class DDPMSchedulerNumpy:
    def __init__(self, num_train_timesteps: int, beta_schedule: str):
        if beta_schedule == "scaled_linear":
            betas = np.linspace(
                np.float32(math.sqrt(0.0001)),
                np.float32(math.sqrt(0.02)),
                num_train_timesteps,
                dtype=np.float32,
            ) ** np.float32(2.0)
        elif beta_schedule == "squaredcos_cap_v2":
            values = []
            for index in range(num_train_timesteps):
                first = index / num_train_timesteps
                second = (index + 1) / num_train_timesteps
                alpha_first = math.cos((first + 0.008) / 1.008 * math.pi / 2) ** 2
                alpha_second = math.cos((second + 0.008) / 1.008 * math.pi / 2) ** 2
                values.append(min(1 - alpha_second / alpha_first, 0.999))
            betas = np.asarray(values, dtype=np.float32)
        else:
            raise ValueError(f"unsupported beta schedule: {beta_schedule}")
        self.betas = betas
        self.alphas = np.float32(1.0) - betas
        self.alphas_cumprod = np.cumprod(self.alphas, dtype=np.float32)
        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1, dtype=np.int64)

    def step(
        self,
        model_output: np.ndarray,
        timestep: int,
        sample: np.ndarray,
        variance_noise: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        t = int(timestep)
        previous_t = t - 1
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[previous_t] if previous_t >= 0 else np.float32(1.0)
        beta_prod_t = np.float32(1.0) - alpha_prod_t
        beta_prod_t_prev = np.float32(1.0) - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = np.float32(1.0) - current_alpha_t

        pred_original = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        pred_original = np.clip(pred_original, np.float32(-1.0), np.float32(1.0))
        original_coeff = np.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
        sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        previous = original_coeff * pred_original + sample_coeff * sample
        if t > 0:
            if variance_noise is None:
                raise ValueError("variance_noise is required for nonzero DDPM timesteps")
            variance = (np.float32(1.0) - alpha_prod_t_prev) / beta_prod_t * current_beta_t
            variance = np.maximum(variance, np.float32(1e-20))
            previous = previous + np.sqrt(variance) * variance_noise
        return np.ascontiguousarray(previous, dtype=np.float32), np.ascontiguousarray(pred_original, dtype=np.float32)


def positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive int, rejecting bool, float, and non-positive input.

    ``bool`` is a subclass of ``int``, so an ``isinstance`` check would silently accept
    ``True`` as ``1``; a float would silently truncate. Both are configuration mistakes
    that must fail at load time rather than produce a degenerate execution plan.
    """
    if type(value) is not int:
        raise ValueError(f"GraspGen {name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"GraspGen {name} must be positive, got {value}")
    return value


def positive_float(value: object, name: str) -> float:
    """Return ``value`` as a finite positive float, rejecting bool, NaN, inf, and <= 0."""
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ValueError(f"GraspGen {name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"GraspGen {name} must be finite, got {number}")
    if number <= 0.0:
        raise ValueError(f"GraspGen {name} must be positive, got {number}")
    return number
