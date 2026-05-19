"""Internal joint state model for the mock.

Single source of truth for joint positions. Action subscribers write into it;
JointState publishers read from it. Thread-safety is provided by the caller
(rclpy single-threaded executor in practice).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class JointModel:
    """Ordered joint position store keyed by joint id (string).

    The order returned by :meth:`positions` always matches the construction
    order of ``joint_ids`` so consumers can publish ``JointState.name`` and
    ``position`` in a stable, contract-compatible layout.
    """

    def __init__(self, joint_ids: Sequence[str], initial: dict[str, float] | None = None):
        if not joint_ids:
            raise ValueError("JointModel requires at least one joint id")
        if len(set(joint_ids)) != len(joint_ids):
            raise ValueError(f"JointModel joint ids must be unique: {list(joint_ids)}")
        self._ids: list[str] = list(joint_ids)
        initial = initial or {}
        self._pos: dict[str, float] = {jid: float(initial.get(jid, 0.0)) for jid in self._ids}

    @property
    def joint_ids(self) -> list[str]:
        return list(self._ids)

    def positions(self) -> list[float]:
        return [self._pos[jid] for jid in self._ids]

    def set_by_index(self, indices: Iterable[int], values: Iterable[float]) -> None:
        """Update positions using parallel index/value sequences.

        Args:
            indices: 0-based indices into :attr:`joint_ids`.
            values:  New positions, same length as ``indices``.

        Raises:
            IndexError: If any index is out of range.
            ValueError: If the two sequences have different lengths.
        """
        idx_list = list(indices)
        val_list = list(values)
        if len(idx_list) != len(val_list):
            raise ValueError(f"set_by_index length mismatch: {len(idx_list)} indices vs {len(val_list)} values")
        n = len(self._ids)
        for i, v in zip(idx_list, val_list, strict=False):
            if not 0 <= i < n:
                raise IndexError(f"Joint index {i} out of range [0, {n})")
            self._pos[self._ids[i]] = float(v)
