"""Drop non-finite points from a simulated lidar cloud (D5).

Gazebo reports gpu_lidar beams without a hit as ``±inf`` and the MuJoCo
lidar plugin as NaN; FAST-LIO's generic handler keeps them, so the provider
bridges the raw cloud on an internal topic and this robot-agnostic node
republishes the finite points on the advertised endpoint with the same
header. Runs on simulation time as part of the physics embodiment.

Output layout (lekiwi-mujoco-runtime-platform D5.3): whenever points are
dropped or a field is added the output is unorganised — ``height 1``,
``width n``, ``point_step``/``row_step``/field offsets recomputed,
``is_dense true`` and ``data`` exactly ``n * point_step`` bytes. An input
that is already finite and needs no field passes through untouched (no
copy, no resharding), and an input that already carries ``intensity``
keeps it.

Supported input layouts (review round 1): the byte parser assumes packed
rows (``row_step == point_step * width`` or unset) and the host byte order.
A cloud that declares padded rows, foreign endianness or a short ``data``
buffer is REJECTED (:class:`UnsupportedCloudLayout`); the node then drops
that frame with a warning instead of filtering it with a misaligned view —
FAST-LIO tolerates a missing frame better than silently wrong points.
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


class UnsupportedCloudLayout(ValueError):
    """An input cloud whose byte layout the filter cannot parse safely."""


def _unsupported_layout_reason(message: PointCloud2) -> str | None:
    count = message.width * message.height
    if message.is_bigendian != (sys.byteorder == "big"):
        return f"is_bigendian={message.is_bigendian} on a {sys.byteorder}-endian host"
    if message.row_step != 0 and message.row_step != message.point_step * message.width:
        return f"row_step {message.row_step} != point_step*width {message.point_step * message.width} (padded rows)"
    if len(message.data) < count * message.point_step:
        return f"data is {len(message.data)} bytes, width*height*point_step needs {count * message.point_step}"
    return None


#: Byte width of every ``sensor_msgs/msg/PointField`` datatype.
_DATATYPE_WIDTH = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}


def _finite_mask(message: PointCloud2) -> np.ndarray | None:
    """The boolean mask of points with finite x/y/z (None when unmappable)."""
    count = message.width * message.height
    if count == 0:
        return np.zeros(0, dtype=bool)
    offsets = {field.name: field.offset for field in message.fields}
    if not {"x", "y", "z"} <= offsets.keys():
        return None
    points = np.frombuffer(bytes(message.data), dtype=np.uint8, count=count * message.point_step).reshape(
        count, message.point_step
    )
    finite = np.ones(count, dtype=bool)
    for axis in ("x", "y", "z"):
        column = np.ascontiguousarray(points[:, offsets[axis] : offsets[axis] + 4])
        finite &= np.isfinite(column.view(np.float32).reshape(-1))
    return finite


def filter_finite_points(message: PointCloud2, *, add_intensity: bool = False) -> PointCloud2:
    """A copy of ``message`` holding only the points with finite x/y/z.

    With ``add_intensity`` a missing ``intensity`` field is appended as
    float32 zeros — the MuJoCo lidar publishes bare ``x y z`` while FAST-LIO's
    generic handler reads ``PointXYZI`` (D5.3). The header and the existing
    fields are preserved; a cloud that is already finite and needs no field
    is returned as-is (no copy, no resharding).
    """
    reason = _unsupported_layout_reason(message)
    if reason is not None:
        raise UnsupportedCloudLayout(reason)
    finite = _finite_mask(message)
    if finite is None:
        return message
    count = message.width * message.height
    has_intensity = any(field.name == "intensity" for field in message.fields)
    append_intensity = add_intensity and not has_intensity
    if finite.all() and not append_intensity:
        return message
    # a field is added or points are dropped: the output is unorganised with a
    # fully recomputed layout (D5.3), the input's field order preserved
    points = np.frombuffer(bytes(message.data), dtype=np.uint8, count=count * message.point_step).reshape(
        count, message.point_step
    )
    kept = points[finite]
    out_point_step = message.point_step + (4 if append_intensity else 0)
    out = np.zeros((kept.shape[0], out_point_step), dtype=np.uint8)
    fields = []
    offset = 0
    for field in message.fields:
        fields.append(PointField(name=field.name, offset=offset, datatype=field.datatype, count=field.count))
        width = _DATATYPE_WIDTH.get(field.datatype, 4) * max(field.count, 1)
        out[:, offset : offset + width] = kept[:, field.offset : field.offset + width]
        offset += width
    if append_intensity:
        fields.append(PointField(name="intensity", offset=offset, datatype=PointField.FLOAT32, count=1))
        offset += 4
    filtered = PointCloud2()
    filtered.header = message.header
    filtered.fields = fields
    filtered.point_step = out_point_step
    filtered.height = 1
    filtered.width = int(kept.shape[0])
    filtered.is_bigendian = message.is_bigendian
    filtered.is_dense = True
    filtered.row_step = out_point_step * filtered.width
    filtered.data = out.tobytes()
    return filtered


class FiniteCloudFilter(Node):
    def __init__(self):
        super().__init__("finite_cloud_filter")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("add_intensity", False)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        if not input_topic or not output_topic:
            raise ValueError("finite_cloud_filter requires the input_topic and output_topic parameters")
        self._add_intensity = bool(self.get_parameter("add_intensity").value)
        self._publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self._subscription = self.create_subscription(PointCloud2, input_topic, self._on_cloud, 10)

    def _on_cloud(self, message: PointCloud2) -> None:
        # a stateless relay: the simulation-time stamps of the source cloud
        # are preserved verbatim in the republished header
        try:
            filtered = filter_finite_points(message, add_intensity=self._add_intensity)
        except UnsupportedCloudLayout as error:
            # A misaligned view would silently corrupt every point; the frame
            # is dropped instead (FAST-LIO tolerates a missing frame) and the
            # configuration error is reported at most once per second
            # (throttle_duration_sec — rclpy loggers take one preformatted
            # message; printf-style positional args raise TypeError, review
            # round 2).
            self.get_logger().warning(
                f"dropping a cloud with an unsupported byte layout: {error}", throttle_duration_sec=1.0
            )
            return
        self._publisher.publish(filtered)


def main(args=None):
    rclpy.init(args=args)
    node = FiniteCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
