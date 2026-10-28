"""finite_cloud_filter layout rules (lekiwi-mujoco-runtime-platform D5.3).

The filter gains ``add_intensity`` (the MuJoCo lidar publishes bare x/y/z
while FAST-LIO's generic handler reads ``PointXYZI``) and explicit output
layout rules: whenever points are dropped or a field is added the output is
unorganised with a recomputed layout; an already-finite cloud that needs no
field passes through untouched.
"""

import struct

import pytest
from sensor_msgs.msg import PointCloud2, PointField

from robot_runtime.finite_cloud_filter import UnsupportedCloudLayout, filter_finite_points


def _cloud(rows, *, fields=("x", "y", "z", "intensity"), width=None, height=1, point_step=None, dense=False):
    message = PointCloud2()
    message.header.frame_id = "livox_frame"
    message.header.stamp.sec = 7
    message.width = width if width is not None else len(rows)
    message.height = height
    message.point_step = point_step if point_step is not None else 4 * len(fields)
    message.is_dense = dense
    message.fields = [
        PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(fields)
    ]
    if height == 1:
        message.data = b"".join(struct.pack("<" + "f" * len(fields), *row) for row in rows)
    else:
        message.data = b"".join(struct.pack("<" + "f" * len(fields), *row) for row in rows)
    return message


def test_all_finite_passes_through_untouched():
    rows = [(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)]
    message = _cloud(rows)
    filtered = filter_finite_points(message, add_intensity=False)
    assert filtered is message, "an already-finite cloud needs no copy"
    # add_intensity on a cloud that already has the field changes nothing
    assert filter_finite_points(message, add_intensity=True) is message


def test_partial_nan_drops_points_and_recomputes_the_layout():
    raw = _cloud(
        [
            (1.0, 2.0, 3.0, 10.0),
            (float("inf"), 1.0, 1.0, 3.0),
            (0.5, -0.5, 0.25, 5.0),
            (1.0, float("nan"), 1.0, 3.0),
        ]
    )
    raw.row_step = raw.point_step * raw.width  # the malformed input carries a stale row_step
    filtered = filter_finite_points(raw)
    assert filtered.width == 2 and filtered.height == 1
    assert filtered.point_step == 16 and filtered.row_step == 32
    assert filtered.is_dense is True
    assert len(bytes(filtered.data)) == 2 * 16
    assert [struct.unpack_from("<ffff", bytes(filtered.data), i * 16) for i in range(2)] == [
        (1.0, 2.0, 3.0, 10.0),
        (0.5, -0.5, 0.25, 5.0),
    ]
    assert [f.name for f in filtered.fields] == ["x", "y", "z", "intensity"]
    assert [f.offset for f in filtered.fields] == [0, 4, 8, 12]


def test_all_nan_yields_an_empty_dense_cloud():
    raw = _cloud([(float("nan"), 0.0, 0.0, 1.0), (float("inf"), 0.0, 0.0, 1.0)])
    filtered = filter_finite_points(raw)
    assert filtered.width == 0 and filtered.height == 1
    assert filtered.is_dense is True and bytes(filtered.data) == b""
    assert filtered.row_step == 0


def test_empty_input_stays_empty():
    message = _cloud([])
    assert filter_finite_points(message) is message
    # an empty cloud without the field still gains it when asked
    bare = _cloud([], fields=("x", "y", "z"))
    filtered = filter_finite_points(bare, add_intensity=True)
    assert filtered is not bare
    assert [f.name for f in filtered.fields] == ["x", "y", "z", "intensity"]
    assert filtered.width == 0 and filtered.is_dense is True


def test_add_intensity_appends_zero_float32_intensity():
    # the MuJoCo plugin publishes bare x y z, point_step 12
    raw = _cloud(
        [(1.0, 2.0, 3.0), (float("nan"), 1.0, 1.0), (0.5, -0.5, 0.25)],
        fields=("x", "y", "z"),
    )
    filtered = filter_finite_points(raw, add_intensity=True)
    assert [f.name for f in filtered.fields] == ["x", "y", "z", "intensity"]
    assert [f.offset for f in filtered.fields] == [0, 4, 8, 12]
    assert filtered.point_step == 16 and filtered.width == 2 and filtered.height == 1
    assert filtered.row_step == 32 and filtered.is_dense is True
    assert len(bytes(filtered.data)) == 32
    assert [struct.unpack_from("<ffff", bytes(filtered.data), i * 16) for i in range(2)] == [
        (1.0, 2.0, 3.0, 0.0),
        (0.5, -0.5, 0.25, 0.0),
    ]
    # the header is preserved verbatim
    assert filtered.header.frame_id == "livox_frame" and filtered.header.stamp.sec == 7


def test_existing_intensity_is_kept_not_duplicated():
    raw = _cloud([(1.0, 2.0, 3.0, 7.5)])
    # all-finite with the field already present: passes through untouched
    assert filter_finite_points(raw, add_intensity=True) is raw
    # and a dropped point does not duplicate the field either
    dropped = _cloud([(1.0, 2.0, 3.0, 7.5), (float("nan"), 0.0, 0.0, 1.0)])
    filtered = filter_finite_points(dropped, add_intensity=True)
    assert [f.name for f in filtered.fields] == ["x", "y", "z", "intensity"]
    assert struct.unpack_from("<ffff", bytes(filtered.data), 0) == (1.0, 2.0, 3.0, 7.5)


def test_organised_2d_input_becomes_unorganised_on_drop():
    rows = [(1.0, 2.0, 3.0, 1.0), (float("nan"), 0.0, 0.0, 1.0), (4.0, 5.0, 6.0, 2.0), (7.0, 8.0, 9.0, 3.0)]
    raw = _cloud(rows, width=2, height=2, dense=False)
    filtered = filter_finite_points(raw)
    assert filtered.height == 1 and filtered.width == 3
    assert filtered.is_dense is True
    assert [struct.unpack_from("<ffff", bytes(filtered.data), i * 16) for i in range(3)] == [
        (1.0, 2.0, 3.0, 1.0),
        (4.0, 5.0, 6.0, 2.0),
        (7.0, 8.0, 9.0, 3.0),
    ]


def test_input_field_order_is_preserved():
    # a field order the runtime never produces itself, but the filter must
    # not reorder: a uint16 ring between z and intensity
    raw = _cloud([(0.0, 0.0, 0.0, 0, 0.0)], fields=("x", "y", "z", "ring", "intensity"))
    raw.fields[3].datatype = PointField.UINT16
    # the true byte layout: x@0 y@4 z@8 ring@12(2B) intensity@14, step 18
    for field, offset in zip(raw.fields, (0, 4, 8, 12, 14), strict=True):
        field.offset = offset
    raw.point_step = 18
    raw.data = b"".join(
        struct.pack("<fffHf", *row) for row in [(1.0, 2.0, 3.0, 4, 10.0), (float("inf"), 0.0, 0.0, 0, 0.0)]
    )
    raw.width = 2
    filtered = filter_finite_points(raw)
    assert [f.name for f in filtered.fields] == ["x", "y", "z", "ring", "intensity"]
    assert [f.offset for f in filtered.fields] == [0, 4, 8, 12, 14]
    assert filtered.point_step == 18 and filtered.width == 1
    assert struct.unpack_from("<fffHf", bytes(filtered.data), 0) == (1.0, 2.0, 3.0, 4, 10.0)


# ---------------------------------------------------------------------------
# Review round 1: layouts the byte parser cannot read safely are rejected
# (never filtered through a misaligned view), and the node drops such a frame
# with a warning instead of publishing silently wrong points.
# ---------------------------------------------------------------------------


def test_padded_rows_are_rejected():
    raw = _cloud([(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)])
    raw.width = 1
    raw.height = 2
    raw.row_step = raw.point_step * raw.width + 4  # 4 bytes of row padding
    raw.data = b"".join(
        struct.pack("<ffff", *row) + b"\x00\x00\x00\x00" for row in [(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)]
    )
    with pytest.raises(UnsupportedCloudLayout, match="row_step"):
        filter_finite_points(raw)


def test_big_endian_clouds_are_rejected():
    raw = _cloud([(1.0, 2.0, 3.0, 10.0)])
    raw.is_bigendian = True
    with pytest.raises(UnsupportedCloudLayout, match="endianness|endian"):
        filter_finite_points(raw)


def test_short_data_buffers_are_rejected():
    raw = _cloud([(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)])
    raw.data = raw.data[: raw.point_step]  # claims 2 points, carries 1
    with pytest.raises(UnsupportedCloudLayout, match="data is"):
        filter_finite_points(raw)


def test_the_node_drops_an_unsupported_layout_with_a_warning():
    import rclpy

    from robot_runtime.finite_cloud_filter import FiniteCloudFilter

    class _Recorder:
        def __init__(self):
            self.published = []

        def publish(self, msg):
            self.published.append(msg)

    class _Logger:
        """The rclpy signature: one preformatted message, keyword-only
        options (review round 2)."""

        def __init__(self):
            self.warnings = []

        def warning(self, message, **kwargs):
            self.warnings.append(message)

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "input_topic:=/layout_in",
            "-p",
            "output_topic:=/layout_out",
            "-p",
            "add_intensity:=false",
        ]
    )
    try:
        node = FiniteCloudFilter()
        recorder, logger = _Recorder(), _Logger()
        node._publisher = recorder
        node.get_logger = lambda: logger
        padded = _cloud([(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)])
        padded.width, padded.height = 1, 2
        padded.row_step = padded.point_step + 4
        node._on_cloud(padded)  # dropped, warned — never published
        assert recorder.published == []
        assert any("row_step" in warning for warning in logger.warnings)
        node._on_cloud(_cloud([(1.0, 2.0, 3.0, 10.0)]))  # a good cloud still flows
        assert len(recorder.published) == 1
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_the_real_node_survives_a_bad_cloud_and_keeps_publishing_good_ones():
    """Review round 2, node level with the REAL rclpy logger (throttled
    warning kwargs): an unsupported layout is dropped through the logging
    path (printf-style positional args would raise TypeError there) and the
    next good cloud still reaches the ROS graph."""
    import time

    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    from robot_runtime.finite_cloud_filter import FiniteCloudFilter

    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "input_topic:=/layout_graph_in",
            "-p",
            "output_topic:=/layout_graph_out",
            "-p",
            "add_intensity:=false",
        ]
    )
    node = None
    collector = None
    executor = None
    try:
        node = FiniteCloudFilter()
        collector = rclpy.create_node("layout_graph_collector")
        received = []
        collector.create_subscription(PointCloud2, "/layout_graph_out", received.append, 10)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(collector)
        padded = _cloud([(1.0, 2.0, 3.0, 10.0), (0.5, -0.5, 0.25, 5.0)])
        padded.width, padded.height = 1, 2
        padded.row_step = padded.point_step + 4
        node._on_cloud(padded)  # the real logger's throttled warning runs here
        good = _cloud([(1.0, 2.0, 3.0, 10.0), (float("inf"), 0.0, 0.0, 0.0)])
        node._on_cloud(good)  # filtered: the non-finite point is dropped
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not received:
            executor.spin_once(timeout_sec=0.05)
        assert len(received) == 1, f"expected exactly the good cloud, got {len(received)}"
        assert received[0].width == 1 and received[0].is_dense is True
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if collector is not None:
            collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
