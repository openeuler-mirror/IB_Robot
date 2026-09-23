"""Tests for bag_to_lerobot helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import yaml
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.utils import DatasetInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.bag_to_lerobot import (  # noqa: E402
    IntegrityReport,
    _build_feature_conversion_table,
    _clean_float_array,
    _contract_from_dataset_metadata,
    _dataset_feature_names_for_spec,
    _estimate_stream_rate_hz,
    _image_to_hwc,
    _log_image_stream_diagnostics,
    _merge_integrity_report,
    _persist_custom_info,
    _plan_streams,
    _preflight_bags,
    _resolve_video_codec,
    _selected_indices_for_ticks,
    discover_video_adapters,
    export_bags_to_lerobot,
    parse_args,
)


def test_resolve_video_codec_prefers_h264_in_auto_mode(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        return DummyCodec(is_encoder=name == "h264")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "h264"


def test_resolve_video_codec_falls_back_to_av1_when_h264_missing(monkeypatch):
    import av

    class DummyCodec:
        def __init__(self, is_encoder: bool):
            self.is_encoder = is_encoder

    def fake_codec(name: str, mode: str):
        assert mode == "w"
        if name == "h264":
            raise ValueError("missing")
        return DummyCodec(is_encoder=name == "libsvtav1")

    monkeypatch.setattr(av.codec, "Codec", fake_codec)

    assert _resolve_video_codec("auto") == "libsvtav1"


def test_estimate_stream_rate_hz_uses_timestamp_span():
    ts = [0, 33_333_333, 66_666_666, 100_000_000]

    assert abs(_estimate_stream_rate_hz(ts) - 30.0) < 0.05


def test_selected_indices_for_ticks_exposes_hold_duplicates_from_phase_offset():
    ts = np.array([0, 34, 68], dtype=np.int64)
    ticks = np.array([0, 33, 66], dtype=np.int64)

    selected = _selected_indices_for_ticks(
        policy="hold",
        ts_ns=ts,
        ticks_ns=ticks,
        step_ns=33,
        tol_ns=0,
    )

    assert selected.tolist() == [0, 0, 1]


def test_dataset_feature_names_for_current_preserve_contract_names():
    class Spec:
        key = "observation.current"
        names = ["current.1", "current.2"]

    assert _dataset_feature_names_for_spec(Spec()) == ["current.1", "current.2"]


def test_plan_streams_uses_external_video_instead_of_rosbag_topic():
    class Spec:
        key = "observation.images.top"
        topic = "/camera/top/image_raw"
        ros_type = "sensor_msgs/msg/Image"
        image_resize = (16, 16)
        is_action = False

    streams, by_topic = _plan_streams(
        [Spec()],
        {"/camera/top/image_raw": "sensor_msgs/msg/Image"},
        {"observation.images.top"},
    )

    assert list(streams) == ["observation.images.top"]
    assert by_topic == {}


def test_rosbag_only_episode_keeps_existing_image_topic_path(tmp_path):
    class Spec:
        key = "observation.images.top"
        topic = "/camera/top/image_raw"
        ros_type = "sensor_msgs/msg/Image"
        image_resize = (16, 16)
        is_action = False

    assert discover_video_adapters(tmp_path) == {}
    streams, by_topic = _plan_streams([Spec()], {Spec.topic: Spec.ros_type})

    assert list(streams) == [Spec.key]
    assert by_topic == {Spec.topic: [Spec.key]}


def test_image_diagnostics_reports_annex_b_alignment_error(capsys):
    class Spec:
        image_resize = (16, 16)
        resample_policy = "hold"
        asof_tol_ms = 0

    stream = type("Stream", (), {"spec": Spec(), "ts": [1_000, 2_000_000]})()

    _log_image_stream_diagnostics(
        streams={"observation.images.top": stream},
        ticks_ns=np.asarray([1_000, 1_001_000], dtype=np.int64),
        step_ns=1_000_000,
        target_fps=1_000,
        video_sources={"observation.images.top": "Annex-B"},
    )

    output = capsys.readouterr().out
    assert "source=Annex-B" in output
    assert "max_alignment_error=1.000 ms" in output


def test_merge_integrity_report_preserves_episode_and_observation_context():
    info = {}

    _merge_integrity_report(info, 0, "observation.images.top", IntegrityReport(clean=True))
    _merge_integrity_report(
        info,
        1,
        "observation.images.wrist",
        IntegrityReport(
            clean=False,
            frame_gaps=[{"frame_index": 7, "lost_packets": 2, "reason": "rtp_sequence_gap"}],
        ),
    )

    assert info["integrity"] == {
        "clean": False,
        "frame_gaps": [
            {
                "episode_index": 1,
                "observation_key": "observation.images.wrist",
                "frame_index": 7,
                "lost_packets": 2,
                "reason": "rtp_sequence_gap",
            }
        ],
    }


def test_persist_custom_info_merges_with_typed_dataset_info(tmp_path):
    dataset_info = DatasetInfo(
        codebase_version="v3.0",
        fps=30,
        features={"action": {"dtype": "float32", "shape": (1,)}},
        total_episodes=1,
    )
    write_info(dataset_info, tmp_path)
    info_path = tmp_path / "meta" / "info.json"

    _persist_custom_info(
        info_path,
        {
            "ibrobot_fingerprint": "fingerprint",
            "integrity": {"clean": False, "frame_gaps": [{"frame_index": 2}]},
        },
    )

    payload = json.loads(info_path.read_text(encoding="utf-8"))
    assert payload["total_episodes"] == 1
    assert payload["ibrobot_fingerprint"] == "fingerprint"
    assert payload["integrity"] == {"clean": False, "frame_gaps": [{"frame_index": 2}]}


def test_export_merges_annex_b_video_with_dds_action_and_state(tmp_path, monkeypatch):
    episode_dir = tmp_path / "episode_000001"
    episode_dir.mkdir()
    (episode_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  storage_identifier: mcap\n  duration:\n    nanoseconds: 200000000\n",
        encoding="utf-8",
    )
    video_path = episode_dir / "observation.images.top.h264"
    codec = av.CodecContext.create("libx264", "w")
    codec.width = 16
    codec.height = 16
    codec.pix_fmt = "yuv420p"
    codec.options = {"profile": "baseline", "tune": "zerolatency", "x264-params": "bframes=0"}
    codec.open()
    packets = []
    for index in range(3):
        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24")
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    video_path.write_bytes(b"".join(packets))
    sidecar_entries = [
        {
            "frame_index": index,
            "capture_timestamp_ns": 1_000_000_000 + index * 100_000_000,
            "rtp_timestamp": 90_000 + index * 9_000,
            "keyframe": index == 0,
            "lost_packets": 0,
            "session_generation": 1,
            "dropped": None,
        }
        for index in range(3)
    ]
    video_path.with_suffix(".h264.json").write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in sidecar_entries), encoding="utf-8"
    )

    @dataclass
    class TopicType:
        name: str
        type: str

    class FakeReader:
        def __init__(self):
            self.messages = [
                ("/state", b"state0", 1_000_000_000),
                ("/action", b"action0", 1_000_000_000),
                ("/state", b"state1", 1_100_000_000),
                ("/action", b"action1", 1_100_000_000),
                ("/state", b"state2", 1_200_000_000),
                ("/action", b"action2", 1_200_000_000),
            ]

        def open(self, *_args):
            pass

        def get_all_topics_and_types(self):
            return [TopicType("/state", "test/State"), TopicType("/action", "test/Action")]

        def has_next(self):
            return bool(self.messages)

        def read_next(self):
            return self.messages.pop(0)

    class FakeInfo:
        """Stand-in for LeRobot 0.6's typed DatasetInfo: attribute access, not a dict."""

        def __init__(self):
            self.total_episodes = 0

        def get(self, key, default=None):
            return getattr(self, key, default)

        def to_dict(self):
            return {"total_episodes": self.total_episodes}

    class FakeMeta:
        def __init__(self, root):
            self.root = Path(root)
            self.info = FakeInfo()

        def update_chunk_settings(self, **_kwargs):
            pass

        def save_episode(self):
            """Mirror LeRobotDatasetMetadata.save_episode(): write_info happens here."""
            self.info.total_episodes += 1
            info_path = self.root / "meta" / "info.json"
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps(self.info.to_dict(), indent=4), encoding="utf-8")

    class FakeDataset:
        last = None

        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
            self.meta = FakeMeta(self.root)
            self.frames = []
            self.saved = 0
            FakeDataset.last = self

        @classmethod
        def create(cls, *, root, **_kwargs):
            return cls(root)

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved += 1
            self.meta.save_episode()

    image_spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top",
        ros_type="sensor_msgs/msg/Image",
        is_action=False,
        names=[],
        image_resize=(16, 16),
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    state_spec = SimpleNamespace(
        key="observation.state",
        topic="/state",
        ros_type="test/State",
        is_action=False,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    action_spec = SimpleNamespace(
        key="action",
        topic="/action",
        ros_type="test/Action",
        is_action=True,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    contract = SimpleNamespace(
        rate_hz=10,
        robot_type="test",
        observations=[],
        actions=[],
    )

    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot._preflight_bags",
        lambda bags: (
            contract,
            [
                {
                    "dataset_meta": {},
                    "info": yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"],
                    "tables": {},
                }
                for bag in bags
            ],
        ),
    )
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.iter_specs", lambda _contract: [image_spec, state_spec, action_spec]
    )
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.feature_from_spec",
        lambda spec, _videos: (
            spec.key,
            {"dtype": "image", "shape": (16, 16, 3)}
            if spec.image_resize
            else {"dtype": "float32", "shape": (1,), "names": ["joint"]},
            spec.image_resize is not None,
        ),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.make_zero_pad", lambda feature: np.zeros(feature["shape"]))
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.rosbag2_py.SequentialReader", FakeReader)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.deserialize_message", lambda data, _type: data)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.get_message", lambda ros_type: ros_type)
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.decode_value",
        lambda _ros_type, data, _spec: np.asarray([0.1 if b"state" in data else 0.2], dtype=np.float32),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.LeRobotDataset", FakeDataset)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.contract_fingerprint", lambda _contract: "fingerprint")

    export_bags_to_lerobot(
        [episode_dir],
        out_root=tmp_path / "output",
        use_videos=False,
    )

    dataset = FakeDataset.last
    assert dataset.saved == 1
    assert len(dataset.frames) == 3
    assert dataset.frames[0]["observation.images.top"].shape == (16, 16, 3)
    np.testing.assert_allclose(dataset.frames[0]["observation.state"], [0.1])
    np.testing.assert_allclose(dataset.frames[0]["action"], [0.2])
    # The typed path writes through meta/info.json, so assert what actually landed on disk.
    persisted = json.loads((dataset.root / "meta" / "info.json").read_text(encoding="utf-8"))
    assert persisted["integrity"] == {"clean": True}
    assert persisted["ibrobot_fingerprint"] == "fingerprint"


def test_clean_float_array_replaces_non_finite_values():
    arr = _clean_float_array([1.0, np.nan, np.inf, -np.inf], np.float32)

    assert arr.dtype == np.float32
    assert arr.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_image_to_hwc_converts_chw_decoded_ros_image():
    chw = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    hwc = _image_to_hwc(chw, (4, 5, 3), feature_name="observation.images.front")

    assert hwc.shape == (4, 5, 3)
    np.testing.assert_array_equal(hwc, np.transpose(chw, (1, 2, 0)))


def test_image_to_hwc_keeps_hwc_image():
    hwc = np.zeros((4, 5, 3), dtype=np.uint8)

    result = _image_to_hwc(hwc, (4, 5, 3))

    assert result.shape == hwc.shape
    np.testing.assert_array_equal(result, hwc)


def test_clean_float_array_warns_for_non_current_features(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.state")
    assert any("observation.state" in rec.message for rec in caplog.records if rec.levelno >= 30)


def test_clean_float_array_silent_for_current(caplog):
    with caplog.at_level("WARNING"):
        _clean_float_array([1.0, np.nan, 2.0], np.float32, feature_name="observation.current")
    assert not any("non-finite" in rec.message for rec in caplog.records if rec.levelno >= 30)


def _write_annex_b_episode(episode_dir: Path, frame_indices: list[int]) -> Path:
    """Write a minimal episode directory with an Annex-B stream and its sidecar.

    ``frame_indices`` is written verbatim so callers can reproduce a sidecar whose
    ``frame_index`` regresses across a session roll, which is what a WiFi session
    flap produces on disk.
    """
    episode_dir.mkdir(parents=True)
    (episode_dir / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  storage_identifier: mcap\n  duration:\n    nanoseconds: 200000000\n",
        encoding="utf-8",
    )
    video_path = episode_dir / "observation.images.top.h264"
    codec = av.CodecContext.create("libx264", "w")
    codec.width = 16
    codec.height = 16
    codec.pix_fmt = "yuv420p"
    codec.options = {"profile": "baseline", "tune": "zerolatency", "x264-params": "bframes=0"}
    codec.open()
    packets = []
    for position in range(len(frame_indices)):
        frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), position * 40, dtype=np.uint8), format="rgb24")
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    video_path.write_bytes(b"".join(packets))
    entries = [
        {
            "frame_index": frame_index,
            "capture_timestamp_ns": 1_000_000_000 + position * 100_000_000,
            "rtp_timestamp": 90_000 + position * 9_000,
            "keyframe": position == 0,
            "lost_packets": 0,
            "session_generation": 1,
            "dropped": None,
        }
        for position, frame_index in enumerate(frame_indices)
    ]
    video_path.with_suffix(".h264.json").write_text(
        "".join(f"{json.dumps(entry)}\n" for entry in entries), encoding="utf-8"
    )
    return episode_dir


class _CountingDataset:
    """Stand-in for LeRobotDataset that records how many episodes were saved."""

    last = None

    def __init__(self, root):
        self.root = Path(root)
        # Mirror LeRobot 0.6: meta.info is a typed object, not a dict.
        self.meta = SimpleNamespace(
            info=SimpleNamespace(total_episodes=0, get=lambda key, default=None: default),
            root=self.root,
            update_chunk_settings=lambda **_kwargs: None,
        )
        self.frames = []
        self.saved = 0
        _CountingDataset.last = self

    @classmethod
    def create(cls, *, root, **_kwargs):
        Path(root).mkdir(parents=True, exist_ok=True)
        return cls(root)

    def add_frame(self, frame):
        self.frames.append(frame)

    def clear_episode_buffer(self, delete_images: bool = True):
        self.frames.clear()

    def save_episode(self):
        self.saved += 1
        self.meta.info.total_episodes += 1
        info_path = self.root / "meta" / "info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps({"total_episodes": self.meta.info.total_episodes}), encoding="utf-8")


def _install_export_stubs(monkeypatch) -> None:
    """Replace the contract/rosbag/LeRobot dependencies of ``export_bags_to_lerobot``."""

    @dataclass
    class TopicType:
        name: str
        type: str

    class FakeReader:
        def __init__(self):
            self.messages = [
                ("/state", b"state0", 1_000_000_000),
                ("/action", b"action0", 1_000_000_000),
                ("/state", b"state1", 1_100_000_000),
                ("/action", b"action1", 1_100_000_000),
                ("/state", b"state2", 1_200_000_000),
                ("/action", b"action2", 1_200_000_000),
            ]

        def open(self, *_args):
            pass

        def get_all_topics_and_types(self):
            return [TopicType("/state", "test/State"), TopicType("/action", "test/Action")]

        def has_next(self):
            return bool(self.messages)

        def read_next(self):
            return self.messages.pop(0)

    image_spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top",
        ros_type="sensor_msgs/msg/Image",
        is_action=False,
        names=[],
        image_resize=(16, 16),
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    state_spec = SimpleNamespace(
        key="observation.state",
        topic="/state",
        ros_type="test/State",
        is_action=False,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    action_spec = SimpleNamespace(
        key="action",
        topic="/action",
        ros_type="test/Action",
        is_action=True,
        names=["joint"],
        image_resize=None,
        resample_policy="hold",
        asof_tol_ms=0,
        stamp_src="header",
    )
    contract = SimpleNamespace(rate_hz=10, robot_type="test", observations=[], actions=[])

    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot._preflight_bags",
        lambda bags: (
            contract,
            [
                {
                    "dataset_meta": {},
                    "info": yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"],
                    "tables": {},
                }
                for bag in bags
            ],
        ),
    )
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.iter_specs", lambda _contract: [image_spec, state_spec, action_spec]
    )
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.feature_from_spec",
        lambda spec, _videos: (
            spec.key,
            {"dtype": "image", "shape": (16, 16, 3)}
            if spec.image_resize
            else {"dtype": "float32", "shape": (1,), "names": ["joint"]},
            spec.image_resize is not None,
        ),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.make_zero_pad", lambda feature: np.zeros(feature["shape"]))
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.rosbag2_py.SequentialReader", FakeReader)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.deserialize_message", lambda data, _type: data)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.get_message", lambda ros_type: ros_type)
    monkeypatch.setattr(
        "dataset_tools.bag_to_lerobot.decode_value",
        lambda _ros_type, data, _spec: np.asarray([0.1 if b"state" in data else 0.2], dtype=np.float32),
    )
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.LeRobotDataset", _CountingDataset)
    monkeypatch.setattr("dataset_tools.bag_to_lerobot.contract_fingerprint", lambda _contract: "fingerprint")


def test_export_skips_a_bag_whose_frames_fail_to_decode(tmp_path, monkeypatch, capsys):
    """A decode failure mid-episode must be skipped and summarized, not abort the run.

    d7e50b5 defined the model: one bad bag is dropped and named, and only an entirely
    failed run raises. A layout error from ``_image_to_hwc`` used to escape that model
    and terminate ``export_bags_to_lerobot`` outright -- bags already written stayed on
    disk, but the one that failed appeared in no summary.
    """
    _install_export_stubs(monkeypatch)
    calls = {"n": 0}
    original = _image_to_hwc

    def fail_first_bag(values, feature_shape, *, feature_name=""):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError(f"unexpected image layout for {feature_name}")
        return original(values, feature_shape, feature_name=feature_name)

    monkeypatch.setattr("dataset_tools.bag_to_lerobot._image_to_hwc", fail_first_bag)
    bad_bag = _write_annex_b_episode(tmp_path / "episode_000001", [0, 1, 2])
    good_bag = _write_annex_b_episode(tmp_path / "episode_000002", [0, 1, 2])

    export_bags_to_lerobot(
        [bad_bag, good_bag],
        out_root=tmp_path / "output",
        use_videos=False,
    )

    captured = capsys.readouterr().out
    assert _CountingDataset.last.saved == 1
    assert "episode_000001" in captured
    assert "[OK] Dataset root" in captured


def test_export_raises_instead_of_reporting_ok_when_every_bag_is_skipped(tmp_path, monkeypatch, capsys):
    """A run where every bag fails must fail loudly, not print [OK] over an empty dataset."""
    _install_export_stubs(monkeypatch)
    # frame_index regresses on the third entry, exactly what a session roll writes.
    bad_bag = _write_annex_b_episode(tmp_path / "episode_000001", [0, 1, 0])

    with pytest.raises(RuntimeError) as excinfo:
        export_bags_to_lerobot(
            [bad_bag],
            out_root=tmp_path / "output",
            use_videos=False,
        )

    assert "episode_000001" in str(excinfo.value)
    captured = capsys.readouterr().out
    assert "[OK] Dataset root" not in captured
    assert _CountingDataset.last.saved == 0


def test_export_summarizes_skipped_bags_when_some_succeed(tmp_path, monkeypatch, capsys):
    """A partially failed run still succeeds but must name every bag it dropped."""
    _install_export_stubs(monkeypatch)
    bad_bag = _write_annex_b_episode(tmp_path / "episode_000001", [0, 1, 0])
    good_bag = _write_annex_b_episode(tmp_path / "episode_000002", [0, 1, 2])

    export_bags_to_lerobot(
        [bad_bag, good_bag],
        out_root=tmp_path / "output",
        use_videos=False,
    )

    assert _CountingDataset.last.saved == 1
    captured = capsys.readouterr().out
    assert "[OK] Dataset root" in captured
    assert "1/2" in captured
    assert "episode_000001" in captured.split("[OK] Dataset root")[1]


def test_export_removes_the_dataset_it_created_when_every_bag_is_skipped(tmp_path, monkeypatch):
    """An all-failed run must not leave a zero-frame dataset behind for someone to trust."""
    _install_export_stubs(monkeypatch)
    bad_bag = _write_annex_b_episode(tmp_path / "episode_000001", [0, 1, 0])
    out_root = tmp_path / "output"

    with pytest.raises(RuntimeError):
        export_bags_to_lerobot(
            [bad_bag],
            out_root=out_root,
            use_videos=False,
        )

    assert not out_root.exists()


def test_export_keeps_a_preexisting_output_directory_when_every_bag_is_skipped(tmp_path, monkeypatch):
    """Cleanup must never delete a directory the run did not create."""
    _install_export_stubs(monkeypatch)
    bad_bag = _write_annex_b_episode(tmp_path / "episode_000001", [0, 1, 0])
    out_root = tmp_path / "output"
    out_root.mkdir()
    (out_root / "preexisting.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError):
        export_bags_to_lerobot(
            [bad_bag],
            out_root=out_root,
            use_videos=False,
        )

    assert (out_root / "preexisting.txt").read_text(encoding="utf-8") == "keep me"


def _public_conversion_meta(joints=None):
    """A recorded public snapshot generated by the real builder (round-trip valid)."""
    from robot_runtime.interface_description import description_digest
    from robot_runtime.model_metadata import build_public_conversion_metadata, joint_conversions_fingerprint

    joints = joints or ["1", "2", "3", "4", "5", "6"]

    def _entry(index: int, name: str) -> dict:
        span, offset = (100.0, 0.0) if name == joints[-1] else (200.0, -100.0)
        return {"min": -1.0 + index, "max": 1.0 + index, "span": span, "offset": offset}

    conversions = {
        "schema_version": 1,
        "joint_names": list(joints),
        "quantities": {name: "position" for name in joints},
        "modes": {
            mode: {name: _entry(index, name) for index, name in enumerate(joints)}
            for mode in ("degrees", "range_m100_100")
        },
    }
    conversions["modes"]["none"] = {}
    model = {
        "schema_version": 1,
        "authority": "calibration",
        "joint_groups": {"all": list(joints), "arm": joints[:5], "gripper": [joints[-1]], "base": []},
        "joint_limits": {name: {"min": -1.0 + index, "max": 1.0 + index} for index, name in enumerate(joints)},
        "home_positions": {},
        "frames": {"base_link": "base", "ee_link": "gripper"},
        "joint_conversions": conversions,
        "joint_conversions_fingerprint": joint_conversions_fingerprint(conversions),
    }
    qos = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5}
    doc = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "test", "runtime_name": "test_robot", "runtime_version": "1.0.0"},
        "execution": "physical",
        "interfaces": {
            "joint.state": {
                "capability": "joint.state",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/unit/joints",
                "message_type": "sensor_msgs/msg/JointState",
                "qos": qos,
                "joint_names": list(joints),
            }
        },
        "states": {},
        "model": model,
    }
    doc["digest"] = description_digest(doc)
    return build_public_conversion_metadata(
        model, list(joints), "degrees", description=doc, feature_names={"observation.state": list(joints)}
    )


def test_public_conversion_table_maps_field_selectors_to_joints():
    # The recorder stores joint names; the offline request carries message
    # field selectors. The table must be built through the selector -> joint
    # mapping, not by passing raw selectors to the model lookup.
    meta = _public_conversion_meta()
    table = _build_feature_conversion_table(
        feature_names=[f"position.{name}" for name in ["1", "2", "3", "4", "5", "6"]],
        conversion_meta=meta,
        feature_kind="state",
    )
    assert [row[0] for row in table] == [-1.0 + index for index in range(6)]
    assert [row[1] for row in table] == [1.0 + index for index in range(6)]


def test_public_conversion_table_detects_real_order_conflicts_through_mapping():
    meta = _public_conversion_meta()
    with pytest.raises(ValueError, match="feature order conflicts"):
        _build_feature_conversion_table(
            feature_names=[f"position.{name}" for name in ["2", "1", "3", "4", "5", "6"]],
            conversion_meta=meta,
            feature_kind="state",
        )


def test_public_conversion_table_maps_action_indices_to_joints():
    meta = _public_conversion_meta()
    table = _build_feature_conversion_table(
        feature_names=["action.0", "action.1", "action.2", "action.3", "action.4", "action.5"],
        conversion_meta=meta,
        feature_kind="action",
    )
    assert len(table) == 6
    assert [row[0] for row in table] == [-1.0 + index for index in range(6)]


# ---------------------------------------------------------------------------
# Dataset-embedded contract snapshots
# ---------------------------------------------------------------------------


def _snapshot_contract_document() -> dict:
    """A resolved contract mapping shaped like what the recorder embeds."""
    return {
        "name": "snapshot_contract",
        "version": 1,
        "rate_hz": 20.0,
        "max_duration_s": 30.0,
        "robot_type": "so_101",
        "timestamp_source": "header",
        "recording": {},
        "process": {},
        "tasks": [],
        "observations": [
            {
                "key": "observation.state",
                "topic": "/joint_states",
                "type": "sensor_msgs/msg/JointState",
                "selector": {"names": ["position.1", "position.2"]},
                "align": {"strategy": "hold", "stamp": "header", "tol_ms": 100},
            }
        ],
        "actions": [
            {
                "key": "action",
                "selector": {"names": ["action.0", "action.1"]},
                "safety_behavior": "hold",
                "publish": {
                    "topic": "/arm_position_controller/commands",
                    "type": "std_msgs/msg/Float64MultiArray",
                },
            }
        ],
    }


def _dataset_metadata_with_snapshot() -> dict:
    from robot_config.contract_utils import contract_fingerprint, contract_from_dict

    document = _snapshot_contract_document()
    contract = contract_from_dict(document)
    return {
        "contract": document,
        "contract_fingerprint": contract_fingerprint(contract),
    }


def test_contract_snapshot_round_trips_through_the_recorder_serializer():
    """contract_to_dict and contract_from_dict must preserve identity."""
    from robot_config.contract_utils import contract_fingerprint, contract_from_dict, contract_to_dict

    original = contract_from_dict(_snapshot_contract_document())
    restored = contract_from_dict(contract_to_dict(original))

    assert contract_fingerprint(restored) == contract_fingerprint(original)
    assert restored.rate_hz == original.rate_hz
    assert restored.robot_type == original.robot_type
    assert [obs.topic for obs in restored.observations] == [obs.topic for obs in original.observations]
    assert [act.publish_topic for act in restored.actions] == [act.publish_topic for act in original.actions]


def test_contract_snapshot_rebuilds_without_the_source_tree():
    contract = _contract_from_dataset_metadata(_dataset_metadata_with_snapshot())

    assert contract is not None
    assert contract.rate_hz == 20.0
    assert contract.observations[0].topic == "/joint_states"
    assert contract.actions[0].publish_topic == "/arm_position_controller/commands"


def test_contract_snapshot_rejects_a_mismatched_fingerprint():
    """A snapshot that no longer round-trips must not be silently used."""
    meta = _dataset_metadata_with_snapshot()
    meta["contract"]["observations"][0]["topic"] = "/somewhere_else"

    with pytest.raises(ValueError, match="does not match its fingerprint"):
        _contract_from_dataset_metadata(meta)


def _recorded_bag(tmp_path, name="dataset", *, mode="none", image_only=False):
    meta = _dataset_metadata_with_snapshot()
    if image_only:
        meta["contract"]["actions"] = []
        meta["contract"]["observations"] = [
            {
                "key": "observation.images.top",
                "topic": "/image",
                "type": "sensor_msgs/msg/Image",
                "image": {"resize": [16, 16]},
            }
        ]
        from robot_config.contract_utils import contract_fingerprint, contract_from_dict

        meta["contract_fingerprint"] = contract_fingerprint(contract_from_dict(meta["contract"]))
    conversion = {"norm_mode": mode}
    meta["lerobot"] = {"default_conversion_fingerprint": "conversion", "conversions": {"conversion": conversion}}
    bag = tmp_path / name / "episodes" / "episode_000001"
    bag.mkdir(parents=True)
    info = {
        "custom_data": {
            "ibrobot.contract_fingerprint": meta["contract_fingerprint"],
            "ibrobot.lerobot_conversion_fingerprint": "conversion",
        }
    }
    _write_recorded_metadata(bag, meta, info)
    return bag, meta, info


def _write_recorded_metadata(bag, meta, info):
    (bag.parent.parent / "dataset.yaml").write_text(yaml.safe_dump(meta))
    (bag / "metadata.yaml").write_text(yaml.safe_dump({"rosbag2_bagfile_information": info}))


@pytest.mark.parametrize("snapshot", [None, {}, [], "bad"])
def test_absent_or_malformed_contract_is_rejected(snapshot):
    with pytest.raises(ValueError, match="contract snapshot"):
        _contract_from_dataset_metadata({"contract": snapshot})


@pytest.mark.parametrize("field", ["name", "rate_hz", "timestamp_source", "observations", "actions", "tasks"])
def test_snapshot_required_fields_do_not_default(field):
    meta = _dataset_metadata_with_snapshot()
    del meta["contract"][field]
    with pytest.raises(ValueError, match=field):
        _contract_from_dataset_metadata(meta)


@pytest.mark.parametrize("rate", [0, -1, 20.5, float("nan"), float("inf"), True, "20"])
def test_invalid_snapshot_rate(rate):
    meta = _dataset_metadata_with_snapshot()
    meta["contract"]["rate_hz"] = rate
    with pytest.raises(ValueError, match="rate_hz"):
        _contract_from_dataset_metadata(meta)


def test_preflight_accepts_explicit_none_without_calibration(tmp_path):
    bag, _, _ = _recorded_bag(tmp_path)
    _, episodes = _preflight_bags([bag])
    assert episodes[0]["tables"] == {"observation.state": [], "action": []}


def test_image_only_needs_no_conversion_metadata(tmp_path):
    bag, meta, info = _recorded_bag(tmp_path, image_only=True)
    del meta["lerobot"]
    del info["custom_data"]["ibrobot.lerobot_conversion_fingerprint"]
    _write_recorded_metadata(bag, meta, info)
    _, episodes = _preflight_bags([bag])
    assert episodes[0]["tables"] == {}


@pytest.mark.parametrize("change", ["episode_contract", "conversion_fp", "lookup", "metadata", "mode"])
def test_second_bag_metadata_failure_precedes_output_creation(tmp_path, monkeypatch, change):
    first, _, _ = _recorded_bag(tmp_path, "first")
    second, meta, info = _recorded_bag(tmp_path, "second")
    if change == "episode_contract":
        info["custom_data"]["ibrobot.contract_fingerprint"] = "wrong"
    elif change == "conversion_fp":
        del info["custom_data"]["ibrobot.lerobot_conversion_fingerprint"]
    elif change == "lookup":
        meta["lerobot"]["conversions"] = {}
    elif change == "metadata":
        del meta["lerobot"]
    else:
        meta["lerobot"]["conversions"]["conversion"] = {}
    _write_recorded_metadata(second, meta, info)

    def fail_create(**kwargs):
        pytest.fail("output created before all episodes passed preflight")

    monkeypatch.setattr("dataset_tools.bag_to_lerobot.LeRobotDataset.create", fail_create)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=str(second)):
        export_bags_to_lerobot([first, second], out_root=out)
    assert not out.exists()


@pytest.mark.parametrize("field", ["description", "joint_names", "feature_names", "joint_conversions", "norm_mode"])
def test_incomplete_normalization_metadata_rejected(tmp_path, field):
    bag, meta, info = _recorded_bag(tmp_path)
    conversion = _public_conversion_meta()
    fingerprint = conversion["conversion_fingerprint"]
    del conversion[field]
    # Even a legacy calibration block must not provide a downgrade path.
    conversion["calibration"] = {"1": {"range_min": 1000, "range_max": 3000}}
    meta["lerobot"]["conversions"] = {fingerprint: conversion}
    info["custom_data"]["ibrobot.lerobot_conversion_fingerprint"] = fingerprint
    _write_recorded_metadata(bag, meta, info)
    with pytest.raises(ValueError):
        _preflight_bags([bag])


def test_mixed_rates_rejected_even_though_fingerprint_excludes_rate(tmp_path):
    first, _, _ = _recorded_bag(tmp_path, "first")
    second, meta, info = _recorded_bag(tmp_path, "second")
    meta["contract"]["rate_hz"] = 30
    _write_recorded_metadata(second, meta, info)
    with pytest.raises(ValueError, match="incompatible contract snapshots"):
        _preflight_bags([first, second])


def test_removed_robot_config_cli_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bag_to_lerobot", "--bag", "bag", "--out", "out", "--robot-config", "robot.yaml"])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_normalization_keeps_velocity_native_and_rejects_missing_positions():
    meta = _public_conversion_meta()
    names = [f"position.{i}" for i in range(1, 7)] + ["velocity.1"]
    table = _build_feature_conversion_table(names, meta, feature_kind="state")
    assert table[-1] == (0.0, 1.0, 1.0, 0.0)
    assert table[0] != table[-1]
    with pytest.raises(ValueError, match="calibration mapping"):
        _build_feature_conversion_table(names + ["position.missing"], meta, feature_kind="state")
    with pytest.raises(ValueError, match="calibration mapping"):
        _build_feature_conversion_table(["action.99"], meta, feature_kind="action")


@pytest.mark.parametrize("joints", [["1", "2", "3", "4", "5", "6"], ["pan", "lift", "elbow", "wrist", "roll", "grip"]])
def test_complete_normalization_snapshot_preflight(tmp_path, joints):
    bag, meta, info = _recorded_bag(tmp_path)
    conversion = _public_conversion_meta(joints)
    fingerprint = conversion["conversion_fingerprint"]
    meta["contract"]["observations"][0]["selector"]["names"] = [f"position.{name}" for name in joints]
    meta["contract"]["actions"][0]["selector"]["names"] = [f"action.{i}" for i in range(6)]
    from robot_config.contract_utils import contract_fingerprint, contract_from_dict

    meta["contract_fingerprint"] = contract_fingerprint(contract_from_dict(meta["contract"]))
    info["custom_data"]["ibrobot.contract_fingerprint"] = meta["contract_fingerprint"]
    info["custom_data"]["ibrobot.lerobot_conversion_fingerprint"] = fingerprint
    meta["lerobot"]["conversions"] = {fingerprint: conversion}
    _write_recorded_metadata(bag, meta, info)
    _, episodes = _preflight_bags([bag])
    tables = episodes[0]["tables"]
    assert tables["observation.state"] == tables["action"]
    assert len(tables["action"]) == 6
    assert tables["action"][0] != (0.0, 1.0, 1.0, 0.0)


def test_none_does_not_bypass_malformed_public_snapshot(tmp_path):
    bag, meta, info = _recorded_bag(tmp_path)
    meta["lerobot"]["conversions"]["conversion"]["description"] = {}
    _write_recorded_metadata(bag, meta, info)
    with pytest.raises(ValueError, match="public conversion metadata"):
        _preflight_bags([bag])


def test_incomplete_second_normalization_snapshot_prevents_output(tmp_path, monkeypatch):
    first, _, _ = _recorded_bag(tmp_path, "first")
    second, meta, info = _recorded_bag(tmp_path, "second", mode="degrees")
    _write_recorded_metadata(second, meta, info)

    def fail_create(**kwargs):
        pytest.fail("created output for incomplete normalization metadata")

    monkeypatch.setattr("dataset_tools.bag_to_lerobot.LeRobotDataset.create", fail_create)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="public conversion metadata"):
        export_bags_to_lerobot([first, second], out_root=out)
    assert not out.exists()


def test_malformed_episode_yaml_reports_bag_context(tmp_path):
    bag, _, _ = _recorded_bag(tmp_path)
    (bag / "metadata.yaml").write_text("broken: [")
    with pytest.raises(ValueError, match=str(bag)):
        _preflight_bags([bag])
