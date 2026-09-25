import hashlib
import json
import tarfile

import pytest

from robot_calibration.capture import (
    REQUIRED_SCENES,
    REQUIRED_TOPICS,
    CaptureError,
    finalize_capture,
    import_legacy_capture,
)
from robot_calibration.store import ArtifactStore, StoreError

# Import rather than transcribe: every fixture in this file is built from these
# two tables, so a verbatim copy let the whole file drift silently when the
# production tuples changed.
SCENES = REQUIRED_SCENES
TOPICS = REQUIRED_TOPICS


def _scene(scene_id: str, *, stationary: bool = True) -> dict:
    return {
        "scene_id": scene_id,
        "role": "test" if scene_id.endswith("test") else "fit",
        "duration_s": 10.0,
        "stationary_windows": 1 if stationary else 0,
        "topics": {topic: {"type": "test/Type", "count": 10, "rate_hz": 10.0} for topic in TOPICS},
        "tf_edges": ["base_link -> body", "body -> camera_front_link"],
        "files": [
            {"path": f"{scene_id}/metadata.yaml"},
            {"path": f"{scene_id}/{scene_id}_0.mcap"},
        ],
    }


def test_finalize_capture_accepts_positive_slow_board_scene_duration(tmp_path):
    scenes = [_scene(scene_id) for scene_id in SCENES]
    scenes[0]["duration_s"] = 8.0
    source = tmp_path / "source"
    _write_scene_files(source, scenes)

    output = finalize_capture(tmp_path, "capture-slow", scenes, {}, source=source)

    assert output.is_dir()


def _write_scene_files(root, scenes):
    for scene in scenes:
        for entry in scene["files"]:
            path = root / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(entry["path"] + "\n", encoding="ascii")


def test_finalize_capture_writes_sealed_manifest_and_rejects_incomplete_scene(tmp_path):
    scenes = [_scene(scene_id) for scene_id in SCENES]
    source = tmp_path / "source"
    _write_scene_files(source, scenes)
    output = finalize_capture(tmp_path, "capture-001", scenes, {"lidar": "L1", "camera": "C1"}, source=source)

    assert (output / "FINALIZED").read_text() == "\n"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["capture_id"] == "capture-001"
    assert manifest["sealed"] is True
    assert set(manifest["scenes"]) == set(SCENES)
    assert (output / "scene-01" / "scene-01_0.mcap").is_file()
    assert len(manifest["files"]) == 8
    assert all(set(entry) == {"path", "size", "sha256"} for entry in manifest["files"])

    with pytest.raises(CaptureError, match="stationary"):
        finalize_capture(
            tmp_path,
            "capture-002",
            [_scene("scene-01", stationary=False)] + scenes[1:],
            {},
            source=source,
        )

    with pytest.raises(CaptureError, match="already exists"):
        finalize_capture(tmp_path, "capture-001", scenes, {}, source=source)


def test_capture_export_import_is_content_addressed(tmp_path):
    scenes = [_scene(scene_id) for scene_id in SCENES]
    source = tmp_path / "source"
    _write_scene_files(source, scenes)
    capture = finalize_capture(tmp_path / "captures", "capture-002", scenes, {}, source=source)
    first_archive = tmp_path / "out" / "capture.tar"
    second_archive = tmp_path / "out" / "capture-again.tar"

    digest = ArtifactStore.export_capture(capture, first_archive)
    second_digest = ArtifactStore.export_capture(capture, second_archive)
    imported = ArtifactStore.import_capture(first_archive, tmp_path / "imported")

    assert digest == imported["sha256"]
    assert digest == second_digest
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert imported["path"].endswith("capture-002")
    imported_root = tmp_path / "imported" / "capture-002"
    assert (imported_root / "scene-04-test" / "scene-04-test_0.mcap").is_file()

    (imported_root / "scene-01" / "scene-01_0.mcap").write_text("tampered\n", encoding="ascii")
    with pytest.raises(StoreError, match="sha256"):
        ArtifactStore.verify_capture(imported_root)


def test_capture_import_rejects_archive_path_traversal(tmp_path):
    archive = tmp_path / "malicious.tar"
    payload = tmp_path / "payload"
    payload.write_text("bad\n", encoding="ascii")
    with tarfile.open(archive, "w") as output:
        output.add(payload, arcname="../outside")

    with pytest.raises(StoreError, match="unsafe"):
        ArtifactStore.import_capture(archive, tmp_path / "imported")


def test_capture_rejects_unsafe_identifier(tmp_path):
    scenes = [_scene(scene_id) for scene_id in SCENES]
    source = tmp_path / "source"
    _write_scene_files(source, scenes)

    with pytest.raises(CaptureError, match="capture_id"):
        finalize_capture(tmp_path / "captures", "../outside", scenes, {}, source=source)

    capture = finalize_capture(tmp_path / "captures", "capture-safe", scenes, {}, source=source)
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["capture_id"] = "../outside"
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256")
    canonical = (json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="ascii"
    )
    with pytest.raises(StoreError, match="capture_id"):
        ArtifactStore.verify_capture(capture)


def test_import_legacy_capture_validates_manifest_and_duration(tmp_path):
    legacy = tmp_path / "legacy"
    files = []
    for scene_id in SCENES:
        metadata = legacy / scene_id / "metadata.yaml"
        storage = legacy / scene_id / f"{scene_id}_0.mcap"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(
            "rosbag2_bagfile_information:\n"
            "  storage_identifier: mcap\n"
            "  duration: {nanoseconds: 10000000000}\n"
            f"  relative_file_paths: [{scene_id}_0.mcap]\n"
            "  topics_with_message_count:\n"
            + "".join(
                f"    - topic_metadata: {{name: {topic}, type: test/Type}}\n      message_count: 10\n"
                for topic in TOPICS
            ),
            encoding="utf-8",
        )
        storage.write_bytes(scene_id.encode("ascii"))
        for path in (metadata, storage):
            content = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(legacy).as_posix(),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    (legacy / "transfer_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_id": "calib-19700101-000154",
                "bundle_type": "fast_calib_capture",
                "files": files,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="ascii",
    )

    imported = import_legacy_capture(legacy, tmp_path / "captures", devices={"lidar": "L1", "camera": "C1"})

    assert imported.name == "calib-19700101-000154"
    assert json.loads((imported / "manifest.json").read_text())["legacy_transfer_manifest_sha256"]

    metadata = legacy / "scene-01" / "metadata.yaml"
    metadata.write_text(metadata.read_text().replace("10000000000", "8000000000"), encoding="utf-8")
    with pytest.raises(CaptureError, match="sha256 mismatch"):
        import_legacy_capture(legacy, tmp_path / "other", devices={})


def test_artifact_install_and_rollback_preserve_immutable_revisions(tmp_path):
    store = ArtifactStore(tmp_path / "calibration")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"artifact_id":"a1","state":"candidate"}\n')
    second.write_text('{"artifact_id":"a2","state":"approved"}\n')

    store.install(first)
    store.install(second)
    assert store.current_artifact()["artifact_id"] == "a2"
    store.rollback("a1")
    assert store.current_artifact()["artifact_id"] == "a1"

    with pytest.raises(StoreError, match="immutable"):
        store.install(first)
