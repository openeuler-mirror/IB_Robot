import asyncio
import json
import socket
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import websockets
from scipy.spatial.transform import Rotation

from robot_teleop.phone.config_phone import PhoneConfig
from robot_teleop.phone.web_phone import WebPhone


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(
    http_port=None,
    websocket_port=None,
    stale_s=0.2,
    optical_flow_fallback_enabled=False,
    binary_protocol_enabled=True,
):
    http_port = http_port or _free_port()
    websocket_port = websocket_port or _free_port()
    while websocket_port == http_port:
        websocket_port = _free_port()
    return PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "optical_flow_fallback_enabled": optical_flow_fallback_enabled,
            "web": {
                "bind_address": "127.0.0.1",
                "http_port": http_port,
                "websocket_port": websocket_port,
                "command_stale_s": stale_s,
                "binary_protocol_enabled": binary_protocol_enabled,
                "tls": {"enabled": False},
            },
        }
    )


def _message(move=False):
    return {
        "pose": np.eye(4).reshape(-1).tolist(),
        "linearVelocity": [0.1, 0.2, 0.3],
        "angularVelocity": [0.4, 0.5, 0.6],
        "trackingMode": "optical_flow",
        "trackingQuality": 1.0,
        "move": move,
        "scale": 1.0,
        "platform": "android",
    }


def _binary_frame(version=1):
    frame = bytearray(99)
    frame[0] = version
    frame[1] = 0x21
    struct.pack_into("<16f", frame, 2, *np.eye(4).reshape(-1))
    struct.pack_into("<3f", frame, 66, 0.1, 0.2, 0.3)
    struct.pack_into("<3f", frame, 78, 0.4, 0.5, 0.6)
    struct.pack_into("<f", frame, 90, 1.0)
    struct.pack_into("<f", frame, 94, 0.9)
    frame[98] = 2
    return bytes(frame)


def _write_web_assets(web_root: Path):
    (web_root / "web_teleop.html").write_text("<html>webphone</html>", encoding="utf-8")
    (web_root / "optical_flow_worker.js").write_text("self.onmessage = () => {};", encoding="utf-8")
    (web_root / "three.min.js").write_text("window.THREE = {};", encoding="utf-8")


def test_binary_protocol_version_and_payload():
    decoded = WebPhone.parse_binary_message(_binary_frame())
    assert decoded["move"] is True
    assert decoded["trackingMode"] == "ar_6dof"
    assert decoded["platform"] == "android"
    with pytest.raises(ValueError, match="version"):
        WebPhone.parse_binary_message(_binary_frame(version=2))
    with pytest.raises(ValueError, match="99"):
        WebPhone.parse_binary_message(b"short")


def test_non_finite_pose_is_rejected():
    phone = WebPhone(_config())
    message = _message()
    message["pose"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        phone._accept_message(message, time.monotonic())


@pytest.mark.parametrize(
    ("pose_update", "match"),
    [
        ({15: 2.0}, "homogeneous"),
        ({0: 2.0}, "orthonormal"),
    ],
)
def test_malformed_pose_transform_is_rejected(pose_update, match):
    phone = WebPhone(_config())
    message = _message()
    for index, value in pose_update.items():
        message["pose"][index] = value
    with pytest.raises(ValueError, match=match):
        phone._accept_message(message, time.monotonic())


def test_json_boolean_strings_are_rejected():
    phone = WebPhone(_config())
    message = _message()
    message["move"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        phone._accept_message(message, time.monotonic())


def test_tracking_quality_outside_unit_range_is_rejected():
    phone = WebPhone(_config())
    message = _message()
    message["trackingQuality"] = 1.1
    with pytest.raises(ValueError, match="range 0..1"):
        phone._accept_message(message, time.monotonic())


def test_home_event_is_latched_until_phone_device_consumes_it():
    phone = WebPhone(_config())
    home_message = _message(move=False)
    home_message["goHome"] = True
    phone._accept_message(home_message, time.monotonic())
    phone._accept_message(_message(move=False), time.monotonic())

    first_action = phone.get_action()
    second_action = phone.get_action()

    assert first_action["phone.raw_inputs"]["goHome"] is True
    assert second_action["phone.raw_inputs"]["goHome"] is False


def test_safety_stop_discards_unconsumed_home_event():
    phone = WebPhone(_config())
    home_message = _message(move=False)
    home_message["goHome"] = True
    phone._accept_message(home_message, time.monotonic())

    phone.require_release("emergency stop")

    assert phone.consume_stop_request() == "emergency stop"
    assert phone.get_action() == {}


def test_stale_stream_requests_stop_and_requires_release():
    phone = WebPhone(_config(stale_s=0.01, optical_flow_fallback_enabled=True))
    phone._accept_message(_message(move=True), time.monotonic() - 1.0)
    assert phone.consume_stop_request().startswith("command stream stale (")

    phone._accept_message(_message(move=True), time.monotonic())
    assert phone.get_action()["phone.enabled"] is False
    phone._accept_message(_message(move=False), time.monotonic())
    assert phone.get_action()["phone.enabled"] is False
    phone._accept_message(_message(move=True), time.monotonic())
    action = phone.get_action()
    assert action["phone.enabled"] is True
    assert "phone.linear_vel" not in action


def test_webphone_rejects_non_pose_tracking_until_release():
    phone = WebPhone(_config())
    phone._accept_message(_message(move=True), time.monotonic())
    assert phone.consume_stop_request() == "WebPhone requires ar_6dof or enabled optical-flow fallback"
    assert phone.get_action()["phone.enabled"] is False

    phone._accept_message(_message(move=False), time.monotonic())
    ar_message = _message(move=True)
    ar_message["trackingMode"] = "ar_6dof"
    phone._accept_message(ar_message, time.monotonic())
    assert phone.get_action()["phone.enabled"] is True


def test_webphone_accepts_enabled_optical_flow_fallback():
    phone = WebPhone(_config(optical_flow_fallback_enabled=True))
    phone._accept_message(_message(move=True), time.monotonic())

    action = phone.get_action()

    assert phone.consume_stop_request() is None
    assert action["phone.enabled"] is True
    assert action["phone.tracking_mode"] == "optical_flow"


def test_optical_flow_uses_the_same_absolute_pose_contract_as_ar():
    phone = WebPhone(_config(optical_flow_fallback_enabled=True))
    pose = np.eye(4)
    release = dict(_message(move=False), pose=pose.reshape(-1).tolist())
    enabled = dict(release, move=True)
    moved_pose = pose.copy()
    moved_pose[:3, 3] = [0.03, -0.02, 0.01]
    moved = dict(enabled, pose=moved_pose.reshape(-1).tolist(), linearVelocity=[9.0, 9.0, 9.0])

    phone._accept_message(release, time.monotonic())
    phone.get_action()
    phone._accept_message(enabled, time.monotonic())
    phone.get_action()
    phone._accept_message(moved, time.monotonic() + 0.02)
    action = phone.get_action()

    assert np.allclose(action["phone.pos"], [0.03, -0.02, 0.01])


def test_optical_flow_rotation_applies_camera_offset_lever_arm_once():
    phone = WebPhone(_config(optical_flow_fallback_enabled=True))
    clutch_pose = np.eye(4)
    clutch_message = dict(_message(move=True), pose=clutch_pose.reshape(-1).tolist())

    phone._accept_message(dict(clutch_message, move=False), time.monotonic())
    phone.get_action()
    phone._accept_message(clutch_message, time.monotonic() + 0.01)
    phone.get_action()

    rolled_pose = np.eye(4)
    rolled_pose[:3, :3] = Rotation.from_euler("z", 60.0, degrees=True).as_matrix()
    phone._accept_message(
        dict(clutch_message, pose=rolled_pose.reshape(-1).tolist()),
        time.monotonic() + 0.02,
    )
    action = phone.get_action()

    offset = phone.phone_config.camera_offset
    rolled = Rotation.from_euler("z", 60.0, degrees=True)
    clutch_offset = offset.copy()
    clutch_offset[1] = 0.0
    rotated_offset = rolled.apply(offset)
    rotated_offset[1] = 0.0
    expected_position = clutch_offset - rotated_offset
    assert np.allclose(action["phone.pos"], expected_position, atol=1e-9)
    assert np.isclose(action["phone.pos"][1], 0.0, atol=1e-9)
    assert np.isclose(np.linalg.norm(action["phone.rot"].as_rotvec()), np.deg2rad(60.0))


def test_optical_flow_pitch_offset_does_not_create_vertical_position():
    phone = WebPhone(_config(optical_flow_fallback_enabled=True))
    clutch_pose = np.eye(4)
    clutch_message = dict(_message(move=True), pose=clutch_pose.reshape(-1).tolist())

    phone._accept_message(dict(clutch_message, move=False), time.monotonic())
    phone.get_action()
    phone._accept_message(clutch_message, time.monotonic() + 0.01)
    phone.get_action()

    pitched_pose = np.eye(4)
    pitched = Rotation.from_euler("x", -60.0, degrees=True)
    pitched_pose[:3, :3] = pitched.as_matrix()
    phone._accept_message(
        dict(clutch_message, pose=pitched_pose.reshape(-1).tolist()),
        time.monotonic() + 0.02,
    )
    action = phone.get_action()

    assert np.isclose(action["phone.pos"][1], 0.0, atol=1e-9)
    assert not np.allclose(action["phone.pos"], np.zeros(3), atol=1e-9)


def test_optical_flow_quality_loss_stops_after_stale_window_and_requires_release():
    phone = WebPhone(_config(stale_s=0.2, optical_flow_fallback_enabled=True))
    message = dict(_message(move=True), trackingQuality=0.1)
    now = time.monotonic()
    phone._accept_message(message, now)

    action = phone.get_action()
    assert action["phone.enabled"] is True
    assert "phone.linear_vel" not in action

    phone._accept_message(message, now + 0.1)
    phone._accept_message(message, now + 0.21)
    assert phone.consume_stop_request() == "optical-flow tracking lost"
    assert phone.get_action()["phone.enabled"] is False

    recovered = dict(message, trackingQuality=1.0)
    phone._accept_message(recovered, now + 0.22)
    assert phone.get_action()["phone.enabled"] is False
    phone._accept_message(dict(recovered, move=False), now + 0.23)
    phone._accept_message(recovered, now + 0.24)
    assert phone.get_action()["phone.enabled"] is True


def test_yaw_alignment_uses_arcore_viewer_back_axis_for_upright_capture():
    rotation = Rotation.from_euler("y", 90.0, degrees=True)

    alignment = WebPhone._yaw_alignment(rotation)

    viewer_back_world = rotation.apply([0.0, 0.0, 1.0])
    world_up = np.array([0.0, 1.0, 0.0])
    viewer_back_horizontal = viewer_back_world - np.dot(viewer_back_world, world_up) * world_up
    viewer_back_horizontal /= np.linalg.norm(viewer_back_horizontal)
    assert np.allclose(alignment @ viewer_back_horizontal, [0.0, 0.0, 1.0], atol=1e-7)
    assert np.allclose(alignment @ world_up, [0.0, 1.0, 0.0], atol=1e-7)


def test_yaw_alignment_remains_stable_at_ninety_degree_viewer_roll():
    rolled = Rotation.from_euler("z", 90.0, degrees=True)

    alignment = WebPhone._yaw_alignment(rolled)

    assert np.allclose(alignment, np.eye(3), atol=1e-7)


def test_tracking_mode_change_while_held_requires_release():
    phone = WebPhone(_config(optical_flow_fallback_enabled=True))
    phone._accept_message(_message(move=True), time.monotonic())
    phone.get_action()
    ar_message = dict(_message(move=True), trackingMode="ar_6dof")

    phone._accept_message(ar_message, time.monotonic())

    assert phone.consume_stop_request() == "tracking mode changed"
    assert phone.get_action()["phone.enabled"] is False


def test_tracking_pose_jump_stops_and_requires_release():
    phone = WebPhone(_config())
    message = dict(_message(move=False), trackingMode="ar_6dof")
    now = time.monotonic()
    phone._accept_message(message, now)
    phone._accept_message(dict(message, move=True), now + 0.01)
    assert phone.get_action()["phone.enabled"] is True

    jumped_pose = np.eye(4)
    jumped_pose[0, 3] = 0.3
    phone._accept_message(dict(message, move=True, pose=jumped_pose.reshape(-1).tolist()), now + 0.02)

    assert phone.consume_stop_request().startswith("tracking pose jumped (")
    assert phone.get_action()["phone.enabled"] is False
    phone._accept_message(dict(message, move=True), now + 0.03)
    assert phone.get_action()["phone.enabled"] is False
    phone._accept_message(message, now + 0.04)
    phone._accept_message(dict(message, move=True), now + 0.05)
    assert phone.get_action()["phone.enabled"] is True


def test_reset_requests_stop_and_held_deadman_cannot_resume():
    phone = WebPhone(_config())
    ar_message = _message(move=True)
    ar_message["trackingMode"] = "ar_6dof"
    phone._accept_message(ar_message, time.monotonic())

    reset_message = dict(ar_message, reset=True)
    phone._accept_message(reset_message, time.monotonic())
    assert phone.consume_stop_request() == "origin reset"
    assert phone.get_action()["phone.enabled"] is False

    phone._accept_message(ar_message, time.monotonic())
    assert phone.get_action()["phone.enabled"] is False
    released = dict(ar_message, move=False, reset=False)
    phone._accept_message(released, time.monotonic())
    phone._accept_message(ar_message, time.monotonic())
    assert phone.get_action()["phone.enabled"] is True


def test_home_release_gate_does_not_stop_the_active_home_motion():
    phone = WebPhone(_config())
    ar_message = _message(move=True)
    ar_message["trackingMode"] = "ar_6dof"
    phone._accept_message(ar_message, time.monotonic())

    phone.require_release("go-home requested", request_stop=False)

    assert phone.consume_stop_request() is None
    assert phone.get_action()["phone.enabled"] is False


def test_pose_free_release_frame_is_accepted_but_enabled_frame_is_rejected():
    phone = WebPhone(_config())
    phone._accept_message({"move": False, "reservedButtonA": True}, time.monotonic())
    action = phone.get_action()
    assert action["phone.enabled"] is False
    assert action["phone.raw_inputs"]["reservedButtonA"] is True
    with pytest.raises(ValueError, match="fresh pose"):
        phone._accept_message({"move": True}, time.monotonic())


class _RejectedClient:
    def __init__(self, *, origin, host):
        self.request_headers = {"Origin": origin, "Host": host}
        self.closed = None

    async def close(self, code, reason):
        self.closed = (code, reason)


class _MessageClient:
    def __init__(self, messages, *, origin, host):
        self.messages = iter(messages)
        self.request_headers = {"Origin": origin, "Host": host}
        self.closed = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self, code, reason):
        self.closed = (code, reason)


def test_second_client_is_rejected():
    config = _config()
    phone = WebPhone(config)
    phone._active_client = object()
    client = _RejectedClient(
        origin=f"http://127.0.0.1:{config.web.http_port}",
        host=f"127.0.0.1:{config.web.websocket_port}",
    )
    asyncio.run(phone._handle_websocket(client))
    assert client.closed == (1008, "another WebPhone client owns control")


def test_websocket_origin_must_match_served_page_host_and_port():
    config = _config()
    phone = WebPhone(config)
    valid = _MessageClient(
        [],
        origin=f"http://127.0.0.1:{config.web.http_port}",
        host=f"127.0.0.1:{config.web.websocket_port}",
    )
    invalid = _MessageClient(
        [],
        origin="https://example.com",
        host=f"127.0.0.1:{config.web.websocket_port}",
    )

    assert phone._origin_allowed(valid) is True
    assert phone._origin_allowed(invalid) is False
    asyncio.run(phone._handle_websocket(invalid))
    assert invalid.closed == (1008, "WebPhone origin is not allowed")


def test_binary_message_is_closed_when_protocol_is_disabled():
    config = _config(binary_protocol_enabled=False)
    phone = WebPhone(config)
    client = _MessageClient(
        [_binary_frame()],
        origin=f"http://127.0.0.1:{config.web.http_port}",
        host=f"127.0.0.1:{config.web.websocket_port}",
    )

    asyncio.run(phone._handle_websocket(client))

    assert client.closed == (1003, "WebPhone binary protocol is disabled")


def test_unknown_binary_protocol_version_closes_and_releases_client():
    config = _config()
    phone = WebPhone(config)
    client = _MessageClient(
        [_binary_frame(version=2)],
        origin=f"http://127.0.0.1:{config.web.http_port}",
        host=f"127.0.0.1:{config.web.websocket_port}",
    )

    asyncio.run(phone._handle_websocket(client))

    assert client.closed == (1003, "invalid WebPhone binary protocol frame")
    assert phone._active_client is None
    assert phone.consume_stop_request() == "client disconnected"


def test_transport_serves_installed_style_asset_and_restarts(tmp_path: Path):
    (tmp_path / "web_teleop.html").write_text("<html>webphone</html>", encoding="utf-8")
    worker_source = tmp_path.parent / f"{tmp_path.name}-optical-flow-worker.js"
    worker_source.write_text("self.onmessage = () => {};", encoding="utf-8")
    (tmp_path / "optical_flow_worker.js").symlink_to(worker_source)
    (tmp_path / "three.min.js").write_text("window.THREE = {};", encoding="utf-8")
    config = _config()
    phone = WebPhone(config, web_root=tmp_path)

    for _ in range(2):
        assert phone.connect() is True
        api_url = f"http://127.0.0.1:{config.web.http_port}/api/config"
        with urllib.request.urlopen(api_url, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["websocket_port"] == config.web.websocket_port
        assert payload["binary_protocol_version"] == 1
        assert "input_mode" not in payload
        page_url = f"http://127.0.0.1:{config.web.http_port}/web_teleop.html"
        with urllib.request.urlopen(page_url, timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
        worker_url = f"http://127.0.0.1:{config.web.http_port}/optical_flow_worker.js"
        with urllib.request.urlopen(worker_url, timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == b"self.onmessage = () => {};"
        three_url = f"http://127.0.0.1:{config.web.http_port}/three.min.js"
        with urllib.request.urlopen(three_url, timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert response.read() == b"window.THREE = {};"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"http://127.0.0.1:{config.web.http_port}/missing.js", timeout=2)
        assert error.value.code == 404
        phone.disconnect()
        assert phone.is_connected is False


def test_live_websocket_accepts_only_the_page_origin(tmp_path: Path):
    _write_web_assets(tmp_path)
    config = _config()
    phone = WebPhone(config, web_root=tmp_path)

    async def send_from_page_origin():
        uri = f"ws://127.0.0.1:{config.web.websocket_port}"
        origin = f"http://127.0.0.1:{config.web.http_port}"
        async with websockets.connect(uri, origin=origin) as websocket:
            await websocket.send(json.dumps(_message(move=False)))
            await asyncio.sleep(0.01)
            return phone.get_action()

    try:
        assert phone.connect() is True
        action = asyncio.run(send_from_page_origin())
        assert action["phone.enabled"] is False
    finally:
        phone.disconnect()


def test_web_optical_fallback_uses_system_fused_device_orientation():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")
    worker = (web_root / "optical_flow_worker.js").read_text(encoding="utf-8")

    assert "deviceOrientationToViewerQuaternion" in page
    assert "const orientationAtCapture = rotationOnly(currentRotationMatrix)" in page
    assert "optical_flow_worker.js?v=15" in page
    assert "RelativeOrientationSensor" not in page
    assert "imuOrientation" not in page
    assert "DeviceMotionEvent" not in page
    assert "系统融合姿态" in page
    assert "DeviceMotion.rotationRate" not in worker
    assert "predictRotationPoints" in worker
    assert "event.data.orientation" in worker
    assert "yawModelUsed" in page
    assert "yawModelUsed" in worker
    assert "rotationFallbackUsed" not in page
    assert "rotationFallbackUsed" not in worker
    assert 'src="/three.min.js"' in page


def test_web_page_warns_about_trusted_internal_network_deployment():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert "仅限受信内部网络" in page
    assert "禁止公网映射、云隧道、访客 Wi-Fi 和不可信 VPN" in page
    assert "cdn.jsdelivr.net" not in page
    init_source = page[
        page.index("async function init()") : page.index(
            "document.addEventListener", page.index("async function init()")
        )
    ]
    assert init_source.index("fetchConfig()") < init_source.index("connectWebSocket()")
    assert "if (!initResults[1])" in init_source
    assert "配置加载失败，请刷新页面重试" in page
    assert "Velocity速度" not in page


def test_web_page_checks_required_tracking_dependencies_without_blocking_optional_features():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert 'id="arCapabilityChecks"' in page
    assert 'id="opticalCapabilityChecks"' in page
    assert 'id="arCapabilitySummary"' in page
    assert 'id="opticalCapabilitySummary"' in page
    assert page.count('class="capability-details" hidden') == 2
    assert "window.isSecureContext === true" in page
    assert "typeof navigator.xr !== 'undefined'" in page
    assert "isSessionSupported('immersive-ar')" in page
    assert "label: '相机/IMU空间追踪', required: true" in page
    assert "label: 'DOM叠加控制', required: true" in page
    assert "label: '地面坐标', required: false" in page
    assert "label: '摄像头接口', required: true" in page
    assert "label: '摄像头权限', required: true" in page
    assert "label: 'Canvas/Worker', required: true" in page
    assert "label: 'DeviceOrientation', required: true" in page
    assert "requiredCapabilitiesReady(requiredMode)" in page
    assert "setCapability('ar', 'spatialTracking', 'ok'" in page
    assert "setCapability('optical', 'cameraPermission', 'ok'" in page
    assert "setCapability('optical', 'orientation', 'ok'" in page
    assert "details.hidden = failures.length === 0" in page
    assert "兼容性问题：" in page
    assert "兼容性提示：" in page
    assert "低速、小范围使用；旋转较稳，平移可能冻结或漂移。" in page
    assert "Chrome 不自带 AR 运行时" in page

    ar_start = page[page.index("start: async function()") : page.index("stop: function()")]
    assert "requiredFeatures:" not in ar_start
    assert "optionalFeatures: ['dom-overlay', 'local-floor']" in ar_start
    assert "domOverlay: { root: document.getElementById('arOverlay') }" in ar_start
    assert "setReferenceSpaceType('local')" in ar_start
    assert "this.session.domOverlayState" in ar_start
    assert "DomOverlayUnavailableError" in ar_start
    assert "requestReferenceSpace('local-floor')" in ar_start


def test_web_page_keeps_unsupported_ar_entry_disabled():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert "function arEntryAllowed()" in page
    assert "arSupported === true" in page
    assert "!arRuntimeRejected" in page
    assert "arToggleBtn.disabled = !arReady || arSessionManager.isStarting" in page
    assert "arToggleBtn.textContent = 'AR 不可用'" in page
    assert "arToggleBtn.textContent = 'AR 运行时不可用'" in page
    assert "arToggleBtn.textContent = 'AR 控制界面不可用'" in page
    assert "华为 AR Engine 不会自动接入 Chrome" in page

    ar_start = page[page.index("start: async function()") : page.index("stop: function()")]
    assert "if (!arEntryAllowed())" in ar_start
    assert "navigator.xr.requestSession" in ar_start
    assert ar_start.index("if (!arEntryAllowed())") < ar_start.index("navigator.xr.requestSession")
    assert "arRuntimeRejected = true" in ar_start
    assert "系统空间追踪运行时不支持该会话" in ar_start


def test_web_page_home_releases_deadman_before_requesting_motion():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")
    handler_start = page.index("homeButtonEl.addEventListener")
    handler = page[handler_start : page.index("debugToggleButton.addEventListener", handler_start)]

    assert handler.index("resetControls()") < handler.index("sendImmediateMessage({ goHome: true })")
    assert "回零中，完成后请重新按住使能" in handler


def test_web_page_sends_safe_idle_heartbeat_before_tracking():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")
    heartbeat = page[
        page.index("function sendControlHeartbeat()") : page.index(
            "setInterval(sendControlHeartbeat", page.index("function sendControlHeartbeat()")
        )
    ]

    assert "trackingMode: 'disabled'" in heartbeat
    assert "trackingQuality: 0" in heartbeat
    assert "linearVelocity: [0, 0, 0]" in heartbeat
    assert "angularVelocity: [0, 0, 0]" in heartbeat


def test_web_page_avoids_false_mobile_blur_release_and_supplements_optical_heartbeat():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert "window.addEventListener('blur'" not in page
    assert "document.addEventListener('visibilitychange'" in page
    assert "window.addEventListener('pagehide', releaseForPageLifecycle)" in page
    assert "document.addEventListener('freeze', releaseForPageLifecycle)" in page

    worker_result = page[
        page.index("_handleWorkerMessage: function(message)") : page.index("resetTrackingBaseline: function()")
    ]
    assert "cameraEnabled && trackingMode === 'optical_flow'" in worker_result
    assert "sendControlHeartbeat()" in worker_result


def test_optical_mode_reuses_the_ar_overlay_layout():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert "body.tracking-active .container" in page
    assert "flex-direction: column-reverse" in page
    assert "body.tracking-active .controls" in page
    assert "body.tracking-active .scale-control" in page
    assert "body.tracking-active .secondary-controls" in page
    assert "document.body.classList.add('ar-active', 'tracking-active')" in page
    assert "document.body.classList.add('camera-active', 'tracking-active')" in page
    assert page.count('id="trackingExitButton"') == 1
    assert 'id="arExitButton"' not in page
    assert 'id="cameraExitButton"' not in page
    assert "body.camera-active .camera-preview-container" in page
    assert 'class="control-mode-ui"' in page
    assert "body.tracking-active .control-mode-ui" in page
    assert "requestCameraFullscreen" in page
    assert page.index('id="cameraPreviewContainer"') > page.index('id="arOverlay"')
    assert "bottom: max(12px, env(safe-area-inset-bottom))" in page


def test_web_page_cleans_up_failed_tracking_resources():
    web_root = Path(__file__).resolve().parents[1] / "web"
    page = (web_root / "web_teleop.html").read_text(encoding="utf-8")

    assert "if (this.isActive || this.isStarting) return;" in page
    assert "await this.session.end();" in page
    assert "this.renderer.setAnimationLoop(null);" in page
    assert "this.renderer.dispose();" in page
    assert "cameraStream.getTracks().forEach(function(t) { t.stop(); });" in page


def test_disconnect_retains_live_websocket_thread_and_blocks_restart():
    class _StuckThread:
        def join(self, timeout):
            assert timeout == 3.0

        def is_alive(self):
            return True

    phone = WebPhone(_config())
    thread = _StuckThread()
    phone._ws_thread = thread

    phone.disconnect()

    assert phone._ws_thread is thread
    with pytest.raises(RuntimeError, match="still shutting down"):
        phone._start_websocket_server()


def test_missing_tls_files_fail_closed(tmp_path: Path):
    _write_web_assets(tmp_path)
    base = _config()
    config = PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "web": {
                "bind_address": "127.0.0.1",
                "http_port": base.web.http_port,
                "websocket_port": base.web.websocket_port,
                "tls": {
                    "enabled": True,
                    "cert_file": str(tmp_path / "missing-cert.pem"),
                    "key_file": str(tmp_path / "missing-key.pem"),
                    "allow_insecure_http": False,
                },
            },
        }
    )
    phone = WebPhone(config, web_root=tmp_path)
    assert phone.connect() is False
    assert phone.is_connected is False


def test_explicit_insecure_fallback_starts_http(tmp_path: Path):
    _write_web_assets(tmp_path)
    base = _config()
    config = PhoneConfig.from_dict(
        {
            "backend": "webphone",
            "web": {
                "bind_address": "127.0.0.1",
                "http_port": base.web.http_port,
                "websocket_port": base.web.websocket_port,
                "tls": {
                    "enabled": True,
                    "cert_file": str(tmp_path / "missing-cert.pem"),
                    "key_file": str(tmp_path / "missing-key.pem"),
                    "allow_insecure_http": True,
                },
            },
        }
    )
    phone = WebPhone(config, web_root=tmp_path)
    try:
        assert phone.connect() is True
        assert phone.page_scheme == "http"
        assert phone.websocket_scheme == "ws"
    finally:
        phone.disconnect()
