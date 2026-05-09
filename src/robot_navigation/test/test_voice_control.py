"""Tests for voice_control node.

Covers keyword matching, destination resolution, stop handling,
hotword extraction, keyword loading, dynamic keyword updates,
and end-to-end topic bridging.
"""

import json
import os
import tempfile

import pytest
import rclpy
from std_msgs.msg import String

# ── test data ───────────────────────────────────────────────────────────────

KEYWORDS_JSON = json.dumps(
    {
        "keywords": {
            "捡.*蓝色方块|拿.*蓝色方块|蓝色方块": {
                "type": "action",
                "info": {"task_description": "Pick up the blue square"},
            },
            "捡.*黑色方块|拿.*黑色方块|黑色方块": {
                "type": "action",
                "info": {"task_description": "Pick up the black square"},
            },
            "捡.*香蕉|拿.*香蕉|香蕉": {
                "type": "action",
                "info": {"task_description": "Pick up the banana"},
            },
            "去.*a点|到.*a点|a点": {
                "type": "destination",
                "info": {"destination": "point_a"},
            },
            "去.*b点|到.*b点|b点": {
                "type": "destination",
                "info": {"destination": "point_b"},
            },
            "回.*起点|到.*起点|起点": {
                "type": "destination",
                "info": {"destination": "origin"},
            },
            "停止|停下": {
                "type": "stop",
                "info": {"task_description": "Stop current action"},
            },
        }
    }
)

DESTINATIONS_JSON = json.dumps(
    {
        "point_a": {"x": 0.0, "y": 0.2, "theta": 1.5708},
        "point_b": {"x": 0.2, "y": 0.0, "theta": 0.0},
        "origin": {"x": 0.0, "y": 0.0, "theta": 0.0},
    }
)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def bridge(rclpy_init):
    """Module-scoped VoiceControl wired to /test/* topics."""
    from robot_navigation.voice_control import VoiceControl

    node = VoiceControl()

    # Load test keywords & destinations
    node.keywords_json = KEYWORDS_JSON
    node.keywords = node._load_keywords()
    node.destinations = json.loads(DESTINATIONS_JSON)

    # Re-create pubs/subs on isolated test topics
    node.keyword_pub = node.create_publisher(String, "/test/keyword_matched", 10)
    node.nav_stop_pub = node.create_publisher(String, "/test/nav_stop", 10)
    node.create_subscription(String, "/test/voice_command", node._text_callback, 10)

    # Cancel hotword timer (no voice_asr_node in test env)
    if node._hotword_timer is not None:
        node.destroy_timer(node._hotword_timer)
        node._hotword_timer = None

    yield node
    node.destroy_node()


# ── helper ──────────────────────────────────────────────────────────────────


def _collect_msg(node, topic, msg_type, trigger_fn, timeout_sec=1.0):
    """Subscribe first, spin to connect, then trigger publish and collect."""
    received = []
    sub = node.create_subscription(msg_type, topic, lambda m: received.append(m), 10)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.02)
    trigger_fn()
    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while not received and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    return received[-1] if received else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. Keyword matching
# ═══════════════════════════════════════════════════════════════════════════


class TestKeywordMatching:
    def test_destination_a(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("去a点")]
        assert any("a点" in p for p in patterns)

    def test_destination_a_with_spaces(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("去 a 点")]
        assert any("a点" in p for p in patterns)

    def test_destination_b_with_spaces(self, bridge):
        """Regression: ASR inserts spaces around English letters."""
        patterns = [p for p, _ in bridge._match_keyword("去 b 点减香蕉")]
        assert any("b点" in p for p in patterns)

    def test_destination_b(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("到b点")]
        assert any("b点" in p for p in patterns)

    def test_destination_origin(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("回起点")]
        assert any("起点" in p for p in patterns)

    def test_action_banana(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("捡起香蕉")]
        assert any("香蕉" in p for p in patterns)

    def test_action_blue_square(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("拿蓝色方块")]
        assert any("蓝色方块" in p for p in patterns)

    def test_action_black_square(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("黑色方块")]
        assert any("黑色方块" in p for p in patterns)

    def test_stop(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("停止")]
        assert any("停止" in p for p in patterns)

    def test_stop_variant(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("停下")]
        assert any("停下" in p for p in patterns)

    def test_case_insensitive_english(self, bridge):
        patterns = [p for p, _ in bridge._match_keyword("去A点")]
        assert any("a点" in p for p in patterns)

    def test_no_match(self, bridge):
        assert bridge._match_keyword("今天天气真好") == []

    def test_empty_text(self, bridge):
        assert bridge._match_keyword("") == []

    def test_multiple_matches(self, bridge):
        matches = bridge._match_keyword("去b点减香蕉")
        patterns = [p for p, _ in matches]
        assert any("b点" in p for p in patterns)
        assert any("香蕉" in p for p in patterns)
        assert len(matches) == 2

    def test_invalid_regex_skipped(self, bridge):
        bridge.keywords["[invalid"] = {"type": "action", "info": {}}
        matches = bridge._match_keyword("hello")
        assert isinstance(matches, list)
        del bridge.keywords["[invalid"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Destination resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestDestinationResolution:
    def test_resolve_point_a(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "去.*a点|到.*a点|a点",
                {"type": "destination", "info": {"destination": "point_a"}},
            ),
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["type"] == "destination"
        assert data["info"]["x"] == pytest.approx(0.0)
        assert data["info"]["y"] == pytest.approx(0.2)
        assert data["info"]["theta"] == pytest.approx(1.5708)

    def test_resolve_point_b(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "去.*b点|到.*b点|b点",
                {"type": "destination", "info": {"destination": "point_b"}},
            ),
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["info"]["x"] == pytest.approx(0.2)
        assert data["info"]["y"] == pytest.approx(0.0)
        assert data["info"]["theta"] == pytest.approx(0.0)

    def test_resolve_origin(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "回.*起点|到.*起点|起点",
                {"type": "destination", "info": {"destination": "origin"}},
            ),
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["info"]["x"] == pytest.approx(0.0)
        assert data["info"]["y"] == pytest.approx(0.0)
        assert data["info"]["theta"] == pytest.approx(0.0)

    def test_unknown_destination_passes_through(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "去.*c点",
                {"type": "destination", "info": {"destination": "point_c"}},
            ),
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["info"] == {"destination": "point_c"}

    def test_action_type_no_resolution(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "捡.*香蕉|拿.*香蕉|香蕉",
                {"type": "action", "info": {"task_description": "Pick up the banana"}},
            ),
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["type"] == "action"
        assert data["info"] == {"task_description": "Pick up the banana"}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stop handling
# ═══════════════════════════════════════════════════════════════════════════


class TestStopHandling:
    def test_stop_publishes_nav_stop(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/nav_stop",
            String,
            lambda: bridge._publish_keyword_matched(
                "停止|停下",
                {"type": "stop", "info": {"task_description": "Stop current action"}},
            ),
            timeout_sec=2.0,
        )
        assert msg is not None
        assert msg.data == "stop"

    def test_stop_publishes_keyword_matched(self, bridge):
        msg = _collect_msg(
            bridge,
            "/test/keyword_matched",
            String,
            lambda: bridge._publish_keyword_matched(
                "停止|停下",
                {"type": "stop", "info": {"task_description": "Stop current action"}},
            ),
            timeout_sec=2.0,
        )
        assert msg is not None
        data = json.loads(msg.data)
        assert data["type"] == "stop"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Hotword extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestHotwordExtraction:
    def test_contains_expected_words(self, bridge):
        hw = bridge._extract_hotwords()
        assert "蓝色方块" in hw
        assert "黑色方块" in hw
        assert "香蕉" in hw
        assert "停止" in hw
        assert "停下" in hw
        assert "a点" in hw
        assert "b点" in hw
        assert "起点" in hw

    def test_no_regex_metacharacters(self, bridge):
        for w in bridge._extract_hotwords():
            assert ".*" not in w
            assert "+" not in w
            assert "(" not in w
            assert ")" not in w

    def test_no_empty_strings(self, bridge):
        for w in bridge._extract_hotwords():
            assert w.strip() != ""

    def test_alternatives_split(self, bridge):
        hw = bridge._extract_hotwords()
        assert "停止" in hw
        assert "停下" in hw

    def test_empty_keywords(self, bridge):
        saved = bridge.keywords
        bridge.keywords = {}
        assert bridge._extract_hotwords() == []
        bridge.keywords = saved


# ═══════════════════════════════════════════════════════════════════════════
# 5. Keyword loading
# ═══════════════════════════════════════════════════════════════════════════


class TestKeywordLoading:
    def _restore(self, bridge, orig_json, orig_file):
        bridge.keywords_json = orig_json
        bridge.keywords_file = orig_file
        bridge.keywords = bridge._load_keywords()

    def test_load_from_json_with_wrapper(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        bridge.keywords_json = KEYWORDS_JSON
        bridge.keywords_file = ""
        result = bridge._load_keywords()
        assert len(result) == 7
        assert "停止|停下" in result
        self._restore(bridge, orig_json, orig_file)

    def test_load_from_json_without_wrapper(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        bridge.keywords_json = json.dumps({"停止|停下": {"type": "stop", "info": {}}})
        bridge.keywords_file = ""
        result = bridge._load_keywords()
        assert "停止|停下" in result
        self._restore(bridge, orig_json, orig_file)

    def test_load_from_file(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(KEYWORDS_JSON)
            tmp = f.name
        try:
            bridge.keywords_json = "{}"
            bridge.keywords_file = tmp
            result = bridge._load_keywords()
            assert len(result) == 7
        finally:
            os.unlink(tmp)
            self._restore(bridge, orig_json, orig_file)

    def test_load_empty_returns_empty(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        bridge.keywords_json = "{}"
        bridge.keywords_file = ""
        assert bridge._load_keywords() == {}
        self._restore(bridge, orig_json, orig_file)

    def test_load_invalid_json_returns_empty(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        bridge.keywords_json = "not valid json{{"
        bridge.keywords_file = ""
        assert bridge._load_keywords() == {}
        self._restore(bridge, orig_json, orig_file)

    def test_load_missing_file_returns_empty(self, bridge):
        orig_json, orig_file = bridge.keywords_json, bridge.keywords_file
        bridge.keywords_json = "{}"
        bridge.keywords_file = "/nonexistent/keywords.json"
        assert bridge._load_keywords() == {}
        self._restore(bridge, orig_json, orig_file)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dynamic keyword updates
# ═══════════════════════════════════════════════════════════════════════════


class TestDynamicKeywordUpdate:
    def test_replace_keywords(self, bridge):
        saved = dict(bridge.keywords)
        msg = String()
        msg.data = json.dumps({"测试词": {"type": "action", "info": {}}})
        bridge._keywords_callback(msg)
        assert "测试词" in bridge.keywords
        assert "停止|停下" not in bridge.keywords
        bridge.keywords = saved

    def test_invalid_json_keeps_keywords(self, bridge):
        saved = dict(bridge.keywords)
        msg = String()
        msg.data = "bad json{"
        bridge._keywords_callback(msg)
        assert bridge.keywords == saved


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-end topic bridging
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def _publish_and_collect(self, bridge, text, topic, timeout=1.0):
        """Publish *text* on /test/voice_command, return first msg on *topic*."""
        received = []
        sub = bridge.create_subscription(String, topic, lambda m: received.append(m), 10)
        pub = bridge.create_publisher(String, "/test/voice_command", 10)

        # Let subscriptions connect
        deadline = bridge.get_clock().now() + rclpy.duration.Duration(seconds=0.3)
        while bridge.get_clock().now() < deadline:
            rclpy.spin_once(bridge, timeout_sec=0.05)

        msg = String()
        msg.data = text
        pub.publish(msg)

        deadline = bridge.get_clock().now() + rclpy.duration.Duration(seconds=timeout)
        while not received and bridge.get_clock().now() < deadline:
            rclpy.spin_once(bridge, timeout_sec=0.05)

        bridge.destroy_subscription(sub)
        bridge.destroy_publisher(pub)
        return received[-1] if received else None

    def test_e2e_destination(self, bridge):
        msg = self._publish_and_collect(bridge, "去a点", "/test/keyword_matched")
        assert msg is not None
        data = json.loads(msg.data)
        assert data["type"] == "destination"
        assert data["info"]["x"] == pytest.approx(0.0)
        assert data["info"]["y"] == pytest.approx(0.2)

    def test_e2e_action(self, bridge):
        msg = self._publish_and_collect(bridge, "捡香蕉", "/test/keyword_matched")
        assert msg is not None
        data = json.loads(msg.data)
        assert data["type"] == "action"
        assert data["info"]["task_description"] == "Pick up the banana"

    def test_e2e_stop(self, bridge):
        msg = self._publish_and_collect(bridge, "停止", "/test/nav_stop", timeout=2.0)
        assert msg is not None
        assert msg.data == "stop"

    def test_e2e_no_match_no_output(self, bridge):
        msg = self._publish_and_collect(bridge, "今天天气不错", "/test/keyword_matched")
        assert msg is None

    def test_e2e_empty_text_no_output(self, bridge):
        msg = self._publish_and_collect(bridge, "", "/test/keyword_matched")
        assert msg is None
