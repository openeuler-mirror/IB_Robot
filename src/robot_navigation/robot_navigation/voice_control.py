"""
Voice Navigation Bridge Node

Bridges voice_asr_service (sherpa-onnx local ASR) to nav2_goal_client
by subscribing to /voice_command, performing keyword matching, and
publishing structured JSON to /voice_asr/keyword_matched.

Replaces funasr_client_node for voice-controlled navigation without
requiring an external FunASR WebSocket server.
"""

import json
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ibrobot_msgs.srv import SetHotwords


class VoiceControl(Node):
    """Bridges voice_asr_node output to nav2_goal_client input."""

    def __init__(self):
        super().__init__("voice_control")

        # Declare parameters
        self.declare_parameter("topic_text", "/voice_command")
        self.declare_parameter("topic_keyword_matched", "/voice_asr/keyword_matched")
        self.declare_parameter("topic_nav_stop", "/voice_asr/nav_stop")
        self.declare_parameter("keywords_json", "{}")
        self.declare_parameter("keywords_file", "")
        self.declare_parameter("destinations_json", "{}")

        # Load keywords
        self.keywords_json = self.get_parameter("keywords_json").value
        self.keywords_file = self.get_parameter("keywords_file").value
        self.keywords = self._load_keywords()

        # Load destinations
        destinations_json = self.get_parameter("destinations_json").value
        try:
            self.destinations = json.loads(destinations_json)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse destinations_json: {e}")
            self.destinations = {}

        # Publishers
        topic_keyword_matched = self.get_parameter("topic_keyword_matched").value
        topic_nav_stop = self.get_parameter("topic_nav_stop").value

        self.keyword_pub = self.create_publisher(String, topic_keyword_matched, 10)
        self.nav_stop_pub = self.create_publisher(String, topic_nav_stop, 10)

        # Service client for stopping evaluation
        self.stop_eval_client = self.create_client(Trigger, "/action_dispatcher/stop_evaluate")

        # Subscriber to voice_asr_node output
        topic_text = self.get_parameter("topic_text").value
        self.sub = self.create_subscription(String, topic_text, self._text_callback, 10)

        # Dynamic keyword updates
        self.keyword_sub = self.create_subscription(String, "/voice_asr/keywords", self._keywords_callback, 10)

        # Hotword registration: push keywords to voice_asr_node at startup
        self._hotword_client = self.create_client(SetHotwords, "/voice_asr_node/set_hotwords")
        self._hotword_timer = self.create_timer(1.0, self._register_hotwords)

        self.get_logger().info(
            f"VoiceControl initialized: "
            f"sub={topic_text}, pub={topic_keyword_matched}, "
            f"{len(self.keywords)} keywords, {len(self.destinations)} destinations"
        )

    def _load_keywords(self) -> dict:
        """Load keywords from keywords_json parameter or keywords_file."""
        if self.keywords_json and self.keywords_json != "{}":
            try:
                data = json.loads(self.keywords_json)
                if isinstance(data, dict) and "keywords" in data:
                    return data["keywords"]
                return data
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse keywords_json: {e}")

        if not self.keywords_file:
            return {}

        try:
            with open(self.keywords_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "keywords" in data:
                return data["keywords"]
            return data
        except FileNotFoundError:
            self.get_logger().warning(f"Keywords file not found: {self.keywords_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse keywords file: {e}")
            return {}

    def _text_callback(self, msg: String):
        """Process recognized text from voice_asr_node."""
        text = msg.data
        if not text:
            return

        self.get_logger().info(f"Received: {text}")

        matches = self._match_keyword(text)
        for pattern, match_info in matches:
            self.get_logger().info(f"Keyword matched: '{pattern}'")
            self._publish_keyword_matched(pattern, match_info)

    def _match_keyword(self, text: str) -> list:
        """Match keywords using regex (case-insensitive for English).

        Returns a list of all matched (pattern, match_info) tuples.
        """
        text_lower = text.lower().replace(" ", "")
        matches = []
        for pattern, match_info in self.keywords.items():
            try:
                if re.search(pattern, text_lower):
                    matches.append((pattern, match_info))
            except re.error:
                self.get_logger().warning(f"Invalid regex: {pattern}")
        return matches

    def _publish_keyword_matched(self, keyword: str, match_info: dict):
        """Publish matched keyword as structured JSON."""
        keyword_type = match_info.get("type", "destination")
        info = match_info.get("info", {})

        # Resolve destination name to coordinates
        if keyword_type == "destination" and "destination" in info:
            dest_name = info["destination"]
            if dest_name in self.destinations:
                coords = self.destinations[dest_name]
                info = {
                    "x": float(coords.get("x", 0.0)),
                    "y": float(coords.get("y", 0.0)),
                    "theta": float(coords.get("theta", 0.0)),
                }
                self.get_logger().info(
                    f"Resolved destination '{dest_name}' -> x={info['x']}, y={info['y']}, theta={info['theta']}"
                )
            else:
                self.get_logger().warning(f"Destination '{dest_name}' not found in destinations config")

        msg_data = {"keyword": keyword, "type": keyword_type, "info": info}

        if keyword_type == "stop":
            self._stop_evaluation()

        msg = String()
        msg.data = json.dumps(msg_data)
        self.keyword_pub.publish(msg)
        self.get_logger().info(f"Published keyword match: {msg.data}")

    def _stop_evaluation(self):
        """Stop lekiwi evaluation and nav2 navigation."""
        if self.stop_eval_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self.stop_eval_client.call_async(request)
            future.add_done_callback(self._stop_eval_callback)
        else:
            self.get_logger().warn("Stop evaluation service not available")

        stop_msg = String()
        stop_msg.data = "stop"
        self.nav_stop_pub.publish(stop_msg)
        self.get_logger().info("Published stop navigation command")

    def _stop_eval_callback(self, future):
        """Callback for stop evaluation service."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Evaluation stopped successfully")
            else:
                self.get_logger().warn(f"Failed to stop evaluation: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Stop evaluation service call failed: {e}")

    def _keywords_callback(self, msg: String):
        """Update keywords from topic."""
        try:
            self.keywords = json.loads(msg.data)
            self.get_logger().info(f"Keywords updated: {len(self.keywords)} patterns")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse keywords: {e}")

    def _extract_hotwords(self) -> list[str]:
        """Extract hotword phrases from keyword regex patterns.

        Splits each pattern on '|' and strips regex metacharacters (e.g. '.*'),
        keeping only meaningful word fragments for ASR hotword boosting.
        """
        hotwords = []
        for pattern in self.keywords:
            for alt in pattern.split("|"):
                # Remove regex metacharacters, keep Chinese/letters/digits
                word = re.sub(r"[.\*\+\?\(\)\[\]\{\}\^\$\\]", "", alt).strip()
                if word:
                    hotwords.append(word)
        return hotwords

    def _register_hotwords(self):
        """One-shot timer callback: register keywords as ASR hotwords."""
        self.destroy_timer(self._hotword_timer)
        self._hotword_timer = None

        hotwords = self._extract_hotwords()
        if not hotwords:
            return

        if not self._hotword_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("voice_asr_node/set_hotwords service not available, skipping hotword registration")
            return

        request = SetHotwords.Request()
        request.hotwords = hotwords
        request.boost_scores = [1.5] * len(hotwords)
        future = self._hotword_client.call_async(request)
        future.add_done_callback(self._hotword_callback)

    def _hotword_callback(self, future):
        """Callback for hotword registration."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Registered {len(self._extract_hotwords())} hotwords to voice_asr_node")
            else:
                self.get_logger().warn(f"Hotword registration failed: {response.error_message}")
        except Exception as e:
            self.get_logger().error(f"Hotword registration service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
