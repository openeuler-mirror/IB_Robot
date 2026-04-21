"""
FunASR WebSocket Client ROS2 Node

Real-time speech recognition from microphone.
Supports regex-based keyword matching and publishes results for navigation.
"""

import asyncio
import json
import re
import ssl
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("Warning: pyaudio not installed, microphone input will not work")

try:
    import websockets
except ImportError:
    websockets = None
    print("Warning: websockets not installed, cannot connect to FunASR server")


class FunasrClientNode(Node):
    """ROS2 Node: FunASR voice recognition client"""

    def __init__(self):
        super().__init__("funasr_client_node")

        # Declare ROS2 parameters
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", "10095")
        self.declare_parameter("use_itn", True)
        self.declare_parameter("chunk_size", [5, 10, 5])
        self.declare_parameter("chunk_interval", 10)
        self.declare_parameter("mode", "2pass")
        self.declare_parameter("hotword_msg", "")
        self.declare_parameter("wav_name", "microphone")
        self.declare_parameter("keywords_file", "")
        self.declare_parameter("keywords_json", "{}")
        self.declare_parameter("destinations_json", "{}")
        self.declare_parameter("auto_start", True)

        # Declare topic name parameters
        self.declare_parameter("topic_text", "/voice_asr/text")
        self.declare_parameter("topic_status", "/voice_asr/status")
        self.declare_parameter("topic_keyword_matched", "/voice_asr/keyword_matched")
        self.declare_parameter("topic_nav_stop", "/robot_navigation/nav/stop")

        # Get parameters
        self.host = self.get_parameter("host").value
        self.port = self.get_parameter("port").value
        self.use_itn = self.get_parameter("use_itn").value
        self.chunk_size = self.get_parameter("chunk_size").value
        self.chunk_interval = self.get_parameter("chunk_interval").value
        self.mode = self.get_parameter("mode").value
        self.hotword_msg = self.get_parameter("hotword_msg").value
        self.wav_name = self.get_parameter("wav_name").value
        self.keywords_file = self.get_parameter("keywords_file").value
        self.keywords_json = self.get_parameter("keywords_json").value
        self.auto_start = self.get_parameter("auto_start").value

        # Load destinations mapping (name -> {x, y, theta})
        destinations_json = self.get_parameter("destinations_json").value
        try:
            self.destinations = json.loads(destinations_json)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse destinations_json: {e}")
            self.destinations = {}

        # Load keywords: prefer keywords_json from launch file, fallback to file
        self.keywords = self._load_keywords()

        # State variables
        self.is_running = False
        self.websocket = None
        self.stream = None
        self.CHUNK = 0
        self.loop = None
        self.async_thread = None

        # Publishers (topic names are configurable via parameters)
        topic_text = self.get_parameter("topic_text").value
        topic_status = self.get_parameter("topic_status").value
        topic_keyword_matched = self.get_parameter("topic_keyword_matched").value
        topic_nav_stop = self.get_parameter("topic_nav_stop").value

        self.text_pub = self.create_publisher(String, topic_text, 10)
        self.status_pub = self.create_publisher(String, topic_status, 10)
        self.keyword_pub = self.create_publisher(String, topic_keyword_matched, 10)
        self.nav_stop_pub = self.create_publisher(String, topic_nav_stop, 10)

        # Subscriber - dynamic keyword updates
        self.keyword_sub = self.create_subscription(String, "/voice_asr/keywords", self.keywords_callback, 10)

        # Services (use private namespace)
        self.start_srv = self.create_service(Trigger, "~/start", self.start_callback)
        self.stop_srv = self.create_service(Trigger, "~/stop", self.stop_callback)

        # Service clients
        self.stop_eval_client = self.create_client(Trigger, "/action_dispatcher/stop_evaluate")

        # Initialize audio
        self._init_audio()

        # Auto-start timer
        self._auto_start_timer = None
        if self.auto_start:
            self._auto_start_timer = self.create_timer(0.5, self._auto_start_callback)

        self.get_logger().info(f"FunASR Client initialized: ws://{self.host}:{self.port}")
        self.get_logger().info(f"Loaded {len(self.keywords)} keyword patterns")
        self.get_logger().info(f"Loaded {len(self.destinations)} destinations: {list(self.destinations.keys())}")

    def _load_keywords(self) -> dict:
        """Load keywords from keywords_json parameter or keywords_file"""
        # First try keywords_json parameter (passed from launch file)
        if self.keywords_json and self.keywords_json != "{}":
            try:
                data = json.loads(self.keywords_json)
                # Handle {"keywords": {...}} structure
                if isinstance(data, dict) and "keywords" in data:
                    return data["keywords"]
                return data
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse keywords_json: {e}")

        # Fallback to keywords_file
        if not self.keywords_file:
            return {}

        try:
            with open(self.keywords_file, encoding="utf-8") as f:
                data = json.load(f)
            self.get_logger().info(f"Loaded keywords from {self.keywords_file}")
            # Handle {"keywords": {...}} structure
            if isinstance(data, dict) and "keywords" in data:
                return data["keywords"]
            return data
        except FileNotFoundError:
            self.get_logger().warning(f"Keywords file not found: {self.keywords_file}")
            return {}
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse keywords file: {e}")
            return {}

    def _init_audio(self):
        """Initialize audio device"""
        if pyaudio is None:
            self.get_logger().warning("pyaudio not available")
            return

        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4800
            )
            self.CHUNK = 4800
            self.get_logger().info("Audio initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize audio: {e}")
            self.stream = None

    def _auto_start_callback(self):
        """Auto-start voice recognition after node initialization"""
        # Destroy the one-shot timer
        if self._auto_start_timer:
            self.destroy_timer(self._auto_start_timer)
            self._auto_start_timer = None

        # Create a fake request/response to call start_callback
        class FakeRequest:
            pass

        class FakeResponse:
            def __init__(self):
                self.success = False
                self.message = ""

        request = FakeRequest()
        response = FakeResponse()
        result = self.start_callback(request, response)

        if result.success:
            self.get_logger().info("Auto-started voice recognition")
        else:
            self.get_logger().warning(f"Auto-start failed: {result.message}")

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def _publish_text(self, text: str):
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)

    def _publish_keyword_matched(self, keyword: str, match_info: dict):
        """Publish matched keyword to /car_asr/keyword_matched

        Keyword types:
        - destination: triggers navigation (includes x, y, theta)
        - action: triggers task execution (includes task_description)
        - stop: stops lekiwi evaluation
        """
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
            # Stop keywords trigger lekiwi evaluation stop
            self._stop_evaluation()

        msg = String()
        msg.data = json.dumps(msg_data)
        self.keyword_pub.publish(msg)
        self.get_logger().info(f"Published keyword match: {msg.data}")

    def keywords_callback(self, msg: String):
        """Update keywords from topic"""
        try:
            self.keywords = json.loads(msg.data)
            self.get_logger().info(f"Keywords updated: {len(self.keywords)} patterns")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse keywords: {e}")

    def _stop_evaluation(self):
        """Call lekiwi_evaluate stop service and stop nav2 navigation"""
        # 停止 lekiwi 评估
        if self.stop_eval_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self.stop_eval_client.call_async(request)
            future.add_done_callback(self._stop_eval_callback)
        else:
            self.get_logger().warn("Stop evaluation service not available")

        # 停止 nav2 导航
        stop_msg = String()
        stop_msg.data = "stop"
        self.nav_stop_pub.publish(stop_msg)
        self.get_logger().info("Published stop navigation command")

    def _stop_eval_callback(self, future):
        """Callback for stop evaluation service"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Evaluation stopped successfully")
            else:
                self.get_logger().warn(f"Failed to stop evaluation: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Stop evaluation service call failed: {e}")

    def start_callback(self, request, response):
        if self.is_running:
            response.success = True
            response.message = "Already running"
            return response

        if websockets is None:
            response.success = False
            response.message = "websockets not installed"
            return response

        if self.stream is None:
            response.success = False
            response.message = "Audio not initialized"
            return response

        self.is_running = True
        self.async_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.async_thread.start()

        response.success = True
        response.message = "Voice recognition started"
        return response

    def stop_callback(self, request, response):
        if not self.is_running:
            response.success = True
            response.message = "Already stopped"
            return response

        self.is_running = False
        self._publish_status("stopped")
        response.success = True
        response.message = "Voice recognition stopped"
        return response

    def _run_async_loop(self):
        # Create a new event loop in this thread (isolated from ROS2's executor)
        try:
            # Try to get existing loop first (may be set by ROS2)
            existing_loop = asyncio.get_event_loop()
            if existing_loop.is_running():
                # If a loop is already running (from ROS2), create a new one
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            else:
                self.loop = existing_loop
        except RuntimeError:
            # No loop exists, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._async_main())
        except Exception as e:
            self.get_logger().error(f"Async loop error: {e}")
        finally:
            # Don't close the loop if it might be reused
            if self.loop != asyncio.get_event_loop():
                self.loop.close()

    async def _async_main(self):
        ssl_context = ssl.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        uri = f"wss://{self.host}:{self.port}"
        self._publish_status("connecting")

        try:
            async with websockets.connect(
                uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
            ) as websocket:
                self.websocket = websocket
                self._publish_status("connected")
                self.get_logger().info("Connected to FunASR server")

                # Send config
                config = {
                    "mode": self.mode,
                    "chunk_size": self.chunk_size,
                    "chunk_interval": self.chunk_interval,
                    "hotword_msg": self.hotword_msg,
                    "wav_name": self.wav_name,
                    "use_itn": self.use_itn,
                }
                await websocket.send(json.dumps(config))

                # Run tasks
                await asyncio.gather(self._record_microphone(), self._recv_text())

        except Exception as e:
            self.get_logger().error(f"Connection error: {e}")
            self._publish_status(f"error: {str(e)}")
        finally:
            self.websocket = None
            self.is_running = False
            self._publish_status("disconnected")

    async def _record_microphone(self):
        while self.is_running and self.websocket:
            try:
                if self.stream:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    await self.websocket.send(data)
                await asyncio.sleep(0.005)
            except Exception as e:
                self.get_logger().error(f"Recording error: {e}")
                break

    async def _recv_text(self):
        while self.is_running and self.websocket:
            try:
                msg = await self.websocket.recv()
                data = json.loads(msg)

                if "text" not in data or "mode" not in data:
                    continue

                text = data["text"]

                if data["mode"] == "2pass-online":
                    self._publish_text(text)
                elif data["mode"] == "2pass-offline":
                    self.get_logger().info(f"Recognized: {text}")
                    self._publish_text(text)

                    # Match keywords (may match multiple)
                    matches = self._match_keyword(text)
                    for pattern, match_info in matches:
                        self.get_logger().info(f"Keyword matched: '{pattern}'")
                        self._publish_keyword_matched(pattern, match_info)

            except websockets.exceptions.ConnectionClosed:
                self.get_logger().warn("WebSocket closed")
                break
            except Exception as e:
                self.get_logger().error(f"Receive error: {e}")
                break

    def _match_keyword(self, text: str) -> list:
        """Match keywords using regex (case-insensitive for English)

        Returns a list of all matched (pattern, match_info) tuples.
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        matches = []
        for pattern, match_info in self.keywords.items():
            try:
                if re.search(pattern, text_lower):
                    matches.append((pattern, match_info))
            except re.error:
                self.get_logger().warning(f"Invalid regex: {pattern}")
        return matches

    def destroy_node(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FunasrClientNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
