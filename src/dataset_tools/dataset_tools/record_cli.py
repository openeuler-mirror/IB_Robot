#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command Line Interface for triggering episodic recordings.
Sends Action goals to the EpisodeRecorderServer.
"""

import sys
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_srvs.srv import Empty, Trigger

# Import the global interface
from ibrobot_msgs.action import RecordEpisode


class RecordCLI(Node):
    def __init__(self):
        super().__init__('record_cli')

        self.declare_parameter('dispatcher_reset_service', '/action_dispatcher/reset')
        self.declare_parameter('policy_reset_service', '/act_inference_node/reset_policy_state')
        self.declare_parameter('reset_timeout_sec', 2.0)
        self.declare_parameter('control_mode', 'teleop')
        self.declare_parameter('reset_before_episode', 'auto')
        self._reset_timeout_s = float(self.get_parameter('reset_timeout_sec').value)
        self._episode_finished_evt = threading.Event()

        # Action client to start recording
        self._action_client = ActionClient(self, RecordEpisode, 'record_episode')

        # Service client to stop recording early
        self._cancel_client = self.create_client(Trigger, 'record_episode/cancel')
        self._dispatcher_reset_client = self.create_client(
            Empty,
            self.get_parameter('dispatcher_reset_service').value,
        )
        self._policy_reset_client = self.create_client(
            Trigger,
            self.get_parameter('policy_reset_service').value,
        )
        # Service client to get dataset info
        self._info_client = self.create_client(Trigger, 'record_episode/get_info')

        self.get_logger().info("Record CLI started. Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Connected to Episode Recorder Server!")

    def send_goal(self, prompt_text: str):
        if self._should_reset_before_episode():
            self.prepare_new_episode()

        goal_msg = RecordEpisode.Goal()
        goal_msg.prompt = prompt_text

        self.get_logger().info(f"Sending goal with prompt: '{prompt_text}'")
        
        # We don't block here so the user can cancel it
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected by server (Is it already recording?)")
            return

        self.get_logger().info("🔴 RECORDING STARTED. (Press Enter to stop early)")
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result_wrapper = future.result()
            result = result_wrapper.result
            if result.success:
                self.get_logger().info(f"✅ RECORDING SAVED: {result.message}")
            else:
                self.get_logger().info(f"⚠️  RECORDING CANCELLED/ENDED: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Action failed to get result: {e}")
            
        print("\n----------------------------------------")
        print("Ready for next episode.")
        self._episode_finished_evt.set()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Optional: Print progress on the same line
        sys.stdout.write(f"\r[Time Left: {feedback.seconds_remaining}s] {feedback.feedback_message}   ")
        sys.stdout.flush()

    def cancel_recording(self):
        if not self._cancel_client.service_is_ready():
            self.get_logger().error("Cancel service not available right now.")
            return
            
        req = Trigger.Request()
        future = self._cancel_client.call_async(req)
        future.add_done_callback(self._cancel_response_callback)

    def _cancel_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Stop signal acknowledged.")
            else:
                self.get_logger().error(f"Stop signal failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Cancel service call failed: {e}")

    def prepare_new_episode(self):
        """Best-effort reset so a new episode reuses clean policy/dispatcher state."""
        # Prefer dispatcher reset because it clears queued action chunks and then
        # asks inference_service to clear policy-local caches. The direct policy
        # reset remains a fallback for deployments that start record_cli without
        # action_dispatch.
        if self._reset_dispatcher_state():
            return
        self._reset_policy_state()

    def _should_reset_before_episode(self) -> bool:
        override = str(self.get_parameter('reset_before_episode').value).strip().lower()
        if override in {'true', '1', 'yes', 'on'}:
            return True
        if override in {'false', '0', 'no', 'off'}:
            return False
        return str(self.get_parameter('control_mode').value).strip() == 'model_inference'

    def _reset_dispatcher_state(self) -> bool:
        service_name = self.get_parameter('dispatcher_reset_service').value
        if not service_name:
            return False
        if not self._dispatcher_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(
                f"Dispatcher reset service unavailable: {service_name}"
            )
            return False

        try:
            future = self._dispatcher_reset_client.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Dispatcher reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if response is not None:
            self.get_logger().info("Dispatcher state reset for new episode.")
            return True

        self.get_logger().warning("Dispatcher reset call failed.")
        return False

    def _reset_policy_state(self) -> bool:
        service_name = self.get_parameter('policy_reset_service').value
        if not service_name:
            return False
        if not self._policy_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(
                f"Policy reset service unavailable: {service_name}"
            )
            return False

        try:
            future = self._policy_reset_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Policy reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if response is not None and response.success:
            self.get_logger().info("Policy runtime state reset for new episode.")
            return True

        message = response.message if response is not None else ""
        self.get_logger().warning(f"Policy reset call failed. {message}")
        return False

    @staticmethod
    def _wait_for_future(future, timeout_sec: float) -> bool:
        """Wait for a future while the node is already spun by the executor thread."""
        if future.done():
            return True
        done_event = threading.Event()
        future.add_done_callback(lambda _future: done_event.set())
        if future.done():
            return True
        return done_event.wait(timeout=timeout_sec)


def cli_loop(node):
    """Run the interactive prompt in a separate thread."""

    print("\nFetching dataset configuration from server...")
    if node._info_client.wait_for_service(timeout_sec=3.0):
        future = node._info_client.call_async(Trigger.Request())
        import time
        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        if future.done() and future.result() is not None:
            try:
                import json
                info = json.loads(future.result().message)
                path = info.get("path", "Unknown")
                count = info.get("episodes", 0)

                print(f"\n========================================")
                print(f"📊 DATASET TARGET INFO")
                print(f"========================================")
                print(f"📁 Path: {path}")
                if count > 0:
                    print(f"⚠️  Found {count} existing episodes in this directory.")
                    print(f"   New recordings will be APPENDED to this dataset.")
                else:
                    print(f"✨ New dataset directory. No existing data found.")
                print(f"")
                print(f"💡 Tip: To change the dataset name, restart the launch server with:")
                print(f"   ros2 launch ... dataset_name:=<new_name> bag_base_dir:=<custom_dir>")
                print(f"========================================")

                ans = input("\nPress Enter to CONFIRM and continue, or 'q' to quit > ")
                if ans.strip().lower() in ['q', 'quit']:
                    rclpy.shutdown()
                    return
            except Exception as e:
                print(f"Failed to parse server info: {e}")
    else:
        print("⚠️  Warning: Could not fetch dataset info from server (timeout).")

    last_prompt = "default_task"

    while rclpy.ok():
        node._episode_finished_evt.clear()

        print("\n========================================")
        print("Dataset Collection CLI")
        print(f"Enter prompt text to start recording. (Press Enter to reuse: '{last_prompt}')")
        print("Type 'q' or 'quit' to exit.")
        print("========================================")
        
        try:
            prompt = input("Prompt > ")
            if prompt.strip().lower() in ['q', 'quit']:
                print("Exiting...")
                rclpy.shutdown()
                break
                
            if not prompt.strip():
                prompt = last_prompt
            else:
                last_prompt = prompt.strip()
                
            # Send the start command
            node.send_goal(prompt)

            # Wait for user to press Enter to stop
            input()

            # Send cancel command
            node.cancel_recording()
            
            # Wait for the server to acknowledge completion before looping.
            # Use a timeout to avoid blocking forever if the recorder crashes.
            if not node._episode_finished_evt.wait(timeout=15.0):
                node.get_logger().warning(
                    "Timed out waiting for recorder to finish. "
                    "The recording server may have crashed — check its logs."
                )

        except EOFError:
            break
        except Exception as e:
            import traceback; traceback.print_exc()


from rclpy.executors import MultiThreadedExecutor

def main(args=None):
    rclpy.init(args=args)
    node = RecordCLI()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    # Run ROS spinning in a background thread so input() doesn't block callbacks
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    
    try:
        # Run the interactive CLI in the main thread
        cli_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join()

if __name__ == '__main__':
    main()
