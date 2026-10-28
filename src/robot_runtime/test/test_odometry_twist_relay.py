"""odometry_twist_relay unit tests (lekiwi-mujoco-runtime-platform D3.3).

The pure relay logic against scripted simulation clocks: validated
forwarding (frame, finiteness, ordering, freshness window) and the reset
epoch — a backwards clock jump starts a zero stream that only a message
fresh on the new clock ends.
"""

from nav_msgs.msg import Odometry

from robot_runtime.odometry_twist_relay import OdometryTwistRelayLogic


def _odom(stamp: float, child: str = "base_link", vx: float = 0.15, vy: float = 0.0, wz: float = 0.0) -> Odometry:
    message = Odometry()
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
    message.child_frame_id = child
    message.twist.twist.linear.x = vx
    message.twist.twist.linear.y = vy
    message.twist.twist.angular.z = wz
    return message


def test_fresh_odometry_forwards_with_its_own_stamp_and_child_frame():
    logic = OdometryTwistRelayLogic("base_link")
    twist, reason = logic.on_odometry(_odom(10.0), now=10.02)
    assert twist is not None and reason == "forwarded"
    assert twist.header.frame_id == "base_link"
    assert twist.header.stamp.sec == 10 and abs(twist.header.stamp.nanosec) < 1
    assert twist.twist.linear.x == 0.15
    assert logic.forwarded == 1 and not logic.resetting


def test_wrong_child_frame_is_dropped_and_counted():
    logic = OdometryTwistRelayLogic("base_link")
    twist, reason = logic.on_odometry(_odom(10.0, child="chassis"), now=10.02)
    assert twist is None and "child_frame_id" in reason
    assert logic.dropped_frames == 1
    # a later correct message still forwards
    assert logic.on_odometry(_odom(10.05), now=10.07)[0] is not None


def test_non_finite_values_are_dropped():
    logic = OdometryTwistRelayLogic("base_link")
    message = _odom(10.0)
    message.twist.twist.linear.y = float("nan")
    twist, reason = logic.on_odometry(message, now=10.02)
    assert twist is None and "non-finite" in reason
    assert logic.dropped_invalid == 1


def test_out_of_order_stamps_are_dropped():
    logic = OdometryTwistRelayLogic("base_link")
    assert logic.on_odometry(_odom(10.0), now=10.02)[0] is not None
    twist, reason = logic.on_odometry(_odom(9.99), now=10.03)
    assert twist is None and "not newer" in reason
    # the strictly-newer rule: an identical stamp is also dropped
    twist, _ = logic.on_odometry(_odom(10.0), now=10.04)
    assert twist is None


def test_stale_and_future_stamps_are_dropped_by_the_window():
    logic = OdometryTwistRelayLogic("base_link", max_future_s=0.02, max_age_s=0.1)
    stale, reason = logic.on_odometry(_odom(9.5), now=10.0)
    assert stale is None and "outside" in reason
    future, reason = logic.on_odometry(_odom(10.05), now=10.0)
    assert future is None and "outside" in reason
    # the window boundaries themselves pass
    assert logic.on_odometry(_odom(9.92), now=10.0)[0] is not None


def test_backwards_clock_jump_starts_the_zero_stream():
    logic = OdometryTwistRelayLogic("base_link")
    assert logic.on_odometry(_odom(20.0), now=20.0)[0] is not None
    # the world resets: the CLOCK observation (not a message stamp) detects it
    assert logic.observe_clock(1.0) is True
    assert logic.resetting
    # the zero stream: zeros with the CURRENT stamp, not the old one
    zero = logic.zero_message(now=1.02)
    assert zero is not None
    assert zero.header.stamp.sec == 1 and abs(zero.header.stamp.nanosec - 0.02e9) < 1000
    assert zero.header.frame_id == "base_link"
    assert zero.twist.linear.x == 0.0 and zero.twist.angular.z == 0.0
    assert logic.zero_message(now=1.04) is not None and logic.zeros == 2
    # no zero once the epoch ends
    assert logic.on_odometry(_odom(1.05), now=1.06)[0] is not None
    assert not logic.resetting
    assert logic.zero_message(now=1.07) is None


def test_the_reset_zero_stream_survives_a_stale_backlog():
    logic = OdometryTwistRelayLogic("base_link")
    assert logic.on_odometry(_odom(20.0), now=20.0)[0] is not None
    assert logic.observe_clock(1.0) is True and logic.resetting
    # queued pre-reset odometry never passes the window on the new clock
    for stamp in (19.5, 19.9, 20.02):
        twist, reason = logic.on_odometry(_odom(stamp), now=1.1)
        assert twist is None and "stale across the reset" in reason
        assert logic.resetting
    # and the zero stream kept running
    assert logic.zero_message(now=1.12) is not None
    # the first fresh message on the new clock ends it
    assert logic.on_odometry(_odom(1.15), now=1.16)[0] is not None
    assert not logic.resetting


def test_a_clock_jump_without_any_odometry_still_zero_streams():
    """Review round 1: a dead odometry publisher after a reset must not leave
    the plugin's cached command driving the base — the clock observation
    alone starts the zero stream, and it keeps flowing indefinitely."""
    logic = OdometryTwistRelayLogic("base_link")
    assert logic.on_odometry(_odom(20.0), now=20.0)[0] is not None
    assert logic.observe_clock(1.0) is True
    # no odometry at all: one minute of zero-rate ticks, still all zeros
    ticks = 0
    now = 1.0
    while now < 61.0:
        logic.observe_clock(now)
        zero = logic.zero_message(now)
        assert zero is not None, f"the zero stream stopped at {now}"
        assert zero.twist.linear.x == 0.0 and zero.twist.angular.z == 0.0
        assert zero.header.stamp.sec == int(now)
        ticks += 1
        now += 0.02
    assert ticks == 3000 and logic.zeros == ticks


def test_a_legally_future_stamp_does_not_trigger_a_reset():
    """A stamp may sit up to ``max_future_s`` AHEAD of now; the old
    stamp-based comparison misread that as a clock jump."""
    logic = OdometryTwistRelayLogic("base_link", max_future_s=0.02, max_age_s=0.1)
    twist, _ = logic.on_odometry(_odom(10.01), now=10.0)  # 10 ms future: legal
    assert twist is not None
    assert logic.observe_clock(10.05) is False  # the clock only moved forward
    assert not logic.resetting
    assert logic.zero_message(now=10.06) is None


def test_dropped_messages_do_not_affect_the_reset_detection():
    """Wrong-frame and non-finite drops leave the clock observation intact."""
    logic = OdometryTwistRelayLogic("base_link")
    logic.observe_clock(10.0)
    assert logic.on_odometry(_odom(10.0, child="chassis"), now=10.01)[0] is None
    poisoned = _odom(10.02)
    poisoned.twist.twist.linear.x = float("nan")
    assert logic.on_odometry(poisoned, now=10.02)[0] is None
    assert logic.observe_clock(0.5) is True
    assert logic.resetting
    assert logic.zero_message(now=0.52) is not None


def test_the_ros_wrapper_zero_streams_after_a_clock_jump_without_odometry():
    """Wrapper level: ``use_sim_time``, a /clock publisher that jumps back and
    NO odometry at all — the relay still publishes the 50 Hz zero stream with
    the new epoch's stamps on its output topic."""
    import threading
    import time as wall_time

    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock

    from robot_runtime.odometry_twist_relay import OdometryTwistRelay

    arguments = [
        "--ros-args",
        "-p",
        "twist_source_topic:=/relay_jump_test/odom",
        "-p",
        "cmd_vel_topic:=/relay_jump_test/cmd",
        "-p",
        "base_frame:=base_link",
        "-p",
        "use_sim_time:=true",
        "-p",
        "zero_rate_hz:=50.0",
    ]
    rclpy.init(args=arguments)
    relay = None
    collector = None
    executor = None
    thread = None
    try:
        relay = OdometryTwistRelay()
        collector = Node("relay_jump_collector")
        received: list[TwistStamped] = []
        collector.create_subscription(TwistStamped, "/relay_jump_test/cmd", received.append, 50)
        clock_publisher = collector.create_publisher(Clock, "/clock", 50)
        executor = SingleThreadedExecutor()
        executor.add_node(relay)
        executor.add_node(collector)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        def advance(start: float, seconds: float) -> None:
            now = start
            deadline = wall_time.monotonic() + seconds
            while wall_time.monotonic() < deadline:
                message = Clock()
                message.clock.sec = int(now)
                message.clock.nanosec = int((now - int(now)) * 1e9)
                clock_publisher.publish(message)
                now += 0.02
                wall_time.sleep(0.02)

        advance(100.0, 0.4)  # the old epoch runs; nothing is published
        assert received == [], "zeros published before any reset"
        advance(5.0, 1.4)  # the clock jumps back to the new epoch
        assert len(received) >= 20, f"only {len(received)} zeros in 1.4 s"
        for message in received:
            stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
            assert 4.9 <= stamp <= 6.5, f"zero stamped in the wrong epoch: {stamp}"
            assert message.twist.linear.x == 0.0 and message.twist.angular.z == 0.0
            assert message.header.frame_id == "base_link"
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=5)
        if relay is not None:
            relay.destroy_node()
        if collector is not None:
            collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_initial_messages_without_a_last_stamp_still_need_freshness():
    logic = OdometryTwistRelayLogic("base_link")
    # the very first message is judged only by the window (no ordering yet)
    assert logic.on_odometry(_odom(9.0), now=10.0)[0] is None
    assert logic.on_odometry(_odom(9.95), now=10.0)[0] is not None
