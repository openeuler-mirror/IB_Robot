"""Run directly in the project environment; local scalar-filter microbenchmark."""

import cProfile
import logging
import pstats
import time

from robot_teleop.safety_filter import SafetyFilter


def main():
    logging.disable(logging.CRITICAL)
    safety = SafetyFilter({str(i): {"min": -1.0, "max": 1.0} for i in range(6)})
    targets = dict.fromkeys(safety.joint_limits, 2.0)
    count = 20_000
    wall = time.perf_counter()
    cpu = time.process_time()
    for _ in range(count):
        safety.apply_limits(targets)
    print(
        f"{count} six-joint calls: wall={(time.perf_counter() - wall) * 1e6 / count:.2f} us/call, "
        f"CPU={(time.process_time() - cpu) * 1e6 / count:.2f} us/call"
    )
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(2000):
        safety.apply_limits(targets)
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(12)


if __name__ == "__main__":
    main()
