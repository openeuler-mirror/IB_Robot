"""Test-support API of the runtime contract layer (openspec D14.1).

Everything under ``robot_runtime.testing`` exists so repository tests can
observe a running simulation the same way; none of it is part of any runtime
launch path. Importing a module here must start no process and create no ROS
entity (D4: nothing ground-truth on the graph).
"""
