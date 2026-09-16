# Copyright 2026 IB_Robot Contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("so101_sdk_py", sys.argv.pop(1))
sdk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk)


class BindingsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "calibration.json"
        self.data = {str(i): {"homing_offset": 0, "range_min": 0, "range_max": 4095} for i in range(1, 7)}
        self.path.write_text(json.dumps(self.data))

    def leader_config(self):
        config = sdk.LeaderConfig()
        config.simulated = True
        config.calibration_file = str(self.path)
        config.calibration_version = 1
        return config

    def test_leader_complete_read_failure_disconnect_and_reconnect(self):
        leader = sdk.LeaderArm(self.leader_config())
        self.assertTrue(leader.connect())
        sim = leader.sim()
        sim.set_position_ticks(6, 4095)
        sim.set_write_ack(1, False)
        sample = leader.read()
        self.assertEqual(set(sample), set(self.data))
        self.assertEqual(sample["6"]["position"], 1.0)
        self.assertFalse(sim.torque_enabled(1))
        sim.set_responsive(6, False)
        self.assertIsNone(leader.read())
        leader.disconnect()
        self.assertIsNone(leader.read())
        self.assertFalse(sim.torque_enabled(1))  # stale inspection handle is inert
        self.assertTrue(leader.connect())
        self.assertEqual(len(leader.read()), 6)

    def test_leader_rejects_unprovisioned_firmware_and_undeclared_version(self):
        config = self.leader_config()
        config.calibration_version = 0
        self.assertFalse(sdk.LeaderArm(config).connect())
        self.data["2"]["homing_offset"] = 42
        self.path.write_text(json.dumps(self.data))
        leader = sdk.LeaderArm(self.leader_config())
        self.assertFalse(leader.connect())
        self.assertIn("provision", leader.health().detail)
        self.assertIsNone(leader.read())

    def test_follower_ownership_and_safe_activation(self):
        config = sdk.ArmConfig()
        config.simulated = True
        config.calibration_file = str(self.path)
        arm = sdk.Arm(config)
        self.assertFalse(arm.hold())
        self.assertTrue(arm.connect())
        sim = arm.sim()
        sim.set_position_ticks(1, 2700)
        self.assertTrue(arm.activate())
        self.assertTrue(arm.write_targets({"1": -1.0}))
        self.assertTrue(arm.stop(sdk.StopPolicy.TorqueOff))
        self.assertTrue(arm.activate())
        arm.read()
        self.assertEqual(sim.position_ticks(1), 2700)
        self.assertFalse(arm.write_targets({"1": float("nan")}))
        del arm
        gc.collect()
        self.assertFalse(sim.torque_enabled(1))


if __name__ == "__main__":
    unittest.main()
