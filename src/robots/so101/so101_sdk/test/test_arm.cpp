// Copyright 2026 IB_Robot Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <limits>
#include <string>

#include "feetech/conversion.hpp"
#include "so101/arm.hpp"

namespace
{

class TempCalibFile
{
public:
  TempCalibFile()
  {
    path_ = "/tmp/so101_sdk_arm_test_" + std::to_string(getpid()) + ".json";
    std::ofstream file(path_);
    file << R"({
      "1": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "2": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "3": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "4": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "5": {"homing_offset": 0, "range_min": 0, "range_max": 4095},
      "6": {"homing_offset": 0, "range_min": 0, "range_max": 4095}
    })";
  }
  ~TempCalibFile() { std::remove(path_.c_str()); }
  const std::string & path() const { return path_; }

private:
  std::string path_;
};

so101::ArmConfig simulated_config(const std::string & calib_path)
{
  so101::ArmConfig config;
  config.simulated = true;
  config.calibration_file = calib_path;
  return config;
}

TEST(Arm, ActivationWithResetPositionsSeedsConfiguredCommands)
{
  TempCalibFile calib;
  so101::ArmConfig config = simulated_config(calib.path());
  config.reset_positions = {{"1", 0.0}, {"2", -1.5}, {"3", 1.5}};

  so101::Arm arm(config);
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Activated);

  // The initial command for joint 2 equals the configured reset position
  // even though the simulated motor started at the center.
  so101::ArmState state;
  ASSERT_TRUE(arm.read(state));
  ASSERT_NE(state.joints.find("2"), state.joints.end());
  EXPECT_EQ(arm.command_targets().at("2"), -1.5);
  EXPECT_EQ(arm.command_targets().at("4"), 0.0);
}

TEST(Arm, ActivationAbortLeavesSafeState)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());

  // The initial feedback sync fails: activation must abort with every motor
  // torque-disabled (fail-closed), never commanding from uninitialized state.
  arm.sim().inject_sync_read_failure();
  EXPECT_FALSE(arm.activate());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Faulted);
  EXPECT_EQ(arm.health().fault, feetech::Fault::SyncReadFailed);

  so101::ArmState state;
  EXPECT_FALSE(arm.read(state));
  EXPECT_TRUE(state.joints.empty());
  for (std::uint8_t id = 1; id <= 6; ++id) {
    EXPECT_FALSE(arm.sim().torque_enabled(id));
  }
}

TEST(Arm, ReadFailureClearsStateNeverReturnsStaleAsFresh)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());

  so101::ArmState good;
  ASSERT_TRUE(arm.read(good));
  ASSERT_EQ(good.joints.size(), 6U);

  arm.sim().inject_sync_read_failure();
  so101::ArmState stale;
  EXPECT_FALSE(arm.read(stale));
  EXPECT_TRUE(stale.joints.empty());
}

TEST(Arm, WriteTargetsByNameAndRejectUnknownJoint)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());

  const std::map<std::string, double> targets = {{"1", 0.1}, {"2", -0.2}};
  EXPECT_TRUE(arm.write_targets(targets));

  const std::map<std::string, double> unknown = {{"9", 0.0}};
  EXPECT_FALSE(arm.write_targets(unknown));
  EXPECT_NE(arm.health().detail.find("unknown joint"), std::string::npos);
}

TEST(Arm, HoldReappliesLastTargets)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());

  const std::map<std::string, double> targets = {{"1", 0.3}};
  ASSERT_TRUE(arm.write_targets(targets));
  EXPECT_TRUE(arm.hold());
}

TEST(Arm, StopPolicies)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  EXPECT_TRUE(arm.sim().torque_enabled(1));
  EXPECT_TRUE(arm.stop(so101::StopPolicy::HoldLast));
  EXPECT_TRUE(arm.sim().torque_enabled(1));  // hold keeps torque
  EXPECT_TRUE(arm.stop(so101::StopPolicy::TorqueOff));
  EXPECT_FALSE(arm.sim().torque_enabled(1));  // torque released
}

TEST(Arm, TorqueOffStopReturnsToConnectedAndReactivates)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Activated);
  ASSERT_TRUE(arm.stop(so101::StopPolicy::TorqueOff));
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Connected);
  EXPECT_FALSE(arm.sim().torque_enabled(1));
  // The hardware lifecycle inactive->active transition re-enters activate()
  // without reconnecting; it must succeed and re-enable torque.
  ASSERT_TRUE(arm.activate());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Activated);
  EXPECT_TRUE(arm.sim().torque_enabled(1));
  so101::ArmState state;
  EXPECT_TRUE(arm.read(state));
}

TEST(Arm, DeactivateIsIdempotentAndClosesBus)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  EXPECT_TRUE(arm.deactivate());
  EXPECT_TRUE(arm.deactivate());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Disconnected);
  so101::ArmState state;
  EXPECT_FALSE(arm.read(state));
}

TEST(Arm, ConnectFailsOnMissingCalibration)
{
  so101::ArmConfig config = simulated_config("/tmp/so101_sdk_missing.json");
  so101::Arm arm(config);
  EXPECT_FALSE(arm.connect());
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Faulted);
  EXPECT_NE(arm.health().detail.find("so101_sdk_missing"), std::string::npos);
}

TEST(Arm, CalibratedRangesInRadians)
{
  TempCalibFile calib;
  so101::ArmConfig config = simulated_config(calib.path());
  config.joint_order = {"1", "2"};
  so101::Arm arm(config);
  ASSERT_TRUE(arm.connect());
  const auto ranges = arm.calibrated_ranges();
  ASSERT_EQ(ranges.size(), 2U);
  // range 0..4095 ticks maps to the full centered span.
  EXPECT_NEAR(ranges.at("1").first, feetech::ticks_to_radians(0), 1e-9);
  EXPECT_NEAR(ranges.at("1").second, feetech::ticks_to_radians(4095), 1e-9);
}

TEST(Arm, TwoArmsOnDistinctBusesAreIndependent)
{
  TempCalibFile calib;
  so101::Arm left(simulated_config(calib.path()));
  so101::Arm right(simulated_config(calib.path()));
  ASSERT_TRUE(left.connect());
  ASSERT_TRUE(right.connect());
  ASSERT_TRUE(left.activate());
  ASSERT_TRUE(right.activate());

  // Deactivating one arm must not affect the other's lifecycle or I/O.
  ASSERT_TRUE(left.deactivate());
  so101::ArmState state;
  EXPECT_FALSE(left.read(state));
  ASSERT_TRUE(right.read(state));
  EXPECT_EQ(state.joints.size(), 6U);
}

TEST(Arm, DefaultActivationHoldsMeasuredPoseIncludingAfterTorqueOff)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  EXPECT_FALSE(arm.hold());
  EXPECT_FALSE(arm.stop());
  ASSERT_TRUE(arm.connect());
  EXPECT_FALSE(arm.hold());
  arm.sim().set_position_ticks(1, 2600);
  ASSERT_TRUE(arm.activate());
  so101::ArmState state;
  ASSERT_TRUE(arm.read(state));
  EXPECT_EQ(arm.sim().position_ticks(1), 2600);
  ASSERT_TRUE(arm.write_targets({{"1", -1.0}}));
  ASSERT_TRUE(arm.stop(so101::StopPolicy::TorqueOff));
  EXPECT_FALSE(arm.write_targets({{"1", 0.0}}));
  EXPECT_FALSE(arm.hold());
  ASSERT_TRUE(arm.activate());
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(arm.read(state));
    EXPECT_EQ(arm.sim().position_ticks(1), 2600);
  }
}

TEST(Arm, PartialCommandsKeepOtherJointTargetsAndRejectNan)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  ASSERT_TRUE(arm.write_targets({{"2", 0.4}}));
  ASSERT_TRUE(arm.write_targets({{"1", 0.3}}));
  EXPECT_EQ(arm.command_targets().size(), 6U);
  EXPECT_EQ(arm.command_targets().at("2"), 0.4);
  EXPECT_FALSE(arm.write_targets({{"1", std::numeric_limits<double>::quiet_NaN()}}));
  EXPECT_EQ(arm.health().fault, feetech::Fault::WriteRejected);
}

TEST(Arm, TorqueReleaseFailureReportsFaultAndBlocksFurtherWrites)
{
  TempCalibFile calib;
  so101::Arm arm(simulated_config(calib.path()));
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  arm.sim().set_write_ack(2, false);
  EXPECT_FALSE(arm.stop(so101::StopPolicy::TorqueOff));
  EXPECT_EQ(arm.health().fault, feetech::Fault::EmergencyPartiallyFailed);
  EXPECT_EQ(arm.health().lifecycle, so101::Lifecycle::Faulted);
  EXPECT_FALSE(arm.hold());
  EXPECT_FALSE(arm.write_targets({{"1", 0.0}}));
  EXPECT_FALSE(arm.sim().torque_enabled(1));
  EXPECT_FALSE(arm.sim().torque_enabled(3));
  arm.sim().clear_injections();
}

TEST(Arm, SupportsExplicitMappedNamesAndRejectsInvalidIds)
{
  TempCalibFile calib;
  auto config = simulated_config(calib.path());
  config.joint_order = {"shoulder", "gripper"};
  config.motor_ids = {{"shoulder", 1}, {"gripper", 6}};
  so101::Arm arm(config);
  ASSERT_TRUE(arm.connect());
  ASSERT_TRUE(arm.activate());
  so101::ArmState state;
  ASSERT_TRUE(arm.read(state));
  EXPECT_EQ(state.joints.size(), 2U);
  EXPECT_TRUE(state.joints.count("gripper"));
  config.motor_ids["gripper"] = 1;
  EXPECT_THROW(so101::Arm invalid(config), std::invalid_argument);
  config.motor_ids.clear();
  config.joint_order = {"257"};
  EXPECT_THROW(so101::Arm invalid(config), std::invalid_argument);
  config.joint_order = {"1garbage"};
  EXPECT_THROW(so101::Arm invalid(config), std::invalid_argument);
}

}  // namespace
