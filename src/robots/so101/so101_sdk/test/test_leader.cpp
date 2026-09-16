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
#include <string>

#include "so101/leader.hpp"

namespace
{

class TempCalibFile
{
public:
  explicit TempCalibFile(const std::string & gripper_extra = "")
  {
    path_ = "/tmp/so101_sdk_leader_test_" + std::to_string(getpid()) + ".json";
    std::string json = "{";
    for (int joint = 1; joint <= 6; ++joint) {
      if (joint > 1) {
        json += ",";
      }
      json += "\"" + std::to_string(joint) +
              "\": {\"homing_offset\": 0, \"range_min\": 0, \"range_max\": 4095";
      if (joint == 6 && !gripper_extra.empty()) {
        json += gripper_extra;
      }
      json += "}";
    }
    json += "}";
    std::ofstream file(path_);
    file << json;
  }
  ~TempCalibFile() { std::remove(path_.c_str()); }
  const std::string & path() const { return path_; }

private:
  std::string path_;
};

so101::LeaderConfig simulated_leader_config(const std::string & calib_path)
{
  so101::LeaderConfig config;
  config.simulated = true;
  config.calibration_file = calib_path;
  config.calibration_version = 1;
  return config;
}

TEST(LeaderArm, ReadsTargetsInSiUnitsWithoutTouchingTorque)
{
  TempCalibFile calib;
  so101::LeaderArm leader(simulated_leader_config(calib.path()));
  ASSERT_TRUE(leader.connect());

  so101::ArmState state;
  ASSERT_TRUE(leader.read(state));
  ASSERT_EQ(state.joints.size(), 6U);
  // Default simulated position is the center tick: arm joints read zero.
  EXPECT_NEAR(state.joints.at("1").position, 0.0, 1e-9);
  // Connecting a leader never enables torque on any motor.
  EXPECT_FALSE(leader.sim().torque_enabled(1));
  EXPECT_FALSE(leader.sim().torque_enabled(6));
}

TEST(LeaderArm, GripperReadsNormalizedOpening)
{
  TempCalibFile calib;
  so101::LeaderArm leader(simulated_leader_config(calib.path()));
  ASSERT_TRUE(leader.connect());

  // Gripper calibration range is the full 0..4095 ticks: normalization
  // follows leader_arm.py semantics, norm = (raw - min) / (max - min).
  leader.sim().set_position_ticks(6, 2048);
  so101::ArmState state;
  ASSERT_TRUE(leader.read(state));
  EXPECT_NEAR(state.joints.at("6").position, 2048.0 / 4095.0, 1e-6);

  leader.sim().set_position_ticks(6, 0);
  ASSERT_TRUE(leader.read(state));
  EXPECT_NEAR(state.joints.at("6").position, 0.0, 1e-6);

  leader.sim().set_position_ticks(6, 4095);
  ASSERT_TRUE(leader.read(state));
  EXPECT_NEAR(state.joints.at("6").position, 1.0, 1e-6);
}

TEST(LeaderArm, DriveModeInvertsGripperNormalization)
{
  TempCalibFile calib(", \"drive_mode\": true");
  so101::LeaderArm leader(simulated_leader_config(calib.path()));
  ASSERT_TRUE(leader.connect());

  leader.sim().set_position_ticks(6, 0);
  so101::ArmState state;
  ASSERT_TRUE(leader.read(state));
  // With drive_mode set, the closed end of the travel reads as fully open.
  EXPECT_NEAR(state.joints.at("6").position, 1.0, 1e-6);
}

TEST(LeaderArm, ReadFailureIsExplicit)
{
  TempCalibFile calib;
  so101::LeaderArm leader(simulated_leader_config(calib.path()));
  ASSERT_TRUE(leader.connect());

  leader.sim().inject_sync_read_failure();
  so101::ArmState state;
  EXPECT_FALSE(leader.read(state));
  EXPECT_TRUE(state.joints.empty());
}

TEST(LeaderArm, RequiresExplicitCalibrationVersion)
{
  TempCalibFile calib;
  auto config = simulated_leader_config(calib.path());
  for (const int version : {0, 2}) {
    config.calibration_version = version;
    so101::LeaderArm leader(config);
    EXPECT_FALSE(leader.connect());
    so101::ArmState state;
    EXPECT_FALSE(leader.read(state));
    EXPECT_EQ(leader.health().fault, feetech::Fault::ConfigFailed);
  }
}

TEST(LeaderArm, IncompleteReadClearsPreviousSampleAndDisconnectCloses)
{
  TempCalibFile calib;
  so101::LeaderArm leader(simulated_leader_config(calib.path()));
  ASSERT_TRUE(leader.connect());
  leader.sim().set_write_ack(1, false);
  so101::ArmState state;
  ASSERT_TRUE(leader.read(state));
  leader.sim().set_responsive(6, false);
  EXPECT_FALSE(leader.read(state));
  EXPECT_TRUE(state.joints.empty());
  leader.sim().clear_injections();
  leader.disconnect();
  EXPECT_FALSE(leader.read(state));
  EXPECT_EQ(leader.health().lifecycle, so101::Lifecycle::Disconnected);
  EXPECT_THROW(leader.sim(), std::logic_error);
}

}  // namespace
